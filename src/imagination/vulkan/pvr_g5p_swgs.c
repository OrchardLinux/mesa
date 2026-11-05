/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * G5P bridge for Mesa common poly GS lowering. Common lowering and its
 * copyright notices remain in src/poly; no AGX packets or helpers are copied.
 */
#include "pvr_g5p_swgs.h"
#include "compiler/nir/nir_builder.h"
#include "util/bitscan.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan_core.h>

/* Symbolic compiler-only regions, never GPU addresses. All memory operations
 * are eliminated or converted to descriptor-relative SSBO operations before
 * the A9 compiler sees the shader. No BDA support is implied.
 */
#define VP_TAG (UINT64_C(1) << 32)
#define INPUT_TAG (UINT64_C(2) << 32)
#define GP_TAG (UINT64_C(3) << 32)

struct bridge {
   struct pvr_g5p_swgs *out;
   struct poly_vertex_params vp;
   struct poly_geometry_params gp;
   bool vertex;
   bool failed;
};

static bool
refuse(struct pvr_g5p_swgs *out, const char *why)
{
   snprintf(out->refusal, sizeof(out->refusal), "G5P refuses: software geometry shader %s", why);
   return false;
}

static void
cleanup(nir_shader *s)
{
   bool progress;
   NIR_PASS(_, s, nir_lower_global_vars_to_local);
   do {
      progress = false;
      NIR_PASS(progress, s, nir_lower_var_copies);
      NIR_PASS(progress, s, nir_lower_vars_to_ssa);
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, nir_opt_cse);
      NIR_PASS(progress, s, nir_opt_dce);
      NIR_PASS(progress, s, nir_opt_dead_cf);
   } while (progress);
}

static void
prepare(nir_shader *s)
{
   NIR_PASS(_, s, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, s, nir_lower_returns);
   NIR_PASS(_, s, nir_inline_functions);
   nir_remove_non_entrypoints(s);
   NIR_PASS(_, s, nir_lower_system_values);
   NIR_PASS(_, s, nir_lower_global_vars_to_local);
   NIR_PASS(_, s, nir_split_var_copies);
   NIR_PASS(_, s, nir_lower_var_copies);
   NIR_PASS(_, s, nir_split_per_member_structs);
   cleanup(s);
   nir_lower_io_passes(s, false);
   cleanup(s);
   nir_shader_gather_info(s, nir_shader_get_entrypoint(s));
}

static bool
admit(nir_shader *s, struct pvr_g5p_swgs *out)
{
   if (s->xfb_info || s->info.has_transform_feedback_varyings)
      return refuse(out, "transform feedback is not qualified");
   if (s->info.outputs_written & ~(VARYING_BIT_POS | VARYING_BIT_PSIZ | BITFIELD64_BIT(VARYING_SLOT_VAR0)))
      return refuse(out, "output slot/layer/clip routing is not qualified");
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(instr);
            switch (i->intrinsic) {
            case nir_intrinsic_store_output:
            case nir_intrinsic_load_per_vertex_input:
            case nir_intrinsic_load_vertex_id:
            case nir_intrinsic_load_vertex_id_zero_base:
            case nir_intrinsic_load_instance_id:
            case nir_intrinsic_load_primitive_id:
            case nir_intrinsic_load_invocation_id:
            case nir_intrinsic_emit_vertex:
            case nir_intrinsic_end_primitive:
               break;
            default:
               snprintf(out->refusal, sizeof(out->refusal),
                        "G5P refuses: software geometry shader source intrinsic %s is not qualified",
                        nir_intrinsic_infos[i->intrinsic].name);
               return false;
            }
         }
      }
   }
   return true;
}

static bool
lower_params(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct bridge *s = data;
   nir_def *value = NULL;
   b->cursor = nir_before_instr(&i->instr);
   switch (i->intrinsic) {
   case nir_intrinsic_load_vertex_param_buffer_poly:
      value = nir_imm_int64(b, VP_TAG); break;
   case nir_intrinsic_load_geometry_param_buffer_poly:
      value = nir_imm_int64(b, GP_TAG); break;
   case nir_intrinsic_load_vertex_id:
   case nir_intrinsic_load_vertex_id_zero_base:
      value = nir_channel(b, nir_load_global_invocation_id(b, 32), 0); break;
   case nir_intrinsic_load_instance_id:
      value = nir_channel(b, nir_load_global_invocation_id(b, 32), 1); break;
   case nir_intrinsic_load_first_vertex:
   case nir_intrinsic_load_base_instance:
   case nir_intrinsic_load_provoking_last:
      value = nir_imm_int(b, 0); break;
   case nir_intrinsic_load_global_constant: {
      if (!nir_src_is_const(i->src[0]))
         return false;
      uint64_t address = nir_src_as_uint(i->src[0]);
      unsigned region = address >> 32, offset = (uint32_t)address;
      const void *buf;
      unsigned size;
      if (region == 1) { buf = &s->vp; size = sizeof(s->vp); }
      else if (region == 3) { buf = &s->gp; size = sizeof(s->gp); }
      else return false;
      unsigned bytes = i->def.bit_size / 8;
      if (i->def.num_components != 1 || (bytes != 4 && bytes != 8) ||
          offset > size || bytes > size - offset) {
         s->failed = true;
         return false;
      }
      uint64_t word = 0;
      memcpy(&word, (const char *)buf + offset, bytes);
      value = nir_imm_intN_t(b, word, i->def.bit_size);
      break;
   }
   default: return false;
   }
   nir_def_replace(&i->def, value);
   return true;
}

/* Recover only the pointer arithmetic emitted by poly, with one known region
 * base plus a 32-bit byte offset. Reject all other provenance. The admitted
 * point-input grids/strides are bounded below; descriptor-range checks guard
 * the final accesses as well. Truncating an arbitrary application pointer is
 * deliberately not an accepted route.
 */
static bool
address(nir_builder *b, nir_def *def, unsigned *region, nir_def **offset,
        unsigned depth)
{
   if (depth > 32 || def->num_components != 1 || def->bit_size != 64)
      return false;
   if (nir_def_is_const(def)) {
      uint64_t v = nir_src_as_uint(nir_src_for_ssa(def));
      *region = v >> 32;
      *offset = nir_imm_int(b, (uint32_t)v);
      return *region <= 3;
   }
   if (nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *a = nir_instr_as_alu(nir_def_instr(def));
   if (a->op == nir_op_u2u64 && a->src[0].src.ssa->bit_size == 32) {
      *region = 0;
      *offset = nir_channel(b, a->src[0].src.ssa, a->src[0].swizzle[0]);
      return true;
   }
   if (a->op != nir_op_iadd && a->op != nir_op_imul && a->op != nir_op_ishl)
      return false;
   unsigned r[2]; nir_def *o[2];
   if (!address(b, a->src[0].src.ssa, &r[0], &o[0], depth + 1))
      return false;
   if (a->op == nir_op_ishl) {
      if (r[0] || a->src[1].src.ssa->bit_size != 32)
         return false;
      *region = 0;
      *offset = nir_ishl(b, o[0], a->src[1].src.ssa);
      return true;
   }
   if (!address(b, a->src[1].src.ssa, &r[1], &o[1], depth + 1) ||
       (r[0] && r[1]) || (a->op == nir_op_imul && (r[0] || r[1])))
      return false;
   *region = r[0] | r[1];
   *offset = a->op == nir_op_iadd ? nir_iadd(b, o[0], o[1]) : nir_imul(b, o[0], o[1]);
   return true;
}

static nir_def *
binding(nir_builder *b, unsigned index)
{
   return nir_vulkan_resource_index(b, 2, 32, nir_imm_int(b, 0),
                                    .desc_set = 0, .binding = index,
                                    .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

static bool
lower_memory(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct bridge *s = data;
   bool load = i->intrinsic == nir_intrinsic_load_global_constant;
   bool store = i->intrinsic == nir_intrinsic_store_global;
   if (!load && !store && i->intrinsic != nir_intrinsic_store_output)
      return false;
   b->cursor = nir_before_instr(&i->instr);
   nir_def *offset;
   unsigned buffer, size, bytes;
   if (load || store) {
      unsigned region;
      if (!address(b, i->src[store ? 1 : 0].ssa, &region, &offset, 0) || region != 2 ||
          load == s->vertex) {
         s->failed = true;
         return false;
      }
      buffer = 0; size = s->out->input_size;
      bytes = load ? i->def.num_components * (i->def.bit_size / 8) :
                     i->src[0].ssa->num_components * (i->src[0].ssa->bit_size / 8);
   } else {
      nir_io_semantics sem = nir_intrinsic_io_semantics(i);
      if (s->vertex || !nir_src_is_const(i->src[1]) || nir_src_as_uint(i->src[1]) ||
          !(s->out->output_mask & BITFIELD64_BIT(sem.location))) {
         s->failed = true;
         return false;
      }
      unsigned slot = util_bitcount64(s->out->output_mask & BITFIELD64_MASK(sem.location));
      nir_def *gid = nir_load_global_invocation_id(b, 32);
      nir_def *linear = nir_iadd(b, nir_channel(b, gid, 0),
         nir_imul_imm(b, nir_channel(b, gid, 1), s->out->geometry_grid[0]));
      offset = nir_iadd_imm(b, nir_imul_imm(b, linear, s->out->output_stride),
                            slot * 16 + nir_intrinsic_component(i) * 4);
      buffer = 1; size = s->out->output_size;
      bytes = i->src[0].ssa->num_components * (i->src[0].ssa->bit_size / 8);
   }
   if (!bytes || bytes > size) { s->failed = true; return false; }
   nir_def *zero = load ? nir_imm_zero(b, i->def.num_components, i->def.bit_size) : NULL;
   nir_push_if(b, nir_ule_imm(b, offset, size - bytes));
   if (load) {
      nir_def *value = nir_load_ssbo(b, i->def.num_components, i->def.bit_size,
                                    binding(b, buffer), offset, .align_mul = 4,
                                    .access = ACCESS_NON_WRITEABLE);
      nir_pop_if(b, NULL);
      nir_def *v = nir_if_phi(b, value, zero);
      nir_def_replace(&i->def, v);
   } else {
      nir_def *value = i->src[0].ssa;
      nir_store_ssbo(b, value, binding(b, buffer), offset, .align_mul = 4,
                     .write_mask = nir_intrinsic_write_mask(i));
      nir_pop_if(b, NULL);
      nir_instr_remove(&i->instr);
   }
   return true;
}

static bool
finish(nir_shader *s, struct bridge *state)
{
   NIR_PASS(_, s, poly_nir_lower_sysvals);
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, s, nir_shader_intrinsics_pass, lower_params,
               nir_metadata_control_flow, state);
      cleanup(s);
   } while (progress);
   NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_memory,
            nir_metadata_control_flow, state);
   if (state->failed)
      return refuse(state->out, "poly memory provenance/layout is not qualified");
   cleanup(s);
   /* Poly's output-ID quotient/remainder uses constant divisors. Reuse NIR's
    * exact integer lowering, including high multiply expanded into 32-bit
    * partial products. Do not approximate primitive IDs with float division.
    */
   NIR_PASS(_, s, nir_opt_idiv_const, 32);
   const nir_shader_compiler_options *saved_options = s->options;
   nir_shader_compiler_options options = *saved_options;
   options.lower_mul_high = true;
   s->options = &options;
   NIR_PASS(_, s, nir_lower_alu);
   cleanup(s);
   s->options = saved_options;
   s->info.stage = MESA_SHADER_COMPUTE;
   memset(&s->info.cs, 0, sizeof(s->info.cs));
   s->info.workgroup_size[0] = s->info.workgroup_size[1] = s->info.workgroup_size[2] = 1;
   s->info.workgroup_size_variable = false;
   s->info.inputs_read = s->info.outputs_written = 0;
   nir_shader_gather_info(s, nir_shader_get_entrypoint(s));
   nir_validate_shader(s, "G5P poly to SSBO compute");
   return true;
}

bool
pvr_g5p_swgs_lower(const nir_shader *vs_, const nir_shader *gs_,
                   unsigned vertices, unsigned instances,
                   void *mem_ctx, struct pvr_g5p_swgs *out)
{
   memset(out, 0, sizeof(*out));
   if (vs_->info.stage != MESA_SHADER_VERTEX || gs_->info.stage != MESA_SHADER_GEOMETRY)
      return refuse(out, "stage pair must be vertex plus geometry");
   if (!vertices || vertices > 64 || !instances || instances > 64)
      return refuse(out, "prototype grid exceeds 64 vertices/instances");
   nir_shader *vs = nir_shader_clone(mem_ctx, vs_);
   nir_shader *gs = nir_shader_clone(mem_ctx, gs_);
   prepare(vs); prepare(gs);
   if (gs->info.gs.input_primitive != MESA_PRIM_POINTS || gs->info.gs.invocations != 1 ||
       gs->info.gs.active_stream_mask != 1)
      return refuse(out, "prototype requires point input, one invocation and stream 0");
   if (!admit(vs, out) || !admit(gs, out)) return false;
   out->input_mask = vs->info.outputs_written;
   out->input_stride = util_bitcount64(out->input_mask) * 16;
   if ((gs->info.inputs_read & out->input_mask) != gs->info.inputs_read)
      return refuse(out, "input slots are not written by the vertex shader");
   out->input_size = vertices * instances * out->input_stride;
   out->vertex_grid[0] = vertices; out->vertex_grid[1] = instances; out->vertex_grid[2] = 1;
   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   poly_nir_lower_gs(gs, &count, &rast, &pre, &out->topology);
   ralloc_free(count); ralloc_free(pre); ralloc_free(gs);
   if (out->topology.xfb || out->topology.multistream || out->topology.count_words ||
       (out->topology.shape != POLY_GS_SHAPE_STATIC_PER_INSTANCE &&
        out->topology.shape != POLY_GS_SHAPE_STATIC_PER_PRIM) ||
       !out->topology.max_indices || out->topology.max_indices > 64) {
      ralloc_free(rast);
      return refuse(out, "dynamic/indexed/empty output topology is not qualified");
   }
   ralloc_steal(mem_ctx, rast);
   out->output_mask = rast->info.outputs_written;
   out->output_stride = util_bitcount64(out->output_mask) * 16;
   out->geometry_grid[0] = poly_gs_rast_vertices(out->topology.shape, out->topology.max_indices, vertices, instances);
   out->geometry_grid[1] = poly_gs_rast_instances(out->topology.shape, out->topology.max_indices, vertices, instances);
   out->geometry_grid[2] = 1;
   out->output_size = out->geometry_grid[0] * out->geometry_grid[1] * out->output_stride;
   struct bridge state = { .out = out };
   state.vp.output_buffer = INPUT_TAG;
   state.vp.outputs = out->input_mask;
   state.vp.verts_per_instance = state.vp.grid[0] = vertices;
   state.vp.grid[1] = instances;
   state.gp.grid[0] = vertices;
   state.gp.grid[1] = instances;
   state.gp.input_topology = MESA_PRIM_POINTS;
   NIR_PASS(_, vs, poly_nir_lower_vs_before_gs);
   NIR_PASS(_, vs, poly_nir_lower_sw_vs);
   state.vertex = true;
   if (!finish(vs, &state)) return false;
   state.vertex = false;
   if (!finish(rast, &state)) return false;
   out->vertex = vs; out->geometry = rast;
   return true;
}
