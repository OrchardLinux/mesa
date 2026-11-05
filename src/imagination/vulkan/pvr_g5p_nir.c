/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P, has_apple_g5p): SPIR-V to NIR for the G5P backend
 * (src/imagination/g5p), as the A9 ICD prepares its graphics stages
 * (g5p_nir.c of the route-b tree: g5p_nir(), graphics_nir(), vs_sysvals()),
 * then pco_g5p_compile_general_link(). The backend's recognised forms depend
 * on this pass sequence, so it is the ICD's, not vk_pipeline's. There is no
 * captured-shape matcher and no out-of-process fallback here: the general
 * backend only. Compiled only with -Dimagination-g5p-experimental=true.
 */

#include <stdio.h>
#include <string.h>

#include "compiler/glsl_types.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_xfb_info.h"
#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv.h"
#include "compiler/spirv/spirv_info.h"
#include "util/u_call_once.h"
#include "util/u_math.h"
#include "vk_shader_module.h"
#include "vk_util.h"

#include "g5p_mrt_out.h"
#include "pco/pco.h"
#include "pco_g5p_general.h"
#include "pvr_g5p.h"
#include "pvr_g5p_vertex.h"

/* nir->options must outlive the shader (the general backend runs NIR passes
 * after the preparation returns), so the options are static. */
static nir_shader_compiler_options g5p_options;

static void
g5p_init_options(void)
{
   g5p_options = *pco_nir_options();
   /* G5P has a measured native FMA; do not inherit PCO's decomposition. */
   g5p_options.float_mul_add32 = nir_float_muladd_support_has_ffma;
   /* Selects are compare/write + inverse MOV pairs (no fused fcsel_*), and
    * fsqrt/fmin/fmax have measured complete lowerings. */
   g5p_options.has_fused_comp_and_csel = false;
   g5p_options.lower_fsqrt = false;
   g5p_options.lower_fminmax_signed_zero = false;
   /* Float results of compares become b2f32 of the compare; integer abs/sign
    * become imax/imin; the multiply-high of nir_lower_idiv becomes 16-bit
    * partial products; extract_u8/u16 stay shifts and masks. Every op they
    * produce has a measured encoding. */
   g5p_options.lower_scmp = true;
   g5p_options.lower_iabs = true;
   g5p_options.lower_isign = true;
   g5p_options.lower_mul_high = true;
   g5p_options.lower_extract_byte = true;
   g5p_options.lower_extract_word = true;
}

/*
 * Zink lowers a GL uniform block to an array of uints, so a vec4 element of
 * a uniform array indexed by a register arrives as four 32-bit loads at
 * (i << 4) + c, c + 4, c + 8 and c + 12. Apple's compiler reads such an
 * element with one 16-byte load (v_cexpr: vec4 at shift 4); the four are
 * merged into that form. Only complete, 16-byte aligned quads in one block
 * with a shift of 4 are merged, so every other load keeps its shape.
 */

/* A term of a UBO offset the backend reads as a constant: an immediate, or
 * the descriptor's offset component, which is 0. */
static bool
ubo_offset_const(nir_scalar s, uint32_t *v)
{
   if (nir_scalar_is_const(s)) {
      *v = nir_scalar_as_uint(s);
      return true;
   }
   if (nir_scalar_is_intrinsic(s) && nir_scalar_intrinsic_op(s) == nir_intrinsic_load_vulkan_descriptor &&
       s.comp == 1) {
      *v = 0;
      return true;
   }
   return false;
}

static bool
dyn_ubo_split(nir_intrinsic_instr *in, nir_scalar *x, uint32_t *c)
{
   if (in->intrinsic != nir_intrinsic_load_ubo || in->def.num_components != 1 || in->def.bit_size != 32)
      return false;
   nir_scalar s = nir_get_scalar(in->src[1].ssa, 0);
   *c = 0;
   for (unsigned guard = 0; guard < 8 && nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd; guard++) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0), b = nir_scalar_chase_alu_src(s, 1);
      uint32_t v;
      if (ubo_offset_const(b, &v)) {
         *c += v;
         s = a;
      } else if (ubo_offset_const(a, &v)) {
         *c += v;
         s = b;
      } else {
         return false;
      }
   }
   if (!nir_scalar_is_alu(s))
      return false;
   nir_op op = nir_scalar_alu_op(s);
   if (op != nir_op_ishl && op != nir_op_imul)
      return false;
   nir_scalar k = nir_scalar_chase_alu_src(s, 1);
   if (!nir_scalar_is_const(k) || nir_scalar_as_uint(k) != (op == nir_op_ishl ? 4u : 16u))
      return false;
   *x = s;
   return true;
}

static bool
vec4_dyn_ubo(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_function_impl (impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block (block, impl) {
         bool again;
         do {
            again = false;
            nir_foreach_instr (instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *lo = nir_instr_as_intrinsic(instr), *q[4] = { lo };
               nir_scalar x, y;
               uint32_t c, d;
               if (!dyn_ubo_split(lo, &x, &c) || c % 16)
                  continue;
               unsigned found = 1;
               for (nir_instr *i = nir_instr_next(instr); i && found < 4; i = nir_instr_next(i)) {
                  if (i->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_instr *in = nir_instr_as_intrinsic(i);
                  if (!dyn_ubo_split(in, &y, &d) || in->src[0].ssa != lo->src[0].ssa || !nir_scalar_equal(x, y) ||
                      d < c + 4 || d > c + 12 || (d - c) % 4 || q[(d - c) / 4])
                     continue;
                  q[(d - c) / 4] = in;
                  found++;
               }
               if (found < 4)
                  continue;
               b.cursor = nir_before_instr(instr);
               /* the first component's offset, (i << 4) + c: vec4 at shift 4 */
               nir_def *v = nir_load_ubo(&b, 4, 32, lo->src[0].ssa, lo->src[1].ssa,
                                         .access = nir_intrinsic_access(lo), .align_mul = 16, .align_offset = 0,
                                         .range_base = 0, .range = ~0u);
               for (unsigned j = 0; j < 4; j++) {
                  nir_def_rewrite_uses(&q[j]->def, nir_channel(&b, v, j));
                  if (j)
                     nir_instr_remove(&q[j]->instr);
               }
               nir_instr_remove(instr);
               progress = again = true;
               break; /* the block changed under the iterator */
            }
         } while (again);
      }
      nir_progress(progress, impl, nir_metadata_control_flow);
   }
   return progress;
}

/*
 * Fragment code (unless PVR_G5P_FS_DYN_UBO=0; the backend refuses any
 * fragment index this does not bound): each register-indexed UBO index is
 * clamped to the block, umin(i, N) with N the last element whose bytes lie
 * inside the UBO variable (robust-access semantics). A negative index taken
 * by a helper lane would otherwise read 4 GiB past the pointer.
 */
static bool
ubo_block_bytes(nir_shader *nir, nir_def *d, unsigned *bytes)
{
   for (unsigned guard = 0; guard < 8; guard++) {
      nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(d, 0));
      if (!nir_scalar_is_intrinsic(s))
         return false;
      nir_intrinsic_instr *in = nir_def_as_intrinsic(s.def);
      if (in->intrinsic == nir_intrinsic_load_vulkan_descriptor) {
         d = in->src[0].ssa;
         continue;
      }
      if (in->intrinsic != nir_intrinsic_vulkan_resource_index || !nir_src_is_const(in->src[0]) ||
          nir_src_as_uint(in->src[0]))
         return false;
      nir_foreach_variable_with_modes (var, nir, nir_var_mem_ubo) {
         if (var->data.descriptor_set != nir_intrinsic_desc_set(in) ||
             var->data.binding != nir_intrinsic_binding(in))
            continue;
         /* Zink declares each block as an array of one; the index is 0 (checked above) */
         *bytes = glsl_get_explicit_size(glsl_without_array(var->type), false);
         return *bytes != 0;
      }
      return false;
   }
   return false;
}

static bool
clamp_dyn_ubo(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_function_impl (impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block (block, impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if (in->intrinsic != nir_intrinsic_load_ubo || in->def.bit_size != 32)
               continue;
            /* offset = shift(x) + c, the shape the backend folds */
            nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(in->src[1].ssa, 0));
            uint32_t c = 0, v;
            bool ok = true;
            for (unsigned guard = 0; guard < 8 && nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd;
                 guard++) {
               nir_scalar a0 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
               nir_scalar a1 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
               if (ubo_offset_const(a1, &v)) {
                  c += v;
                  s = a0;
               } else if (ubo_offset_const(a0, &v)) {
                  c += v;
                  s = a1;
               } else {
                  ok = false;
                  break;
               }
            }
            if (!ok || ubo_offset_const(s, &v))
               continue; /* a constant offset (the descriptor term reads 0) */
            unsigned shift = 0;
            nir_scalar x = s;
            if (nir_scalar_is_alu(s) && (nir_scalar_alu_op(s) == nir_op_ishl || nir_scalar_alu_op(s) == nir_op_imul)) {
               nir_scalar k = nir_scalar_chase_alu_src(s, 1);
               if (!nir_scalar_is_const(k))
                  continue;
               uint32_t kv = nir_scalar_as_uint(k);
               if (nir_scalar_alu_op(s) == nir_op_imul) {
                  if (!util_is_power_of_two_nonzero(kv))
                     continue;
                  kv = util_logbase2(kv);
               }
               if (kv > 31)
                  continue;
               shift = kv;
               x = nir_scalar_chase_alu_src(s, 0);
            }
            unsigned size, bytes = in->def.num_components * 4;
            if (!ubo_block_bytes(nir, in->src[0].ssa, &size) || c > size || size - c < bytes)
               continue; /* c may be a folded negative */
            uint32_t last = (size - c - bytes) >> shift;
            b.cursor = nir_before_instr(instr);
            nir_def *i = nir_umin(&b, nir_channel(&b, x.def, x.comp), nir_imm_int(&b, last));
            nir_def *off = nir_iadd_imm(&b, nir_ishl_imm(&b, i, shift), c);
            nir_src_rewrite(&in->src[1], off);
            progress = true;
         }
      }
      nir_progress(progress, impl, nir_metadata_control_flow);
   }
   return progress;
}

static nir_shader *g5p_nir_prepare(nir_shader *nir);

/* SPIR-V -> optimised NIR (the ICD's g5p_nir()) */
static nir_shader *
g5p_nir(const uint32_t *spv, size_t words, mesa_shader_stage stage, const char *entry)
{
   static util_once_flag once = UTIL_ONCE_FLAG_INIT;
   util_call_once(&once, g5p_init_options);
   struct spirv_capabilities caps = { .Shader = true, .Float16 = true, .Int16 = true, .Int8 = true,
                                      .Int64 = true, .Float64 = true, .TransformFeedback = true };
   struct spirv_to_nir_options opts = {
      .environment = NIR_SPIRV_VULKAN,
      .capabilities = &caps,
      .ubo_addr_format = nir_address_format_32bit_index_offset,
      .ssbo_addr_format = nir_address_format_32bit_index_offset,
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .push_const_addr_format = nir_address_format_32bit_offset,
      .shared_addr_format = nir_address_format_32bit_offset,
      .temp_addr_format = nir_address_format_32bit_offset,
      .skip_os_break_in_debug_build = true,
   };
   opts.workarounds.force_exact_glsl_fma = true;
   nir_shader *nir = spirv_to_nir(spv, words, NULL, stage, entry, &opts, &g5p_options);
   if (!nir)
      return NULL;
   return g5p_nir_prepare(nir);
}

/* The ICD's passes after spirv_to_nir(), for any stage's NIR */
static nir_shader *
g5p_nir_prepare(nir_shader *nir)
{
   const mesa_shader_stage stage = nir->info.stage;
   nir_lower_variable_initializers(nir, nir_var_function_temp);
   nir_lower_returns(nir);
   nir_inline_functions(nir);
   nir_remove_non_entrypoints(nir);
   /* gl_PerVertex: one variable per member, each at its builtin's slot, as
    * the Vulkan runtime does (vk_nir.c); without it a member keeps its index
    * as its location (glslang's gl_PointSize became slot 1, COL0) */
   nir_split_var_copies(nir);
   nir_split_per_member_structs(nir);
   nir_lower_vars_to_ssa(nir);
   nir_opt_deref(nir);
   nir_lower_system_values(nir);
   /* XFB decorations live on the variables: gather them before IO lowering,
    * which then attaches them to the store_output intrinsics. */
   if (nir->info.has_transform_feedback_varyings)
      nir_shader_gather_xfb_info(nir);
   nir_lower_io_passes(nir, false);
   nir_lower_explicit_io(nir, nir_var_mem_ssbo | nir_var_mem_ubo, nir_address_format_32bit_index_offset);
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
   } while (progress);
   progress = false;
   NIR_PASS(progress, nir, vec4_dyn_ubo);
   if (stage == MESA_SHADER_FRAGMENT && pco_g5p_fs_dyn_ubo())
      NIR_PASS(progress, nir, clamp_dyn_ubo);
   if (progress) {
      NIR_PASS(_, nir, nir_opt_copy_prop);
      NIR_PASS(_, nir, nir_opt_dce);
   }
   nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
   return nir;
}

/* Outputs at locations DATA0+N..DATA7 never have an attachment and are
 * discarded (Zink broadcasts gl_FragColor to all 8). */
static bool
drop_unattached_output(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   if (in->intrinsic != nir_intrinsic_store_output)
      return false;
   unsigned loc = nir_intrinsic_io_semantics(in).location;
   if (loc < FRAG_RESULT_DATA0 || loc > FRAG_RESULT_DATA7 ||
       g5p_mrt_slot_is_attached(loc - FRAG_RESULT_DATA0, *(const unsigned *)data))
      return false;
   nir_instr_remove(&in->instr);
   return true;
}

/* SampleMask at one sample: gl_SampleMaskIn is 1, and a written mask without
 * bit 0 removes the fragment's only sample, a demote (the discard path's
 * rules apply). */
static bool
lower_sample_mask_1x(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   bool *used = data;
   b->cursor = nir_before_instr(&in->instr);
   if (in->intrinsic == nir_intrinsic_load_sample_mask_in) {
      *used = true;
      nir_def_replace(&in->def, nir_imm_int(b, 1));
      return true;
   }
   if (in->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(in).location != FRAG_RESULT_SAMPLE_MASK)
      return false;
   *used = true;
   nir_def *mask = nir_channel(b, in->src[0].ssa, 0);
   nir_demote_if(b, nir_ieq_imm(b, nir_iand_imm(b, mask, 1), 0));
   nir_instr_remove(&in->instr);
   return true;
}

/* gl_PointSize has no effect outside point pipelines (polygon mode FILL
 * only): its store is dropped unless transform feedback captures it. A
 * point pipeline keeps it (the TA reads it after the position), and gets
 * 1.0 when the program writes none (maintenance5's default). */
static bool
drop_point_size(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   if (in->intrinsic != nir_intrinsic_store_output || nir_intrinsic_io_semantics(in).location != VARYING_SLOT_PSIZ)
      return false;
   nir_instr_remove(&in->instr);
   return true;
}

/* The program stores gl_PointSize (its intrinsics: outputs_written is not
 * kept up to date by the passes before) */
static bool
find_point_size(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   if (in->intrinsic == nir_intrinsic_store_output && nir_intrinsic_io_semantics(in).location == VARYING_SLOT_PSIZ)
      *(bool *)data = true;
   return false;
}

static bool
xfb_captures(const nir_shader *nir, gl_varying_slot slot)
{
   if (!nir->xfb_info)
      return false;
   for (unsigned i = 0; i < nir->xfb_info->output_count; i++)
      if (nir->xfb_info->outputs[i].location == slot)
         return true;
   return false;
}

/* The graphics lowerings of a prepared stage (the ICD's graphics_nir()) */
static nir_shader *
graphics_lower(nir_shader *nir, unsigned color_targets, bool *sample_mask, bool point_size, bool *psize_dropped)
{
   const mesa_shader_stage stage = nir ? nir->info.stage : MESA_SHADER_NONE;

   if (nir && stage == MESA_SHADER_VERTEX && !point_size && !xfb_captures(nir, VARYING_SLOT_PSIZ) &&
       nir_shader_intrinsics_pass(nir, drop_point_size, nir_metadata_control_flow, NULL)) {
      /* polygon mode POINT needs to know (pco_g5p_program::psize_dropped) */
      if (psize_dropped)
         *psize_dropped = true;
      nir->info.outputs_written &= ~BITFIELD64_BIT(VARYING_SLOT_PSIZ);
      nir_opt_dce(nir);
      nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
   }
   bool writes_psize = false;
   if (nir && stage == MESA_SHADER_VERTEX && point_size)
      nir_shader_intrinsics_pass(nir, find_point_size, nir_metadata_all, &writes_psize);
   if (nir && stage == MESA_SHADER_VERTEX && point_size && !writes_psize) {
      nir_function_impl *impl = nir_shader_get_entrypoint(nir);
      nir_builder b = nir_builder_at(nir_after_impl(impl));
      nir_store_output(&b, nir_imm_float(&b, 1.0f), nir_imm_int(&b, 0), .base = 0, .component = 0,
                       .write_mask = 1, .src_type = nir_type_float32,
                       .io_semantics = (nir_io_semantics){ .location = VARYING_SLOT_PSIZ, .num_slots = 1 });
      nir->info.outputs_written |= BITFIELD64_BIT(VARYING_SLOT_PSIZ);
      nir_index_ssa_defs(impl);
   }
   if (nir && stage == MESA_SHADER_FRAGMENT &&
       nir_shader_intrinsics_pass(nir, drop_unattached_output, nir_metadata_control_flow, &color_targets)) {
      nir_opt_dce(nir);
      nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
   }
   if (nir && stage == MESA_SHADER_FRAGMENT &&
       nir_shader_intrinsics_pass(nir, lower_sample_mask_1x, nir_metadata_control_flow, sample_mask)) {
      nir->info.outputs_written &= ~BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK);
      BITSET_CLEAR(nir->info.system_values_read, SYSTEM_VALUE_SAMPLE_MASK_IN);
      nir_opt_constant_folding(nir);
      nir_opt_algebraic(nir);
      nir_opt_dce(nir);
      nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
   }
   return nir;
}

static nir_shader *
graphics_nir(const uint32_t *spv, size_t words, mesa_shader_stage stage, const char *entry, unsigned color_targets,
             bool *sample_mask, bool point_size, bool *psize_dropped)
{
   return graphics_lower(g5p_nir(spv, words, stage, entry), color_targets, sample_mask, point_size, psize_dropped);
}

/* Vertex system values: the hardware index the PDS writes is the draw's
 * first vertex + i, which is gl_VertexIndex; the base instance is 0, as the
 * instance index already holds it (the fetch program adds the first instance
 * to t1, Apple's g3b-basevertex form). */
static bool
lower_vs_sysvals(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   b->cursor = nir_before_instr(&in->instr);
   switch (in->intrinsic) {
   case nir_intrinsic_load_raw_vertex_id:
      nir_def_replace(&in->def, nir_load_vertex_id(b));
      return true;
   case nir_intrinsic_load_raw_vertex_offset:
   case nir_intrinsic_load_base_instance:
      nir_def_replace(&in->def, nir_imm_int(b, 0));
      return true;
   default:
      return false;
   }
}

static void
vs_sysvals(nir_shader *nir)
{
   if (nir_shader_intrinsics_pass(nir, lower_vs_sysvals, nir_metadata_control_flow, NULL)) {
      nir_opt_constant_folding(nir);
      nir_opt_algebraic(nir);
      nir_opt_dce(nir);
   }
   nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
}

/* The SPIR-V module declares ExecutionMode Xfb (execution modes precede
 * every function) */
static bool
spirv_xfb(const uint32_t *spv, size_t words)
{
   for (size_t i = 5; i < words;) {
      const unsigned op = spv[i] & 0xffff, n = spv[i] >> 16;
      if (!n)
         return false;
      if (op == SpvOpExecutionMode && n >= 3 && i + 2 < words && spv[i + 2] == SpvExecutionModeXfb)
         return true;
      if (op == SpvOpFunction)
         return false;
      i += n;
   }
   return false;
}

/* The transform feedback variant (xfb != NULL): the capture layout from the
 * shader's xfb_info, then nir_lower_xfb_to_stores (64-bit global addresses,
 * outputs kept, so the same draw rasterizes). The backend turns those stores
 * into 4-byte vertex stores at the per-draw block's addresses
 * (PCO_G5P_UPLOAD_XFB). The A9 ICD's pvr_g5p_general_xfb(). */
static bool
xfb_lower(nir_shader *nir, struct g5p_xfb_layout *layout, char *refusal, size_t refusal_size)
{
   const nir_xfb_info *x = nir->xfb_info;
   memset(layout, 0, sizeof(*layout));
   if (!x || !x->buffers_written)
      return true; /* nothing captured: no variant */
   if (x->output_count > G5P_XFB_MAX_RANGES || x->streams_written > 1) {
      snprintf(refusal, refusal_size, "transform feedback: %u outputs (at most %u), streams %#x (stream 0 only)",
               x->output_count, G5P_XFB_MAX_RANGES, x->streams_written);
      return false;
   }
   layout->buffers = x->buffers_written;
   for (unsigned b = 0; b < G5P_XFB_BUFFERS; b++)
      layout->stride[b] = x->buffers[b].stride;
   for (unsigned i = 0; i < x->output_count; i++) {
      const nir_xfb_output_info *o = &x->outputs[i];
      if (o->data_is_16bit) {
         snprintf(refusal, refusal_size, "transform feedback of 16-bit data");
         return false;
      }
      const unsigned r = layout->nranges++;
      layout->ranges[r].buffer = o->buffer;
      layout->ranges[r].offset = o->offset;
      layout->ranges[r].size = 4 * util_bitcount(o->component_mask);
   }
   const nir_lower_xfb_to_stores_options opts = { .address_format = nir_address_format_64bit_global,
                                                  .keep_outputs = true };
   NIR_PASS(_, nir, nir_lower_xfb_to_stores, &opts);
   return true;
}

static bool
compile_stage(const VkPipelineShaderStageCreateInfo *stage_info, bool half_output, uint32_t samples,
              struct pvr_g5p_vertex_layout *vertex, const struct pco_g5p_fs_key *key,
              const struct pco_g5p_vs_link *link, struct g5p_xfb_layout *xfb, struct pco_g5p_program *out)
{
   memset(out, 0, sizeof(*out));

   const mesa_shader_stage stage = stage_info->stage == VK_SHADER_STAGE_VERTEX_BIT     ? MESA_SHADER_VERTEX
                                   : stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT ? MESA_SHADER_FRAGMENT
                                                                                       : MESA_SHADER_NONE;
   if (stage == MESA_SHADER_NONE) {
      snprintf(out->refusal, sizeof(out->refusal), "a %#x stage (vertex and fragment only)", stage_info->stage);
      return false;
   }
   if (stage_info->pSpecializationInfo && stage_info->pSpecializationInfo->mapEntryCount) {
      snprintf(out->refusal, sizeof(out->refusal), "specialization constants");
      return false;
   }

   /* the SPIR-V words: the module's, or an inline module (maintenance5); or the driver's own NIR (an internal
    * module, vk_shader_module_handle_from_nir(): the clear quad's stages) */
   VK_FROM_HANDLE(vk_shader_module, module, stage_info->module);
   const VkShaderModuleCreateInfo *inline_ci = vk_find_struct_const(stage_info->pNext, SHADER_MODULE_CREATE_INFO);
   const nir_shader *internal = module ? module->nir : NULL;
   const uint32_t *spv = NULL;
   size_t bytes = 0;
   if (module && !module->nir) {
      spv = (const uint32_t *)module->data;
      bytes = module->size;
   } else if (!internal && inline_ci) {
      spv = inline_ci->pCode;
      bytes = inline_ci->codeSize;
   } else if (!internal) {
      snprintf(out->refusal, sizeof(out->refusal), "a shader stage without SPIR-V");
      return false;
   }
   if (!internal && (bytes < 20 || bytes % 4)) {
      snprintf(out->refusal, sizeof(out->refusal), "SPIR-V of %zu bytes", bytes);
      return false;
   }
   if (internal && internal->info.stage != stage) {
      snprintf(out->refusal, sizeof(out->refusal), "an internal module of another stage");
      return false;
   }
   if (xfb && (stage != MESA_SHADER_VERTEX || internal || !spirv_xfb(spv, bytes / 4))) {
      memset(xfb, 0, sizeof(*xfb)); /* no capture: no variant */
      return true;
   }

   glsl_type_singleton_init_or_ref();
   const char *entry = stage_info->pName ? stage_info->pName : "main";
   const unsigned color_targets = key && key->color_targets ? key->color_targets : 1;
   bool sample_mask = false;
   const bool point_size = stage == MESA_SHADER_VERTEX && link && link->point_size;
   bool psize_dropped = false;
   nir_shader *nir = internal ? graphics_lower(g5p_nir_prepare(nir_shader_clone(NULL, internal)), color_targets,
                                               &sample_mask, point_size, &psize_dropped)
                              : graphics_nir(spv, bytes / 4, stage, entry, color_targets, &sample_mask, point_size,
                                             &psize_dropped);
   bool ok = false;
   if (!nir) {
      snprintf(out->refusal, sizeof(out->refusal), "SPIR-V -> NIR failed");
   } else if (sample_mask && samples > 1) {
      /* the lowering above is the one-sample rule; at 4x the mask needs the program's coverage feedback */
      snprintf(out->refusal, sizeof(out->refusal), "gl_SampleMask or gl_SampleMaskIn at %u samples "
               "(the coverage feedback is not built)", samples);
   } else if (stage == MESA_SHADER_VERTEX && vertex && !pvr_g5p_vertex_lower(nir, vertex)) {
      /* the vertex input layout (pvr_g5p_vertex_nir.c) */
      snprintf(out->refusal, sizeof(out->refusal), "%s", vertex->refusal);
   } else if (xfb && !xfb_lower(nir, xfb, out->refusal, sizeof(out->refusal))) {
      /* refused by name */
   } else if (xfb && !xfb->buffers) {
      ok = true; /* Xfb mode, nothing captured: no variant */
   } else {
      if (stage == MESA_SHADER_VERTEX)
         vs_sysvals(nir);
      ok = pco_g5p_compile_general_link(nir, half_output, vertex ? vertex->components : NULL, key, link, out);
      out->psize_dropped = psize_dropped;
   }
   ralloc_free(nir);
   glsl_type_singleton_decref();
   return ok;
}

bool
pvr_g5p_compile_stage(const VkPipelineShaderStageCreateInfo *stage_info, bool half_output, uint32_t samples,
                      struct pvr_g5p_vertex_layout *vertex, const struct pco_g5p_fs_key *key,
                      const struct pco_g5p_vs_link *link, struct pco_g5p_program *out)
{
   return compile_stage(stage_info, half_output, samples, vertex, key, link, NULL, out);
}

bool
pvr_g5p_stage_has_xfb(const VkPipelineShaderStageCreateInfo *stage_info)
{
   VK_FROM_HANDLE(vk_shader_module, module, stage_info->module);
   const VkShaderModuleCreateInfo *inline_ci = vk_find_struct_const(stage_info->pNext, SHADER_MODULE_CREATE_INFO);
   if (stage_info->stage != VK_SHADER_STAGE_VERTEX_BIT)
      return false;
   if (module && !module->nir)
      return spirv_xfb((const uint32_t *)module->data, module->size / 4);
   return inline_ci && spirv_xfb(inline_ci->pCode, inline_ci->codeSize / 4);
}

bool
pvr_g5p_compile_vs_xfb(const VkPipelineShaderStageCreateInfo *stage_info, struct pvr_g5p_vertex_layout *vertex,
                       const struct pco_g5p_vs_link *link, struct g5p_xfb_layout *layout,
                       struct pco_g5p_program *out)
{
   return compile_stage(stage_info, false, 1, vertex, NULL, link, layout, out);
}

/*
 * The background program of a render whose colour attachment is loaded
 * (loadOp LOAD): the attachment, bound as a combined image/sampler at set 0
 * binding 0, read at the fragment's own pixel,
 *   color = texelFetch(prev, ivec2(gl_FragCoord.xy), 0),
 * built with nir_builder (Mesa's way for a driver's own shaders), then the
 * same preparation and backend as every graphics stage, with the
 * last-read source groups of every fragment program. Its uploads are the
 * image state and the sampler at that binding (out->uploads).
 */
bool
pvr_g5p_compile_load(struct pco_g5p_program *out)
{
   static util_once_flag once = UTIL_ONCE_FLAG_INIT;
   util_call_once(&once, g5p_init_options);
   memset(out, 0, sizeof(*out));
   glsl_type_singleton_init_or_ref();

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &g5p_options, "g5p-load");
   nir_variable *prev = nir_variable_create(b.shader, nir_var_uniform,
                                            glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
                                            "prev");
   prev->data.descriptor_set = 0;
   prev->data.binding = 0;
   nir_variable *color = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "color");
   color->data.location = FRAG_RESULT_DATA0;
   /* texelFetch (Mesa's form of a load): with the last-read group bits it reads every channel (the LOAD probe's
    * one-channel result was the default group rule, not the read op) */
   nir_def *xy = nir_f2i32(&b, nir_trim_vector(&b, nir_load_frag_coord(&b), 2));
   nir_def *texel = nir_txf(&b, xy, .texture_deref = nir_build_deref_var(&b, prev), .lod = nir_imm_int(&b, 0),
                            .dim = GLSL_SAMPLER_DIM_2D, .dest_type = nir_type_float32);
   nir_store_var(&b, color, texel, 0xf);

   bool sample_mask = false; /* the load program reads no sample mask */
   nir_shader *nir = graphics_lower(g5p_nir_prepare(b.shader), 1, &sample_mask, false, NULL);
   /* Apple's source groups, as every pipeline's fragment program (g5p_check_state()): 1 only on a register's last
    * read. The default, 1 on every read, returns a wrong value on a re-read: the LOAD probe's texelFetch wrote only
    * output component 0 (B G R A = 0 0 R 0). */
   struct pco_g5p_fs_key key = PCO_G5P_FS_KEY_DEFAULT;
   key.last_read_groups = true;
   bool ok = nir && pco_g5p_compile_general_link(nir, true, NULL, &key, NULL, out);
   if (!nir)
      snprintf(out->refusal, sizeof(out->refusal), "the load program's NIR");
   ralloc_free(nir);
   glsl_type_singleton_decref();
   return ok;
}

/* The clear quad's stages (pvr_g5p.h) */
nir_shader *
pvr_g5p_clear_nir(bool vertex)
{
   static util_once_flag once = UTIL_ONCE_FLAG_INIT;
   util_call_once(&once, g5p_init_options);
   glsl_type_singleton_init_or_ref();

   nir_builder b = nir_builder_init_simple_shader(vertex ? MESA_SHADER_VERTEX : MESA_SHADER_FRAGMENT, &g5p_options,
                                                  vertex ? "g5p-clear-vs" : "g5p-clear-fs");
   if (vertex) {
      nir_variable *pos = nir_variable_create(b.shader, nir_var_shader_in, glsl_vec4_type(), "pos");
      pos->data.location = VERT_ATTRIB_GENERIC0;
      nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
      out->data.location = VARYING_SLOT_POS;
      nir_store_var(&b, out, nir_load_var(&b, pos), 0xf);
   } else {
      nir_variable *color = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "color");
      color->data.location = FRAG_RESULT_DATA0;
      nir_store_var(&b, color, nir_load_push_constant(&b, 4, 32, nir_imm_int(&b, 0), .base = 0, .range = 16), 0xf);
      b.shader->num_uniforms = 16;
   }
   return b.shader;
}

void
pvr_g5p_clear_nir_free(nir_shader *nir)
{
   ralloc_free(nir);
   glsl_type_singleton_decref();
}
