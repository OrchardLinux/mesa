/* SPDX-License-Identifier: MIT */
/*
 * G3: compute for the Apple A7 through upstream pco's whole pipeline (NIR ->
 * pco IR -> RA -> schedule -> encode), for the shaders g3_compute.c's narrow
 * path refuses (loads, ALU, control flow).
 *
 * The A7 deltas:
 *  - encoding: a single lower or upper source with a 3-bit bank is 2 bytes
 *    (pco_binary.c and pco_group_instrs.c with pco_igrp_a7_short_src());
 *  - stores: pco already builds a st's address and data as one register
 *    vector, {addr lo, addr hi, data..} (trans_store_global()), which is the
 *    A7's rule (the data are the registers after the address pair), puts a wdf
 *    after every st (pco_schedule()), and on the A7 stores write-through, the
 *    only cache mode that ran. g3_pco_check() refuses a 4-byte (extended) st;
 *    compute/tools/a7dec.py checks the rest on the bytes;
 *  - no bitwise groups: pco's bitwise encoding does not parse the bitwise
 *    groups of Apple's A7 or Imagination's 6XT programs (compute/R7.md). Shifts
 *    left by a constant become multiplies, 32-bit constants outside Rogue's
 *    table (0..31) come from the unified store (a pool the PDS writes), and
 *    anything else that needs a bitwise group (other shifts, and/or/xor, ..)
 *    is refused after grouping;
 *  - inputs: no coefficient registers, no common (shared) store. Everything a
 *    shader reads comes from the unified store, written by the kernel's PDS
 *    program (g3_pds_code()): vi0 = the workgroup ID along x (CPU-folded, one
 *    CDM record per workgroup as the narrow path does), vi1 = the workgroup
 *    count along x, vi4 = the lane (PDS temps q0), vi8 + 2k = storage buffer
 *    k's address, then the constant pool. Shaders use them through
 *    load_vtxin_pco.
 *
 * Refused by name: shared memory, barriers, atomics, push constants, uniform
 * buffers, images, textures, subgroup operations, buffer sizes, spilling and
 * scratch, workgroups other than Nx1x1 with N <= 32.
 */
#include "g3_compute.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "pco.h"
#include "pco_builder.h"
#include "pco_internal.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_math.h"

#define G3_PCO_WGID 0u
#define G3_PCO_NUM_WG 1u
#define G3_PCO_LANE 4u
#define G3_PCO_BUFFER0 8u

static bool
refuse(struct g3_program *p, const char *fmt, ...) PRINTFLIKE(2, 3);

static bool
refuse(struct g3_program *p, const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   int n = snprintf(p->refusal, sizeof(p->refusal), "pco: ");
   vsnprintf(p->refusal + n, sizeof(p->refusal) - n, fmt, ap);
   va_end(ap);
   return false;
}

const struct spirv_to_nir_options *
g3_pco_spirv_options(void)
{
   static struct spirv_to_nir_options o;
   static bool init;
   if (!init) {
      o = *pco_spirv_options();
      /* storage buffers by address: the descriptor is its base, in vi8 + 2k */
      o.ssbo_addr_format = nir_address_format_2x32bit_global;
      init = true;
   }
   return &o;
}

const struct nir_shader_compiler_options *
g3_pco_nir_options(void)
{
   return pco_nir_options();
}

struct lower_state {
   struct g3_program *p;
   bool failed;
};

static nir_def *
vtxin(nir_builder *b, unsigned reg, unsigned comps)
{
   return nir_load_vtxin_pco(b, comps, nir_imm_int(b, reg));
}

/* the buffer's index in p->buffers (added on first use) */
static int
buffer_index(struct g3_program *p, unsigned set, unsigned binding)
{
   for (unsigned i = 0; i < p->num_buffers; i++)
      if (p->buffers[i].set == set && p->buffers[i].binding == binding)
         return i;
   if (p->num_buffers == G3_MAX_BUFFERS)
      return -1;
   p->buffers[p->num_buffers].set = set;
   p->buffers[p->num_buffers].binding = binding;
   return p->num_buffers++;
}

static bool
lower_descriptor(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower_state *st = data;

   if (intr->intrinsic != nir_intrinsic_load_vulkan_descriptor)
      return false;
   nir_intrinsic_instr *res = nir_src_as_intrinsic(intr->src[0]);
   if (!res || res->intrinsic != nir_intrinsic_vulkan_resource_index) {
      st->failed = !refuse(st->p, "a descriptor that is not a plain binding");
      return false;
   }
   if (!nir_src_is_const(res->src[0]) || nir_src_as_uint(res->src[0])) {
      st->failed = !refuse(st->p, "an element of a descriptor array");
      return false;
   }
   if (nir_intrinsic_desc_type(res) != nir_descriptor_type_storage_buffer) {
      st->failed = !refuse(st->p, "a descriptor of type %u (storage buffers only)", nir_intrinsic_desc_type(res));
      return false;
   }
   const int k = buffer_index(st->p, nir_intrinsic_desc_set(res), nir_intrinsic_binding(res));
   if (k < 0) {
      st->failed = !refuse(st->p, "more than %u storage buffers", G3_MAX_BUFFERS);
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, vtxin(b, G3_PCO_BUFFER0 + 2 * k, 2));
   return true;
}

static bool
lower_sysvals(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *v;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_local_invocation_index:
      v = vtxin(b, G3_PCO_LANE, 1);
      break;
   case nir_intrinsic_load_workgroup_id:
      v = nir_vec3(b, vtxin(b, G3_PCO_WGID, 1), nir_imm_int(b, 0), nir_imm_int(b, 0));
      break;
   case nir_intrinsic_load_num_workgroups:
      v = nir_vec3(b, vtxin(b, G3_PCO_NUM_WG, 1), nir_imm_int(b, 1), nir_imm_int(b, 1));
      break;
   default:
      return false;
   }
   if (intr->def.bit_size != 32)
      return false;
   nir_def_replace(&intr->def, nir_trim_vector(b, v, intr->def.num_components));
   return true;
}

/* pco's instance check (pco_nir_compute_instance_check()) is lane < local size: always true here, as the CDM
 * starts only local size instances (KERNEL8 size_x; the 16-lane rungs ran with 16 stores) */
static bool
drop_instance_check(nir_builder *b, nir_instr *instr, void *data)
{
   const unsigned *local = data;
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if ((alu->op != nir_op_ilt && alu->op != nir_op_ult) || !nir_src_is_const(alu->src[1].src))
      return false;
   nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(alu->src[0].src.ssa, alu->src[0].swizzle[0]));
   if (!nir_scalar_is_intrinsic(s) || nir_scalar_intrinsic_op(s) != nir_intrinsic_load_local_invocation_index)
      return false;
   if (nir_src_comp_as_uint(alu->src[1].src, alu->src[1].swizzle[0]) < *local)
      return false;
   b->cursor = nir_before_instr(instr);
   nir_def_replace(&alu->def, nir_imm_true(b));
   return true;
}

/* nir_lower_explicit_io gives storage buffer loads in nir_address_format_2x32bit_global a load_global with a
 * 2x32 address (stores get store_global_2x32): make them load_global_2x32 */
static bool
fix_load_global(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_global || intr->src[0].ssa->num_components != 2 ||
       intr->src[0].ssa->bit_size != 32)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *v = nir_load_global_2x32(b, intr->def.num_components, intr->def.bit_size, intr->src[0].ssa,
                                     .access = nir_intrinsic_access(intr), .align_mul = nir_intrinsic_align_mul(intr),
                                     .align_offset = nir_intrinsic_align_offset(intr));
   nir_def_replace(&intr->def, v);
   return true;
}

/*
 * pco encodes shifts, logical ops and 32-bit immediates in bitwise groups, whose encoding does not parse Apple's or
 * Imagination's Series6/6XT bitwise groups (compute/R7.md): on the A7 none is emitted. A shift left by a constant
 * becomes a multiply (main ALU), and every 32-bit constant outside Rogue's constant table (0..31) comes from the
 * unified store (the pool after the buffers, written by the PDS). Other bitwise work is refused after grouping.
 */
struct pool_state {
   struct g3_program *p;
   unsigned base;
   bool failed;
};

static bool
shl_to_mul(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_ishl || alu->def.bit_size != 32 || !nir_src_is_const(alu->src[1].src))
      return false;
   const unsigned k = nir_src_comp_as_uint(alu->src[1].src, alu->src[1].swizzle[0]) & 31;
   b->cursor = nir_before_instr(instr);
   nir_def *x = nir_mov_alu(b, alu->src[0], alu->def.num_components);
   nir_def_replace(&alu->def, nir_imul(b, x, nir_imm_int(b, 1u << k)));
   return true;
}

/* pco's min (O_MIN) tests "less" in its own encoding: as a select of a comparison, which trans_cmp() swaps */
static bool
lower_min(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_imin && alu->op != nir_op_umin && alu->op != nir_op_fmin)
      return false;
   b->cursor = nir_before_instr(instr);
   nir_def *x = nir_mov_alu(b, alu->src[0], alu->def.num_components);
   nir_def *y = nir_mov_alu(b, alu->src[1], alu->def.num_components);
   nir_def *lt = alu->op == nir_op_imin ? nir_ilt(b, x, y) : alu->op == nir_op_umin ? nir_ult(b, x, y)
                                                                                     : nir_flt(b, x, y);
   nir_def_replace(&alu->def, nir_bcsel(b, lt, x, y));
   return true;
}

static bool
const_to_pool(nir_builder *b, nir_instr *instr, void *data)
{
   struct pool_state *st = data;
   if (instr->type != nir_instr_type_load_const)
      return false;
   nir_load_const_instr *lc = nir_instr_as_load_const(instr);
   if (lc->def.bit_size != 32 || lc->def.num_components != 1)
      return false;
   const uint32_t v = lc->value[0].u32;
   if (v < 32)
      return false;
   /* not where an intrinsic needs a constant (load_vtxin_pco's offset), nor an if condition */
   nir_foreach_use_including_if (use, &lc->def) {
      if (nir_src_is_if(use))
         return false;
      nir_instr *u = nir_src_use_instr(use);
      if (u->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(u)->intrinsic != nir_intrinsic_store_global_2x32)
         return false;
      if (u->type != nir_instr_type_alu && u->type != nir_instr_type_intrinsic)
         return false;
   }
   unsigned i;
   for (i = 0; i < st->p->num_pool && st->p->pool[i] != v; i++)
      ;
   if (i == st->p->num_pool) {
      if (i == G3_MAX_POOL) {
         st->failed = true;
         return false;
      }
      st->p->pool[st->p->num_pool++] = v;
   }
   b->cursor = nir_after_instr(instr);
   nir_def *r = vtxin(b, st->base + i, 1);
   nir_def_rewrite_uses_after(&lc->def, r);
   return true;
}

/* What may reach pco on the A7: refuse every intrinsic outside these by name */
static bool
check_intrinsics(nir_shader *nir, struct g3_program *p)
{
   nir_foreach_function_impl (impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type == nir_instr_type_tex)
               return refuse(p, "a texture instruction");
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            const nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            switch (in->intrinsic) {
            case nir_intrinsic_load_global_2x32:
            case nir_intrinsic_store_global_2x32:
            case nir_intrinsic_load_vtxin_pco:
            case nir_intrinsic_decl_reg:
            case nir_intrinsic_load_reg:
            case nir_intrinsic_store_reg:
               break;
            default:
               return refuse(p, "intrinsic %s", nir_intrinsic_infos[in->intrinsic].name);
            }
         }
      }
   }
   return true;
}

/*
 * The A7's rules on the final program (after grouping): no bitwise group, no
 * 4-byte (tiled/extended) st, a wdf right after every st.
 */
static bool
g3_pco_check(pco_shader *s, struct g3_program *p)
{
   pco_foreach_func_in_shader (func, s) {
      pco_foreach_block_in_func (block, func) {
         pco_foreach_igrp_in_block (igrp, block) {
            if (igrp->hdr.alutype == PCO_ALUTYPE_BITWISE) {
               const pco_instr *bi = igrp->instrs[PCO_OP_PHASE_0] ? igrp->instrs[PCO_OP_PHASE_0]
                                     : igrp->instrs[PCO_OP_PHASE_1] ? igrp->instrs[PCO_OP_PHASE_1]
                                                                    : igrp->instrs[PCO_OP_PHASE_2];
               return refuse(p, "a bitwise group (%s; pco's bitwise encoding does not parse the A7-era groups, "
                             "compute/R7.md)", bi ? pco_op_info[bi->op].str : "?");
            }
            /* the A7's test op is 3 bits (see trans_cmp()): no less/lequal, whose fourth bit it ignores */
            const pco_instr *tst =
               igrp->hdr.alutype == PCO_ALUTYPE_MAIN ? igrp->instrs[PCO_OP_PHASE_2_TST] : NULL;
            if (tst && pco_instr_has_mod(tst, PCO_OP_MOD_TST_OP_MAIN)) {
               const enum pco_tst_op_main op = pco_instr_get_tst_op_main(tst);
               if (op == PCO_TST_OP_MAIN_LESS || op == PCO_TST_OP_MAIN_LEQUAL)
                  return refuse(p, "a %s test (the A7's test op has no fourth bit)",
                                op == PCO_TST_OP_MAIN_LESS ? "less" : "lequal");
            }
            pco_instr *be = igrp->hdr.alutype == PCO_ALUTYPE_MAIN ? igrp->instrs[PCO_OP_PHASE_BACKEND] : NULL;
            if (!be || be->op != PCO_OP_ST32)
               continue;
            if (igrp->variant.instr[PCO_OP_PHASE_BACKEND].backend != PCO_BACKEND_ST_IMMBL)
               return refuse(p, "a st in form %u (only the 3-byte st_immbl ran on the A7)",
                             igrp->variant.instr[PCO_OP_PHASE_BACKEND].backend);
            pco_igrp *next = pco_next_igrp(igrp);
            if (!next || next->hdr.alutype != PCO_ALUTYPE_CONTROL || !next->instrs[PCO_OP_PHASE_CTRL] ||
                next->instrs[PCO_OP_PHASE_CTRL]->op != PCO_OP_WDF)
               return refuse(p, "a st not followed by its wdf (the A7 reads a store's data late)");
         }
      }
   }
   return true;
}

bool
g3_compile_pco(void *pco, struct nir_shader *nir, struct g3_program *out)
{
   pco_ctx *ctx = pco;
   void *mem_ctx = ralloc_context(NULL);
   pco_data data = { 0 };
   struct lower_state st = { .p = out };
   bool ok = false;

   memset(out, 0, sizeof(*out));
   out->pco = true;
   if (nir->info.stage != MESA_SHADER_COMPUTE) {
      refuse(out, "a non-compute shader");
      goto out;
   }
   if (nir->info.workgroup_size_variable || nir->info.workgroup_size[1] != 1 ||
       nir->info.workgroup_size[2] != 1 || !nir->info.workgroup_size[0] || nir->info.workgroup_size[0] > 32) {
      refuse(out, "a %ux%ux%u workgroup (Nx1x1 with N <= 32: one task)", nir->info.workgroup_size[0],
             nir->info.workgroup_size[1], nir->info.workgroup_size[2]);
      goto out;
   }
   nir_foreach_variable_with_modes (var, nir, nir_var_mem_shared) {
      refuse(out, "shared memory (Rogue keeps it in the common store; no common-store path on the A7 yet)");
      goto out;
   }
   if (nir->info.shared_size) {
      refuse(out, "shared memory (Rogue keeps it in the common store; no common-store path on the A7 yet)");
      goto out;
   }
   out->local_size = nir->info.workgroup_size[0];
   for (unsigned i = 0; i < 3; i++)
      data.cs.workgroup_size[i] = nir->info.workgroup_size[i];

   /* the device reports a subgroup size of 1 (the Apple paths); subgroup operations are refused below */
   nir->info.api_subgroup_size = 0;
   pco_preprocess_nir(ctx, nir, &data);
   NIR_PASS(_, nir, nir_shader_instructions_pass, drop_instance_check, nir_metadata_control_flow,
            &out->local_size);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_dead_cf);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_sysvals, nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_descriptor, nir_metadata_control_flow, &st);
   if (st.failed)
      goto out;
   NIR_PASS(_, nir, nir_opt_dce);
   {
      /* one pass both, so that NIR validates only the fixed loads */
      bool progress = false;
      nir_lower_explicit_io(nir, nir_var_mem_ssbo, nir_address_format_2x32bit_global);
      progress |= nir_shader_intrinsics_pass(nir, fix_load_global, nir_metadata_control_flow, NULL);
      (void)progress;
      nir_validate_shader(nir, "after the A7's storage buffer lowering");
   }
   pco_lower_nir(ctx, nir, &data);
   pco_postprocess_nir(ctx, nir, &data);
   {
      /* no bitwise groups on the A7 (see shl_to_mul()) */
      struct pool_state ps = { .p = out, .base = G3_PCO_BUFFER0 + 2 * out->num_buffers };
      nir_shader_instructions_pass(nir, shl_to_mul, nir_metadata_control_flow, NULL);
      nir_shader_instructions_pass(nir, lower_min, nir_metadata_control_flow, NULL);
      nir_shader_instructions_pass(nir, const_to_pool, nir_metadata_control_flow, &ps);
      nir_opt_dce(nir);
      if (ps.failed) {
         refuse(out, "more than %u 32-bit constants outside 0..31", G3_MAX_POOL);
         goto out;
      }
   }
   if (!check_intrinsics(nir, out))
      goto out;

   pco_shader *s = pco_trans_nir(ctx, nir, &data, mem_ctx);
   if (!s) {
      refuse(out, "pco_trans_nir failed");
      goto out;
   }
   pco_process_ir(ctx, s);
   pco_encode_ir(ctx, s);

   const pco_data *d = pco_shader_data(s);
   if (d->common.spilled_temps || d->common.scratch) {
      refuse(out, "register spilling or scratch memory");
      goto out;
   }
   if (d->common.coeffs || d->common.shareds) {
      refuse(out, "coefficient or shared registers (%u, %u)", d->common.coeffs, d->common.shareds);
      goto out;
   }
   if (s->a7_unencodable) {
      refuse(out, "a short (2-byte) source the A7 cannot encode (an index above 255 or a mux)");
      goto out;
   }
   if (!g3_pco_check(s, out))
      goto out;
   const unsigned size = pco_shader_binary_size(s);
   if (size > sizeof(out->code)) {
      refuse(out, "a %u-byte program (at most %zu)", size, sizeof(out->code));
      goto out;
   }
   memcpy(out->code, pco_shader_binary_data(s), size);
   out->code_size = size;
   out->entry_offset = d->common.entry_offset;
   out->usc_temps = d->common.temps;
   if (out->usc_temps > 4 * 0x3f) {
      refuse(out, "%u temporaries (DOUTU allocates at most 252)", out->usc_temps);
      ok = false;
      goto out;
   }
   out->uses_wgid = true;
   out->unified_regs = G3_PCO_BUFFER0 + 2 * out->num_buffers + ALIGN_POT(out->num_pool, 2);

   /* the PDS writes: {wgid, groups} -> vi0, buffer k -> vi8 + 2k, the lane -> vi4 (temps last, as R6) */
   out->num_writes = 0;
   out->writes[out->num_writes++] = (struct g3_pds_write){ G3_PDS_WGID, 0, 0, G3_PCO_WGID };
   for (unsigned k = 0; k < out->num_buffers; k++)
      out->writes[out->num_writes++] = (struct g3_pds_write){ G3_PDS_BUFFER, k, 0, G3_PCO_BUFFER0 + 2 * k };
   for (unsigned i = 0; i < out->num_pool; i += 2)
      out->writes[out->num_writes++] =
         (struct g3_pds_write){ G3_PDS_POOL, 0, i / 2, G3_PCO_BUFFER0 + 2 * out->num_buffers + i };
   out->writes[out->num_writes++] = (struct g3_pds_write){ G3_PDS_LANE, 0, 0, G3_PCO_LANE };
   if (getenv("PVR_G3_PCO_PRINT"))
      pco_print_shader(s, stderr, "a7");
   ok = true;

out:
   ralloc_free(mem_ctx);
   return ok;
}
