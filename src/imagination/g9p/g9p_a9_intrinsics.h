/* SPDX-License-Identifier: MIT */
#ifdef PVR_APPLE_G5P
#include "../g5p/g5p_barrier.h"
/* Static threadgroup operands: A9 host-compiler forms (compute.md §4).
 * Reserve TG0/1 for IDs and TG2/3 for the future software-barrier counters.
 * CDM allocation covers system words plus the native shared block. */
/* S and RI address forms from compute-isa-shared-atomic.md §1.3.
 * RI reads a temporary half index for the last time (short parcel8800).
 * The uint32->half conversion uses the helper's explicitly inferred1189 form. */
static bool a9_shared(struct g9p_ctx *ctx, nir_intrinsic_instr *in)
{
   bool load = in->intrinsic == nir_intrinsic_load_shared;
   nir_src offset = in->src[load ? 0 : 1];
   unsigned bits = load ? in->def.bit_size : nir_src_bit_size(in->src[0]);
   unsigned comps = load ? in->def.num_components : nir_src_num_components(in->src[0]);
   if (!ctx->out->a9 || bits != 32 || comps != 1 || (nir_intrinsic_base(in) & 3))
      return false;
   unsigned word = 4 + nir_intrinsic_base(in) / 4;
   bool constant = nir_src_is_const(offset);
   int index = -1;
   if (constant) {
      unsigned byte = nir_src_as_uint(offset);
      if (byte & 3) return false;
      word += byte / 4;
   } else {
      int src = src_reg(ctx, nir_get_scalar(offset.ssa, 0));
      int shift = const_reg(ctx, 2);
      index = alloc_temp(ctx);
      if (src < 0 || shift < 0 || index < 0) return false;
      g9p_shift(ctx->c, index, src, shift, true);
      G9P_EMIT(ctx->c, 0x1189, 0x0300 | 2 * index,
               0x0101 | index << 10, 0x0120, 0x0fa1, 0x4000);
   }
   int reg = load ? alloc_reg(ctx) : src_reg(ctx, nir_get_scalar(in->src[0].ssa, 0));
   if (reg < 0 || word >= 65536) return false;
   unsigned w1 = 0x0100 | (!load ? 0x80 : 0) | 2 * reg;
   if (constant) {
      G9P_EMIT(ctx->c, 0x2314, w1, 0x8100 | (word & 255), 0x0200 | (word >> 8));
   } else {
      w1 |= (2 * index) << 9;
      if (word < 256) G9P_EMIT(ctx->c, 0x2314, w1, 0x8800 | word);
      else G9P_EMIT(ctx->c, 0x2314, w1, 0x8900 | (word & 255), word >> 8);
   }
   g9p_wait(ctx->c);
   release(ctx, &in->instr);
   if (load) ctx->reg[key(&in->def, 0)] = reg;
   return true;
}
/* Keep the election's local-index source live for every task barrier. */
static void a9_generate_init(struct g9p_ctx *ctx, nir_function_impl *impl)
{
   ctx->a9_barriers_remaining = 0;
   if (!ctx->out->a9) return;
   unsigned n = ctx->out->local_size[0] * ctx->out->local_size[1] * ctx->out->local_size[2];
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic) continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic == nir_intrinsic_barrier &&
             nir_intrinsic_execution_scope(in) == SCOPE_WORKGROUP)
            ctx->a9_barriers_remaining++;
      }
   }
   ctx->out->a9_has_barriers = n > 32 && ctx->a9_barriers_remaining;
   if (ctx->out->a9_has_barriers) ctx->busy |= ctx->pinned |= BITFIELD64_BIT(0);
}
static bool a9_barrier(struct g9p_ctx *ctx, nir_intrinsic_instr *in)
{
   if (!ctx->out->a9) return false;
   unsigned n = ctx->out->local_size[0] * ctx->out->local_size[1] * ctx->out->local_size[2];
   unsigned scope = nir_intrinsic_execution_scope(in);
   if ((nir_intrinsic_memory_modes(in) & ~nir_var_mem_shared) ||
       (scope != SCOPE_NONE && scope != SCOPE_INVOCATION && scope != SCOPE_WORKGROUP)) {
      fprintf(stderr, "g5p: device/image memory barrier not implemented\n");
      return false;
   }
   if (scope == SCOPE_WORKGROUP) {
      unsigned N = ctx->a9_barriers_remaining--;
      if (!N || N > 31) return false;
      if (n > 32) {
         int r = alloc_temp(ctx), t = alloc_temp(ctx);
         if (r < 0 || t < 0) return false;
         /* Local size is fixed per pipeline: specialize Apple's target uniform. */
         g9p_mov_imm(ctx->c, t, DIV_ROUND_UP(n, 32) - 1);
         g5p_shared_barrier(ctx->c, r, 2 * t, 2, ctx->st, N);
         release(ctx, &in->instr);
         return true;
      }
      /* Apple's <=32 compiler form; cross-lane hardware qualification pending. */
   }
   g9p_wait(ctx->c);
   return true;
}
#define G9P_A9_INTRINSICS \
   case nir_intrinsic_load_shared: case nir_intrinsic_store_shared: return a9_shared(ctx, intr); \
   case nir_intrinsic_barrier: return a9_barrier(ctx, intr);
#else
#define G9P_A9_INTRINSICS
#endif
