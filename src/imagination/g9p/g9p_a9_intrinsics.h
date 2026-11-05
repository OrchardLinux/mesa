/* SPDX-License-Identifier: MIT */
#ifdef PVR_APPLE_G5P
/* Static threadgroup operands: A9 host-compiler forms (compute.md §4).
 * Reserve TG0/1 for IDs and TG2/3 for the future software-barrier counters.
 * Current one-common-chunk path admits at most32 shared bytes. */
static bool a9_shared(struct g9p_ctx *ctx, nir_intrinsic_instr *in)
{
   bool load = in->intrinsic == nir_intrinsic_load_shared;
   nir_src offset = in->src[load ? 0 : 1];
   unsigned bits = load ? in->def.bit_size : nir_src_bit_size(in->src[0]);
   unsigned comps = load ? in->def.num_components : nir_src_num_components(in->src[0]);
   if (!ctx->out->a9 || !nir_src_is_const(offset) || bits != 32 || comps != 1)
      return false;
   unsigned byte = nir_src_as_uint(offset) + nir_intrinsic_base(in);
   if ((byte & 3) || byte >= 32) return false;
   unsigned word = 4 + byte / 4;
   if (load) {
      release(ctx, &in->instr);
      int dst = alloc_reg(ctx);
      if (dst < 0) return false;
      g9p_tg_load(ctx->c, dst, word);
      g9p_wait(ctx->c);
      ctx->reg[key(&in->def, 0)] = dst;
   } else {
      int src = src_reg(ctx, nir_get_scalar(in->src[0].ssa, 0));
      if (src < 0) return false;
      G9P_EMIT(ctx->c, 0x2314, 0x0180 | 2 * src, 0x8100 | word, 0x0200);
      g9p_wait(ctx->c);
      release(ctx, &in->instr);
   }
   return true;
}
static bool a9_barrier(struct g9p_ctx *ctx, nir_intrinsic_instr *in)
{
   if (!ctx->out->a9) return false;
   unsigned n = ctx->out->local_size[0] * ctx->out->local_size[1] * ctx->out->local_size[2];
   if ((nir_intrinsic_memory_modes(in) & ~nir_var_mem_shared) ||
       ((nir_intrinsic_execution_scope(in) != SCOPE_NONE &&
         nir_intrinsic_execution_scope(in) != SCOPE_INVOCATION) && n != 1)) {
      fprintf(stderr, "g5p: inter-invocation barrier not implemented; no wait-as-barrier fallback\n");
      return false;
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
