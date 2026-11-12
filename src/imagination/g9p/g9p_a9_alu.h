/* SPDX-License-Identifier: MIT */
#ifndef G9P_A9_ALU_H
#define G9P_A9_ALU_H
#ifdef PVR_APPLE_G5P
#   include "../g5p/g5p_usc.h"
/* Coalesce only an immediate, sole store_reg use. No intervening instruction
 * may observe the old register value; no indirect or vector register writes. */
static int
a9_alu_store_destination(struct g9p_ctx *ctx, nir_def *def)
{
   nir_instr *next = nir_instr_next(nir_def_instr(def));
   if (!next || next->type != nir_instr_type_intrinsic)
      return -1;
   nir_intrinsic_instr *store = nir_instr_as_intrinsic(next);
   if (store->intrinsic != nir_intrinsic_store_reg || store->num_components != 1 ||
       store->src[0].ssa != def || nir_intrinsic_base(store) ||
       nir_intrinsic_write_mask(store) != 1 || nir_intrinsic_legacy_fsat(store) ||
       nir_def_used_by_if(def))
      return -1;
   unsigned uses = 0;
   nir_foreach_use(src, def) {
      if (nir_src_use_instr(src) != next)
         return -1;
      uses++;
   }
   if (uses != 1)
      return -1;
   nir_def *decl = store->src[1].ssa;
   if (!nir_def_is_intrinsic(decl) ||
       nir_def_as_intrinsic(decl)->intrinsic != nir_intrinsic_decl_reg ||
       (ctx->home && ctx->home[key(decl, 0)] >= 32))
      return -1;
   return nir_reg(ctx, decl);
}

static int
a9_alu_single_destination(struct g9p_ctx *ctx, nir_alu_instr *alu)
{
   const char *enabled = getenv("PVR_G5P_FMA_LITERALS");
   int dst = ctx->out->a9 && (!enabled || strcmp(enabled, "0"))
                ? a9_alu_store_destination(ctx, &alu->def) : -1;
   return dst >= 0 ? dst : alloc_reg(ctx);
}

static bool
a9_alu_fma_literals(struct g9p_ctx *ctx, nir_alu_instr *alu, nir_scalar s[3], bool *handled)
{
   uint32_t multiplier, addend;
   *handled = false;
   if (!ctx->out->a9 || alu->op != nir_op_ffma ||
       !is_const(s[2], &addend) || (addend & 0x200) ||
       (getenv("PVR_G5P_FMA_LITERALS") && !strcmp(getenv("PVR_G5P_FMA_LITERALS"), "0")))
      return true;
   unsigned literal = is_const(s[0], &multiplier) ? 0 : 1;
   if (!is_const(s[literal], &multiplier) || nir_scalar_is_const(s[1 - literal]))
      return true;
   *handled = true;
   int src = src_reg(ctx, s[1 - literal]);
   if (src < 0)
      return false;
   release(ctx, &alu->instr);
   int dst = a9_alu_store_destination(ctx, &alu->def);
   if (dst < 0)
      dst = alloc_reg(ctx);
   if (dst < 0)
      return false;
   uint8_t raw[G5P_USC_MAX];
   unsigned bytes = g5p_usc_fma_f32_literals(raw, 2 * dst, 2 * src,
                                              multiplier, addend, 1);
   if (!bytes)
      return false;
   for (unsigned i = 0; i < bytes; i += 2)
      G9P_EMIT(ctx->c, raw[i] | (uint16_t)raw[i + 1] << 8);
   ctx->reg[key(&alu->def, 0)] = dst;
   return true;
}
#   define G9P_A9_BRANCH_IMMEDIATE() do {                                \
      uint32_t value;                                                    \
      const char *enabled = getenv("PVR_G5P_FMA_LITERALS");                \
      if (ctx->out->a9 && !is_float && a >= 0 &&                          \
          (!enabled || strcmp(enabled, "0")) &&                         \
          is_const(nir_get_scalar(cmp->src[1].src.ssa,                    \
                                  cmp->src[1].swizzle[0]), &value) &&    \
          value < 1024) {                                               \
         g9p_cmp_imm(ctx->c, cond, a, value);                             \
         ctx->busy &= ~ctx->temps;                                       \
         ctx->temps = 0;                                                 \
         return true;                                                   \
      }                                                                 \
   } while (0)
#   define G9P_A9_ALU_SINGLE_DEST() a9_alu_single_destination(ctx, alu)
#   define G9P_A9_ALU_LITERAL_FMA() do {                       \
      bool handled;                                          \
      if (!a9_alu_fma_literals(ctx, alu, s, &handled))          \
         return false;                                       \
      if (handled)                                           \
         return true;                                        \
   } while (0)
#else
#   define G9P_A9_BRANCH_IMMEDIATE()
#   define G9P_A9_ALU_SINGLE_DEST() alloc_reg(ctx)
#   define G9P_A9_ALU_LITERAL_FMA()
#endif
#endif
