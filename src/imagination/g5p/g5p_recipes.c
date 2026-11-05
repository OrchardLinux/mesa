/* SPDX-License-Identifier: MIT
 * Round 19: complete reference lowerings and the premultiplied blend
 * epilogue on the round-2 field encoders (no NIR; shared by the general
 * backend and g5p-usc-test).
 */
#include "g5p_usc.h"

#include <string.h>

/* Apple's complete reference lowerings (isa-g5p.md ISA round
 * 2 §3), every instruction with the field values of the fast-corpus
 * f_math_*_0 programs. Arguments are 16-bit half indices: d = result,
 * x = input (d == x: the recipe overwrites its input, as Apple's does),
 * y = second input (min/max), t = scratch. g5p-usc-test --recipes prints
 * them with Apple's register assignment; check-recipes-r19.py compares each
 * against the corresponding slice of the compiled f_math_*_0 program. */
struct rbuf {
   uint8_t *out;
   unsigned n, max;
   const char *err;
};

static void rput(struct rbuf *b, const uint8_t *raw, unsigned len, const char *what)
{
   if (b->err)
      return;
   if (!len) {
      b->err = what;
      return;
   }
   if (b->n + len > b->max) {
      b->err = "recipe overflow";
      return;
   }
   memcpy(b->out + b->n, raw, len);
   b->n += len;
}

#define R(what, expr) do { uint8_t _r[G5P_USC_MAX]; unsigned _l = (expr); rput(b, _r, _l, what); } while (0)
#define S(h, g) ((struct g5p_src){.half = (h), .group = (g)})
#define K(hd, x, dc) ((struct g5p_ctl){.header = (hd), .source_extension = (x), .dst_control = (dc)})

const char *const g5p_recipe_name[G5P_RC_COUNT] = {"rcp", "rsqrt", "sqrt", "sign", "min", "max", "ceil", "trunc",
                                                   "rint", "sin", "cos"};

static struct g5p_lit3 lit3(unsigned op, unsigned d, uint32_t value, unsigned b, unsigned bg, struct g5p_ctl k,
                            unsigned ctl, int routing)
{
   return (struct g5p_lit3){.opcode = op, .dst_half = d, .value = value, .b_half = b, .b_group = bg, .k = k,
                            .operation_control = ctl, .routing = routing, .long_form = -1, .c_long_form = -1};
}

static void recipe(struct rbuf *b, enum g5p_recipe r, unsigned d, unsigned x, unsigned y, unsigned t,
                   unsigned flags)
{
   struct g5p_lit3 l;
   struct g5p_alu f;
   if (r != G5P_RC_RSQ && !(r == G5P_RC_TRUNC && (flags & G5P_RC_VERTEX)) && d != x) {
      b->err = "recipe overwrites its input";
      return;
   }
   /* Vertex programs (tools/mesa-a9-hyprland-gpu/vsops-probes.py: v_rcp,
    * v_rsq, v_min, v_max): every instruction is the fragment one with
    * destination bank 1 and the header's bank bit for each register operand
    * (A 0x20, B 0x10, C 0x08), the bits Apple sets per form: VB() below. */
   const bool vs = flags & G5P_RC_VERTEX;
   const unsigned bank = vs;
   /* sin and cos: v_sin, v_cos (bench-gaps.md), every temp in bank 1. */
   if (vs && r != G5P_RC_RCP && r != G5P_RC_RSQ && r != G5P_RC_MIN && r != G5P_RC_MAX && r != G5P_RC_SIN &&
       r != G5P_RC_COS && r != G5P_RC_TRUNC) {
      b->err = "no measured vertex form";
      return;
   }
#define VB(hd, bits) ((hd) | (vs ? (bits) : 0))
#define KV(hd, bits, x, dc) ((struct g5p_ctl){.header = VB(hd, bits), .source_extension = (x), \
                                              .dst_control = (dc), .bank = bank})
   switch (r) {
   case G5P_RC_RCP: /* f_math_rcp_0: core 80 + literal FMA 58 + FMA 0c + NaN fixup */
      R("rcp core", g5p_usc_sfu_core(_r, G5P_RCP_APPROX, t, S(x, 0), KV(1, 0x20, 4, 0x80), false));
      l = lit3(0x4c, x, 0x3c00, x, 1, KV(0x81, 0x18, 4, 0), 0x90, 0x58);
      l.c_half = t;
      l.c_group = 0;
      R("rcp refine", g5p_usc_literal_ternary(_r, &l));
      R("rcp repeat fma", g5p_usc_repeat_fma(_r, x, S(x, 1), S(t, 0), KV(0x41, 0x30, 4, 0)));
      l = lit3(0xb9, x, 0x7fc00000, x, 1, KV(1, 0x18, 0, 0), 0x20, -1);
      l.c_half = t;
      l.c_group = 1;
      R("rcp fixup", g5p_usc_literal_ternary(_r, &l));
      break;
   case G5P_RC_RSQ: /* f_math_rsqrt_0: -0 -> -inf test, predicated core 81 */
      l = lit3(0xbd, d, 0x80000000, x, 0, KV(0x41, 0x10, 4, 0), 0x60, -1);
      l.c_literal = true;
      l.c_value = 0xff800000;
      R("rsqrt -0 test", g5p_usc_literal_ternary(_r, &l));
      R("rsqrt core", g5p_usc_predicated_sfu(_r, d, S(x, !(flags & G5P_RC_X_LIVE)), bank, VB(1, 0x20)));
      break;
   case G5P_RC_SQRT: /* f_math_sqrt_0: x * rsqrt(x) with the 0x80800000 fixup */
      R("sqrt core", g5p_usc_sfu_core(_r, G5P_RSQRT_APPROX, t, S(x, 0), K(1, 4, 0), false));
      l = lit3(0x59, t, 0x80800000, x, 0, K(0x41, 4, 0), 0, 0x9070);
      l.c_half = t;
      l.c_group = 1;
      R("sqrt fixup", g5p_usc_literal_ternary(_r, &l));
      f = G5P_ALU(x);
      f.a = S(t, 1);
      f.b = S(x, 1);
      R("sqrt multiply", g5p_usc_binary(_r, G5P_BIN_FMUL, &f));
      break;
   case G5P_RC_SIGN: /* f_math_sign_0 */
      l = lit3(0x5d, t, 0x80000000, x, 0, K(0x41, 4, 0), 0, 0x70);
      l.c_literal = true;
      l.c_value = 0x3f800000;
      R("sign test", g5p_usc_literal_ternary(_r, &l));
      l = lit3(0x58, x, 0xbf800000, x, 1, K(1, 0, 0), 0, 0x2058);
      l.c_half = t;
      l.c_group = 1;
      R("sign select", g5p_usc_literal_ternary(_r, &l));
      break;
   case G5P_RC_MIN:
   case G5P_RC_MAX: /* assemble_float_minmax_finite (fast pattern) */
      R("minmax test", g5p_usc_float_minmax_test(_r, r == G5P_RC_MAX, x, S(y, 0), S(x, 1), bank, VB(1, 0x30)));
      R("minmax merge", g5p_usc_minmax_merge(_r, x, S(x, 1), bank, VB(1, 0x20)));
      /* a y read again: group 0, and in the vertex form no extension (v_min:
       * Apple's y is read again by the outputs, extension 0) */
      R("minmax move", flags & G5P_RC_Y_LIVE ? g5p_usc_predicated_move(_r, x, S(y, 0), KV(1, 0x20, vs ? 0 : 4, 0))
                                             : g5p_usc_predicated_move(_r, x, S(y, 1), KV(1, 0x20, 0, 0)));
      break;
   case G5P_RC_CEIL: /* f_math_ceil_0: -floor(-x) */
      R("ceil floor", g5p_usc_unary2(_r, G5P_FLOOR, x, x, 0, 1, 1, true, 0, 0, false));
      R("ceil negate", g5p_usc_binary_literal2(_r, G5P_BIN_FADD, x, 0x8000, x, 0, 1, 1, G5P_LIT_F16, -1, 0, 0,
                                               true));
      break;
   case G5P_RC_TRUNC:
      if (vs) {
         /* v_trunc_rr (tools/g5p-1b-dgl/vs-canary/vsops2-probes.py): the fragment round trip in bank 1 with
          * the header's bank bit per register operand, into the destination (d != x, d its own temporary);
          * x is read again by the select (group 0), as Apple's form does whether or not x dies. */
         R("round convert", g5p_usc_convert2(_r, G5P_F2I32, d, x, 1, 0, 0x51, 4, 0));
         R("round back", g5p_usc_convert2(_r, G5P_I2F32, d, d, 1, 1, 0x11, 0, 0));
         l = lit3(0x5c, d, 0x4b000000, x, 0, KV(1, 0x18, 0, 0), 0x40, 0x4000);
         l.c_half = d;
         l.c_group = 1;
         R("round select", g5p_usc_literal_ternary(_r, &l));
         break;
      }
      __attribute__((fallthrough));
   case G5P_RC_RINT: /* f_math_trunc_0 / f_math_rint_0: int round trip, |x| >= 2^23 keeps x */
      R("round convert", g5p_usc_convert2(_r, r == G5P_RC_TRUNC ? G5P_F2I32 : G5P_F2I32_RTE, t, x, 0, 0, 0x41, 4, 0));
      R("round back", g5p_usc_convert2(_r, G5P_I2F32, t, t, 0, 1, 1, 0, 0));
      l = lit3(0x5c, x, 0x4b000000, x, 1, K(1, 0, 0), 0x40, 0x4000);
      l.c_half = t;
      l.c_group = 1;
      R("round select", g5p_usc_literal_ternary(_r, &l));
      break;
   case G5P_RC_SIN:
   case G5P_RC_COS: /* f_math_sin_0 / f_math_cos_0 (fast namespace range reduction); vertex: v_sin / v_cos */
      if (r == G5P_RC_SIN) {
         R("sin scale", g5p_usc_binary_literal2(_r, G5P_BIN_FMUL, t, 0x3e22f983, x, bank, 0, VB(0x41, 0x10),
                                                G5P_LIT_F32, -1, 4, 0, false));
         R("sin floor", g5p_usc_unary2(_r, G5P_FLOOR, t, t, bank, 1, VB(1, 0x20), false, 0, 0, false));
         l = lit3(0x4c, x, 0x3e22f983, x, 1, KV(1, 0x18, 0, 0), 0x20, 0x70);
         l.c_half = t;
         l.c_group = 1;
         R("sin fract", g5p_usc_literal_ternary(_r, &l));
      } else {
         l = lit3(0x4c, x, 0x3e22f983, x, 1, KV(1, 0x10, 0, 0), 0x10, 0x70);
         l.c_literal = true;
         l.c_value = 0x3400;
         l.c_long_form = 0;
         R("cos scale", g5p_usc_literal_ternary(_r, &l));
         R("cos floor", g5p_usc_unary2(_r, G5P_FLOOR, t, x, bank, 0, VB(1, 0x20), false, 4, 0, false));
         f = G5P_ALU(x);
         f.bank = bank;
         f.header = VB(1, 0x30);
         f.a = S(x, 1);
         f.b = S(t, 1);
         f.b.neg = true;
         R("cos fract", g5p_usc_binary(_r, G5P_BIN_FADD, &f));
      }
      R("trig scale", g5p_usc_binary_literal2(_r, G5P_BIN_FMUL, x, 0x4400, x, bank, 1, VB(1, 0x10), G5P_LIT_F16, -1,
                                              0, 0, false));
      R("sin core", g5p_usc_sfu_core(_r, G5P_SIN_CORE, x, S(x, 1), KV(1, 0x20, 0, 0), false));
      R("cos core", g5p_usc_sfu_core(_r, G5P_COS_CORE, t, S(x, 0), KV(1, 0x20, 4, 0), false));
      f = G5P_ALU(x);
      f.bank = bank;
      f.header = VB(1, 0x30);
      f.a = S(x, 1);
      f.b = S(t, 1);
      R("trig product", g5p_usc_binary(_r, G5P_BIN_FMUL, &f));
      break;
   default:
      b->err = "unknown recipe";
   }
#undef KV
#undef VB
}

unsigned g5p_recipe(enum g5p_recipe r, uint8_t *out, unsigned max, unsigned d, unsigned x, unsigned y,
                    unsigned t, unsigned flags, const char **err)
{
   struct rbuf b = {.out = out, .max = max};
   if (r >= G5P_RC_COUNT)
      b.err = "unknown recipe";
   else
      recipe(&b, r, d, x, y, t, flags);
   if (err)
      *err = b.err;
   return b.err ? 0 : b.n;
}

/* The premultiplied epilogue after the colour sits in halves 0..3
 * (assemble_premult_blend_fs from the tile read on; ds-isa-state.md §3.2):
 * destination in halves 4..7, 1-srcA in half 12, scratch halves 4/5. */
static void blend_epilogue(struct rbuf *b)
{
   static const unsigned scratch[4] = {4, 5, 5, 4};
   R("tile read", g5p_usc_tile_read(_r, 4, 2));
   R("wait", g5p_usc_wait(_r, 0));
   for (unsigned lane = 0; lane < 4; lane++) {
      R("tile component", g5p_usc_tile_component(_r, scratch[lane], 4 + lane));
      if (!lane)
         R("1 - src alpha", g5p_usc_half_one_minus(_r, 12, 3));
      R("blend fma", g5p_usc_half_fma(_r, lane, S(scratch[lane], 1), S(12, lane < 3 ? 0 : 1), S(lane, 1),
                                      K(lane < 3 ? 0x41 : 1, lane < 3 ? 4 : 0, 0), 0x80));
   }
}

unsigned g5p_blend_epilogue(uint8_t *out, unsigned max)
{
   struct rbuf b = {.out = out, .max = max};
   blend_epilogue(&b);
   return b.err ? 0 : b.n;
}
