/* SPDX-License-Identifier: MIT */
/* Generated-by: LLM
 *
 * The vertex forms of the g3-fail-a-r13 refusal families against Apple's applegpu_g5p vertex programs (offline
 * host compiler, tools/g5p-1b-dgl/vs-canary/vsalu-probes.py; registers and banks as Apple allocated them): the
 * dynamic shift (amount & 31, then the register-amount shift or the two-register ishr), fmul with source
 * modifiers, ffma with operands read again, and the rint/sign/ceil lowerings in the vertex stage.
 */
#include <stdio.h>
#include <string.h>

#include "g5p_usc.h"

static int check(const char *name, const uint8_t *got, unsigned n, const char *want)
{
   char hex[512] = {0};
   for (unsigned i = 0; i < n && 2 * i + 2 < sizeof(hex); i++)
      snprintf(hex + 2 * i, 3, "%02x", got[i]);
   const int ok = n && !strcmp(hex, want);
   printf("%s %s got=%s want=%s\n", ok ? "PASS" : "FAIL", name, hex, want);
   return !ok;
}

static struct g5p_alu alu(unsigned dst, unsigned header, struct g5p_src a, struct g5p_src b)
{
   struct g5p_alu f = G5P_ALU(dst);
   f.bank = 1;
   f.header = header;
   f.a = a;
   f.b = b;
   return f;
}

#define SRC(h, g, n, ab) ((struct g5p_src){.half = (h), .group = (g), .neg = (n), .abs = (ab)})

int main(void)
{
   uint8_t b[256];
   const char *err = NULL;
   int fails = 0;
   unsigned n;
   struct g5p_alu f;

   /* v_ishl_dyn_rr: amount (r5) & 31 -> r7 (bank 1), then x (r4, read again) << r7 */
   n = g5p_usc_binary_literal2(b, G5P_BIN_AND, 14, 31, 10, 1, 0, 0x11, G5P_LIT_U32, -1, 0, 0, false);
   fails += check("v_ishl_dyn_rr mask", b, n, "99110e033f152001c8017000");
   n = g5p_usc_shift(b, true, 14, 8, 14, false, 1, 0, 1, 0x19);
   fails += check("v_ishl_dyn_rr shift", b, n, "9a190e03011120019c01f004");
   /* v_ishl_dyn, v_ushr_dyn: the masked amount in bank 0 (r1, header 0x11), x (r4) dies */
   n = g5p_usc_binary_literal2(b, G5P_BIN_AND, 2, 31, 10, 0, 0, 0x11, G5P_LIT_U32, -1, 0, 0, false);
   fails += check("v_ishl_dyn mask (bank 0)", b, n, "991102013f152001c8017000");
   n = g5p_usc_shift(b, true, 8, 8, 2, false, 1, 1, 1, 0x11);
   fails += check("v_ishl_dyn shift", b, n, "9a110803019120018401f004");
   n = g5p_usc_shift(b, false, 8, 8, 2, false, 1, 1, 1, 0x11);
   fails += check("v_ushr_dyn shift", b, n, "9a110803019120018401f404");
   /* v_ishr_dyn: the two-register ishr, value A (bank 1), amount B (bank 0) */
   f = alu(8, 0x21, SRC(8, 1, 0, 0), SRC(2, 1, 0, 0));
   n = g5p_usc_binary(b, G5P_BIN_ISHR, &f);
   fails += check("v_ishr_dyn", b, n, "992108039085fc00");

   /* fmul with source modifiers (x r4, y r5) */
   f = alu(8, 0x31, SRC(8, 1, 1, 0), SRC(10, 0, 0, 0));
   n = g5p_usc_binary(b, G5P_BIN_FMUL, &f);
   fails += check("v_fmul_neg", b, n, "9931080390154880");
   f = alu(8, 0x31, SRC(8, 1, 0, 1), SRC(10, 0, 0, 0));
   n = g5p_usc_binary(b, G5P_BIN_FMUL, &f);
   fails += check("v_fmul_abs", b, n, "99310803901548010080");
   f = alu(8, 0x31, SRC(8, 1, 0, 1), SRC(10, 0, 1, 0));
   n = g5p_usc_binary(b, G5P_BIN_FMUL, &f);
   fails += check("v_fmul_negabs", b, n, "99310803901548410080");
   f = alu(14, 0x31, SRC(8, 0, 1, 0), SRC(10, 0, 0, 0));
   n = g5p_usc_binary(b, G5P_BIN_FMUL, &f);
   fails += check("v_fmul_neg_rr", b, n, "99310e0310154880");
   f = alu(8, 0x31, SRC(10, 0, 0, 1), SRC(8, 1, 0, 0));
   n = g5p_usc_binary(b, G5P_BIN_FMUL, &f);
   fails += check("v_fmul_abs_b", b, n, "99310803149148010080");

   /* ffma, operands in order, groups their last reads, {h39 x0} (x r4, y r5, z r6) */
   static const struct { const char *name; unsigned dst, ga, gb, gc; const char *want; } fm[] = {
      {"v_ffma_rr", 14, 0, 0, 0, "9a390e03101518014c00"},
      {"v_ffma_rr2", 12, 0, 0, 1, "9a390c03101598014c00"},
      {"v_ffma_rr3", 10, 0, 1, 0, "9a390a03109518014c00"},
      {"v_ffma_rr4", 8, 1, 0, 0, "9a390803901518014c00"},
   };
   for (unsigned i = 0; i < sizeof(fm) / sizeof(fm[0]); i++) {
      f = alu(fm[i].dst, 0x39, SRC(8, fm[i].ga, 0, 0), SRC(10, fm[i].gb, 0, 0));
      f.c = SRC(12, fm[i].gc, 0, 0);
      n = g5p_usc_fma(b, &f);
      fails += check(fm[i].name, b, n, fm[i].want);
   }

   /* ceil: v_ceil, x (r4) in place, exactly the vertex lowering */
   n = g5p_recipe(G5P_RC_CEIL, b, sizeof(b), 8, 8, 0, 0, G5P_RC_VERTEX, &err);
   fails += check("v_ceil", b, n, "9821080390014080" "991108030191a00144897000");
   /* rint, sign: Apple's _rr programs write the result into the temporary (x is read again); the lowering
    * writes it over x. The temporary's instructions are Apple's bytes exactly; the selects are checked with
    * Apple's own operands. */
   n = g5p_recipe(G5P_RC_RINT, b, sizeof(b), 8, 8, 0, 14, G5P_RC_VERTEX, &err);
   fails += check("v_rint_rr round trip", b, n >= 20 ? 20 : 0, "9d510e03171120016004" "99110e03179d20016804");
   struct g5p_lit3 l = {.opcode = 0x5c, .dst_half = 14, .value = 0x4b000000, .b_half = 8, .b_group = 0,
                        .c_half = 14, .c_group = 1, .k = {.header = 0x19, .bank = 1}, .operation_control = 0x40,
                        .routing = 0x4000, .long_form = -1, .c_long_form = -1};
   n = g5p_usc_literal_ternary(b, &l);
   fails += check("v_rint_rr select", b, n, "9a190e0301112181004b9c015c410040");
   n = g5p_recipe(G5P_RC_SIGN, b, sizeof(b), 8, 8, 0, 14, G5P_RC_VERTEX, &err);
   fails += check("v_sign_rr test", b, n >= 20 ? 20 : 0, "9e510e0301112101008101012181803f5d017000");
   l = (struct g5p_lit3){.opcode = 0x58, .dst_half = 14, .value = 0xbf800000, .b_half = 8, .b_group = 0,
                         .c_half = 14, .c_group = 1, .k = {.header = 0x19, .bank = 1}, .routing = 0x2058,
                         .long_form = -1, .c_long_form = -1};
   n = g5p_usc_literal_ternary(b, &l);
   fails += check("v_sign_rr select", b, n, "9a190e030111218180bf9c0158015820");
   /* fragment lowerings unchanged (f_ceil-shaped x half 0, f_rint/f_sign x half 0, temporary half 10) */
   printf("G5P VS-ALU %s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
   return fails != 0;
}
