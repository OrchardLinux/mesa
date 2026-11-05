/* SPDX-License-Identifier: MIT */
/* Generated-by: LLM
 *
 * The vertex encodings of the canary2 GLES2 vertex refusals, against Apple's applegpu_g5p vertex programs
 * (offline host compiler, tools/g5p-1b-dgl/vs-canary/vsops2-probes.py; decoded instructions, registers as
 * Apple allocated them): float->int/uint conversion, 32-bit not, and the vertex u32 literal form of a boolean.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "g5p_usc.h"

static int fails;

static void check(const char *name, const uint8_t *got, unsigned n, const char *want_hex)
{
   char hex[2 * G5P_USC_MAX + 1] = {0};
   for (unsigned i = 0; i < n; i++)
      snprintf(hex + 2 * i, 3, "%02x", got[i]);
   const bool ok = !strcmp(hex, want_hex);
   printf("%s %s got=%s want=%s\n", ok ? "PASS" : "FAIL", name, hex, want_hex);
   fails += !ok;
}

int main(void)
{
   uint8_t b[G5P_USC_MAX];
   unsigned n;
   /* v_f2i / v_f2u: convert, bank 1, header 0x11, r4 <- r4, last read */
   n = g5p_usc_convert(b, G5P_F2I32, 8, 8, 1, 1, 0x11);
   check("v_f2i", b, n, "99110803139120016004");
   n = g5p_usc_convert(b, G5P_F2U32, 8, 8, 1, 1, 0x11);
   check("v_f2u", b, n, "99110803119120016004");
   /* v_f2i_rr: the source read again (group 0), r7 <- r4 */
   n = g5p_usc_convert(b, G5P_F2I32, 14, 8, 1, 0, 0x11);
   check("v_f2i_rr", b, n, "99110e03131120016004");
   /* v_f2i_i2f: the i2f32 of the same program (the existing vertex i2f form) */
   n = g5p_usc_convert(b, G5P_I2F32, 8, 8, 1, 1, 0x11);
   check("v_f2i_i2f", b, n, "99110803179120016804");
   /* v_inot32: unary not, bank 1, header 0x21 */
   n = g5p_usc_unary2(b, G5P_NOT, 8, 8, 1, 1, 0x21, false, 0, 0, false);
   check("v_inot32", b, n, "982108039001c500");
   /* v_bnot: the vertex u32 binary literal (and 1), bank 1, header 0x11, the form the boolean not's XOR takes */
   n = g5p_usc_binary_literal2(b, G5P_BIN_AND, 8, 1, 8, 1, 1, 0x11, G5P_LIT_U32, -1, 0, 0, false);
   check("v_bnot_and", b, n, "9911080303912001c8017000");
   /* v_iand_lit, v_iand_big, v_ior_big, v_ixor_big, v_ixor_big_rr: the u32 binary literal, bank 1, header 0x11;
    * x ^ 0x3f800000 is the boolean not of a materialized 1.0/0.0 */
   n = g5p_usc_binary_literal2(b, G5P_BIN_AND, 8, 0x7f, 8, 1, 1, 0x11, G5P_LIT_U32, -1, 0, 0, false);
   check("v_iand_lit", b, n, "991108037f912201c8017000");
   n = g5p_usc_binary_literal2(b, G5P_BIN_AND, 8, 0x12345678, 8, 1, 1, 0x11, G5P_LIT_U32, -1, 0, 0, false);
   check("v_iand_big", b, n, "99310803719173213413c8017000");
   n = g5p_usc_binary_literal2(b, G5P_BIN_OR, 8, 0x40000000, 8, 1, 1, 0x11, G5P_LIT_U32, -1, 0, 0, false);
   check("v_ior_big", b, n, "99110803019121010041ce017000");
   n = g5p_usc_binary_literal2(b, G5P_BIN_XOR, 8, 0x3f800000, 8, 1, 1, 0x11, G5P_LIT_U32, -1, 0, 0, false);
   check("v_ixor_big (boolean not)", b, n, "9911080301912181803fc6017000");
   n = g5p_usc_binary_literal2(b, G5P_BIN_XOR, 14, 0x3f800000, 8, 1, 0, 0x11, G5P_LIT_U32, -1, 0, 0, false);
   check("v_ixor_big_rr (boolean not, read again)", b, n, "99110e0301112181803fc6017000");
   printf("G5P VSOPS ENCODERS %s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
   return fails != 0;
}
