/* SPDX-License-Identifier: MIT */
/* Generated-by: LLM
 *
 * The vertex trunc lowering against Apple's applegpu_g5p v_trunc_rr (offline host compiler,
 * tools/g5p-1b-dgl/vs-canary/vsops2-probes.py; registers as Apple allocated them: x = r4, result/temporary r7):
 * f2i32 (bank 1, header 0x51, extension 4), i2f32 (bank 1, header 0x11), the |x| < 2^23 literal select (bank 1,
 * header 0x19). The fragment form is checked unchanged against f_trunc.
 */
#include <stdio.h>
#include <string.h>

#include "g5p_usc.h"

static int check(const char *name, const uint8_t *got, unsigned n, const char *want)
{
   char hex[512] = {0};
   for (unsigned i = 0; i < n && 2 * i + 2 < sizeof(hex); i++)
      snprintf(hex + 2 * i, 3, "%02x", got[i]);
   const int ok = !strcmp(hex, want);
   printf("%s %s got=%s want=%s\n", ok ? "PASS" : "FAIL", name, hex, want);
   return !ok;
}

int main(void)
{
   uint8_t b[256];
   const char *err = NULL;
   int fails = 0;
   /* vertex: x half 8, destination/temporary half 14 */
   unsigned n = g5p_recipe(G5P_RC_TRUNC, b, sizeof(b), 14, 8, 0, 0, G5P_RC_VERTEX, &err);
   fails += check("v_trunc_rr", b, n, "9d510e03131120016004" "99110e03179d20016804" "9a190e0301112181004b9c015c410040");
   /* fragment, unchanged: x half 0 in place, temporary half 10 (f_trunc) */
   n = g5p_recipe(G5P_RC_TRUNC, b, sizeof(b), 0, 0, 0, 10, 0, &err);
   fails += check("f_trunc", b, n, "9d410a01130120016004" "99010a01179520016804" "9a01000101812181004b94015c410040");
   printf("G5P VS-TRUNC %s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
   return fails != 0;
}
