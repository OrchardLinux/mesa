/* SPDX-License-Identifier: MIT */
/* Generated-by: LLM
 *
 * Dynamic 32-bit shifts against Apple's applegpu_g5p fragment programs
 * f_ishl_dyn, f_ushr_dyn and f_ishr_dyn (fast and strict produce the same
 * bytes).  GLSL only defines amounts 0..31.  Apple nevertheless masks the
 * runtime amount with 31 before all three operations; verify that sequence
 * byte-for-byte and check every defined amount for positive and negative
 * operands against an independent integer reference.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "g5p_usc.h"

static int
check_bytes(const char *name, const uint8_t *got, unsigned n, const char *want)
{
   char hex[128] = {0};

   for (unsigned i = 0; i < n && 2 * i + 2 < sizeof(hex); i++)
      snprintf(hex + 2 * i, 3, "%02x", got[i]);

   const bool ok = n && !strcmp(hex, want);
   printf("%s %s got=%s want=%s\n", ok ? "PASS" : "FAIL", name, hex, want);
   return !ok;
}

static uint32_t
reference_ishr(uint32_t value, unsigned amount)
{
   const int64_t signed_value = (int32_t)value;
   const int64_t divisor = INT64_C(1) << amount;

   if (signed_value >= 0)
      return (uint32_t)(signed_value / divisor);

   /* Arithmetic right shift is floor(value / 2^amount), while C integer
    * division truncates toward zero. */
   return (uint32_t)-(((-signed_value) + divisor - 1) / divisor);
}

static uint32_t
model_ishr(uint32_t value, unsigned amount)
{
   if (!amount)
      return value;

   const uint32_t sign = (value & UINT32_C(0x80000000))
                            ? UINT32_MAX << (32 - amount)
                            : 0;
   return (value >> amount) | sign;
}

int
main(void)
{
   uint8_t raw[G5P_USC_MAX];
   int failures = 0;

   /* f_*_dyn: amount r1 & 31 -> r5, then value r0 op r5 -> r0. */
   unsigned n = g5p_usc_binary_literal2(raw,
                                         G5P_BIN_AND,
                                         10,
                                         31,
                                         2,
                                         0,
                                         0,
                                         1,
                                         G5P_LIT_U32,
                                         -1,
                                         0,
                                         0,
                                         false);
   failures += check_bytes("fragment amount & 31", raw, n, "99010a013f052001c8017000");

   struct g5p_alu ishr = G5P_ALU(0);
   ishr.a = (struct g5p_src){ .half = 0, .group = 1 };
   ishr.b = (struct g5p_src){ .half = 10, .group = 1 };
   n = g5p_usc_binary(raw, G5P_BIN_ISHR, &ishr);
   failures += check_bytes("fragment dynamic ishr", raw, n, "990100018095fc00");

   n = g5p_usc_shift(raw, true, 0, 0, 10, false, 0, 1, 1, 1);
   failures += check_bytes("fragment dynamic ishl", raw, n, "9a010001018120019401f004");
   n = g5p_usc_shift(raw, false, 0, 0, 10, false, 0, 1, 1, 1);
   failures += check_bytes("fragment dynamic ushr", raw, n, "9a010001018120019401f404");

   static const uint32_t values[] = {
      0,
      1,
      UINT32_C(0x40000001),
      UINT32_C(0x7fffffff),
      UINT32_C(0x80000000),
      UINT32_C(0x87654321),
      UINT32_C(0xffffffff),
   };

   for (unsigned amount = 0; amount < 32; amount++) {
      const uint64_t divisor = UINT64_C(1) << amount;

      for (unsigned i = 0; i < sizeof(values) / sizeof(values[0]); i++) {
         const uint32_t value = values[i];
         const uint32_t want_shl = (uint32_t)((uint64_t)value * divisor);
         const uint32_t want_ushr = (uint32_t)((uint64_t)value / divisor);
         const uint32_t got_shl = value << amount;
         const uint32_t got_ushr = value >> amount;
         const uint32_t got_ishr = model_ishr(value, amount);
         const uint32_t want_ishr = reference_ishr(value, amount);

         if (got_shl != want_shl || got_ushr != want_ushr || got_ishr != want_ishr) {
            fprintf(stderr,
                    "FAIL math value=%08" PRIx32 " amount=%u shl=%08" PRIx32 "/%08" PRIx32
                    " ushr=%08" PRIx32 "/%08" PRIx32 " ishr=%08" PRIx32 "/%08" PRIx32 "\n",
                    value,
                    amount,
                    got_shl,
                    want_shl,
                    got_ushr,
                    want_ushr,
                    got_ishr,
                    want_ishr);
            failures++;
         }
      }
   }

   printf("G5P DYNAMIC SHIFT %s (%d failed, 32 amounts, %zu signed/unsigned values)\n",
          failures ? "FAIL" : "PASS",
          failures,
          sizeof(values) / sizeof(values[0]));
   return failures != 0;
}
