/* SPDX-License-Identifier: MIT */
/* g3_selftest: re-encode the A7's R6 programs (g3_selftest()) and print the empty program */
#include <stdio.h>

#include "g3_compute.h"

int
main(void)
{
   char why[256];

   if (!g3_selftest(why, sizeof(why))) {
      printf("g3 self-test FAIL %s\n", why);
      return 1;
   }
   printf("g3 self-test PASS (R6 store32, lanes, combo and nop.end re-encoded byte for byte)\n");
   return 0;
}
