/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * Exact native Apple A9 iv_reg_pop/reverse, last and repeated source reads.
 * tools/ios-witness-r1/l7-integer.metal, apple-r4 register pipeline captures.
 */
#include "g5p_usc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void)
{
   const unsigned char expected[][8]={
      {0x98,0x21,0x08,0x03,0x90,0x01,0xe8,0x00},
      {0x98,0x21,0x0e,0x03,0x10,0x01,0xe8,0x00},
      {0x98,0x21,0x08,0x03,0x90,0x01,0xe4,0x00},
      {0x98,0x21,0x0e,0x03,0x10,0x01,0xe4,0x00},
   };
   for(unsigned i=0;i<4;i++) {
      unsigned char raw[G5P_USC_MAX];
      unsigned n=g5p_usc_unary2(raw,i<2?G5P_POPCOUNT:G5P_BITREVERSE,
                               i%2?14:8,8,1,i%2?0:1,0x21,false,0,0,false);
      if(n!=8 || memcmp(raw,expected[i],8)){fprintf(stderr,"native integer encoding mismatch %u\n",i);abort();}
   }
   puts("INTEGER ENCODINGS PASS 4 exact native Apple A9 vertex programs");
   return 0;
}
