/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * Exact Apple A9 fragment derivatives, including tokens within uniform loops.
 * Source: tools/ios-witness-r1/witness.metal fw_l7_*; apple-r3 captures.
 */
#include "g5p_usc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void)
{
   const unsigned char expected[][10]={
      {0x98,0x01,0x00,0x01,0x80,0x01,0x44,0x01,0x00,0x02}, /* dfdx_dead */
      {0x98,0x01,0x0a,0x01,0x94,0x01,0x44,0x01,0x00,0x04}, /* dfdy_loop_dead */
      {0x9c,0x01,0x0a,0x01,0x08,0x01,0x44,0x01,0x00,0x02}, /* both_loop dx */
      {0x98,0x01,0x08,0x01,0x08,0x01,0x44,0x01,0x00,0x04}, /* both_loop dy */
   };
   const unsigned dst[]={0,10,10,8},src[]={0,10,4,4},grp[]={1,1,0,0},ext[]={0,0,4,0};
   for(unsigned i=0;i<4;i++){
      unsigned char raw[G5P_USC_MAX];
      unsigned n=g5p_usc_derivative(raw,i==1||i==3,dst[i],src[i],grp[i],1,ext[i],0);
      if(n!=10 || memcmp(raw,expected[i],10)){fprintf(stderr,"native derivative mismatch %u\n",i);abort();}
   }
   puts("DERIVATIVE ENCODINGS PASS 4 exact native A9 instructions, including uniform loop");
   return 0;
}
