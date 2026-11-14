/* SPDX-License-Identifier: MIT
 * Generated-by: LLM. Exact observer-mapped native PBE words, plus linear ABI.
 */
#include "g5p_pbe.h"
#include <assert.h>
#include <stdio.h>
#include <inttypes.h>
int main(int argc,char **argv)
{
 assert(argc==2);FILE *f=fopen(argv[1],"r");assert(f);
 uint64_t q[3],got[3];unsigned n=0;
 while(fscanf(f,"%"SCNx64" %"SCNx64" %"SCNx64,&q[0],&q[1],&q[2])==3) {
  assert(q[0]&8);uint32_t w=((q[0]>>27)&0x3fff)+1,h=((q[0]>>41)&0x3fff)+1;
  uint64_t va=(q[1]&((1ull<<39)-1))<<1;
  g5p_pbe_2d(got,q[0]&((1u<<27)-1),(q[1]>>45)&1,va,w,h,0,true);
  for(unsigned i=0;i<3;i++)assert(got[i]==q[i]);n++;
 }
 assert(!ferror(f));fclose(f);assert(n>=8);
 for(unsigned s=16;s<=1048576;s*=2) {
  const unsigned stride=s/16-1;
  g5p_pbe_2d(got,0x344050a,false,0x100038000ull,37,29,s,false);
  assert(got[0]==((28ull<<41)|(36ull<<27)|0x3440502));
  assert(got[1]==((0x100038000ull>>1)|((uint64_t)(stride&0x3fffc)<<45)|(1ull<<63)));
  assert(got[2]==(uint64_t)(stride&3)<<62);
 }
 printf("PBE PASS %u native twiddled descriptors, preserved linear pitch/address ABI\n",n);
}
