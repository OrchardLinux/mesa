/* SPDX-License-Identifier: MIT
 * Generated-by: LLM. A9 native 2D PBE layout: linear rows or twiddled.
 * Address/dimensions are independent of the physical channel source ABI.
 */
#ifndef G5P_PBE_H
#define G5P_PBE_H
#include <stdbool.h>
#include <stdint.h>
static inline void g5p_pbe_2d(uint64_t q[3],uint32_t key,bool srgb,uint64_t va,
                              uint32_t width,uint32_t height,uint32_t stride,bool twiddled)
{
   const uint32_t s=twiddled?0:stride/16-1;
   q[0]=(uint64_t)(height-1)<<41|(uint64_t)(width-1)<<27|(key&~8u)|(twiddled?8u:0u);
   q[1]=va>>1|(uint64_t)(s&0x3fffc)<<45|(uint64_t)srgb<<45|1ull<<63;
   q[2]=(uint64_t)(s&3)<<62;
}
#endif
