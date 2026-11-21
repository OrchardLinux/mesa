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

/*
 * The A8's colour target words: Rogue's PBESTATE_REG_WORD0/1, as AGXMetalA8 packs them (iOS 12.4,
 * sub_1a13fae24). The kernel copies them to
 * the 3D record's W+0x118 + 16 i (and +0x4c0, +0x540). Word 0: tilerelative (bit 57), the swizzle (three bits
 * per channel from bit 20; RGBA8 0o3210, BGRA8 0o3012), memlayout TWIDDLE_2D (bit 32) or the linear row stride
 * in two-pixel units (bits 34-49). Word 1: maxclip x (bits 32-45) and y (0-13), and for a twiddled target
 * the log2 sizes (x at 60, y at 28). Captured on the A8 for 64x64 BGRA8 and RGBA8 twiddled targets:
 * 0x0200000160a00000 / 0x0200000168800000 and 0x6000003f6000003f.
 */
#define G5P_PBE_A8_SWIZZLE_RGBA 03210u
#define G5P_PBE_A8_SWIZZLE_BGRA 03012u
/* An unused target: an empty clip rectangle, minclip (1, 1) and maxclip (0, 0) */
#define G5P_PBE_A8_EMPTY0 0x40ull
#define G5P_PBE_A8_EMPTY1 (0x4000ull << 32)

static inline unsigned g5p_pbe_a8_log2_ceil(uint32_t n)
{
   unsigned l = 0;
   while ((1u << l) < n)
      l++;
   return l;
}

static inline void g5p_pbe_a8_reg_words(uint64_t w[2], uint32_t width, uint32_t height, unsigned swizzle,
                                        bool twiddled, uint32_t stride_px)
{
   w[0] = 1ull << 57 | (uint64_t)(swizzle & 07777) << 20;
   if (twiddled)
      w[0] |= 1ull << 32;
   else
      w[0] |= (uint64_t)(((stride_px - 1) >> 1) & 0xffff) << 34;
   w[1] = (uint64_t)((width - 1) & 0x3fff) << 32 | ((height - 1) & 0x3fff);
   if (twiddled)
      w[1] |= (uint64_t)(g5p_pbe_a8_log2_ceil(width) & 0xf) << 60 | (uint64_t)(g5p_pbe_a8_log2_ceil(height) & 0xf) << 28;
}
#endif
