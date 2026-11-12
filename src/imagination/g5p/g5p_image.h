/* SPDX-License-Identifier: MIT */
#ifndef G5P_IMAGE_H
#define G5P_IMAGE_H
#include "../g9p/g9p_isa.h"
/* RE §4 raw32-bit-channel image access. Format conversion is not implicit:
 * only the pipeline's witnessed/probed raw format may use this form.
 *
 * The coordinate halfword's bit 15 (the sample encoder's coord_control) is
 * set on reads only. Apple's A9 compiler (build/gpu/compute-isa-re out-img,
 * out-image32: every texture write kernel) writes through integer
 * coordinates as `f51a 01c0 0900 ...` with that bit clear; the bit appears on
 * a write only with opcode f518 after the coordinates were converted to
 * float. The old `f51a ... 8900` store mixed the two: its pixels reached
 * memory, but a later job's imageLoad read pre-store data for the last
 * workgroups written (release9d imgdiag1: 384 of 4096 words, always whole
 * 8x8 groups of the tail, with or without glFinish between the jobs). */
static inline void
g5p_image_raw32(struct g9p_code *c, bool store, unsigned data,
                unsigned coords, unsigned descriptor_dword, unsigned lod_dword)
{
   assert(!c->fragment && data+3<32 && coords+1<32 && !(descriptor_dword&1) && lod_dword<32);
   G9P_EMIT(c,0xf51a,0x0100 | (descriptor_dword/2)<<9 | (store?0xc0:0x40) | 2*data,
            (store?0x0100:0x8100) | coords<<9 | (store?0:8),0x0140 | lod_dword,
            store?0x8100:0x8180,0x0008);
   if(!store)g9p_wait(c);
}
#endif
