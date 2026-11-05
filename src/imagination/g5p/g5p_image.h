/* SPDX-License-Identifier: MIT */
#ifndef G5P_IMAGE_H
#define G5P_IMAGE_H
#include "../g9p/g9p_isa.h"
/* RE §4 raw32-bit-channel image access. Format conversion is not implicit:
 * only the pipeline's witnessed/probed raw format may use this form. */
static inline void
g5p_image_raw32(struct g9p_code *c, bool store, unsigned data,
                unsigned coords, unsigned descriptor_dword, unsigned lod_dword)
{
   assert(!c->fragment && data+3<32 && coords+1<32 && !(descriptor_dword&1) && lod_dword<32);
   G9P_EMIT(c,0xf51a,0x0100 | (descriptor_dword/2)<<9 | (store?0xc0:0x40) | 2*data,
            0x8100 | coords<<9 | (store?0:8),0x0140 | lod_dword,
            store?0x8100:0x8180,0x0008);
   if(!store)g9p_wait(c);
}
#endif
