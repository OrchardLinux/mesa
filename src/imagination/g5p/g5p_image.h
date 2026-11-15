/* SPDX-License-Identifier: MIT */
#ifndef G5P_IMAGE_H
#define G5P_IMAGE_H
#include "../g9p/g9p_isa.h"
/* Apple's captured image and array shaders pack 16/8-bit channels into two dwords (the sample class with bit 1
 * clear, as g5p_usc_sample()'s narrow result) or supply four raw dwords; the captured array shaders use the 2D
 * array dimension with integer x,y,layer coordinates. W0: that class byte and the dimension byte (all four
 * components in its high nibble and the dimension in its low one, as g5p_usc_sample_ext()); W1: the data
 * register's half, the integer-coordinate bit 6 (texture_base()'s read bit), the store bit 7 and the descriptor's
 * qword at bit 9. Keep raw32's qualified coordinate control and completion unchanged. */
#define G5P_IMAGE_CLASS          0x1a /* texture_base()'s sample class */
#define G5P_IMAGE_WIDE           0x02 /* clear: 16/8-bit channels packed into two dwords */
#define G5P_IMAGE_RGBA           0xf0
#define G5P_IMAGE_DIM_2D         0x5
#define G5P_IMAGE_INTEGER_COORDS 0x40
#define G5P_IMAGE_STORE          0x80
static inline void
g5p_image_access(struct g9p_code *c, bool store, bool packed, bool array,
                 unsigned data, unsigned coords, unsigned descriptor_dword, unsigned lod_dword)
{
   assert(!c->fragment && data + (packed ? 1 : 3) < 32 && coords + (array ? 2 : 1) < 32 &&
          !(descriptor_dword & 1) && lod_dword < 32);
   const unsigned dim = G5P_IMAGE_RGBA | (array ? G9P_TEX_2D_ARRAY : G5P_IMAGE_DIM_2D);
   G9P_EMIT(c, dim << 8 | (G5P_IMAGE_CLASS & ~(packed ? G5P_IMAGE_WIDE : 0)),
            G9P_CONT | (descriptor_dword / 2) << 9 | G5P_IMAGE_INTEGER_COORDS | (store ? G5P_IMAGE_STORE : 0) |
               2 * data,
            (store?0x0100:0x8100) | coords<<9 | (store?0:8),0x0140 | lod_dword,
            store?0x8100:0x8180,0x0008);
   if(!store)g9p_wait(c);
}
static inline void
g5p_image_raw32(struct g9p_code *c, bool store, unsigned data,
                unsigned coords, unsigned descriptor_dword, unsigned lod_dword)
{
   g5p_image_access(c, store, false, false, data, coords, descriptor_dword, lod_dword);
}
#endif
