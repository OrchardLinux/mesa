/* SPDX-License-Identifier: MIT */
/*
 * The A8's fetch sampler: the Rogue sampler words a texelFetch drawn as a sample
 * reads, point filtered, non-normalized, clamped to the edge and to one level, so the texture's own sampler state
 * cannot change the texel. From the binding's sampler words (its border index, its comparison off) or from zero.
 * Also the A8's image-size upload (pvr_g5p_a8_tex_size()).
 * Include after pvr_csb.h.
 */
#ifndef PVR_G5P_A8_FETCH_H
#define PVR_G5P_A8_FETCH_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "pvr_device_info.h"
#include "pvr_g5p_tex_encode.h"

static inline void
pvr_g5p_a8_fetch_sampler(uint32_t out[4], const uint32_t *app, unsigned level)
{
   uint64_t w0 = 0, w1 = 0;
   struct ROGUE_TEXSTATE_SAMPLER_WORD0 s;

   if (app) {
      memcpy(&w0, app, sizeof(w0));
      memcpy(&w1, app + 2, sizeof(w1));
   }
   ROGUE_TEXSTATE_SAMPLER_WORD0_unpack(&w0, &s);
   s.minfilter = ROGUE_TEXSTATE_FILTER_POINT;
   s.magfilter = ROGUE_TEXSTATE_FILTER_POINT;
   s.mipfilter = false;
   s.anisoctl = ROGUE_TEXSTATE_ANISOCTL_DISABLED;
   s.addrmode_u = ROGUE_TEXSTATE_ADDRMODE_CLAMP_TO_EDGE;
   s.addrmode_v = ROGUE_TEXSTATE_ADDRMODE_CLAMP_TO_EDGE;
   s.addrmode_w = ROGUE_TEXSTATE_ADDRMODE_CLAMP_TO_EDGE;
   s.non_normalized_coords = true;
   /* the level the fetch reads (0, or a constant level with PVR_A8_TEXFETCH_LEVEL) */
   s.minlod = level << ROGUE_TEXSTATE_CLAMP_FRACTIONAL_BITS;
   s.maxlod = level << ROGUE_TEXSTATE_CLAMP_FRACTIONAL_BITS;
   s.dadjust = ROGUE_TEXSTATE_DADJUST_ZERO_UINT;
   ROGUE_TEXSTATE_SAMPLER_WORD0_pack(&w0, &s);
   memcpy(out, &w0, sizeof(w0));
   memcpy(out + 2, &w1, sizeof(w1));
}

/* The sampler of PVR_A8_TEXFETCH_LEVEL=apple (PCO_G5P_UPLOAD_SAMPLER_READ): point, mip nearest, CLAMP_TO_EDGE,
 * normalized, LOD 0..MAX, no LOD bias (0x02002401df800fff, word 1 0), as read from Apple's compiler binary; with byte 6 bit 2 and the level
 * in a register it read levels 1 and 2 right on the iPod. Apple's own read sampler, in an iOS capture, is
 * the fetch sampler's words instead (0x0202240000000fff: non-normalized, LOD 0..0): pvr_g5p_a8_fetch_sampler(). */
static inline void
pvr_g5p_a8_read_sampler(uint32_t out[4])
{
   uint64_t w0 = 0;
   struct ROGUE_TEXSTATE_SAMPLER_WORD0 s;

   memset(&s, 0, sizeof(s));
   s.minfilter = ROGUE_TEXSTATE_FILTER_POINT;
   s.magfilter = ROGUE_TEXSTATE_FILTER_POINT;
   s.mipfilter = false;
   s.anisoctl = ROGUE_TEXSTATE_ANISOCTL_DISABLED;
   s.addrmode_u = ROGUE_TEXSTATE_ADDRMODE_CLAMP_TO_EDGE;
   s.addrmode_v = ROGUE_TEXSTATE_ADDRMODE_CLAMP_TO_EDGE;
   s.addrmode_w = ROGUE_TEXSTATE_ADDRMODE_CLAMP_TO_EDGE;
   s.non_normalized_coords = false;
   s.minlod = ROGUE_TEXSTATE_CLAMP_MIN;
   s.maxlod = ROGUE_TEXSTATE_CLAMP_MAX;
   s.dadjust = ROGUE_TEXSTATE_DADJUST_ZERO_UINT;
   ROGUE_TEXSTATE_SAMPLER_WORD0_pack(&w0, &s);
   memcpy(out, &w0, sizeof(w0));
   out[2] = out[3] = 0;
}

/*
 * The A8's image-size upload (PCO_G5P_UPLOAD_IMAGE_SIZE: textureSize at `level`, textureQueryLevels), the
 * counterpart of pvr_g5p_tex_size() for the A9's words. An A8 view keeps Rogue's image words
 * (pvr_arch_image.c, pvr_arch_tex_state.c): width - 1 and height - 1 in IMAGE_WORD0 bits 34-47 and 48-61, which
 * AGXMetalA8's own states have as well (its 4x4 RGBA32I, f8000a61 0003000d: width and height fields 3). In
 * IMAGE_WORD1: the base level, the view's level count and the depth (3D) or layer count - 1; a linear (STRIDE)
 * state has one level and no depth. pvr_g5p_tex_size() reads the A9's fields (width at bit 27, height at bit 41),
 * which on these words give sizes in the thousands: an atlas coordinate divided by textureSize() came out near
 * zero. Returns false for a state that isn't valid.
 */
static inline bool
pvr_g5p_a8_tex_size(const uint32_t state[4], unsigned level, uint32_t out[4])
{
   uint64_t w0, w1;
   struct ROGUE_TEXSTATE_IMAGE_WORD0 i0;
   unsigned base = 0, levels = 1, depth = 1;

   memcpy(&w0, state, sizeof(w0));
   memcpy(&w1, state + 2, sizeof(w1));
   if (!w0)
      return false;
   ROGUE_TEXSTATE_IMAGE_WORD0_unpack(&w0, &i0);
   if (i0.textype == ROGUE_TEXSTATE_TEXTYPE_STRIDE) {
      struct ROGUE_TEXSTATE_STRIDE_IMAGE_WORD1 s1;

      ROGUE_TEXSTATE_STRIDE_IMAGE_WORD1_unpack(&w1, &s1);
      levels = s1.num_mip_levels ? s1.num_mip_levels : 1;
   } else {
      struct ROGUE_TEXSTATE_IMAGE_WORD1 i1;

      ROGUE_TEXSTATE_IMAGE_WORD1_unpack(&w1, &i1);
      base = i1.baselevel;
      levels = i1.num_mip_levels ? i1.num_mip_levels : 1;
      depth = i1.depth + 1;
   }

   const unsigned l = base + level;
   const uint32_t w = i0.width + 1, h = i0.height + 1;

   out[0] = w >> l ? w >> l : 1;
   out[1] = h >> l ? h >> l : 1;
   switch (i0.textype) {
   case ROGUE_TEXSTATE_TEXTYPE_3D:
      out[2] = depth >> l ? depth >> l : 1;
      break;
   case ROGUE_TEXSTATE_TEXTYPE_CUBE:
      out[2] = 6;
      break;
   default: /* 1D, 2D, 2D array (the layer count), linear */
      out[2] = depth;
      break;
   }
   out[3] = levels;
   return true;
}

/* The image-size upload from a descriptor's words on either GPU */
static inline bool
pvr_g5p_image_size(const struct pvr_device_info *dev_info, const uint32_t state[4], unsigned level, uint32_t out[4])
{
   return PVR_HAS_FEATURE(dev_info, apple_g4p) ? pvr_g5p_a8_tex_size(state, level, out)
                                               : pvr_g5p_tex_size(state, level, out);
}

/* Apple's vertex read() sampler, from an iOS capture (AGXMetalA8's vread2_f32: the program's constant data, DMA'd to
 * vertex common dwords 4..7): 0x04004801df800fff, word 1 0 -- point, mip off, CLAMP_TO_BORDER on u/v/w,
 * normalized, LOD 0..MAX, no LOD bias. The sample's read bit (byte 6 bit 2) addresses texels; the level is Apple's
 * constant 0.0 in the common store. */
static inline void
pvr_g5p_a8_vread_sampler(uint32_t out[4])
{
   uint64_t w0 = 0;
   struct ROGUE_TEXSTATE_SAMPLER_WORD0 s;

   memset(&s, 0, sizeof(s));
   s.minfilter = ROGUE_TEXSTATE_FILTER_POINT;
   s.magfilter = ROGUE_TEXSTATE_FILTER_POINT;
   s.mipfilter = false;
   s.anisoctl = ROGUE_TEXSTATE_ANISOCTL_DISABLED;
   s.addrmode_u = ROGUE_TEXSTATE_ADDRMODE_CLAMP_TO_BORDER;
   s.addrmode_v = ROGUE_TEXSTATE_ADDRMODE_CLAMP_TO_BORDER;
   s.addrmode_w = ROGUE_TEXSTATE_ADDRMODE_CLAMP_TO_BORDER;
   s.non_normalized_coords = false;
   s.minlod = ROGUE_TEXSTATE_CLAMP_MIN;
   s.maxlod = ROGUE_TEXSTATE_CLAMP_MAX;
   s.dadjust = ROGUE_TEXSTATE_DADJUST_ZERO_UINT;
   ROGUE_TEXSTATE_SAMPLER_WORD0_pack(&w0, &s);
   memcpy(out, &w0, sizeof(w0));
   out[2] = out[3] = 0;
}

#endif
