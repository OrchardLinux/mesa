/* SPDX-License-Identifier: MIT */
/*
 * The A8's fetch sampler: the Rogue sampler words a texelFetch drawn as a sample
 * reads, point filtered, non-normalized, clamped to the edge and to one level, so the texture's own sampler state
 * cannot change the texel. From the binding's sampler words (its border index, its comparison off) or from zero.
 * Include after pvr_csb.h.
 */
#ifndef PVR_G5P_A8_FETCH_H
#define PVR_G5P_A8_FETCH_H

#include <stdint.h>
#include <string.h>

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
