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

#endif
