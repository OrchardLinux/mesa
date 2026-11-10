/* SPDX-License-Identifier: MIT */
/* The per-pass programs and words of a render pass with N colour targets.
 *
 * Plain data, free of Vulkan and UAPI types, so a host test can compare every
 * byte with Apple's own N-target pass (our iOS captures of one-, two- and
 * four-target passes, 64x64 BGRA8; the comparison is a private host test,
 * not part of this tree). The programs are built by these functions from
 * their instruction fields; no captured program is stored here.
 *
 * Clear (background object, BGRND0/2): the uniform loader DMAs the clear
 * colours, four f32 per target, to common 0..4N-1. The USC program converts
 * common h to output half h, components 3..0 of each target in turn, then
 * one emitpix writes all 2N output dwords.
 *
 * End of tile (EVENT_PIXEL_PDS): one DOUTD moves the PBE state to common
 * 0..8N-2: seven dwords per target (the three g5p_pbe_encode() qwords and the
 * target's output-half offset 4t), 0xffffffff between targets so each target
 * starts on an eight-dword boundary. The USC program loads the targets' state
 * in reverse order, emits one PBE write per target and branches to an END at
 * the next 64-byte boundary. The PDS program (DOUTD, DOUTU, HALT) is the same
 * for every N; only its data and EVENT_PIXEL_PDS_INFO change.
 */
#ifndef G5P_MRT_PROGRAMS_H
#define G5P_MRT_PROGRAMS_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define G5P_MRT_MAX 4
#define G5P_MRT_PBE_DWORDS 7      /* per target */
#define G5P_MRT_PBE_STRIDE 8      /* dwords per target in the common store */
#define G5P_MRT_CLEAR_DWORDS 4    /* f32 clear colour per target */
#define G5P_MRT_CLEAR_USC_MAX (16 * 4 * G5P_MRT_MAX + 8)
#define G5P_MRT_EOT_USC_MAX 130

/* Target counts with an Apple witness for the whole pass: 1, 2 and 4. */
static inline bool g5p_mrt_count_ok(unsigned n)
{
   return n == 1 || n == 2 || n == 4;
}

/* Pixel output width of the pass (UAPI pixel_output_width): two output
 * registers per 8-bit RGBA target. iOS record+0x6dc: 2, 4, 8. */
static inline uint32_t g5p_mrt_pixel_width(unsigned n)
{
   return 2 * n;
}

/* PBE dwords the end-of-tile DOUTD moves: 7, 15, 31. */
static inline unsigned g5p_mrt_pbe_dwords(unsigned n)
{
   return G5P_MRT_PBE_STRIDE * n - 1;
}

/* EVENT_PIXEL_PDS_INFO (eot[2]): const_size 2 (x4 dwords of PDS data),
 * usc_sr_size 2 for one or two targets and 4 for four (captured words
 * 0x402, 0x402, 0x802; Apple's sizes are twice DIV_ROUND_UP(dwords, 16)). */
static inline uint32_t g5p_mrt_eot_info(unsigned n)
{
   return n == 4 ? 0x802 : 0x402;
}

/* Partial-render restore SIZEINFO (restore[1], record+0x51c): usc_sharedsize
 * of Apple's N-image restore loader, 0x800000 per target (1, 2, 4 units). */
static inline uint32_t g5p_mrt_restore1(unsigned n)
{
   return 0x800000u * n;
}

/* The PBE block: target t's three qwords at dword 8t, its output-half offset
 * at dword 8t + 6, 0xffffffff at dword 8t + 7 except after the last. */
static inline unsigned g5p_mrt_pbe_block(uint32_t *out, unsigned n, const uint64_t pbe[][3])
{
   for (unsigned t = 0; t < n; t++) {
      uint32_t *w = out + G5P_MRT_PBE_STRIDE * t;
      for (unsigned q = 0; q < 3; q++) {
         w[2 * q] = (uint32_t)pbe[t][q];
         w[2 * q + 1] = (uint32_t)(pbe[t][q] >> 32);
      }
      w[6] = 4 * t;
      if (t + 1 < n)
         w[7] = 0xffffffff;
   }
   return g5p_mrt_pbe_dwords(n);
}

/* The clear program. Returns its size, 16 * 4N + 6. */
static inline unsigned g5p_mrt_clear_usc(uint8_t *out, unsigned n)
{
   unsigned o = 0;
   for (unsigned t = 0; t < n; t++)
      for (int c = 3; c >= 0; c--) {
         uint8_t h = 4 * t + c;
         const uint8_t ins[16] = {0x8a, 0x01, h, 0x01, 0x01, 0x01, 0xa0, 0x01,
                                  (uint8_t)(4 * h + 1), 0x01, 0x40, 0x41, 0x04, 0x01, 0xb0, 0x00};
         memcpy(out + o, ins, sizeof(ins));
         o += sizeof(ins);
      }
   /* emitpix of 2N words (the 3-bit field wraps: 8 is 0), then END */
   const uint8_t tail[6] = {0x1c, (uint8_t)(((2 * n) & 7) << 4 | 1), 0x00, 0x08, 0x00, 0x80};
   memcpy(out + o, tail, sizeof(tail));
   return o + sizeof(tail);
}

/* The end-of-tile program. Returns its size: 66 for one or two targets, 130
 * for four. */
static inline unsigned g5p_mrt_eot_usc(uint8_t *out, unsigned n)
{
   unsigned o = 0;
   for (unsigned k = 0; k < n; k++) {
      uint8_t t = n - 1 - k;
      const uint8_t ins[12] = {0x88, 0x01, (uint8_t)(2 * t), 0x01, (uint8_t)(0x99 + 0x20 * k), 0x01,
                               0x00, 0x41, 0x28, 0x01, 0x39, 0x00};
      memcpy(out + o, ins, sizeof(ins));
      o += sizeof(ins);
   }
   for (unsigned k = 0; k < n; k++) {
      bool last = k + 1 == n;
      uint8_t t = n - 1 - k;
      const uint8_t ins[8] = {0x1c, 0x05, (uint8_t)(2 * t), last ? 0x51 : 0x71, (uint8_t)(2 * k), 0x08,
                              0x4c, 0x04};
      memcpy(out + o, ins, last ? 6 : 8);
      o += last ? 6 : 8;
   }
   unsigned end = (o + 6 + 63) & ~63u;
   const uint8_t br[6] = {0x00, 0x01, (uint8_t)(end - o), 0x01, 0x00, 0x00};
   memcpy(out + o, br, sizeof(br));
   for (o += sizeof(br); o < end; o += 2) {
      out[o] = 0x00;
      out[o + 1] = 0xc0;
   }
   out[o] = 0x00;
   out[o + 1] = 0x80;
   return o + 2;
}

#endif
