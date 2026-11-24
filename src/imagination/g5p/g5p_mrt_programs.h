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

#include "g5p_usc.h"

#define G5P_MRT_MAX 8 /* 5..8: compact UNORM8 targets */
#define G5P_MRT_PBE_DWORDS 7      /* per target */
#define G5P_MRT_PBE_STRIDE 8      /* dwords per target in the common store */
#define G5P_MRT_PBE_HALF_OFFSET 6 /* a target's output-half offset in its PBE block */
#define G5P_MRT_CLEAR_DWORDS 4    /* f32 clear colour per target */
#define G5P_MRT_CLEAR_USC_MAX (16 * 4 * G5P_MRT_MAX + 8)
#define G5P_MRT_EOT_USC_MAX 194

/* Target counts with an Apple witness for the whole pass: 1, 2 and 4; 5..8
 * as compact UNORM8 (verified on the A9 over 104 test images). */
static inline bool g5p_mrt_count_ok(unsigned n)
{
   return n == 1 || n == 2 || (n >= 4 && n <= 8);
}

/* Pixel output width of the pass (UAPI pixel_output_width): two output
 * registers per 8-bit RGBA target. iOS record+0x6dc: 2, 4, 8. */
static inline uint32_t g5p_mrt_pixel_width(unsigned n)
{
   return n > 4 ? 8 : 2 * n; /* compact: one dword per target, the kernel's width 8 */
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
   /* the same values for 1, 2, 4; 0xc02 and 0x1002 for 5..8 (39..63 dwords) */
   return ((g5p_mrt_pbe_dwords(n) + 15) / 16) * 0x400 | 2;
}

/* The A8's EVENT_PIXEL_PDS_INFO: usc_sr_size in Rogue's unit, DIV_ROUND_UP(dwords,
 * 16) at bit 9, not twice that. Captured 0x202 for a one-target 64x64 BGRA8 pass
 * (iPod touch 6, 3D record +0x3c4, from a live iOS dump),
 * where the A9 writes 0x402. Two or more targets are not captured on the A8. */
static inline uint32_t g5p_mrt_eot_info_g4p(unsigned n)
{
   return ((g5p_mrt_pbe_dwords(n) + 15) / 16) << 9 | 2;
}

/* Partial-render restore SIZEINFO (restore[1], record+0x51c): usc_sharedsize
 * of Apple's N-image restore loader, 0x800000 per target (1, 2, 4 units). */
static inline uint32_t g5p_mrt_restore1(unsigned n)
{
   return 0x800000u * n;
}

/* The PBE block: target t's three qwords at dword 8t, its output-half offset
 * at dword 8t + 6, 0xffffffff at dword 8t + 7 except after the last.
 * halves: output halves per target, 4 for an 8-bit RGBA target (two packed
 * f16x2 dwords), 8 for a 32-bit integer target of the raw32 output (four
 * raw words, so target 1 starts at output dword 4). */
static inline unsigned g5p_mrt_pbe_block_halves(uint32_t *out, unsigned n, const uint64_t pbe[][3], unsigned halves)
{
   for (unsigned t = 0; t < n; t++) {
      uint32_t *w = out + G5P_MRT_PBE_STRIDE * t;
      for (unsigned q = 0; q < 3; q++) {
         w[2 * q] = (uint32_t)pbe[t][q];
         w[2 * q + 1] = (uint32_t)(pbe[t][q] >> 32);
      }
      w[6] = halves * t;
      if (t + 1 < n)
         w[7] = 0xffffffff;
   }
   return g5p_mrt_pbe_dwords(n);
}
static inline unsigned g5p_mrt_pbe_block(uint32_t *out, unsigned n, const uint64_t pbe[][3])
{
   return g5p_mrt_pbe_block_halves(out, n, pbe, 4);
}

/* 5..8 UNORM targets: one packed dword per target; BGRA8
 * byte order is B,G,R,A. Each f32 clear channel in common dword 4t + c is
 * converted to UNORM8 into one byte of an output half (g5p_usc_unorm8_byte(),
 * the packing form recovered from controlled probes), then one emitpix
 * writes the N packed dwords and the program ends. Returns its size,
 * 16 * 4N + 6, or 0 when an encoder refuses (never for 5..8 targets). */
static inline unsigned g5p_mrt_clear_compact(uint8_t *out, unsigned n, bool bgra)
{
   unsigned o = 0, len;
   for (unsigned t = 0; t < n; t++)
      for (unsigned h = 0; h < 2; h++)
         for (unsigned byte = 0; byte < 2; byte++) {
            const unsigned c = h * 2 + byte;
            const unsigned channel = bgra && !(c & 1) ? 2 - c : c;
            const unsigned src = 4 * t + channel, dst = 2 * t + h;
            len = g5p_usc_unorm8_byte(out + o, dst, byte, src);
            if (!len)
               return 0;
            o += len;
         }
   /* emitpix of the N packed dwords from output half 0, then END */
   len = g5p_usc_emit_mrt(out + o, n, 0);
   if (!len)
      return 0;
   o += len;
   return o + g5p_usc_special(out + o, true);
}

/* The clear program. Returns its size, 16 * 4N + 6. */
/*
 * The A8's background program for one R8 (halves 1) or RG8 (halves 2) target: AGXMetalA8's (iOS 12.4, a8-passcap):
 * the clear colour's low channels converted as for four (g5p_mrt_clear_usc()), for R8 a zero into half 1
 * (0188 0101 0101 0120 0128 0039), then the one-word emitpix 111c 0900 8200 and END
 */
static inline unsigned g5p_mrt_clear_usc_g4p_narrow(uint8_t *out, unsigned halves)
{
   unsigned o = 0;
   for (int c = (int)halves - 1; c >= 0; c--) {
      const uint8_t h = (uint8_t)c;
      const uint8_t ins[16] = {0x8a, 0x01, h, 0x01, 0x01, 0x01, 0xa0, 0x01,
                               (uint8_t)(4 * h + 1), 0x01, 0x40, 0x41, 0x04, 0x01, 0xb0, 0x00};
      memcpy(out + o, ins, sizeof(ins));
      o += sizeof(ins);
   }
   if (halves == 1) {
      static const uint8_t zero_g[12] = {0x88, 0x01, 0x01, 0x01, 0x01, 0x01, 0x20, 0x01, 0x28, 0x01, 0x39, 0x00};
      memcpy(out + o, zero_g, sizeof(zero_g));
      o += sizeof(zero_g);
   }
   const unsigned len = g5p_usc_emit(out + o, 1, 0);
   if (!len)
      return 0;
   return o + len + g5p_usc_special(out + o + len, true);
}

/* The A8's RGBA8UI background program (AGXMetalA8, a8-passcap rgba8u): the four clear uints in common-store
 * dwords 0..3 clamped to 255 and packed into one output word R | G << 8 | B << 16 | A << 24, high half first:
 * r0 = umin(255, B), r1 = umin(255, A), h1 = h0 | h2 << 8; r1 = umin(255, R), r2 = umin(255, G),
 * h0 = h2 | h4 << 8; emitpix one word. The second clamp carries Apple's destination control 0x80. */
static inline unsigned g5p_mrt_clear_usc_g4p_rgba8ui(uint8_t *out)
{
   static const struct { uint8_t dst, ctl, dword; } clamp[2][2] = {
      { { 0, 0, 2 }, { 2, 0x80, 3 } },
      { { 2, 0, 0 }, { 4, 0, 1 } },
   };
   unsigned o = 0, len;
   for (unsigned i = 0; i < 2; i++) {
      for (unsigned k = 0; k < 2; k++) {
         len = g5p_usc_minmax_literal_shared(out + o, G5P_UMIN, clamp[i][k].dst, clamp[i][k].ctl, 255,
                                             clamp[i][k].dword);
         if (!len)
            return 0;
         o += len;
      }
      len = g5p_usc_pack_byte_half(out + o, 1 - i, clamp[i][0].dst, clamp[i][1].dst, 8);
      if (!len)
         return 0;
      o += len;
   }
   len = g5p_usc_emit(out + o, 1, 0);
   if (!len)
      return 0;
   return o + len + g5p_usc_special(out + o + len, true);
}

/* The A8's R32F background program (AGXMetalA8, a8-passcap r32f): the clear colour's red f32 (common-store
 * dword 0) moved into output word 0, emitpix one word. */
static inline unsigned g5p_mrt_clear_usc_g4p_r32f(uint8_t *out)
{
   unsigned o = g5p_usc_mov_shared(out, 0, 0, 0), len;
   if (!o)
      return 0;
   len = g5p_usc_emit(out + o, 1, 0);
   if (!len)
      return 0;
   return o + len + g5p_usc_special(out + o + len, true);
}

/* The A8's colour LOAD background (AGXMetalA8, a8-passcap load: passcap USC heap 0x2180, the same bytes before and
 * after commit): the fragment position converted and transformed by the 2x3 matrix in common-store dwords 8..13, the A8
 * narrow 2D sample of the texture state in dwords 0..3 with the sampler in dwords 4..7, emitpix of two words (the f16
 * pair BGRA8's end of tile takes). Its inputs are the three DOUTDs of the background data (pvr_arch_g5p.c). The bytes are
 * Apple's: the conversion and FMA forms with common-store operands have no encoder here yet. */
#define G5P_A8_LOAD_USC_BYTES 124
static inline unsigned g5p_mrt_load_usc_g4p(uint8_t *out)
{
   static const uint8_t usc[G5P_A8_LOAD_USC_BYTES] = {
      0x1c, 0x21, 0x00, 0x68, 0x4d, 0x00, 0x99, 0x01, 0x00, 0x01, 0x15, 0x81, 0x20, 0x01, 0x68, 0x04, 0x99, 0x01,
      0x04, 0x81, 0x01, 0x81, 0x30, 0x51, 0x48, 0x09, 0x70, 0x00, 0x99, 0x01, 0x00, 0x01, 0x15, 0x85, 0x20, 0x01,
      0x68, 0x04, 0x99, 0x01, 0x00, 0x81, 0x01, 0x81, 0x30, 0x51, 0x48, 0x09, 0x70, 0x00, 0x9e, 0x41, 0x02, 0x01,
      0xa5, 0x09, 0x00, 0x41, 0xb5, 0x01, 0x00, 0x41, 0x4c, 0x00, 0x9e, 0x41, 0x02, 0x01, 0xad, 0x01, 0x00, 0x41,
      0x84, 0x01, 0x4c, 0x00, 0x9a, 0x01, 0x04, 0x01, 0xa1, 0x89, 0x00, 0x41, 0xb1, 0x01, 0x00, 0x41, 0x4c, 0x00,
      0x9a, 0x01, 0x00, 0x01, 0xa9, 0x81, 0x00, 0x41, 0x88, 0x01, 0x4c, 0x00, 0x1a, 0x59, 0x00, 0x41, 0x00, 0x01,
      0x88, 0x01, 0x00, 0x01, 0x00, 0x80, 0x4d, 0x00, 0x1c, 0x21, 0x00, 0x09, 0x00, 0x82, 0x00, 0x80
   };
   memcpy(out, usc, sizeof(usc));
   return sizeof(usc);
}

/* The A8's LOAD background for one R8 target (AGXMetalA8, a8-passcap r8_load: an R8Unorm target with loadAction
 * Load, USC heap 0x1fc0; the same bytes are in the r8 pass's heap at 0x2140): BGRA8's program up to the wait after the
 * sample (its first 116 bytes), then, as R8's clear and fragment programs, a zero into half 1
 * (0188 0101 0101 0120 0128 0039) and the one-word emitpix 111c 0900 8200 and END. 136 bytes. The texture state, the
 * sampler, the matrix and their three DOUTDs (10000004 10008004 90010006) are BGRA8's; only the state's format
 * differs. */
#define G5P_A8_LOAD_R8_USC_BYTES 136
static inline unsigned g5p_mrt_load_usc_g4p_r8(uint8_t *out)
{
   static const uint8_t zero_g[12] = {0x88, 0x01, 0x01, 0x01, 0x01, 0x01, 0x20, 0x01, 0x28, 0x01, 0x39, 0x00};
   uint8_t bgra[G5P_A8_LOAD_USC_BYTES];
   g5p_mrt_load_usc_g4p(bgra);
   unsigned o = G5P_A8_LOAD_USC_BYTES - 8; /* up to and with the wait (4d 00); BGRA8's two-word emitpix and END after */
   memcpy(out, bgra, o);
   memcpy(out + o, zero_g, sizeof(zero_g));
   o += sizeof(zero_g);
   const unsigned len = g5p_usc_emit(out + o, 1, 0);
   if (!len)
      return 0;
   return o + len + g5p_usc_special(out + o + len, true);
}

static inline unsigned g5p_mrt_clear_usc(uint8_t *out, unsigned n)
{
   if (n > 4)
      return g5p_mrt_clear_compact(out, n, true);
   unsigned o = 0;
   /* The A8 (AGXMetalA8, a8-passcap mrt2) converts the last target first and reads target t's clear colour from
    * slot n - 1 - t of the clear DMA (its colours in reverse target order); the A9 in order */
   for (unsigned i = 0; i < n; i++)
      for (int c = 3; c >= 0; c--) {
         const unsigned t = g5p_usc_g4p ? n - 1 - i : i, slot = g5p_usc_g4p ? n - 1 - t : t;
         uint8_t h = 4 * t + c;
         const uint8_t src = (uint8_t)(4 * slot + c);
         const uint8_t ins[16] = {0x8a, 0x01, h, 0x01, 0x01, 0x01, 0xa0, 0x01,
                                  (uint8_t)(4 * src + 1), 0x01, 0x40, 0x41, 0x04, 0x01, 0xb0, 0x00};
         memcpy(out + o, ins, sizeof(ins));
         o += sizeof(ins);
      }
   /* emitpix of 2N words (the 3-bit field wraps: 8 is 0), then END. The A8 takes the long emitpix
    * (g5p_usc_emit()); its background program for one target is otherwise these bytes (Apple's, iOS 12.4) */
   if (g5p_usc_g4p) {
      const unsigned len = g5p_usc_emit(out + o, 2 * n, 0);
      if (!len)
         return 0;
      return o + len + g5p_usc_special(out + o + len, true);
   }
   const uint8_t tail[6] = {0x1c, (uint8_t)(((2 * n) & 7) << 4 | 1), 0x00, 0x08, 0x00, 0x80};
   memcpy(out + o, tail, sizeof(tail));
   return o + sizeof(tail);
}

/* The end-of-tile program. Returns its size: 66 for one or two targets, 130
 * for four; 0 when an encoder refuses (never for the counts g5p_mrt_count_ok() accepts). */
static inline unsigned g5p_mrt_eot_usc_mode(uint8_t *out, unsigned n, bool integer)
{
   unsigned o = 0;
   if (g5p_usc_g4p) {
      /* The A8's end of tile for one target: 1e 01 00 70, which takes the target from the common store
       * (its address and 2, 0: the EOT DMA), then the A9's branch to an END at the next 64-byte boundary
       * (Apple's program, iOS 12.4, a8-shadercap). Other target counts and the integer form are not
       * captured. */
      if ((n != 1 && n != 2) || integer) {
         g5p_usc_error = "A8 end of tile captured for one or two float targets only";
         return 0;
      }
      /* two targets (Apple's, a8-passcap mrt2): 1c 01 00 70 (target 0), 4c 04, 1e 01 00 71 08 00 (target 1, its
       * state from the second EOT DMA block), then the branch to the END at byte 64 */
      static const uint8_t g4_1[10] = {0x1e, 0x01, 0x00, 0x70, 0x00, 0x01, 0x3c, 0x01, 0x00, 0x00};
      static const uint8_t g4_2[18] = {0x1c, 0x01, 0x00, 0x70, 0x4c, 0x04, 0x1e, 0x01, 0x00, 0x71,
                                       0x08, 0x00, 0x00, 0x01, 0x34, 0x01, 0x00, 0x00};
      const uint8_t *g4 = n == 2 ? g4_2 : g4_1;
      const unsigned g4_len = n == 2 ? sizeof(g4_2) : sizeof(g4_1);
      memcpy(out, g4, g4_len);
      for (o = g4_len; o < 64; o += 2) {
         out[o] = 0x00;
         out[o + 1] = 0xc0;
      }
      out[o] = 0x00;
      out[o + 1] = 0x80;
      return o + 2;
   }
   for (unsigned k = 0; k < n; k++) {
      /* output half 2 * (n - 1 - k) = target k's output-half offset (in its PBE block) */
      const unsigned len = g5p_usc_mov_shared_half(out + o, 2 * (n - 1 - k),
                                                   G5P_MRT_PBE_STRIDE * k + G5P_MRT_PBE_HALF_OFFSET);
      if (!len)
         return 0;
      o += len;
   }
   for (unsigned k = 0; k < n; k++) {
      bool last = k + 1 == n;
      uint8_t t = n - 1 - k;
      const uint8_t ins[8] = {0x1c, (uint8_t)(0x05 | (integer ? 0x10 : 0)), (uint8_t)(2 * t), last ? 0x51 : 0x71, (uint8_t)(2 * k), 0x08,
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

/* Integer source mode (bit 4) comes from the A9 builtin EOT compiler for
 * unsigned integer attachments. Signed 32-bit stores use the identical
 * unsigned physical layout to preserve their two's-complement bit patterns.
 * The original float entry point and all of its bytes remain unchanged.
 */
static inline unsigned g5p_mrt_eot_usc(uint8_t *out, unsigned n)
{
   /* 5..8: packed bytes through a physical R32UI store, the integer source mode */
   return g5p_mrt_eot_usc_mode(out, n, n > 4);
}
#endif
