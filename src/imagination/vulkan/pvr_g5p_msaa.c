/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): 4x multisampling words (pvr_g5p_msaa.h). Plain
 * arithmetic, checked by a host test against our iOS captures msaa4-store
 * and msaa4-resolve (2026-11-01); no captured word is stored here. The
 * hardware result of the same words: all 8 cases of our A9 ICD's 4x run
 * exact (2026-11-02).
 */

#include <string.h>

#include "g5p_mrt_programs.h"
#include "pvr_g5p_msaa.h"

/* the one-sample pattern: every sample at the pixel centre (8, 8) */
#define G5P_POSITIONS_1X 0x88u
/* DOUTU src0[34:33]: the fragment program's sample rate */
#define G5P_DOUTU_SAMPLE_RATE_SHIFT 33
#define G5P_DOUTU_SAMPLE_RATE_MASK (3ull << G5P_DOUTU_SAMPLE_RATE_SHIFT)
#define G5P_DOUTU_SAMPLE_RATE_4X 2ull
/* PBE q0: the image layout field [3:0] (bit 3 twiddled; 2D multisample is
 * layout 4) */
#define G5P_PBE_LAYOUT_MASK 0xfu
#define G5P_PBE_LAYOUT_MS_TWIDDLED (0x8u | 0x4u)
/* PBE q1: VA >> 1, bit 63 PBE, bit 45 sRGB; q2 bit 42: resolve */
#define G5P_PBE_Q1_PBE (1ull << 63)
#define G5P_PBE_Q1_SRGB (1ull << 45)
#define G5P_PBE_Q2_RESOLVE (1ull << 42)
/* emitpix's second byte: [1:0] the emit form, [3:2] the samples it writes
 * (1: one per pixel, 2: all four, a multisample store) */
#define G5P_EMIT_SAMPLES_SHIFT 2
#define G5P_EMIT_SAMPLES_MASK (3u << G5P_EMIT_SAMPLES_SHIFT)
#define G5P_EMIT_SAMPLES_MS 2u

/* one sample's position byte: x | y << 4, in 1/16 pixel */
#define G5P_POSITION(s, x, y) ((uint64_t)((x) | (y) << 4) << (8 * (s)))

uint64_t
pvr_g5p_msaa_positions(uint32_t samples)
{
   if (samples != 4)
      return G5P_POSITIONS_1X;
   /* Vulkan's standard sample locations for 4 samples (the spec's table):
    * (0.375, 0.125) (0.875, 0.375) (0.125, 0.625) (0.625, 0.875) */
   return G5P_POSITION(0, 6, 2) | G5P_POSITION(1, 14, 6) | G5P_POSITION(2, 2, 10) | G5P_POSITION(3, 10, 14);
}

struct pvr_g5p_msaa_words
pvr_g5p_msaa_pass(uint32_t samples, bool resolve)
{
   struct pvr_g5p_msaa_words m = { 0 };

   if (samples == 4) {
      m.samples = 4;
      m.flags = resolve ? PVR_G5P_MSAA_FLAG_RESOLVE : 0;
      m.positions = pvr_g5p_msaa_positions(4);
   }
   return m;
}

VkSampleCountFlags
pvr_g5p_msaa_sample_counts(bool kernel_msaa)
{
   return VK_SAMPLE_COUNT_1_BIT | (kernel_msaa ? VK_SAMPLE_COUNT_4_BIT : 0);
}

static uint32_t
g5p_pot(uint32_t v)
{
   uint32_t p = 1;
   while (p < v)
      p <<= 1;
   return p;
}

static unsigned
g5p_log2(uint32_t v)
{
   unsigned l = 0;
   while (v > 1)
      v >>= 1, l++;
   return l;
}

/* Morton order over power-of-two sides w, h: y in bit 0 for the square part,
 * the longer side's remaining bits above it */
static uint64_t
g5p_twiddle2d(uint32_t x, uint32_t y, uint32_t w, uint32_t h)
{
   const unsigned common = g5p_log2(w < h ? w : h);
   uint64_t r = 0;
   for (unsigned b = 0; b < common; b++)
      r |= (uint64_t)((y >> b) & 1) << (2 * b) | (uint64_t)((x >> b) & 1) << (2 * b + 1);
   return r | (uint64_t)((x >> common) | (y >> common)) << (2 * common);
}

uint64_t
pvr_g5p_msaa_layout(uint32_t width, uint32_t height, uint32_t cpp)
{
   if (cpp != 4 || !width || !height)
      return 0;
   return (uint64_t)g5p_pot(width) * g5p_pot(height) * cpp * 4;
}

uint64_t
pvr_g5p_msaa_sample_offset(uint32_t width, uint32_t height, uint32_t cpp, uint32_t x, uint32_t y, uint32_t s)
{
   return g5p_twiddle2d(x, y, g5p_pot(width), g5p_pot(height)) * cpp * 4 + (uint64_t)s * cpp;
}

void
pvr_g5p_msaa_pbe_store(uint64_t q[3], uint32_t key, uint64_t va, uint32_t width, uint32_t height, bool srgb)
{
   q[0] = (uint64_t)(height - 1) << 41 | (uint64_t)(width - 1) << 27 | ((key & ~G5P_PBE_LAYOUT_MASK) |
                                                                         G5P_PBE_LAYOUT_MS_TWIDDLED);
   q[1] = va >> 1 | G5P_PBE_Q1_PBE | (srgb ? G5P_PBE_Q1_SRGB : 0);
   q[2] = 0;
}

void
pvr_g5p_msaa_pbe_resolve(uint64_t q[3])
{
   q[2] |= G5P_PBE_Q2_RESOLVE;
}

unsigned
pvr_g5p_msaa_eot_usc(uint8_t *out)
{
   const unsigned n = g5p_mrt_eot_usc(out, 1);
   /* the one-target program: one 12-byte state load, then the emitpix */
   const unsigned emit = 12;

   if (out[emit] != 0x1c)
      return 0;
   out[emit + 1] = (uint8_t)((out[emit + 1] & ~G5P_EMIT_SAMPLES_MASK) | G5P_EMIT_SAMPLES_MS << G5P_EMIT_SAMPLES_SHIFT);
   return n;
}

uint64_t
pvr_g5p_msaa_doutu(uint64_t src0, uint32_t samples)
{
   if (samples != 4)
      return src0;
   return (src0 & ~G5P_DOUTU_SAMPLE_RATE_MASK) | G5P_DOUTU_SAMPLE_RATE_4X << G5P_DOUTU_SAMPLE_RATE_SHIFT;
}

uint32_t
pvr_g5p_msaa_upass(uint32_t samples)
{
   return samples == 4 ? 7 : 15;
}

void
pvr_g5p_msaa_resolve_cpu(const uint8_t *src, uint8_t *dst, uint32_t dst_stride, uint32_t width, uint32_t height,
                         uint32_t cpp)
{
   for (uint32_t y = 0; y < height; y++)
      for (uint32_t x = 0; x < width; x++) {
         const uint8_t *p = src + pvr_g5p_msaa_sample_offset(width, height, cpp, x, y, 0);
         uint8_t *d = dst + (uint64_t)y * dst_stride + (uint64_t)x * cpp;
         for (uint32_t k = 0; k < cpp; k++)
            d[k] = (uint8_t)((p[k] + p[cpp + k] + p[2 * cpp + k] + p[3 * cpp + k] + 2) / 4);
      }
}
