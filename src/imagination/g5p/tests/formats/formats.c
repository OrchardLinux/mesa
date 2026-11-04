/* SPDX-License-Identifier: MIT */
/*
 * formats: a hardware test of the colour formats a Vulkan device advertises
 * (README.md). Plain Vulkan 1.0, no driver internals: the same program checks
 * the A9 (G5P) and A10 (G9P) paths of the PowerVR driver, or any device.
 *
 * For each format it runs the rungs the device's optimal-tiling features ask
 * for, and skips the others:
 *   s-<fmt>   SAMPLED_IMAGE: an 8x8 texture uploaded by vkCmdCopyBufferToImage,
 *             sampled nearest (integers: texture() of an integer sampler, one
 *             rung per byte of the channel width) into a 64x64 UNORM8 target;
 *   f-<fmt>   SAMPLED_IMAGE_FILTER_LINEAR: the same texture sampled halfway
 *             between two texels (weights 1/2, 1/2);
 *   m-<fmt>   a two-level texture, textureLod() of level 1 (twiddled layouts
 *             of each texel size);
 *   r-<fmt>   COLOR_ATTACHMENT: a clear, then a constant colour over the left
 *             half, read back by vkCmdCopyImageToBuffer;
 *   b-<fmt>   COLOR_ATTACHMENT_BLEND: the same with SRC_ALPHA,
 *             ONE_MINUS_SRC_ALPHA (alpha ONE, ONE_MINUS_SRC_ALPHA).
 * Every pixel is compared exactly with a CPU reference. Texel values are
 * chosen so that every expected 8-bit output is at least a quarter step from
 * a rounding boundary (an implementation may write 8-bit UNORM through f16
 * rounded toward zero, up to an eighth of a step low), and blends and render
 * values are exact in f16.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "formats-spirv.h"

#define W 64
#define H 64
#define TW 8
#define TH 8
#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

/* ---- formats ---- */

enum ty {
   T_NONE, /* the channel is absent: 0, or 1 for alpha */
   T_UN,   /* unorm */
   T_SN,   /* snorm */
   T_UI,   /* unsigned integer */
   T_SI,   /* signed integer */
   T_F,    /* 16 or 32-bit float */
   T_UF,   /* unsigned 11/10-bit float (5-bit exponent) */
   T_E5,   /* 9-bit mantissa of a shared-exponent format */
   T_SRGB, /* 8-bit sRGB */
};

struct ch {
   uint8_t off, bits, ty;
};

/* Block-compressed codecs (4x4 blocks; cpp is the block's bytes) */
enum codec { C_NONE, C_ETC2, C_ETC2A1, C_ETC2A8, C_EAC_R, C_EAC_RG };

struct fmt {
   const char *name;
   VkFormat vk;
   uint8_t cpp;
   struct ch c[4]; /* R, G, B, A; a compressed format's say only unorm, snorm or sRGB */
   uint8_t codec;
};

#define UN(o, b) { o, b, T_UN }
#define SN(o, b) { o, b, T_SN }
#define UI(o, b) { o, b, T_UI }
#define SI(o, b) { o, b, T_SI }
#define FL(o, b) { o, b, T_F }
#define UF(o, b) { o, b, T_UF }
#define E5(o) { o, 9, T_E5 }
#define SR(o) { o, 8, T_SRGB }
#define NO { 0, 0, T_NONE }
#define F(n, cpp, r, g, b, a) { #n, VK_FORMAT_##n, cpp, { r, g, b, a }, C_NONE }
#define FC(n, cpp, codec, r, g, b, a) { #n, VK_FORMAT_##n, cpp, { r, g, b, a }, codec }

static const struct fmt fmts[] = {
   F(B8G8R8A8_UNORM, 4, UN(16, 8), UN(8, 8), UN(0, 8), UN(24, 8)),
   F(R8G8B8A8_UNORM, 4, UN(0, 8), UN(8, 8), UN(16, 8), UN(24, 8)),
   F(A8B8G8R8_UNORM_PACK32, 4, UN(0, 8), UN(8, 8), UN(16, 8), UN(24, 8)),
   F(B8G8R8A8_SRGB, 4, SR(16), SR(8), SR(0), UN(24, 8)),
   F(R8G8B8A8_SRGB, 4, SR(0), SR(8), SR(16), UN(24, 8)),
   F(A8B8G8R8_SRGB_PACK32, 4, SR(0), SR(8), SR(16), UN(24, 8)),
   F(R8_UNORM, 1, UN(0, 8), NO, NO, NO),
   F(R8G8_UNORM, 2, UN(0, 8), UN(8, 8), NO, NO),
   F(R8_SNORM, 1, SN(0, 8), NO, NO, NO),
   F(R8G8_SNORM, 2, SN(0, 8), SN(8, 8), NO, NO),
   F(R8G8B8A8_SNORM, 4, SN(0, 8), SN(8, 8), SN(16, 8), SN(24, 8)),
   F(A8B8G8R8_SNORM_PACK32, 4, SN(0, 8), SN(8, 8), SN(16, 8), SN(24, 8)),
   F(R16_UNORM, 2, UN(0, 16), NO, NO, NO),
   F(R16G16_UNORM, 4, UN(0, 16), UN(16, 16), NO, NO),
   F(R16G16B16A16_UNORM, 8, UN(0, 16), UN(16, 16), UN(32, 16), UN(48, 16)),
   F(R16_SNORM, 2, SN(0, 16), NO, NO, NO),
   F(R16G16_SNORM, 4, SN(0, 16), SN(16, 16), NO, NO),
   F(R16G16B16A16_SNORM, 8, SN(0, 16), SN(16, 16), SN(32, 16), SN(48, 16)),
   F(R16_SFLOAT, 2, FL(0, 16), NO, NO, NO),
   F(R16G16_SFLOAT, 4, FL(0, 16), FL(16, 16), NO, NO),
   F(R16G16B16A16_SFLOAT, 8, FL(0, 16), FL(16, 16), FL(32, 16), FL(48, 16)),
   F(R32_SFLOAT, 4, FL(0, 32), NO, NO, NO),
   F(R32G32_SFLOAT, 8, FL(0, 32), FL(32, 32), NO, NO),
   F(R32G32B32A32_SFLOAT, 16, FL(0, 32), FL(32, 32), FL(64, 32), FL(96, 32)),
   F(A2B10G10R10_UNORM_PACK32, 4, UN(0, 10), UN(10, 10), UN(20, 10), UN(30, 2)),
   F(A2R10G10B10_UNORM_PACK32, 4, UN(20, 10), UN(10, 10), UN(0, 10), UN(30, 2)),
   F(B10G11R11_UFLOAT_PACK32, 4, UF(0, 11), UF(11, 11), UF(22, 10), NO),
   F(E5B9G9R9_UFLOAT_PACK32, 4, E5(0), E5(9), E5(18), NO),
   F(R5G6B5_UNORM_PACK16, 2, UN(11, 5), UN(5, 6), UN(0, 5), NO),
   F(B4G4R4A4_UNORM_PACK16, 2, UN(4, 4), UN(8, 4), UN(12, 4), UN(0, 4)),
   F(R8_UINT, 1, UI(0, 8), NO, NO, NO),
   F(R8G8_UINT, 2, UI(0, 8), UI(8, 8), NO, NO),
   F(R8G8B8A8_UINT, 4, UI(0, 8), UI(8, 8), UI(16, 8), UI(24, 8)),
   F(A8B8G8R8_UINT_PACK32, 4, UI(0, 8), UI(8, 8), UI(16, 8), UI(24, 8)),
   F(R16_UINT, 2, UI(0, 16), NO, NO, NO),
   F(R16G16_UINT, 4, UI(0, 16), UI(16, 16), NO, NO),
   F(R16G16B16A16_UINT, 8, UI(0, 16), UI(16, 16), UI(32, 16), UI(48, 16)),
   F(R32_UINT, 4, UI(0, 32), NO, NO, NO),
   F(R32G32_UINT, 8, UI(0, 32), UI(32, 32), NO, NO),
   F(R32G32B32A32_UINT, 16, UI(0, 32), UI(32, 32), UI(64, 32), UI(96, 32)),
   F(A2B10G10R10_UINT_PACK32, 4, UI(0, 10), UI(10, 10), UI(20, 10), UI(30, 2)),
   F(R8_SINT, 1, SI(0, 8), NO, NO, NO),
   F(R8G8_SINT, 2, SI(0, 8), SI(8, 8), NO, NO),
   F(R8G8B8A8_SINT, 4, SI(0, 8), SI(8, 8), SI(16, 8), SI(24, 8)),
   F(A8B8G8R8_SINT_PACK32, 4, SI(0, 8), SI(8, 8), SI(16, 8), SI(24, 8)),
   F(R16_SINT, 2, SI(0, 16), NO, NO, NO),
   F(R16G16_SINT, 4, SI(0, 16), SI(16, 16), NO, NO),
   F(R16G16B16A16_SINT, 8, SI(0, 16), SI(16, 16), SI(32, 16), SI(48, 16)),
   F(R32_SINT, 4, SI(0, 32), NO, NO, NO),
   F(R32G32_SINT, 8, SI(0, 32), SI(32, 32), NO, NO),
   F(R32G32B32A32_SINT, 16, SI(0, 32), SI(32, 32), SI(64, 32), SI(96, 32)),
   FC(ETC2_R8G8B8_UNORM_BLOCK, 8, C_ETC2, UN(0, 8), UN(0, 8), UN(0, 8), NO),
   FC(ETC2_R8G8B8_SRGB_BLOCK, 8, C_ETC2, SR(0), SR(0), SR(0), NO),
   FC(ETC2_R8G8B8A1_UNORM_BLOCK, 8, C_ETC2A1, UN(0, 8), UN(0, 8), UN(0, 8), UN(0, 8)),
   FC(ETC2_R8G8B8A1_SRGB_BLOCK, 8, C_ETC2A1, SR(0), SR(0), SR(0), UN(0, 8)),
   FC(ETC2_R8G8B8A8_UNORM_BLOCK, 16, C_ETC2A8, UN(0, 8), UN(0, 8), UN(0, 8), UN(0, 8)),
   FC(ETC2_R8G8B8A8_SRGB_BLOCK, 16, C_ETC2A8, SR(0), SR(0), SR(0), UN(0, 8)),
   FC(EAC_R11_UNORM_BLOCK, 8, C_EAC_R, UN(0, 11), NO, NO, NO),
   FC(EAC_R11_SNORM_BLOCK, 8, C_EAC_R, SN(0, 11), NO, NO, NO),
   FC(EAC_R11G11_UNORM_BLOCK, 16, C_EAC_RG, UN(0, 11), UN(0, 11), NO, NO),
   FC(EAC_R11G11_SNORM_BLOCK, 16, C_EAC_RG, SN(0, 11), SN(0, 11), NO, NO),
};

/* ---- ETC2/EAC (Khronos Data Format Specification, ETC2 and EAC) ---- */

static double srgb_to_linear(double s);

static const int etc1_mod[8][2] = { { 2, 8 },   { 5, 17 },  { 9, 29 },  { 13, 42 },
                                    { 18, 60 }, { 24, 80 }, { 33, 106 }, { 47, 183 } };
static const int eac_mod[16][8] = {
   { -3, -6, -9, -15, 2, 5, 8, 14 }, { -3, -7, -10, -13, 2, 6, 9, 12 }, { -2, -5, -8, -13, 1, 4, 7, 12 },
   { -2, -4, -6, -13, 1, 3, 5, 12 }, { -3, -6, -8, -12, 2, 5, 7, 11 }, { -3, -7, -9, -11, 2, 6, 8, 10 },
   { -4, -7, -8, -11, 3, 6, 7, 10 }, { -3, -5, -8, -11, 2, 4, 7, 10 }, { -2, -6, -8, -10, 1, 5, 7, 9 },
   { -2, -5, -8, -10, 1, 4, 7, 9 },  { -2, -4, -8, -10, 1, 3, 7, 9 },  { -2, -5, -7, -10, 1, 4, 6, 9 },
   { -3, -4, -7, -10, 2, 3, 6, 9 },  { -1, -2, -3, -10, 0, 1, 2, 9 },  { -4, -6, -8, -9, 3, 5, 7, 8 },
   { -3, -5, -7, -9, 2, 4, 6, 8 },
};

static uint64_t
be64(const uint8_t *p)
{
   uint64_t v = 0;
   for (unsigned i = 0; i < 8; i++)
      v = v << 8 | p[i];
   return v;
}

static void
put_be64(uint8_t *p, uint64_t v)
{
   for (unsigned i = 0; i < 8; i++)
      p[i] = (uint8_t)(v >> (56 - 8 * i));
}

static int
clampi(int v, int lo, int hi)
{
   return v < lo ? lo : v > hi ? hi : v;
}

/* The base colour channel c of subblock sub of an ETC block (individual or
 * differential mode, the only modes written here) */
static int
etc_base(uint64_t v, bool diff, unsigned sub, unsigned c)
{
   if (!diff) {
      const int n = (int)(v >> (sub ? 56 - 8 * c : 60 - 8 * c) & 15);
      return n << 4 | n;
   }
   int b = (int)(v >> (59 - 8 * c) & 31);
   if (sub)
      b += ((int)(v >> (56 - 8 * c) & 7) ^ 4) - 4;
   return b << 3 | b >> 2;
}

/* An ETC2 RGB block's texel (x, y): RGB; punch: RGB8A1, always differential */
static void
etc_rgb(const uint8_t *blk, unsigned x, unsigned y, bool punch, int out[3])
{
   const uint64_t v = be64(blk);
   const bool diff = punch || (v >> 33 & 1), flip = v >> 32 & 1;
   const unsigned sub = flip ? y >= 2 : x >= 2, i = x * 4 + y;
   const unsigned tbl = (unsigned)(v >> (sub ? 34 : 37) & 7), msb = v >> (16 + i) & 1, lsb = v >> i & 1;
   const int m = etc1_mod[tbl][lsb] * (msb ? -1 : 1);
   for (unsigned c = 0; c < 3; c++)
      out[c] = clampi(etc_base(v, diff, sub, c) + m, 0, 255);
}

/* An EAC block's texel: alpha (8-bit), or R11 unsigned (0..2047) / signed (-1023..1023) */
static int
eac(const uint8_t *blk, unsigned x, unsigned y, unsigned bits, bool sgn)
{
   const uint64_t v = be64(blk);
   const int base = (int)(v >> 56 & 255), mult = (int)(v >> 52 & 15), ti = (int)(v >> 48 & 15);
   const int mod = eac_mod[ti][v >> (45 - 3 * (x * 4 + y)) & 7];
   if (bits == 8)
      return clampi(base + mod * mult, 0, 255);
   const int m = mult ? mult * 8 : 1;
   if (!sgn)
      return clampi(base * 8 + 4 + mod * m, 0, 2047);
   const int b = (int8_t)base == -128 ? -127 : (int8_t)base;
   return clampi(b * 8 + mod * m, -1023, 1023);
}

/* Channel c of texel (x, y) of a compressed level, w texels wide */
static double
block_texel(const struct fmt *f, const uint8_t *t, unsigned w, unsigned x, unsigned y, unsigned c)
{
   const uint8_t *blk = t + ((size_t)(y / 4) * ((w + 3) / 4) + x / 4) * f->cpp;
   const bool sgn = f->c[0].ty == T_SN;
   int rgb[3];

   x %= 4, y %= 4;
   switch (f->codec) {
   case C_EAC_R:
   case C_EAC_RG:
      if (c == 3)
         return 1;
      if (c == 2 || (c == 1 && f->codec == C_EAC_R))
         return 0;
      return sgn ? eac(blk + 8 * c, x, y, 11, true) / 1023.0 : eac(blk + 8 * c, x, y, 11, false) / 2047.0;
   case C_ETC2A8:
      if (c == 3)
         return eac(blk, x, y, 8, false) / 255.0;
      blk += 8;
      /* fallthrough */
   default:
      if (c == 3)
         return 1; /* RGB8; RGB8A1 written opaque */
      etc_rgb(blk, x, y, f->codec == C_ETC2A1, rgb);
      return f->c[c].ty == T_SRGB ? srgb_to_linear(rgb[c] / 255.0) : rgb[c] / 255.0;
   }
}

static bool
is_int(const struct fmt *f)
{
   return f->c[0].ty == T_UI || f->c[0].ty == T_SI;
}

static unsigned
int_bits(const struct fmt *f)
{
   unsigned b = 0;
   for (unsigned c = 0; c < 4; c++)
      b = f->c[c].bits > b ? f->c[c].bits : b;
   return b;
}

static uint64_t
get_bits(const uint8_t *t, unsigned off, unsigned bits)
{
   uint64_t v = 0;
   for (unsigned i = 0; i < bits; i++)
      v |= (uint64_t)((t[(off + i) / 8] >> ((off + i) % 8)) & 1) << i;
   return v;
}

static void
put_bits(uint8_t *t, unsigned off, unsigned bits, uint64_t v)
{
   for (unsigned i = 0; i < bits; i++) {
      const unsigned b = off + i;
      t[b / 8] = (uint8_t)((t[b / 8] & ~(1u << (b % 8))) | ((v >> i) & 1) << (b % 8));
   }
}

static double
half_to_double(uint32_t h)
{
   const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
   const double v = e ? ldexp(1024 + m, e - 25) : ldexp(m, -24);
   return s ? -v : v;
}

static double
uf_to_double(uint32_t v, unsigned bits)
{
   const unsigned mb = bits - 5, e = v >> mb, m = v & ((1u << mb) - 1);
   return e ? ldexp((double)((1u << mb) + m), (int)e - 15 - (int)mb) : ldexp(m, -14 - (int)mb);
}

static double
srgb_to_linear(double s)
{
   return s <= 0.04045 ? s / 12.92 : pow((s + 0.055) / 1.055, 2.4);
}

static double
linear_to_srgb(double l)
{
   return l <= 0.0031308 ? l * 12.92 : 1.055 * pow(l, 1 / 2.4) - 0.055;
}

/* A texel's channel as the shader reads it: float formats in d, integers in i */
static void
decode(const struct fmt *f, const uint8_t *t, unsigned c, double *d, int64_t *i)
{
   const struct ch *ch = &f->c[c];
   const uint64_t v = ch->bits ? get_bits(t, ch->off, ch->bits) : 0;

   *i = 0;
   switch (ch->ty) {
   case T_NONE: *d = c == 3; *i = c == 3; break;
   case T_UN: *d = (double)v / (double)((1ull << ch->bits) - 1); break;
   case T_SN: {
      const int64_t s = (int64_t)(v << (64 - ch->bits)) >> (64 - ch->bits);
      *d = fmax((double)s / (double)((1ll << (ch->bits - 1)) - 1), -1.0);
      break;
   }
   case T_UI: *i = (int64_t)v; break;
   case T_SI: *i = (int64_t)(v << (64 - ch->bits)) >> (64 - ch->bits); break;
   case T_F: {
      if (ch->bits == 16) {
         *d = half_to_double((uint32_t)v);
      } else {
         float x;
         const uint32_t u = (uint32_t)v;
         memcpy(&x, &u, 4);
         *d = x;
      }
      break;
   }
   case T_UF: *d = uf_to_double((uint32_t)v, ch->bits); break;
   case T_E5: *d = ldexp((double)v, (int)get_bits(t, 27, 5) - 15 - 9); break;
   case T_SRGB: *d = srgb_to_linear((double)v / 255.0); break;
   }
}

/* ---- deterministic values ---- */

static uint32_t rng_state;

static uint32_t
rng(void)
{
   rng_state = rng_state * 1664525u + 1013904223u;
   return rng_state >> 8 ^ rng_state << 13;
}

/* A random value of a channel's raw bits, finite and of a modest size */
static uint64_t
rand_raw(const struct ch *ch)
{
   const uint64_t r = (uint64_t)rng() << 32 | rng();

   switch (ch->ty) {
   case T_F:
      if (ch->bits == 16) /* |v| in [2^-5, 8): exponent 10..17 */
         return (r & 0x83ff) | (uint64_t)(10 + (r >> 20) % 8) << 10;
      return (r & 0x807fffff) | (uint64_t)(122 + (r >> 40) % 8) << 23; /* the same in f32 */
   case T_UF: { /* [2^-5, 8) */
      const unsigned mb = ch->bits - 5;
      return (r & ((1u << mb) - 1)) | (uint64_t)(10 + (r >> 20) % 8) << mb;
   }
   default:
      return ch->bits >= 64 ? r : r & ((1ull << ch->bits) - 1);
   }
}

/* The shader's scale and bias of a float format's values into [0, 1]. The
 * signed ones are centred on 128/255, so that 0 (an absent channel) is the
 * byte 128 exactly, not 127.5 */
static void
scale_bias(const struct fmt *f, float *scale, float *bias)
{
   *scale = 1, *bias = 0;
   for (unsigned c = 0; c < 4; c++) {
      if (f->c[c].ty == T_SN)
         *scale = 0.5f, *bias = 128.0f / 255;
      else if (f->c[c].ty == T_F)
         *scale = 1.0f / 16, *bias = 128.0f / 255;
      else if (f->c[c].ty == T_UF || f->c[c].ty == T_E5)
         *scale = 1.0f / 16, *bias = 0;
   }
}

/* Whether an expected UNORM8 output v (in [0, 1]) of channel c is safe: a
 * quarter step from rounding, or 0.4 of a step for a channel decoded from
 * sRGB (implementations decode sRGB with less precision: lavapipe's and the
 * A9's differ from the exact curve by up to a third of a step) */
static bool
safe8(const struct fmt *f, unsigned c, double v)
{
   if (v <= 0 || v >= 1)
      return true;
   const double s = v * 255, fr = s - floor(s);
   return fabs(fr - 0.5) >= (f->c[c].ty == T_SRGB ? 0.4 : 0.25);
}

static uint8_t
unorm8(double v)
{
   v = v < 0 ? 0 : v > 1 ? 1 : v;
   return (uint8_t)lround(v * 255.0);
}

/* ---- Vulkan ---- */

static VkPhysicalDevice pd;
static VkDevice dev;
static VkQueue queue;
static uint32_t queue_family;
static VkCommandPool pool;
static bool verify = true;
static const char *cur = "setup";

#define CHECK(x)                                                                                        \
   do {                                                                                                 \
      VkResult r_ = (x);                                                                                \
      if (r_ != VK_SUCCESS) {                                                                           \
         printf("FORMATS %s FAIL: %s = %d (line %d)\n", cur, #x, r_, __LINE__);                          \
         return 1;                                                                                      \
      }                                                                                                 \
   } while (0)

static int
host_memory(const VkMemoryRequirements *req, VkDeviceMemory *mem)
{
   VkPhysicalDeviceMemoryProperties mp;
   uint32_t type = UINT32_MAX;

   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; i++) {
      if ((req->memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
         type = i;
   }
   if (type == UINT32_MAX) {
      printf("FORMATS %s FAIL: no host-visible memory type\n", cur);
      return 1;
   }
   const VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req->size,
                                      .memoryTypeIndex = type };
   CHECK(vkAllocateMemory(dev, &mai, NULL, mem));
   return 0;
}

struct buf {
   VkBuffer b;
   VkDeviceMemory mem;
   uint8_t *map;
};

static int
make_buffer(struct buf *b, VkDeviceSize size, VkBufferUsageFlags usage)
{
   const VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage };
   VkMemoryRequirements req;

   CHECK(vkCreateBuffer(dev, &bci, NULL, &b->b));
   vkGetBufferMemoryRequirements(dev, b->b, &req);
   if (host_memory(&req, &b->mem))
      return 1;
   CHECK(vkBindBufferMemory(dev, b->b, b->mem, 0));
   CHECK(vkMapMemory(dev, b->mem, 0, VK_WHOLE_SIZE, 0, (void **)&b->map));
   return 0;
}

static void
free_buffer(struct buf *b)
{
   vkUnmapMemory(dev, b->mem);
   vkDestroyBuffer(dev, b->b, NULL);
   vkFreeMemory(dev, b->mem, NULL);
}

static int
sync_mem(VkDeviceMemory mem, bool invalidate)
{
   const VkMappedMemoryRange range = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = mem,
                                       .size = VK_WHOLE_SIZE };
   if (invalidate)
      CHECK(vkInvalidateMappedMemoryRanges(dev, 1, &range));
   else
      CHECK(vkFlushMappedMemoryRanges(dev, 1, &range));
   return 0;
}

struct img {
   VkImage image;
   VkDeviceMemory mem;
   VkImageView view;
   VkDeviceSize size;
};

/* An image and a view of all of it: 2D, a 2D array of d layers, or 3D of depth d */
static int
make_image_ex(struct img *im, VkFormat format, VkImageViewType vt, unsigned w, unsigned h, unsigned d,
              unsigned levels, VkImageUsageFlags usage)
{
   const bool is3d = vt == VK_IMAGE_VIEW_TYPE_3D;
   const unsigned layers = vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY ? d : 1;
   const VkImageCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                   .imageType = is3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D, .format = format,
                                   .extent = { w, h, is3d ? d : 1 }, .mipLevels = levels, .arrayLayers = layers,
                                   .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                   .usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VkMemoryRequirements req;

   CHECK(vkCreateImage(dev, &ici, NULL, &im->image));
   vkGetImageMemoryRequirements(dev, im->image, &req);
   im->size = req.size;
   if (host_memory(&req, &im->mem))
      return 1;
   CHECK(vkBindImageMemory(dev, im->image, im->mem, 0));
   const VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im->image,
                                       .viewType = vt, .format = format,
                                       .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers } };
   CHECK(vkCreateImageView(dev, &vci, NULL, &im->view));
   return 0;
}

static int
make_image(struct img *im, VkFormat format, unsigned w, unsigned h, unsigned levels, VkImageUsageFlags usage)
{
   return make_image_ex(im, format, VK_IMAGE_VIEW_TYPE_2D, w, h, 1, levels, usage);
}

static void
free_image(struct img *im)
{
   vkDestroyImageView(dev, im->view, NULL);
   vkDestroyImage(dev, im->image, NULL);
   vkFreeMemory(dev, im->mem, NULL);
}

static void
barrier(VkCommandBuffer cmd, VkImage image, unsigned levels, VkImageLayout from, VkImageLayout to,
        VkAccessFlags src, VkAccessFlags dst, VkPipelineStageFlags ss, VkPipelineStageFlags ds)
{
   const VkImageMemoryBarrier imb = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = src,
                                      .dstAccessMask = dst, .oldLayout = from, .newLayout = to,
                                      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = image,
                                      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0,
                                                            VK_REMAINING_ARRAY_LAYERS } };
   vkCmdPipelineBarrier(cmd, ss, ds, 0, 0, NULL, 0, NULL, 1, &imb);
}

static int
begin(VkCommandBuffer *cmd)
{
   const VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                             .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                             .commandBufferCount = 1 };
   const VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkAllocateCommandBuffers(dev, &cai, cmd));
   CHECK(vkBeginCommandBuffer(*cmd, &cbi));
   return 0;
}

/* End, submit alone and wait */
static int
submit(VkCommandBuffer cmd)
{
   const VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   const VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                             .pCommandBuffers = &cmd };
   VkFence fence;

   CHECK(vkEndCommandBuffer(cmd));
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));
   CHECK(vkQueueSubmit(queue, 1, &si, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull));
   vkDestroyFence(dev, fence, NULL);
   vkFreeCommandBuffers(dev, pool, 1, &cmd);
   return 0;
}

/* ---- rungs ---- */

enum kind { K_SAMPLE, K_FILTER, K_MIP, K_FETCH, K_RENDER, K_BLEND, K_BLIT, K_LAYOUT };
static const char kind_prefix[] = { 's', 'f', 'm', 'x', 'r', 'b', 'k', 'l' };

struct layout_case {
   const char *name;
   VkImageViewType vt;
   unsigned w, h, d; /* d: the layers of an array, the depth of a 3D image */
   unsigned levels, level;
   const uint32_t *fs;
   size_t fs_size;
};

#define LAY(n, vt, w, h, d, levels, level, fs) { n, VK_IMAGE_VIEW_TYPE_##vt, w, h, d, levels, level, spv_##fs##_frag, \
                                                  sizeof(spv_##fs##_frag) }
static const struct layout_case layouts[] = {
   LAY("l-npot-0", 2D, 12, 10, 1, 4, 0, lay2d_0),
   LAY("l-npot-1", 2D, 12, 10, 1, 4, 1, lay2d_1),
   LAY("l-npot-2", 2D, 12, 10, 1, 4, 2, lay2d_2),
   LAY("l-npot-3", 2D, 12, 10, 1, 4, 3, lay2d_3),
   LAY("l-wide-1", 2D, 16, 4, 1, 3, 1, lay2d_1),
   LAY("l-tall-2", 2D, 4, 16, 1, 3, 2, lay2d_2),
   LAY("l-arr-mip-1", 2D_ARRAY, 8, 8, 2, 4, 1, layarr_1),
   LAY("l-arr-npot-0", 2D_ARRAY, 12, 10, 3, 3, 0, layarr_0),
   LAY("l-arr-npot-2", 2D_ARRAY, 12, 10, 3, 3, 2, layarr_2),
   LAY("l-3d-4x4x2", 3D, 4, 4, 2, 1, 0, lay3d_2_0),
   LAY("l-3d-8", 3D, 8, 8, 8, 1, 0, lay3d_8_0),
   LAY("l-3d-npot", 3D, 6, 5, 3, 1, 0, lay3d_3_0),
   LAY("l-3d-mip-1", 3D, 8, 8, 8, 4, 1, lay3d_4_1),
   LAY("l-3d-mip-2", 3D, 8, 8, 8, 4, 2, lay3d_2_2),
   LAY("l-3d-npot-1", 3D, 6, 5, 3, 2, 1, lay3d_1_1),
   LAY("l-3d-big", 3D, 16, 16, 16, 1, 0, lay3d_16_0),
};
#undef LAY

struct rung {
   char name[64];
   const struct fmt *f;
   enum kind kind;
   unsigned shift; /* integers: the byte of the channels read */
   const struct layout_case *lay; /* K_LAYOUT */
};

static struct rung rungs[512];
static unsigned nrungs;

static VkFormatFeatureFlags
feature_of(enum kind k)
{
   switch (k) {
   case K_SAMPLE: case K_MIP: case K_FETCH: case K_LAYOUT: return VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
   case K_FILTER: return VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
   case K_RENDER: return VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
   case K_BLIT: return VK_FORMAT_FEATURE_BLIT_SRC_BIT;
   default: return VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
   }
}

static void
add(const struct fmt *f, enum kind k, unsigned shift)
{
   struct rung *r = &rungs[nrungs++];
   r->f = f, r->kind = k, r->shift = shift;
   if (shift)
      snprintf(r->name, sizeof(r->name), "%c-%s-%u", kind_prefix[k], f->name, shift);
   else
      snprintf(r->name, sizeof(r->name), "%c-%s", kind_prefix[k], f->name);
}

/* The render candidates: the formats a fragment program's four f16 outputs
 * and a PBE could write (8-bit RGBA, sRGB, 10_10_10_2, R/RG 8, half float,
 * 11_11_10 float). RGBA32 integer targets have separate typed raw-output
 * rungs, added in the integer branch of build_rungs(). */
static bool
render_candidate(const struct fmt *f)
{
   switch (f->vk) {
   case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
   case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
      return true;
   default:
      return false;
   }
}

static void
build_rungs(void)
{
   for (unsigned i = 0; i < ARRAY_LEN(fmts); i++) {
      const struct fmt *f = &fmts[i];
      if (is_int(f)) {
         for (unsigned s = 0; s < int_bits(f); s += 8)
            add(f, K_SAMPLE, s);
         if (f->vk == VK_FORMAT_R32_UINT || f->vk == VK_FORMAT_R8G8B8A8_UINT)
            add(f, K_FETCH, 0);
         if (f->vk == VK_FORMAT_R32G32B32A32_UINT || f->vk == VK_FORMAT_R32G32B32A32_SINT)
            add(f, K_RENDER, 0);
         continue;
      }
      add(f, K_SAMPLE, 0);
      add(f, K_FILTER, 0);
      if (f->vk == VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK)
         add(f, K_MIP, 0);
      if (f->codec)
         add(f, K_BLIT, 0);
      if (f->vk == VK_FORMAT_R8_UNORM || f->vk == VK_FORMAT_R8G8_UNORM || f->vk == VK_FORMAT_R16G16B16A16_SFLOAT ||
          f->vk == VK_FORMAT_R32G32B32A32_SFLOAT || f->vk == VK_FORMAT_R8G8B8A8_SRGB ||
          f->vk == VK_FORMAT_B10G11R11_UFLOAT_PACK32)
         add(f, K_MIP, 0);
      if (render_candidate(f)) {
         add(f, K_RENDER, 0);
         add(f, K_BLEND, 0);
      }
   }
   const struct fmt *rgba8 = NULL;
   for (unsigned i = 0; i < ARRAY_LEN(fmts); i++) {
      if (fmts[i].vk == VK_FORMAT_R8G8B8A8_UNORM)
         rgba8 = &fmts[i];
   }
   for (unsigned i = 0; i < ARRAY_LEN(layouts); i++) {
      struct rung *r = &rungs[nrungs++];
      r->f = rgba8, r->kind = K_LAYOUT, r->shift = 0, r->lay = &layouts[i];
      snprintf(r->name, sizeof(r->name), "%s", layouts[i].name);
   }
}

/* ---- texture rungs ---- */

/* Whether the decoded value d of channel c is safe after the rung's scale and bias */
static bool
safe_val(const struct fmt *f, unsigned c, double d)
{
   float scale, bias;
   scale_bias(f, &scale, &bias);
   return safe8(f, c, d * scale + bias);
}

/* Whether block (bx, by) of a compressed level decodes to safe texels: each
 * value, and for a filter rung each texel's mean with its left neighbour
 * (possibly in the block to the left), in both sRGB filtering orders */
/* k- rungs of signed EAC: decoded values are kept a quarter f16 ulp from rounding */
static bool half_margin;
static uint16_t half_bits(float v);

static bool
block_ok(const struct fmt *f, const uint8_t *t, unsigned w, unsigned bx, unsigned by, bool filter)
{
   for (unsigned y = 4 * by; y < 4 * by + 4; y++) {
      for (unsigned x = 4 * bx; x < 4 * bx + 4; x++) {
         for (unsigned c = 0; c < 4; c++) {
            const double a = block_texel(f, t, w, x, y, c);
            if (!safe_val(f, c, a))
               return false;
            if (half_margin && half_bits((float)(a * (1 + 1.0 / 4096))) != half_bits((float)(a * (1 - 1.0 / 4096))))
               return false;
            if (!filter || !x)
               continue;
            const double b = block_texel(f, t, w, x - 1, y, c);
            if (!safe_val(f, c, (a + b) / 2))
               return false;
            if (f->c[c].ty == T_SRGB && fabs(a - b) > 1e-12)
               return false; /* sRGB pairs are equal (both filtering orders agree) */
         }
      }
   }
   return true;
}

/* Whether channel c of every texel of subblock sub of ETC block v is safe */
static bool
etc_sub_ok(const struct fmt *f, uint64_t v, unsigned sub, unsigned c)
{
   uint8_t b[8];
   const bool flip = v >> 32 & 1;

   put_be64(b, v);
   for (unsigned x = 0; x < 4; x++) {
      for (unsigned y = 0; y < 4; y++) {
         int rgb[3];
         if ((flip ? y >= 2 : x >= 2) != sub)
            continue;
         etc_rgb(b, x, y, f->codec == C_ETC2A1, rgb);
         if (!safe8(f, c, srgb_to_linear(rgb[c] / 255.0)))
            return false;
      }
   }
   return true;
}

/* sRGB: each channel's base colours searched until both subblocks' texels
 * are safe (a random base is safe for a fifth of the sRGB bytes only) */
static uint64_t
etc_srgb_bases(const struct fmt *f, uint64_t v)
{
   const bool diff = f->codec == C_ETC2A1;

   for (unsigned c = 0; c < 3; c++) {
      if (!diff) {
         for (unsigned sub = 0; sub < 2; sub++) {
            const unsigned sh = sub ? 56 - 8 * c : 60 - 8 * c, start = rng();
            for (unsigned k = 0; k < 16; k++) {
               v = (v & ~(15ull << sh)) | (uint64_t)((start + k) % 16) << sh;
               if (etc_sub_ok(f, v, sub, c))
                  break;
            }
         }
         continue;
      }
      const unsigned start = rng();
      for (unsigned k = 0; k < 32; k++) {
         const unsigned b = (start + k) % 32;
         bool found = false;
         v = (v & ~(31ull << (59 - 8 * c))) | (uint64_t)b << (59 - 8 * c);
         if (!etc_sub_ok(f, v, 0, c))
            continue;
         for (int d = -4; d < 4 && !found; d++) {
            if ((int)b + d < 0 || (int)b + d > 31)
               continue;
            v = (v & ~(7ull << (56 - 8 * c))) | (uint64_t)(d & 7) << (56 - 8 * c);
            found = etc_sub_ok(f, v, 1, c);
         }
         if (found)
            break;
      }
   }
   return v;
}

/* A random ETC block in the modes the decoder above reads: individual
 * (RGB8, RGBA8's colour) or differential, opaque (RGB8A1). Constraints keep
 * the rejection loop short: per-column indices (sRGB: each column one value),
 * per-row indices and flip (row-constant blocks, repeated across the row),
 * or even bases with tables {2,8}, {18,60}, {24,80} (every texel even, so
 * filtered means are whole bytes) */
static uint64_t
gen_etc(const struct fmt *f, bool filter)
{
   const bool srgb = f->c[0].ty == T_SRGB, diff = f->codec == C_ETC2A1;
   const bool rowconst = filter && (srgb || diff), colconst = srgb && !filter;
   const bool flip = rowconst ? true : colconst ? false : rng() & 1;
   static const unsigned even_tbl[3] = { 0, 4, 5 };
   const unsigned t1 = filter && !rowconst ? even_tbl[rng() % 3] : rng() % 8;
   const unsigned t2 = filter && !rowconst ? even_tbl[rng() % 3] : rng() % 8;
   uint64_t v = (uint64_t)t1 << 37 | (uint64_t)t2 << 34 | (uint64_t)diff << 33 | (uint64_t)flip << 32;
   uint32_t idx = rng();
   if (rowconst || colconst) {
      /* one index per row (rowconst) or per column (colconst) */
      const unsigned per[4] = { rng() & 3, rng() & 3, rng() & 3, rng() & 3 };
      idx = 0;
      for (unsigned x = 0; x < 4; x++) {
         for (unsigned y = 0; y < 4; y++) {
            const unsigned k = per[rowconst ? y : x], i = x * 4 + y;
            idx |= (k >> 1 & 1) << (16 + i) | (k & 1) << i;
         }
      }
   }
   v |= idx;
   for (unsigned c = 0; c < 3; c++) {
      if (diff) {
         const unsigned b = rng() % 32;
         int d = (int)(rng() % 8) - 4;
         if ((int)b + d < 0 || (int)b + d > 31)
            d = 0;
         v |= (uint64_t)b << (59 - 8 * c) | (uint64_t)(d & 7) << (56 - 8 * c);
      } else {
         for (unsigned sub = 0; sub < 2; sub++) {
            unsigned n = rng() % 16;
            if (filter && !rowconst) {
               const unsigned tb = sub ? t2 : t1;
               do
                  n = 2 * (rng() % 8);
               while (n * 17 + (unsigned)etc1_mod[tb][1] > 255);
            }
            v |= (uint64_t)n << (sub ? 56 - 8 * c : 60 - 8 * c);
         }
      }
   }
   if (srgb)
      v = etc_srgb_bases(f, v);
   return v;
}

/* A random EAC block: alpha (8-bit; filter: even base, multiplier 2, so every
 * value is even) or R11 (one index per column) */
static uint64_t
gen_eac(bool alpha, bool filter)
{
   const uint64_t base = alpha && filter ? 32 + 2 * (rng() % 96) : rng() % 256;
   const uint64_t mult = alpha && filter ? 2 : 1 + rng() % 15, ti = rng() % 16;
   uint64_t v = base << 56 | mult << 52 | ti << 48;
   const unsigned per[4] = { rng() & 7, rng() & 7, rng() & 7, rng() & 7 };
   for (unsigned i = 0; i < 16; i++)
      v |= (uint64_t)(alpha ? rng() & 7 : per[i / 4]) << (45 - 3 * i);
   return v;
}

/* A compressed level: each block drawn until block_ok() */
static void
gen_blocks(const struct rung *r, uint8_t *out, unsigned w, unsigned h)
{
   const struct fmt *f = r->f;
   const bool filter = r->kind == K_FILTER, srgb = f->c[0].ty == T_SRGB;
   const bool repeat = filter && (srgb || f->codec == C_ETC2A1); /* row-constant: one block per row */

   half_margin = r->kind == K_BLIT && f->c[0].ty == T_SN;
   const unsigned bw = (w + 3) / 4, bh = (h + 3) / 4;

   for (unsigned by = 0; by < bh; by++) {
      for (unsigned bx = 0; bx < bw; bx++) {
         uint8_t *blk = out + ((size_t)by * bw + bx) * f->cpp;
         unsigned tries;
         if (repeat && bx) {
            memcpy(blk, blk - f->cpp, f->cpp);
            continue;
         }
         for (tries = 0; tries < 20000; tries++) {
            switch (f->codec) {
            case C_EAC_R: put_be64(blk, gen_eac(false, filter)); break;
            case C_EAC_RG:
               put_be64(blk, gen_eac(false, filter));
               put_be64(blk + 8, gen_eac(false, filter));
               break;
            case C_ETC2A8:
               put_be64(blk, gen_eac(true, filter));
               put_be64(blk + 8, gen_etc(f, filter));
               break;
            default: put_be64(blk, gen_etc(f, filter)); break;
            }
            if (block_ok(f, out, w, bx, by, filter))
               break;
         }
         if (tries == 20000)
            printf("FORMATS %s: no safe block %u,%u (reference not exact-safe)\n", r->name, bx, by);
      }
   }
}

/* The texels of a rung's texture, level l, raw; every expected output safe */
static void
make_texels(const struct rung *r, unsigned l, uint8_t *out, unsigned w, unsigned h)
{
   const struct fmt *f = r->f;
   float scale, bias;

   scale_bias(f, &scale, &bias);
   rng_state = 0x9e3779b9u ^ (uint32_t)f->vk * 7919u ^ l * 104729u;
   if (f->codec) {
      gen_blocks(r, out, w, h);
      return;
   }
   memset(out, 0, (size_t)w * h * f->cpp);
   for (unsigned t = 0; t < w * h; t++) {
      uint8_t *p = out + (size_t)t * f->cpp;
      for (unsigned tries = 0; tries < 1000; tries++) {
         for (unsigned c = 0; c < 4; c++) {
            if (f->c[c].ty == T_E5) {
               put_bits(p, f->c[c].off, 9, rand_raw(&f->c[c]));
               put_bits(p, 27, 5, 12 + rng() % 6); /* 2^-12 .. 2^2 */
            } else if (f->c[c].bits) {
               put_bits(p, f->c[c].off, f->c[c].bits, rand_raw(&f->c[c]));
            }
         }
         if (is_int(f))
            break;
         bool ok = true;
         for (unsigned c = 0; c < 4 && ok; c++) {
            double d;
            int64_t i;
            decode(f, p, c, &d, &i);
            ok = safe8(f, c, d * scale + bias);
         }
         if (ok)
            break;
         if (tries == 999)
            printf("FORMATS %s: no safe texel %u (reference not exact-safe)\n", r->name, t);
      }
   }
   if (r->kind != K_FILTER)
      return;
   /* linear filtering: each horizontal pair's mean must be safe too; each
    * channel of a texel is redrawn, left to right, until it is */
   for (unsigned y = 0; y < h; y++) {
      for (unsigned x = 1; x < w; x++) {
         uint8_t *p = out + ((size_t)y * w + x) * f->cpp, *q = p - f->cpp;
         for (unsigned c = 0; c < 4; c++) {
            unsigned tries;
            for (tries = 0; tries < 1000; tries++) {
               double a, b;
               int64_t i;
               decode(f, p, c, &a, &i);
               decode(f, q, c, &b, &i);
               bool ok = safe8(f, c, a * scale + bias) && safe8(f, c, (a + b) / 2 * scale + bias);
               /* sRGB: filtering may happen before or after the decode (lavapipe's is near the
                * former); take pairs where both give the same byte */
               if (ok && f->c[c].ty == T_SRGB) {
                  const double ka = get_bits(p, f->c[c].off, 8), kb = get_bits(q, f->c[c].off, 8);
                  ok = unorm8(srgb_to_linear((ka + kb) / 2 / 255)) == unorm8((a + b) / 2) &&
                       safe8(f, c, srgb_to_linear((ka + kb) / 2 / 255));
               }
               if (ok || !f->c[c].bits)
                  break;
               if (f->c[c].ty == T_E5) {
                  put_bits(p, f->c[c].off, 9, rand_raw(&f->c[c]));
               } else if (f->c[c].ty == T_SRGB) {
                  /* near its left neighbour, where both filtering orders agree; in the end
                   * equal to it (a safe value: the first pass took safe texels only) */
                  const int k = (int)get_bits(q, f->c[c].off, 8) + (tries < 900 ? (int)(rng() % 17) - 8 : 0);
                  put_bits(p, f->c[c].off, 8, (uint64_t)(k < 0 ? 0 : k > 255 ? 255 : k));
               } else {
                  put_bits(p, f->c[c].off, f->c[c].bits, rand_raw(&f->c[c]));
               }
            }
            if (tries == 1000)
               printf("FORMATS %s: no safe texel at %u,%u channel %u (reference not exact-safe)\n", r->name, x, y, c);
         }
      }
   }
}

/* Channel c of texel (x, y) of a level, w texels wide: float in *d, integer in *i */
static void
texel_value(const struct fmt *f, const uint8_t *t, unsigned w, unsigned x, unsigned y, unsigned c, double *d,
            int64_t *i)
{
   if (f->codec) {
      *i = 0;
      *d = block_texel(f, t, w, x, y, c);
   } else {
      decode(f, t + ((size_t)y * w + x) * f->cpp, c, d, i);
   }
}

/* The expected UNORM8 RGBA of a texture rung at pixel x, y (level-l texels t, w x h) */
static void
expect_tex(const struct rung *r, const uint8_t *t, unsigned w, unsigned h, unsigned x, unsigned y, uint8_t out[4])
{
   const struct fmt *f = r->f;
   float scale, bias;

   scale_bias(f, &scale, &bias);
   if (r->kind == K_FILTER) {
      /* u = x / 8 at pixel x: s = x - 0.5, taps x-1 and x (clamped), row min(y, 7) */
      const unsigned j = y < h ? y : h - 1;
      const unsigned i1 = x < w ? x : w - 1, i0 = x ? (x - 1 < w ? x - 1 : w - 1) : 0;
      for (unsigned c = 0; c < 4; c++) {
         double a, b;
         int64_t i;
         texel_value(f, t, w, i0, j, c, &a, &i);
         texel_value(f, t, w, i1, j, c, &b, &i);
         out[c] = unorm8((a + b) / 2 * scale + bias);
      }
      return;
   }
   const unsigned i = x * w / W, j = y * h / H;
   for (unsigned c = 0; c < 4; c++) {
      double d;
      int64_t v;
      texel_value(f, t, w, i, j, c, &d, &v);
      if (is_int(f))
         out[c] = (uint8_t)((uint64_t)v >> r->shift & 0xff);
      else
         out[c] = unorm8(d * scale + bias);
   }
}

static int
read_target_wh(VkImage image, unsigned w, unsigned h, unsigned cpp, struct buf *rb)
{
   VkCommandBuffer cmd;
   if (make_buffer(rb, (VkDeviceSize)w * h * cpp, VK_BUFFER_USAGE_TRANSFER_DST_BIT))
      return 1;
   if (begin(&cmd))
      return 1;
   const VkBufferImageCopy reg = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                   .imageExtent = { w, h, 1 } };
   vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb->b, 1, &reg);
   const VkBufferMemoryBarrier bmb = { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                       .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                       .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
                                       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .buffer = rb->b,
                                       .size = VK_WHOLE_SIZE };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &bmb, 0,
                        NULL);
   if (submit(cmd))
      return 1;
   return sync_mem(rb->mem, true);
}

static int
read_target(VkImage image, VkFormat format, unsigned cpp, struct buf *rb)
{
   (void)format;
   return read_target_wh(image, W, H, cpp, rb);
}

/* A graphics pipeline over one colour attachment, a two-attribute quad */
static int
make_pipeline(VkRenderPass rp, VkPipelineLayout layout, const uint32_t *fs, size_t fs_size, bool blend,
              VkPipeline *pipe)
{
   VkShaderModuleCreateInfo smi = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                    .codeSize = sizeof(spv_quad_vert), .pCode = spv_quad_vert };
   VkShaderModule vsm, fsm;
   CHECK(vkCreateShaderModule(dev, &smi, NULL, &vsm));
   smi.codeSize = fs_size;
   smi.pCode = fs;
   CHECK(vkCreateShaderModule(dev, &smi, NULL, &fsm));
   const VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vsm, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fsm, .pName = "main" },
   };
   const VkVertexInputBindingDescription vib = { 0, 16, VK_VERTEX_INPUT_RATE_VERTEX };
   const VkVertexInputAttributeDescription attr[2] = { { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 },
                                                       { 1, 0, VK_FORMAT_R32G32_SFLOAT, 8 } };
   const VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vib, .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = attr };
   const VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP };
   const VkViewport vpt = { 0, 0, W, H, 0, 1 };
   const VkRect2D sc = { { 0, 0 }, { W, H } };
   const VkPipelineViewportStateCreateInfo vps = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                                   .viewportCount = 1, .pViewports = &vpt, .scissorCount = 1,
                                                   .pScissors = &sc };
   const VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
   const VkPipelineMultisampleStateCreateInfo ms = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                     .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   const VkPipelineColorBlendAttachmentState cba = {
      .blendEnable = blend, .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0xf };
   const VkPipelineColorBlendStateCreateInfo cb = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                    .attachmentCount = 1, .pAttachments = &cba };
   const VkGraphicsPipelineCreateInfo gci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cb, .layout = layout, .renderPass = rp };
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gci, NULL, pipe));
   vkDestroyShaderModule(dev, vsm, NULL);
   vkDestroyShaderModule(dev, fsm, NULL);
   return 0;
}

static int
make_pass(VkFormat format, VkImageView view, VkAttachmentLoadOp load, VkRenderPass *rp, VkFramebuffer *fb)
{
   const VkAttachmentDescription att = { .format = format, .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = load,
                                         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                         .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL };
   const VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
   const VkSubpassDescription sp = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     .colorAttachmentCount = 1, .pColorAttachments = &ref };
   const VkSubpassDependency dep = { .srcSubpass = 0, .dstSubpass = VK_SUBPASS_EXTERNAL,
                                     .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                     .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT };
   const VkRenderPassCreateInfo rci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1,
                                        .pAttachments = &att, .subpassCount = 1, .pSubpasses = &sp,
                                        .dependencyCount = 1, .pDependencies = &dep };
   CHECK(vkCreateRenderPass(dev, &rci, NULL, rp));
   const VkFramebufferCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = *rp,
                                         .attachmentCount = 1, .pAttachments = &view, .width = W, .height = H,
                                         .layers = 1 };
   CHECK(vkCreateFramebuffer(dev, &fci, NULL, fb));
   return 0;
}

/* A quad's vertices (position, uv), in a host buffer */
static int
quad(struct buf *vb, float x1, float u0, float u1, float v0, float v1)
{
   const float q[16] = { -1, -1, u0, v0, x1, -1, u1, v0, -1, 1, u0, v1, x1, 1, u1, v1 };
   if (make_buffer(vb, sizeof(q), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT))
      return 1;
   memcpy(vb->map, q, sizeof(q));
   return sync_mem(vb->mem, false);
}

struct result {
   unsigned bad, fx, fy;
   uint8_t got[16], want[16];
   unsigned bytes;
};

static void
note(struct result *res, unsigned x, unsigned y, const uint8_t *got, const uint8_t *want, unsigned n)
{
   if (!memcmp(got, want, n))
      return;
   if (!res->bad++) {
      res->fx = x, res->fy = y, res->bytes = n;
      memcpy(res->got, got, n);
      memcpy(res->want, want, n);
   }
}

static int
report_n(const struct rung *r, const struct result *res, unsigned pixels)
{
   if (!verify) {
      printf("FORMATS %s PASS: submitted and retired (not compared)\n", r->name);
      return 0;
   }
   if (!res->bad) {
      printf("FORMATS %s PASS: %u pixels exact\n", r->name, pixels);
      return 0;
   }
   char g[64] = "", w[64] = "";
   for (unsigned i = 0; i < res->bytes; i++) {
      snprintf(g + strlen(g), sizeof(g) - strlen(g), "%02x", res->got[i]);
      snprintf(w + strlen(w), sizeof(w) - strlen(w), "%02x", res->want[i]);
   }
   printf("FORMATS %s FAIL: %u of %u pixels differ, first at %u,%u: got %s, expected %s\n", r->name, res->bad,
          pixels, res->fx, res->fy, g, w);
   return 1;
}

static int
report(const struct rung *r, const struct result *res)
{
   return report_n(r, res, W * H);
}

/* The bytes of a w x h level: texels, or 4x4 blocks */
static size_t
level_bytes(const struct fmt *f, unsigned w, unsigned h)
{
   return f->codec ? (size_t)((w + 3) / 4) * ((h + 3) / 4) * f->cpp : (size_t)w * h * f->cpp;
}

/* Draw the 64x64 quad sampling view with fs into a BGRA8 target and read it back into rb: nearest, uv 0..1
 * over the target, or linear at half-texel offsets (the quad below) */
static int
draw_sampled(VkImageView view, bool linear, float max_lod, const uint32_t *fs, size_t fs_size, struct buf *rb)
{
   struct img target;
   struct buf vb;
   VkCommandBuffer cmd;

   const VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                     .magFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
                                     .minFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
                                     .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                     .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                     .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                     .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                     .maxLod = max_lod };
   VkSampler sampler;
   CHECK(vkCreateSampler(dev, &sci, NULL, &sampler));
   const VkDescriptorSetLayoutBinding b = { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
   const VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                  .bindingCount = 1, .pBindings = &b };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl));
   const VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
   const VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
                                             .poolSizeCount = 1, .pPoolSizes = &ps };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dpool));
   const VkDescriptorSetAllocateInfo dai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                             .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl };
   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(dev, &dai, &set));
   const VkDescriptorImageInfo dii = { sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
   const VkWriteDescriptorSet wds = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
                                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      .pImageInfo = &dii };
   vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

   if (make_image(&target, VK_FORMAT_B8G8R8A8_UNORM, W, H, 1,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
      return 1;
   VkRenderPass rp;
   VkFramebuffer fb;
   if (make_pass(VK_FORMAT_B8G8R8A8_UNORM, target.view, VK_ATTACHMENT_LOAD_OP_CLEAR, &rp, &fb))
      return 1;
   const VkPipelineLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                            .setLayoutCount = 1, .pSetLayouts = &dsl };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
   VkPipeline pipe;
   if (make_pipeline(rp, layout, fs, fs_size, false, &pipe))
      return 1;
   /* nearest: uv 0..1 over the target, 8 pixels a texel. Linear: at pixel
    * (x, y), u = x/8 and v = (y + 1/2)/8, so s = x - 1/2 (halfway between
    * texels x-1 and x) and t = y (row y's centre); both clamp past the
    * texture */
   if (linear ? quad(&vb, 1, -0.5f / TW, 63.5f / TW, 0, 64.0f / TH) : quad(&vb, 1, 0, 1, 0, 1))
      return 1;

   if (begin(&cmd))
      return 1;
   const VkClearValue cv = { .color.float32 = { 0, 0, 0, 0 } };
   const VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp,
                                       .framebuffer = fb, .renderArea = { { 0, 0 }, { W, H } }, .clearValueCount = 1,
                                       .pClearValues = &cv };
   vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
   const VkDeviceSize zero = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vb.b, &zero);
   vkCmdDraw(cmd, 4, 1, 0, 0);
   vkCmdEndRenderPass(cmd);
   if (submit(cmd) || read_target(target.image, VK_FORMAT_B8G8R8A8_UNORM, 4, rb))
      return 1;

   free_buffer(&vb);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkDestroySampler(dev, sampler, NULL);
   free_image(&target);
   return 0;
}

static int
run_texture(const struct rung *r)
{
   const struct fmt *f = r->f;
   const unsigned levels = r->kind == K_MIP ? 2 : 1, lvl = levels - 1;
   const unsigned w = TW >> lvl, h = TH >> lvl;
   uint8_t texels[2][TW * TH * 16];
   struct img tex;
   struct buf staging, rb;
   VkCommandBuffer cmd;

   /* the texture, uploaded in a submission of its own */
   if (make_image(&tex, f->vk, TW, TH, levels, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
      return 1;
   if (make_buffer(&staging, (VkDeviceSize)TW * TH * f->cpp * 2, VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
      return 1;
   VkBufferImageCopy regs[2];
   VkDeviceSize off = 0;
   for (unsigned l = 0; l < levels; l++) {
      make_texels(r, l, texels[l], TW >> l, TH >> l);
      memcpy(staging.map + off, texels[l], level_bytes(f, TW >> l, TH >> l));
      regs[l] = (VkBufferImageCopy){ .bufferOffset = off, .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1 },
                                     .imageExtent = { TW >> l, TH >> l, 1 } };
      off += level_bytes(f, TW >> l, TH >> l);
   }
   if (sync_mem(staging.mem, false) || begin(&cmd))
      return 1;
   barrier(cmd, tex.image, levels, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
           VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
   vkCmdCopyBufferToImage(cmd, staging.b, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, levels, regs);
   barrier(cmd, tex.image, levels, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
   if (submit(cmd))
      return 1;

   /* the fragment program: the format class's compiled-in scale and bias, the integer byte */
   const bool linear = r->kind == K_FILTER;
   const uint32_t *fs;
   size_t fs_size;
#define FS(n) (fs = spv_##n##_frag, fs_size = sizeof(spv_##n##_frag))
   const bool lod = r->kind == K_MIP;
   float scale, bias;
   scale_bias(f, &scale, &bias);
   if (r->kind == K_FETCH)
      FS(ufetch);
   else if (f->c[0].ty == T_UI)
      r->shift == 0 ? FS(usamp_0) : r->shift == 8 ? FS(usamp_8) : r->shift == 16 ? FS(usamp_16) : FS(usamp_24);
   else if (f->c[0].ty == T_SI)
      r->shift == 0 ? FS(isamp_0) : r->shift == 8 ? FS(isamp_8) : r->shift == 16 ? FS(isamp_16) : FS(isamp_24);
   else if (scale == 0.5f)
      lod ? FS(samplod_sn) : FS(samp_sn);
   else if (scale != 1 && bias != 0)
      lod ? FS(samplod_f) : FS(samp_f);
   else if (scale != 1)
      lod ? FS(samplod_uf) : FS(samp_uf);
   else
      lod ? FS(samplod_un) : FS(samp_un);
#undef FS
   if (draw_sampled(tex.view, linear, levels > 1 ? VK_LOD_CLAMP_NONE : 0.25f, fs, fs_size, &rb))
      return 1;

   struct result res = { 0 };
   for (unsigned y = 0; verify && y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
         const uint8_t *p = rb.map + ((size_t)y * W + x) * 4;
         uint8_t want[4];
         const uint8_t got[4] = { p[2], p[1], p[0], p[3] };
         expect_tex(r, texels[lvl], w, h, x, y, want);
         note(&res, x, y, got, want, 4);
      }
   }
   const int ret = report(r, &res);

   free_buffer(&rb);
   free_buffer(&staging);
   free_image(&tex);
   return ret;
}

/* ---- layout rungs ---- */

/* An RGBA8 texture whose texels name themselves (layout_texel()), read at
 * one level: NPOT and non-square mip chains (each level twiddled over its
 * own extent padded to powers of two), arrays of mip chains (a layer is the
 * image's own levels rounded) and 3D images (4x4x4 bricks, the 16 KiB big
 * layer). 2D rungs read the level over the whole target; array and 3D rungs
 * read layer or slice c in cell c of a 4x4 grid of 16x16 pixels. Every
 * level is at most 16 texels a side, so each texel is read at its centre
 * and none is read at a texel boundary. */

static unsigned
minify(unsigned v, unsigned l)
{
   return v >> l ? v >> l : 1;
}

/* The bytes of texel (x, y, z) of level l: x, y, z < 16, l < 16 */
static void
layout_texel(unsigned x, unsigned y, unsigned z, unsigned l, uint8_t out[4])
{
   out[0] = (uint8_t)(x << 4 | l);
   out[1] = (uint8_t)(y << 4 | z);
   out[2] = (uint8_t)(z << 4 | ((x + 3 * y) & 15));
   out[3] = (uint8_t)(0xf0 ^ (7 * z + 5 * l));
}

/* Whether the device takes the case's image */
static bool
layout_supported(const struct layout_case *c)
{
   VkImageFormatProperties p;
   const bool is3d = c->vt == VK_IMAGE_VIEW_TYPE_3D;
   if (vkGetPhysicalDeviceImageFormatProperties(pd, VK_FORMAT_R8G8B8A8_UNORM,
                                                is3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0,
                                                &p) != VK_SUCCESS)
      return false;
   return p.maxMipLevels >= c->levels && p.maxExtent.width >= c->w && p.maxExtent.height >= c->h &&
          (is3d ? p.maxExtent.depth >= c->d : p.maxArrayLayers >= (c->vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY ? c->d : 1));
}

static int
run_layout(const struct rung *r)
{
   const struct layout_case *c = r->lay;
   const bool is3d = c->vt == VK_IMAGE_VIEW_TYPE_3D, array = c->vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY;
   struct img tex;
   struct buf staging, rb;
   VkCommandBuffer cmd;
   VkBufferImageCopy regs[8];
   VkDeviceSize off = 0;

   if (make_image_ex(&tex, VK_FORMAT_R8G8B8A8_UNORM, c->vt, c->w, c->h, c->d, c->levels,
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
      return 1;
   /* every level of every layer or slice, x fastest, then y, then z */
   if (make_buffer(&staging, (VkDeviceSize)c->w * c->h * c->d * 4 * 2, VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
      return 1;
   for (unsigned l = 0; l < c->levels; l++) {
      const unsigned lw = minify(c->w, l), lh = minify(c->h, l), ld = is3d ? minify(c->d, l) : c->d;
      regs[l] = (VkBufferImageCopy){ .bufferOffset = off,
                                     .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, l, 0, array ? c->d : 1 },
                                     .imageExtent = { lw, lh, is3d ? ld : 1 } };
      for (unsigned z = 0; z < ld; z++) {
         for (unsigned y = 0; y < lh; y++) {
            for (unsigned x = 0; x < lw; x++, off += 4)
               layout_texel(x, y, z, l, staging.map + off);
         }
      }
   }
   if (sync_mem(staging.mem, false) || begin(&cmd))
      return 1;
   barrier(cmd, tex.image, c->levels, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
           VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
   vkCmdCopyBufferToImage(cmd, staging.b, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, c->levels, regs);
   barrier(cmd, tex.image, c->levels, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
   if (submit(cmd))
      return 1;
   /* FORMATS_DUMP=DIR: the image's memory as the upload left it, DIR/<rung>.bin, for a host check of the
    * driver's layout (the texture memory is host-visible) */
   const char *dump = getenv("FORMATS_DUMP");
   if (dump) {
      char path[512];
      void *map;
      snprintf(path, sizeof(path), "%s/%s.bin", dump, r->name);
      CHECK(vkMapMemory(dev, tex.mem, 0, VK_WHOLE_SIZE, 0, &map));
      FILE *fp = fopen(path, "wb");
      if (!fp || fwrite(map, 1, tex.size, fp) != tex.size) {
         printf("FORMATS %s FAIL: dump to %s\n", r->name, path);
         return 1;
      }
      fclose(fp);
      vkUnmapMemory(dev, tex.mem);
   }
   if (draw_sampled(tex.view, false, VK_LOD_CLAMP_NONE, c->fs, c->fs_size, &rb))
      return 1;

   const unsigned l = c->level, lw = minify(c->w, l), lh = minify(c->h, l), ld = is3d ? minify(c->d, l) : c->d;
   struct result res = { 0 };
   for (unsigned y = 0; verify && y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
         const uint8_t *p = rb.map + ((size_t)y * W + x) * 4;
         const uint8_t got[4] = { p[2], p[1], p[0], p[3] };
         unsigned tx, ty, tz = 0;
         uint8_t want[4];

         if (c->vt == VK_IMAGE_VIEW_TYPE_2D) {
            /* u = (x + 1/2) / 64 */
            tx = (2 * x + 1) * lw / (2 * W), ty = (2 * y + 1) * lh / (2 * H);
         } else {
            /* cell (x / 16, y / 16), u = (x % 16 + 1/2) / 16; the layer or slice clamps */
            const unsigned cell = x / 16 + 4 * (y / 16);
            tx = (2 * (x % 16) + 1) * lw / 32, ty = (2 * (y % 16) + 1) * lh / 32;
            tz = cell < ld ? cell : ld - 1;
         }
         layout_texel(tx, ty, tz, l, want);
         note(&res, x, y, got, want, 4);
      }
   }
   const int ret = report(r, &res);

   free_buffer(&rb);
   free_buffer(&staging);
   free_image(&tex);
   return ret;
}

/* ---- render rungs ---- */

/* The raw value of a channel nearest to v, and whether it is exact-safe:
 * render and blend values are chosen exact in f16, so a target's stored
 * channel is the rounding of the exact value, a quarter step from a
 * boundary */
static bool
encode(const struct ch *ch, double v, uint64_t *raw)
{
   switch (ch->ty) {
   case T_NONE: *raw = 0; return true;
   case T_UN: {
      const double s = v * (double)((1ull << ch->bits) - 1), fr = s - floor(s);
      *raw = (uint64_t)lround(s);
      return fabs(fr - 0.5) >= 0.25 || fr == 0;
   }
   case T_SRGB: {
      const double s = linear_to_srgb(v) * 255, fr = s - floor(s);
      *raw = (uint64_t)lround(s);
      return fabs(fr - 0.5) >= 0.25;
   }
   case T_F: { /* f16 targets only: v is exact in f16 */
      int e;
      const double m = frexp(fabs(v), &e);
      if (v == 0) {
         *raw = 0;
         return true;
      }
      const uint64_t mant = (uint64_t)llround(m * 2048) - 1024; /* 1.m with 10 bits */
      *raw = (v < 0 ? 0x8000u : 0) | (uint64_t)(e - 1 + 15) << 10 | mant;
      return half_to_double((uint32_t)*raw) == v;
   }
   case T_UF: {
      const unsigned mb = ch->bits - 5;
      int e;
      const double m = frexp(v, &e);
      if (v == 0) {
         *raw = 0;
         return true;
      }
      *raw = (uint64_t)(e - 1 + 15) << mb | ((uint64_t)llround(m * (2 << mb)) - (1u << mb));
      return uf_to_double((uint32_t)*raw, ch->bits) == v;
   }
   default: return false;
   }
}

/* The render rung's clear (dst) and colour (src), and the blend rung's.
 * Every value, and every blend result (source alpha 1/4: s/4 + 3d/4), is
 * exact in f16 and in an 11/10-bit float, and a quarter step from rounding
 * in UNORM8, 10-bit, 2-bit and sRGB (encode() checks). The blend rung's dst
 * is 0 or 1, the values sRGB keeps exactly. */
static const double CLEAR[4] = { 0.25, 0.9375, 0.0, 1.0 };
static const double SRC[4] = { 0.9375, 0.1875, 0.125, 0.25 }; /* solid.frag (build.sh) */
static const double CLEAR_B[4] = { 1.0, 0.0, 1.0, 0.0 };
static const double SRC_B[4] = { 0.9375, 0.1875, 0.75, 0.25 }; /* solid-b.frag */

static int
run_render(const struct rung *r)
{
   const struct fmt *f = r->f;
   const bool blend = r->kind == K_BLEND;
   const bool raw32 = f->vk == VK_FORMAT_R32G32B32A32_UINT || f->vk == VK_FORMAT_R32G32B32A32_SINT;
   const double *dst = blend ? CLEAR_B : CLEAR, *src = blend ? SRC_B : SRC;
   struct img target;
   struct buf vb, rb;
   VkCommandBuffer cmd;

   /* the expected texels: the clear on the right, the colour (or the blend) on the left */
   uint8_t left[16] = { 0 }, right[16] = { 0 };
   const uint32_t raw_src[4] = { 0x80000001u, 0x7fffffffu, 0x12345678u, 0xffffffffu };
   const uint32_t raw_dst[4] = { 0, 17, 0x10001u, 0x76543210u };
   if (raw32) {
      memcpy(left, raw_src, sizeof(left));
      memcpy(right, raw_dst, sizeof(right));
   }
   for (unsigned c = 0; !raw32 && c < 4; c++) {
      const struct ch *ch = &f->c[c];
      const double sa = src[3];
      const double lv = !blend ? src[c] : c == 3 ? src[3] + dst[3] * (1 - sa) : src[c] * sa + dst[c] * (1 - sa);
      uint64_t raw;

      if (!ch->bits)
         continue;
      if (!encode(ch, dst[c], &raw)) {
         printf("FORMATS %s FAIL: reference: clear %g not exact-safe in channel %u\n", r->name, dst[c], c);
         return 1;
      }
      put_bits(right, ch->off, ch->bits, raw);
      if (!encode(ch, lv, &raw)) {
         printf("FORMATS %s FAIL: reference: value %g not exact-safe in channel %u\n", r->name, lv, c);
         return 1;
      }
      put_bits(left, ch->off, ch->bits, raw);
   }

   if (make_image(&target, f->vk, W, H, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
      return 1;
   VkRenderPass rp;
   VkFramebuffer fb;
   if (make_pass(f->vk, target.view, VK_ATTACHMENT_LOAD_OP_CLEAR, &rp, &fb))
      return 1;
   const VkPipelineLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
   /* the colour is compiled into solid.frag / solid-b.frag (SRC, SRC_B: build.sh) */
   VkPipeline pipe;
   const uint32_t *fs = raw32 ? (f->vk == VK_FORMAT_R32G32B32A32_UINT ? spv_solid_u32_frag : spv_solid_i32_frag)
                             : (blend ? spv_solid_b_frag : spv_solid_frag);
   const size_t fs_size = raw32 ? (f->vk == VK_FORMAT_R32G32B32A32_UINT ? sizeof(spv_solid_u32_frag) : sizeof(spv_solid_i32_frag))
                               : (blend ? sizeof(spv_solid_b_frag) : sizeof(spv_solid_frag));
   if (make_pipeline(rp, layout, fs, fs_size, blend, &pipe))
      return 1;
   if (quad(&vb, 0, 0, 0, 0, 0))
      return 1;
   if (begin(&cmd))
      return 1;
   VkClearValue cv = { .color.float32 = { (float)dst[0], (float)dst[1], (float)dst[2], (float)dst[3] } };
   if (raw32)
      memcpy(cv.color.uint32, raw_dst, sizeof(raw_dst));
   const VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp,
                                       .framebuffer = fb, .renderArea = { { 0, 0 }, { W, H } }, .clearValueCount = 1,
                                       .pClearValues = &cv };
   vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   const VkDeviceSize zero = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vb.b, &zero);
   vkCmdDraw(cmd, 4, 1, 0, 0);
   vkCmdEndRenderPass(cmd);
   if (submit(cmd) || read_target(target.image, f->vk, f->cpp, &rb))
      return 1;

   struct result res = { 0 };
   for (unsigned y = 0; verify && y < H; y++) {
      for (unsigned x = 0; x < W; x++)
         note(&res, x, y, rb.map + ((size_t)y * W + x) * f->cpp, x < W / 2 ? left : right, f->cpp);
   }
   const int ret = report(r, &res);

   free_buffer(&rb);
   free_buffer(&vb);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   free_image(&target);
   return ret;
}

/* ---- blit rungs ---- */

static uint16_t
half_bits(float v)
{
   _Float16 h = (_Float16)v;
   uint16_t b;
   memcpy(&b, &h, 2);
   return b;
}

/* k-<fmt>: an 8x8 compressed texture blitted (nearest, 1:1) into a UNORM8
 * image (an RGBA16F one for signed EAC), read back: the decoded texels */
static int
run_blit(const struct rung *r)
{
   const struct fmt *f = r->f;
   const bool sgn = f->c[0].ty == T_SN;
   const VkFormat dfmt = sgn ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_B8G8R8A8_UNORM;
   const unsigned dcpp = sgn ? 8 : 4;
   uint8_t texels[TW * TH * 16];
   struct img src, dst;
   struct buf staging, rb;
   VkCommandBuffer cmd;

   if (make_image(&src, f->vk, TW, TH, 1, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
       make_image(&dst, dfmt, TW, TH, 1, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
       make_buffer(&staging, level_bytes(f, TW, TH), VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
      return 1;
   make_texels(r, 0, texels, TW, TH);
   memcpy(staging.map, texels, level_bytes(f, TW, TH));
   if (sync_mem(staging.mem, false) || begin(&cmd))
      return 1;
   const VkBufferImageCopy reg = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                                   .imageExtent = { TW, TH, 1 } };
   barrier(cmd, src.image, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
           VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
   vkCmdCopyBufferToImage(cmd, staging.b, src.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
   barrier(cmd, src.image, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
           VK_PIPELINE_STAGE_TRANSFER_BIT);
   barrier(cmd, dst.image, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
           VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
   const VkImageBlit blit = { .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                              .srcOffsets = { { 0, 0, 0 }, { TW, TH, 1 } },
                              .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                              .dstOffsets = { { 0, 0, 0 }, { TW, TH, 1 } } };
   vkCmdBlitImage(cmd, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.image,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
   barrier(cmd, dst.image, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
           VK_PIPELINE_STAGE_TRANSFER_BIT);
   if (submit(cmd) || read_target_wh(dst.image, TW, TH, dcpp, &rb))
      return 1;

   struct result res = { 0 };
   for (unsigned y = 0; verify && y < TH; y++) {
      for (unsigned x = 0; x < TW; x++) {
         uint8_t want[8];
         for (unsigned c = 0; c < 4; c++) {
            const double d = block_texel(f, texels, TW, x, y, c);
            if (sgn) {
               const uint16_t hb = half_bits((float)d);
               memcpy(want + 2 * c, &hb, 2);
            } else {
               want[c == 0 ? 2 : c == 2 ? 0 : c] = unorm8(d); /* B, G, R, A */
            }
         }
         note(&res, x, y, rb.map + ((size_t)y * TW + x) * dcpp, want, dcpp);
      }
   }
   const int ret = report_n(r, &res, TW * TH);
   free_buffer(&rb);
   free_buffer(&staging);
   free_image(&dst);
   free_image(&src);
   return ret;
}

/* ---- main ---- */

/* --coverage: every colour format whose optimal features offer sampling or
 * rendering beyond what a rung here checks is a failure (nothing advertised
 * untested). Core formats 1..184. */
static unsigned
coverage(void)
{
   unsigned bad = 0;
   for (int v = 1; v <= 184; v++) {
      VkFormatProperties p;
      vkGetPhysicalDeviceFormatProperties(pd, (VkFormat)v, &p);
      VkFormatFeatureFlags want = p.optimalTilingFeatures &
                                  (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                   VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT);
      /* a blit source: checked for the block-compressed formats (k- rungs); the
       * uncompressed ones' blits are the transfer path's, not a format's */
      if (v >= VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK && v <= VK_FORMAT_EAC_R11G11_SNORM_BLOCK)
         want |= p.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT;
      if (!want || (v >= VK_FORMAT_D16_UNORM && v <= VK_FORMAT_D32_SFLOAT_S8_UINT))
         continue;
      for (unsigned i = 0; i < nrungs; i++) {
         if (rungs[i].f->vk == (VkFormat)v)
            want &= ~feature_of(rungs[i].kind);
      }
      if (want) {
         printf("FORMATS coverage FAIL: format %d advertises %#x that no rung checks\n", v, want);
         bad++;
      }
   }
   return bad;
}

int
main(int argc, char **argv)
{
   unsigned device_index = 0, nsel = 0, pass = 0, ran = 0, skipped = 0;
   const char *sel[64];
   bool list = false, cover = false;

   setvbuf(stdout, NULL, _IOLBF, 0);
   build_rungs();
   for (int a = 1; a < argc; a++) {
      if (!strcmp(argv[a], "--list"))
         list = true;
      else if (!strcmp(argv[a], "--device") && a + 1 < argc)
         device_index = (unsigned)atoi(argv[++a]);
      else if (!strcmp(argv[a], "--no-verify"))
         verify = false;
      else if (!strcmp(argv[a], "--coverage"))
         cover = true;
      else if (argv[a][0] != '-' && nsel < ARRAY_LEN(sel))
         sel[nsel++] = argv[a];
      else {
         fprintf(stderr, "usage: %s [--list] [--device N] [--no-verify] [--coverage] [RUNG|PREFIX*...]\n", argv[0]);
         return 2;
      }
   }
   if (list && !cover) {
      for (unsigned i = 0; i < nrungs; i++)
         printf("%s\n", rungs[i].name);
      return 0;
   }

   const VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "formats",
                                   .apiVersion = VK_API_VERSION_1_0 };
   const VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 0;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
   VkPhysicalDevice pds[16];
   n = n > 16 ? 16 : n;
   CHECK(vkEnumeratePhysicalDevices(inst, &n, pds));
   if (device_index >= n) {
      printf("FORMATS RESULT FAIL: no Vulkan device %u (%u found)\n", device_index, n);
      return 1;
   }
   pd = pds[device_index];
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pd, &props);
   printf("FORMATS device %u: %s, Vulkan %u.%u, driver %#x\n", device_index, props.deviceName,
          VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion), props.driverVersion);

   /* what the device advertises for each format of the list: S sampled, F linear, C attachment, B blend */
   for (unsigned i = 0; i < ARRAY_LEN(fmts); i++) {
      VkFormatProperties p;
      vkGetPhysicalDeviceFormatProperties(pd, fmts[i].vk, &p);
      const VkFormatFeatureFlags o = p.optimalTilingFeatures;
      printf("FORMATS advertised %-26s %c%c%c%c\n", fmts[i].name, o & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT ? 'S' : '-',
             o & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT ? 'F' : '-',
             o & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT ? 'C' : '-',
             o & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT ? 'B' : '-');
   }
   unsigned uncovered = cover ? coverage() : 0;
   if (list) {
      printf("FORMATS RESULT %s coverage\n", uncovered ? "FAIL" : "PASS");
      return uncovered ? 1 : 0;
   }

   VkQueueFamilyProperties qf[8];
   uint32_t nq = 8;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
   queue_family = UINT32_MAX;
   for (uint32_t i = 0; i < nq && queue_family == UINT32_MAX; i++) {
      if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
         queue_family = i;
   }
   if (queue_family == UINT32_MAX) {
      printf("FORMATS RESULT FAIL: no graphics queue\n");
      return 1;
   }
   const float prio = 1.0f;
   const VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                         .queueFamilyIndex = queue_family, .queueCount = 1, .pQueuePriorities = &prio };
   const VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                                    .pQueueCreateInfos = &qci };
   CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   vkGetDeviceQueue(dev, queue_family, 0, &queue);
   const VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                         .queueFamilyIndex = queue_family };
   CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));

   for (unsigned i = 0; i < nrungs; i++) {
      const struct rung *r = &rungs[i];
      bool want = !nsel;
      for (unsigned k = 0; k < nsel; k++) {
         const size_t l = strlen(sel[k]);
         want |= l && sel[k][l - 1] == '*' ? !strncmp(sel[k], r->name, l - 1) : !strcmp(sel[k], r->name);
      }
      if (!want)
         continue;
      VkFormatProperties p;
      vkGetPhysicalDeviceFormatProperties(pd, r->f->vk, &p);
      if (!(p.optimalTilingFeatures & feature_of(r->kind)) || (r->kind == K_LAYOUT && !layout_supported(r->lay))) {
         printf("FORMATS %s SKIP: not advertised\n", r->name);
         skipped++;
         continue;
      }
      cur = r->name;
      ran++;
      if (!(r->kind == K_RENDER || r->kind == K_BLEND ? run_render(r)
            : r->kind == K_BLIT                    ? run_blit(r)
            : r->kind == K_LAYOUT                  ? run_layout(r)
                                                   : run_texture(r)))
         pass++;
   }
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   const bool ok = ran && pass == ran && !uncovered;
   printf("FORMATS RESULT %s %u of %u (%u not advertised%s)\n", ok ? "PASS" : "FAIL", pass, ran, skipped,
          uncovered ? ", coverage FAILED" : "");
   return ok ? 0 : 1;
}
