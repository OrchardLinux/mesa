/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): texture state and sampler words, as pure functions
 * over plain values. The driver (pvr_arch_g5p_tex.c) and the host field
 * test (tools: g5p-tex-fields) include this one copy.
 *
 * Texture state, three qwords (the image-state layout the A9's PBE shares;
 * our decode of AGXMetalA9's ImageStateFields and PBEStateEncoderGen3::
 * StateFields, checked against the iOS captures named below):
 *   q0 = (height - 1) << 41 | (width - 1) << 27 | key, where the key is
 *        layout (bits 0-2: 2 2D, 3 2D array, 5 3D, 6 cube) | twiddled
 *        (bit 3) | format code << 5 | channel selectors R, G, B, A at bits
 *        15, 18, 21, 24 (0-3 a component, 4 one, 5 zero); twiddled images
 *        add the base level at bit 57;
 *   q1 = VA >> 1 (64-byte aligned) | sRGB at bit 45; linear: the row
 *        stride S = stride / 16 - 1, S[17:2] at bits 47-62; twiddled: the
 *        last level (bits 0-3), 0x10 when the image has mips, the layer or
 *        slice count - 1 at bit 47 (cube: 0), bit 61 when a layer is 16 KiB
 *        or more;
 *   q2 = linear: S[1:0] at bits 62-63; twiddled: 0.
 * Captures: pass/pass-rgba (linear 2D), tex-array/3d/cube (layouts 3/5/6,
 * bit 47), g3b-samp-mip and g3b-rt-mip2 (level fields 0x14, 0x16),
 * g3b-rt-layer1 (bit 61), g3b-samp-swizzle (selectors), fmt-srgb8 (bit 45).
 *
 * Sampler word (64 bits; our decode of the iOS sampler states of the
 * gles3/textypes captures): min LOD (bits 0-9) and max LOD (10-19) in
 * 1/64 units clamped to 14, log2 anisotropy at 20, linear magnification
 * at 23, linear minification at 25, mip-nearest at 27, mip-linear at 28,
 * address modes U/V/W at 29/32/35 (0 clamp to edge, 1 repeat, 2 mirrored
 * repeat, 3 clamp to transparent black), the compare function at 39 (code
 * 0 LessEqual with bit 42 set: the default 0x00c00780 high half).
 */

#ifndef PVR_G5P_TEX_ENCODE_H
#define PVR_G5P_TEX_ENCODE_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

/* The image-state key's fields */
#define PVR_G5P_TEX_LAYOUT_2D 2u
#define PVR_G5P_TEX_LAYOUT_2D_ARRAY 3u
#define PVR_G5P_TEX_LAYOUT_3D 5u
#define PVR_G5P_TEX_LAYOUT_CUBE 6u
#define PVR_G5P_TEX_TWIDDLED 8u
#define PVR_G5P_TEX_SEL_ONE 4u
#define PVR_G5P_TEX_SEL_ZERO 5u
#define PVR_G5P_TEX_SEL_SHIFT(c) (15u + 3u * (c))
#define PVR_G5P_TEX_KEY(code, r, g, b, a)                                                               \
   (PVR_G5P_TEX_LAYOUT_2D | PVR_G5P_TEX_TWIDDLED | (uint32_t)(code) << 5 |                            \
    (uint32_t)(r) << PVR_G5P_TEX_SEL_SHIFT(0) | (uint32_t)(g) << PVR_G5P_TEX_SEL_SHIFT(1) |           \
    (uint32_t)(b) << PVR_G5P_TEX_SEL_SHIFT(2) | (uint32_t)(a) << PVR_G5P_TEX_SEL_SHIFT(3))

/* Linear ImageStateFields (AGXMetalA9 ImageStateEncoderGen2, G5 revision):
 * base[38:6] at q1[37:5]; pitch_code=(bytes/16)-1, high bits at q1[62:47].
 * q1[63] enables q2's pitch_code[1:0] and base[5:4]. Without it, hardware
 * supplies pitch low bits=3 and base low bits=0. Source decode sites:
 * 0x207d17c3c..0x207d17c84 and0x207d17de4, differential shmtex field tests.
 * This encodes fields; it contains no captured state/program words.
 */
#define PVR_G5P_TEX_LINEAR_BASE_BLOCK 64ull
#define PVR_G5P_TEX_LINEAR_BASE_LOW_UNIT 16u
#define PVR_G5P_TEX_LINEAR_PITCH_UNIT 16u
#define PVR_G5P_TEX_LINEAR_PITCH_BITS 18u
#define PVR_G5P_TEX_LINEAR_LOW_MASK 3u
#define PVR_G5P_TEX_LINEAR_PITCH_HIGH_SHIFT 45u
#define PVR_G5P_TEX_LINEAR_EXT_ENABLE_BIT 63u
#define PVR_G5P_TEX_LINEAR_PITCH_LOW_SHIFT 62u
#define PVR_G5P_TEX_LINEAR_BASE_LOW_SHIFT 58u
#define PVR_G5P_TEX_SRGB_BIT 45
#define PVR_G5P_TEX_DEPTH_SHIFT 47
#define PVR_G5P_TEX_BIG_LAYER_BIT 61
#define PVR_G5P_TEX_HAS_MIPS 0x10u
#define PVR_G5P_TEX_BASE_LEVEL_SHIFT 57
/* A layer of this size or more is aligned to 16 KiB and sets bit 61 */
#define PVR_G5P_TEX_BIG_LAYER 16384u

/* The sampler word's fields */
#define PVR_G5P_SMP_LOD_MAX 896u /* 14 levels in 1/64 units */
#define PVR_G5P_SMP_MAXLOD_SHIFT 10
#define PVR_G5P_SMP_ANISO_SHIFT 20
#define PVR_G5P_SMP_MAG_LINEAR_BIT 23
#define PVR_G5P_SMP_MIN_LINEAR_BIT 25
#define PVR_G5P_SMP_MIP_NEAREST_BIT 27
#define PVR_G5P_SMP_MIP_LINEAR_BIT 28
#define PVR_G5P_SMP_ADDR_SHIFT(c) (29u + 3u * (c))
#define PVR_G5P_SMP_COMPARE_SHIFT 39
/* The fixed fields of every sampler word: the high half 0x00c00780 */
#define PVR_G5P_SMP_BASE 0x00c0078000000000ull

/*
 * A view's component mapping composed into the key's channel selectors,
 * Apple's rule (g3b-samp-swizzle: (B, ONE, R, G) on RGBA8 is [2, 4, 0, 1]).
 */
static inline uint32_t
pvr_g5p_tex_swizzle_key(uint32_t key, const VkComponentMapping *m)
{
   const VkComponentSwizzle s[4] = { m->r, m->g, m->b, m->a };
   uint32_t out = key;

   for (unsigned i = 0; i < 4; i++) {
      unsigned sel;

      switch (s[i]) {
      case VK_COMPONENT_SWIZZLE_ZERO:
         sel = PVR_G5P_TEX_SEL_ZERO;
         break;
      case VK_COMPONENT_SWIZZLE_ONE:
         sel = PVR_G5P_TEX_SEL_ONE;
         break;
      case VK_COMPONENT_SWIZZLE_R:
      case VK_COMPONENT_SWIZZLE_G:
      case VK_COMPONENT_SWIZZLE_B:
      case VK_COMPONENT_SWIZZLE_A:
         sel = (key >> PVR_G5P_TEX_SEL_SHIFT(s[i] - VK_COMPONENT_SWIZZLE_R)) & 7;
         break;
      default: /* identity */
         sel = (key >> PVR_G5P_TEX_SEL_SHIFT(i)) & 7;
         break;
      }
      out = (out & ~(7u << PVR_G5P_TEX_SEL_SHIFT(i))) | sel << PVR_G5P_TEX_SEL_SHIFT(i);
   }
   return out;
}

/*
 * The sampled state of a linear2D image. Bit63 enables extended base/pitch
 * fields here; PBE uses its own always-enabled form. Rows/base are16-byte units.
 */
static inline void
pvr_g5p_tex_linear(uint64_t q[3], uint32_t key, bool srgb, uint64_t va, uint32_t width,
                   uint32_t height, uint32_t row_stride)
{
   const uint32_t pitch = row_stride / PVR_G5P_TEX_LINEAR_PITCH_UNIT - 1;
   const uint32_t high_mask = ((1u << PVR_G5P_TEX_LINEAR_PITCH_BITS) - 1) &
                              ~PVR_G5P_TEX_LINEAR_LOW_MASK;
   const uint32_t base_low = (va / PVR_G5P_TEX_LINEAR_BASE_LOW_UNIT) & PVR_G5P_TEX_LINEAR_LOW_MASK;
   const uint32_t pitch_low = pitch & PVR_G5P_TEX_LINEAR_LOW_MASK;

   q[0] = (uint64_t)(height - 1) << 41 | (uint64_t)(width - 1) << 27 | (key & ~PVR_G5P_TEX_TWIDDLED);
   q[1] = (va & ~(PVR_G5P_TEX_LINEAR_BASE_BLOCK - 1)) >> 1 |
          (uint64_t)(pitch & high_mask) << PVR_G5P_TEX_LINEAR_PITCH_HIGH_SHIFT |
          (uint64_t)srgb << PVR_G5P_TEX_SRGB_BIT;
   q[2] = (uint64_t)pitch_low << PVR_G5P_TEX_LINEAR_PITCH_LOW_SHIFT;
   if (pitch_low != PVR_G5P_TEX_LINEAR_LOW_MASK || base_low) {
      q[1] |= 1ull << PVR_G5P_TEX_LINEAR_EXT_ENABLE_BIT;
      q[2] |= (uint64_t)base_low << PVR_G5P_TEX_LINEAR_BASE_LOW_SHIFT;
   }
}

/* A view of a twiddled image */
struct pvr_g5p_tex_twiddled_view {
   uint32_t key;         /* swizzled; its layout bits are replaced by `layout` */
   unsigned layout;      /* PVR_G5P_TEX_LAYOUT_* */
   uint32_t width, height; /* level 0 of the image, in texels */
   uint32_t depth;       /* layers of an array view, slices of a 3D image, else 1 */
   unsigned base_level, level_count;
   bool image_has_mips;  /* the image has more than one level */
   bool big_layer;       /* layered image with layers of 16 KiB or more */
   bool srgb;
   uint64_t va;          /* the chain's base, layer-offset for array views */
};

static inline void
pvr_g5p_tex_twiddled(uint64_t q[3], const struct pvr_g5p_tex_twiddled_view *v)
{
   const bool layered = v->layout == PVR_G5P_TEX_LAYOUT_2D_ARRAY || v->layout == PVR_G5P_TEX_LAYOUT_3D;

   q[0] = (uint64_t)(v->height - 1) << 41 | (uint64_t)(v->width - 1) << 27 | (v->key & ~7u) | v->layout |
          (uint64_t)v->base_level << PVR_G5P_TEX_BASE_LEVEL_SHIFT;
   q[1] = v->va >> 1 | (uint64_t)(layered ? v->depth - 1 : 0) << PVR_G5P_TEX_DEPTH_SHIFT |
          (uint64_t)v->srgb << PVR_G5P_TEX_SRGB_BIT |
          (uint64_t)(v->big_layer && v->layout != PVR_G5P_TEX_LAYOUT_2D) << PVR_G5P_TEX_BIG_LAYER_BIT |
          (v->image_has_mips ? PVR_G5P_TEX_HAS_MIPS : 0) | (v->base_level + v->level_count - 1);
   q[2] = 0;
}

static inline unsigned
pvr_g5p_smp_lod(float lod)
{
   const float x = lod * 64.0f;

   return x <= 0.0f ? 0 : x >= (float)PVR_G5P_SMP_LOD_MAX ? PVR_G5P_SMP_LOD_MAX : (unsigned)x;
}

static inline int
pvr_g5p_smp_address(VkSamplerAddressMode m, VkBorderColor border)
{
   switch (m) {
   case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
      return 0;
   case VK_SAMPLER_ADDRESS_MODE_REPEAT:
      return 1;
   case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
      return 2;
   case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
      return border == VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK ||
                   border == VK_BORDER_COLOR_INT_TRANSPARENT_BLACK
                ? 3
                : -1;
   default: /* mirror clamp to edge */
      return -1;
   }
}

/*
 * One form of a sampler's word, or a refusal (NULL: *word is valid).
 * mip_nearest: Vulkan NEAREST mipmapping as Apple's mip-nearest (bit 27,
 * the LOD clamps applied, as LINEAR's bit 28); false gives the single-level
 * form (no mip filter, LOD 0..14), the word for one-level views. The two
 * forms are equal unless mipmapMode is NEAREST. `extension`: the create
 * info chains YCbCr conversion, a reduction mode or a custom border colour.
 */
static inline const char *
pvr_g5p_sampler_encode(const VkSamplerCreateInfo *ci, bool extension, bool mip_nearest, uint64_t *word)
{
   static const uint8_t compare_code[] = {
      [VK_COMPARE_OP_NEVER] = 7,         [VK_COMPARE_OP_LESS] = 2,
      [VK_COMPARE_OP_EQUAL] = 4,         [VK_COMPARE_OP_LESS_OR_EQUAL] = 0,
      [VK_COMPARE_OP_GREATER] = 3,       [VK_COMPARE_OP_NOT_EQUAL] = 5,
      [VK_COMPARE_OP_GREATER_OR_EQUAL] = 1, [VK_COMPARE_OP_ALWAYS] = 6,
   };
   int addr[3];
   uint64_t w = PVR_G5P_SMP_BASE;

   *word = 0;
   if (ci->compareEnable && (unsigned)ci->compareOp >= sizeof(compare_code))
      return "compare op";
   if (ci->unnormalizedCoordinates)
      return "unnormalized coordinates";
   if (ci->mipLodBias != 0.0f)
      return "LOD bias (no field in the sampler word)";
   if (extension)
      return "YCbCr conversion, reduction mode or custom border colour";
   addr[0] = pvr_g5p_smp_address(ci->addressModeU, ci->borderColor);
   addr[1] = pvr_g5p_smp_address(ci->addressModeV, ci->borderColor);
   addr[2] = pvr_g5p_smp_address(ci->addressModeW, ci->borderColor);
   for (unsigned i = 0; i < 3; i++) {
      if (addr[i] < 0)
         return "border colour other than transparent black, or mirror-clamp-to-edge (no such address "
                "mode on A9)";
   }

   const bool mip = ci->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR;
   const bool nearest = mip_nearest && ci->mipmapMode == VK_SAMPLER_MIPMAP_MODE_NEAREST;
   const unsigned min_lod = pvr_g5p_smp_lod(ci->minLod), max_lod = pvr_g5p_smp_lod(ci->maxLod);

   w |= mip || nearest ? min_lod : 0;
   w |= (uint64_t)(mip || nearest ? (max_lod > min_lod ? max_lod : min_lod) : PVR_G5P_SMP_LOD_MAX)
        << PVR_G5P_SMP_MAXLOD_SHIFT;
   w |= (uint64_t)nearest << PVR_G5P_SMP_MIP_NEAREST_BIT;
   if (ci->anisotropyEnable) {
      const float a = ci->maxAnisotropy < 1.0f ? 1.0f : ci->maxAnisotropy > 16.0f ? 16.0f : ci->maxAnisotropy;
      unsigned log2 = 0;

      while ((1u << log2) < (unsigned)a)
         log2++;
      w |= (uint64_t)log2 << PVR_G5P_SMP_ANISO_SHIFT;
   }
   w |= (uint64_t)(ci->magFilter == VK_FILTER_LINEAR) << PVR_G5P_SMP_MAG_LINEAR_BIT;
   w |= (uint64_t)(ci->minFilter == VK_FILTER_LINEAR) << PVR_G5P_SMP_MIN_LINEAR_BIT;
   w |= (uint64_t)mip << PVR_G5P_SMP_MIP_LINEAR_BIT;
   for (unsigned i = 0; i < 3; i++)
      w |= (uint64_t)addr[i] << PVR_G5P_SMP_ADDR_SHIFT(i);
   if (ci->compareEnable)
      w = (w & ~(7ull << PVR_G5P_SMP_COMPARE_SHIFT)) |
          (uint64_t)compare_code[ci->compareOp] << PVR_G5P_SMP_COMPARE_SHIFT;
   *word = w;
   return NULL;
}

/*
 * The backend's image-size upload (PCO_G5P_UPLOAD_IMAGE_SIZE: textureSize at
 * `level` and textureQueryLevels) from the view's state words alone: width
 * and height at the level, the slices at the level (3D) or the layers (an
 * array: the depth field + 1, cube: 6, else 1), and the view's level count.
 * Returns false for a state that isn't valid.
 */
static inline bool
pvr_g5p_tex_size(const uint32_t state[4], unsigned level, uint32_t out[4])
{
   const uint64_t q0 = state[0] | (uint64_t)state[1] << 32, q1 = state[2] | (uint64_t)state[3] << 32;
   const unsigned layout = q0 & 7;
   const bool twiddled = q0 & PVR_G5P_TEX_TWIDDLED;
   const unsigned base = twiddled ? (q0 >> PVR_G5P_TEX_BASE_LEVEL_SHIFT) & 0xf : 0;
   const unsigned l = base + level;
   const uint32_t w0 = ((q0 >> 27) & 0x3fff) + 1, h0 = ((q0 >> 41) & 0x3fff) + 1;
   const uint32_t d0 = twiddled ? ((q1 >> PVR_G5P_TEX_DEPTH_SHIFT) & 0x3fff) + 1 : 1;

   if (!q0)
      return false;
   out[0] = w0 >> l ? w0 >> l : 1;
   out[1] = h0 >> l ? h0 >> l : 1;
   switch (layout) {
   case PVR_G5P_TEX_LAYOUT_3D:
      out[2] = d0 >> l ? d0 >> l : 1;
      break;
   case PVR_G5P_TEX_LAYOUT_2D_ARRAY:
      out[2] = d0;
      break;
   case PVR_G5P_TEX_LAYOUT_CUBE:
      out[2] = 6;
      break;
   default:
      out[2] = 1;
      break;
   }
   out[3] = twiddled ? (unsigned)(q1 & 0xf) - base + 1 : 1;
   return true;
}

/* A descriptor's state as the draw path uploads it: a valid image state
 * never has q0 = 0 (its layout bits), a valid sampler word is never 0. */
static inline bool
pvr_g5p_tex_state_valid(const uint32_t dwords[2])
{
   return dwords[0] || dwords[1];
}

static inline bool
pvr_g5p_sampler_valid(const uint32_t dwords[2])
{
   return dwords[0] || dwords[1];
}

#endif /* PVR_G5P_TEX_ENCODE_H */
