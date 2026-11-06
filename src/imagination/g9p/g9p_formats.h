/* SPDX-License-Identifier: MIT */
/*
 * Colour formats of Apple's G9P (A10): the image-state key, the same in a
 * texture's state and a render target's. Header-only; the Vulkan driver's
 * image states and its format features both read it, so a format is offered
 * only with its key.
 *
 * The keys are PR #6's A9 table (pvr_g5p_format_table.h, from iOS captures and
 * tests on the A9); the rule reproduces the A10's own captures of BGRA8, RGBA8
 * and RGBA16F. The code is a bit layout | 0x80 signed | 0x100 integer | 0x200
 * float. Layouts: 0x00 8, 0x0a 8_8, 0x28 8_8_8_8, 0x09 16, 0x23 16_16,
 * 0x32 16_16_16_16, 0x21 32, 0x31 32_32, 0x38 32_32_32_32, 0x26 10_10_10_2,
 * 0x25 11_11_10, 0x27 9_9_9_e5, 0x0b 5_6_5, 0x0c 4_4_4_4. Selectors, for R, G,
 * B, A: 0-3 a stored component (0 the lowest field of a packed format), 4 one,
 * 5 zero. Zink and Mesa pick another format for what is not here.
 */
#ifndef G9P_FORMATS_H
#define G9P_FORMATS_H

#include <stdbool.h>
#include <stdint.h>

#include <vulkan/vulkan_core.h>

#define G9P_FORMAT_KEY(code, r, g, b, a)                                                          \
   ((uint32_t)(code) << 5 | (uint32_t)(r) << 15 | (uint32_t)(g) << 18 | (uint32_t)(b) << 21 |     \
    (uint32_t)(a) << 24)
#define G9P_RGBA(code) G9P_FORMAT_KEY(code, 0, 1, 2, 3)
#define G9P_BGRA(code) G9P_FORMAT_KEY(code, 2, 1, 0, 3)
#define G9P_RG01(code) G9P_FORMAT_KEY(code, 0, 1, 5, 4)
#define G9P_R001(code) G9P_FORMAT_KEY(code, 0, 5, 5, 4)
#define G9P_RGB1(code) G9P_FORMAT_KEY(code, 0, 1, 2, 4)

/* The image-state key of @format (bits 5-26), 0 if it has none */
static inline uint32_t
g9p_image_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
      return G9P_BGRA(0x28);
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
      return G9P_RGBA(0x28);
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      return G9P_RGBA(0x26);
   case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
      return G9P_BGRA(0x26);
   case VK_FORMAT_A2B10G10R10_UINT_PACK32:
      return G9P_RGBA(0x126);
   case VK_FORMAT_R8_UNORM:
      return G9P_R001(0x00);
   case VK_FORMAT_R8G8_UNORM:
      return G9P_RG01(0x0a);
   case VK_FORMAT_R8_SNORM:
      return G9P_R001(0x80);
   case VK_FORMAT_R8G8_SNORM:
      return G9P_RG01(0x8a);
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
      return G9P_RGBA(0xa8);
   case VK_FORMAT_R16_SFLOAT:
      return G9P_R001(0x209);
   case VK_FORMAT_R16G16_SFLOAT:
      return G9P_RG01(0x223);
   case VK_FORMAT_R16G16B16A16_SFLOAT:
      return G9P_RGBA(0x232);
   case VK_FORMAT_R32_SFLOAT:
      return G9P_R001(0x221);
   case VK_FORMAT_R32G32_SFLOAT:
      return G9P_RG01(0x231);
   case VK_FORMAT_R32G32B32A32_SFLOAT:
      return G9P_RGBA(0x238);
   case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
      return G9P_RGB1(0x225);
   case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
      return G9P_RGB1(0x227);
   case VK_FORMAT_R5G6B5_UNORM_PACK16:
      return G9P_FORMAT_KEY(0x0b, 2, 1, 0, 4);
   case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
      return G9P_FORMAT_KEY(0x0c, 1, 2, 3, 0);
   case VK_FORMAT_R8_UINT:
      return G9P_R001(0x100);
   case VK_FORMAT_R8G8_UINT:
      return G9P_RG01(0x10a);
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_A8B8G8R8_UINT_PACK32:
      return G9P_RGBA(0x128);
   case VK_FORMAT_R16_UINT:
      return G9P_R001(0x109);
   case VK_FORMAT_R16G16_UINT:
      return G9P_RG01(0x123);
   case VK_FORMAT_R16G16B16A16_UINT:
      return G9P_RGBA(0x132);
   case VK_FORMAT_R32_UINT:
      return G9P_R001(0x121);
   case VK_FORMAT_R32G32_UINT:
      return G9P_RG01(0x131);
   case VK_FORMAT_R32G32B32A32_UINT:
      return G9P_RGBA(0x138);
   case VK_FORMAT_R8_SINT:
      return G9P_R001(0x180);
   case VK_FORMAT_R8G8_SINT:
      return G9P_RG01(0x18a);
   case VK_FORMAT_R8G8B8A8_SINT:
   case VK_FORMAT_A8B8G8R8_SINT_PACK32:
      return G9P_RGBA(0x1a8);
   case VK_FORMAT_R16_SINT:
      return G9P_R001(0x189);
   case VK_FORMAT_R16G16_SINT:
      return G9P_RG01(0x1a3);
   case VK_FORMAT_R16G16B16A16_SINT:
      return G9P_RGBA(0x1b2);
   case VK_FORMAT_R32_SINT:
      return G9P_R001(0x1a1);
   case VK_FORMAT_R32G32_SINT:
      return G9P_RG01(0x1b1);
   case VK_FORMAT_R32G32B32A32_SINT:
      return G9P_RGBA(0x1b8);
   default:
      return 0;
   }
}

/* Rendered through the fragment shader's four f16 and the PBE's packing */
static inline bool
g9p_format_renderable(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
      return true;
   default:
      return false;
   }
}

/* The hardware filters it: not integers, not 32-bit float */
static inline bool
g9p_format_filterable(VkFormat format)
{
   const uint32_t code = (g9p_image_format(format) >> 5) & 0x3ff;

   return code && !(code & 0x100) && code != 0x221 && code != 0x231 && code != 0x238;
}

#endif /* G9P_FORMATS_H */
