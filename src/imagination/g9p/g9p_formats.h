/* SPDX-License-Identifier: MIT */
/*
 * Image formats of the Apple A10 GPU (applegpu_g9p): bytes 1-2 of an image
 * state's dword 0, for render targets and textures alike (the A10 notes'
 * RENDER.md "Render target captures" and "Varyings and textures with our
 * encodings"). Header-only; the Vulkan driver's image states and its format
 * features both read it, so a format is offered only once it is decoded.
 */
#ifndef G9P_FORMATS_H
#define G9P_FORMATS_H

#include <stdint.h>

#include <vulkan/vulkan_core.h>

/* The format bits of @format, 0 if not decoded yet (decoded + tested) */
static inline uint32_t
g9p_image_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
      return 0x0505;
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
      return 0x4405;
   case VK_FORMAT_R16G16B16A16_SFLOAT:
      return 0x4446;
   default:
      return 0;
   }
}

#endif /* G9P_FORMATS_H */
