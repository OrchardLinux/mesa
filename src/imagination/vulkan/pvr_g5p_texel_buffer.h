/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * A9 buffer textures use the ordinary linear 2D state over buffer memory.
 * AGXA9FamilyDevice minimumTextureBufferAlignmentForPixelFormat tail-calls
 * minimumLinearTextureAlignmentForPixelFormat (0x207cd0898); AGXTexture
 * initWithBuffer:desc:offset:bytesPerRow: passes the buffer/pitch to the
 * ordinary layout constructor (0x207dd1c20). No captured words are embedded.
 */
#ifndef PVR_G5P_TEXEL_BUFFER_H
#define PVR_G5P_TEXEL_BUFFER_H
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan_core.h>
#include "g5p_texel_defs.h"
/* GL3.1's non-RGB32 texture-buffer set. No sRGB, depth or compressed views. */
static inline bool pvr_g5p_tbo_format(VkFormat f)
{
   switch (f) {
#define TBO_TYPES(prefix) \
   case VK_FORMAT_##prefix##_UNORM: \
   case VK_FORMAT_##prefix##_UINT: \
   case VK_FORMAT_##prefix##_SINT:
   TBO_TYPES(R8) TBO_TYPES(R8G8) TBO_TYPES(R8G8B8A8)
   TBO_TYPES(R16) TBO_TYPES(R16G16) TBO_TYPES(R16G16B16A16)
#undef TBO_TYPES
   case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32B32A32_SFLOAT:
   case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32B32A32_UINT:
   case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32A32_SINT:
      return true;
   default: return false;
   }
}
static inline VkFormatFeatureFlags2 pvr_g5p_tbo_features(VkFormat f, VkFormatFeatureFlags2 flags)
{
   flags &= ~(VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT | VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT |
              VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT);
   if (pvr_g5p_tbo_probe() && pvr_g5p_tbo_format(f))
      flags |= VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT;
   /* Storage/atomic admission needs its own stage/read/write qualification. */
   return flags;
}
struct pvr_device;
struct pvr_texture_state_info;
struct pvr_image_descriptor;
VkResult pvr_g5p_tbo_pack(struct pvr_device *, const struct pvr_texture_state_info *, struct pvr_image_descriptor *);
bool pvr_g5p_tbo_upload(unsigned kind, unsigned level, unsigned dwords, const uint32_t desc[8], uint32_t out[8]);
#endif
