/* SPDX-License-Identifier: MIT */
/* Explicit imported linear surfaces can be sampled in16-byte units. PBE
 * render admission separately retains its64-byte base requirement.
 */
#ifndef PVR_G5P_SHM_H
#define PVR_G5P_SHM_H
#include <vulkan/vulkan_core.h>
static inline const VkImageDrmFormatModifierExplicitCreateInfoEXT *
pvr_g5p_shm_explicit(const VkImageCreateInfo *info)
{
   if (info->tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT)
      return NULL;
   for (const VkBaseInStructure *ext = info->pNext; ext; ext = ext->pNext)
      if (ext->sType == VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT)
         return (const VkImageDrmFormatModifierExplicitCreateInfoEXT *)ext;
   return NULL;
}
#endif
