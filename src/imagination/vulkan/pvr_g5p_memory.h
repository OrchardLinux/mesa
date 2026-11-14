/* SPDX-License-Identifier: MIT */
/* Apple A9: large application images prefer a dedicated Vk allocation so a
 * panel-sized image does not occupy a power-of-two Zink slab.
 */
#ifndef PVR_G5P_MEMORY_H
#define PVR_G5P_MEMORY_H

#include <vulkan/vulkan_core.h>

/* From this size an image prefers memory of its own. Zink suballocates up to
 * 1 MiB itself, from 2 MiB parents. */
#define PVR_G5P_DEDICATED_THRESHOLD (1ull << 20)

static inline void
pvr_g5p_memory_dedicated(VkMemoryRequirements2 *requirements)
{
   if (requirements->memoryRequirements.size < PVR_G5P_DEDICATED_THRESHOLD)
      return;
   for (VkBaseOutStructure *ext = requirements->pNext; ext; ext = ext->pNext)
      if (ext->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS)
         ((VkMemoryDedicatedRequirements *)ext)->prefersDedicatedAllocation = VK_TRUE;
   /* requiresDedicatedAllocation is unchanged: external-memory requirements
    * remain mandatory; ordinary large images only express a preference. */
}

#endif
