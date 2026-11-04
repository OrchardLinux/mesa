/* SPDX-License-Identifier: MIT */
/* Apple A9 data allocation policy. Internal USC/PDS/VDM allocations are NOT
 * application VkDeviceMemory and must stay in the 32-bit code window.
 * Large application images prefer a dedicated Vk allocation so a panel-sized
 * image does not occupy a power-of-two Zink slab. No Zink changes are needed.
 */
#ifndef PVR_G5P_MEMORY_H
#define PVR_G5P_MEMORY_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

#define PVR_G5P_DATA_THRESHOLD (2ull << 20)
/* A private winsys type, selected only by pvr_AllocateMemory's G5P hook.
 * G9P's GPU/DISPLAY enum and internal pvr_bo_alloc callers are unchanged. */
#define PVR_G5P_BO_TYPE_DATA ((enum pvr_winsys_bo_type)2)

static inline bool
pvr_g5p_memory_is_large(uint64_t size)
{
   return size >= PVR_G5P_DATA_THRESHOLD;
}

static inline void
pvr_g5p_memory_dedicated(VkMemoryRequirements2 *requirements)
{
   if (!pvr_g5p_memory_is_large(requirements->memoryRequirements.size))
      return;
   for (VkBaseOutStructure *ext = requirements->pNext; ext; ext = ext->pNext)
      if (ext->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS)
         ((VkMemoryDedicatedRequirements *)ext)->prefersDedicatedAllocation = VK_TRUE;
   /* requiresDedicatedAllocation is unchanged: external-memory requirements
    * remain mandatory; ordinary large images only express a preference. */
}

#endif
