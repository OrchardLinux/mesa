/* SPDX-License-Identifier: MIT */
#ifndef PVR_G5P_QUERY_H
#define PVR_G5P_QUERY_H

#include <vulkan/vulkan_core.h>

struct pvr_cmd_buffer;
struct pvr_device;
struct pvr_query_info;

#ifdef PVR_PER_ARCH
VkResult PVR_PER_ARCH(g5p_query_programs_create)(struct pvr_device *device);
void PVR_PER_ARCH(g5p_query_programs_destroy)(struct pvr_device *device);
VkResult
   PVR_PER_ARCH(g5p_add_query_program)(struct pvr_cmd_buffer *cmd_buffer,
                                       const struct pvr_query_info *query_info);

#   define pvr_arch_g5p_query_programs_create \
      PVR_PER_ARCH(g5p_query_programs_create)
#   define pvr_arch_g5p_query_programs_destroy \
      PVR_PER_ARCH(g5p_query_programs_destroy)
#   define pvr_arch_g5p_add_query_program PVR_PER_ARCH(g5p_add_query_program)
#endif

#endif
