/* SPDX-License-Identifier: MIT */
/*
 * Apple Rogue-derived GPUs (G9P, has_apple_g9p): compute pipelines,
 * dispatches and the end of a compute pass (pvr_arch_g9p.c).
 */
#ifndef PVR_G9P_H
#define PVR_G9P_H

#include <stdint.h>

#include <vulkan/vulkan_core.h>

#include "pvr_macros.h"

struct pvr_cmd_buffer;
struct pvr_compute_pipeline;
struct pvr_device;
struct pvr_sub_cmd_compute;

#ifdef PVR_PER_ARCH

VkResult PVR_PER_ARCH(g9p_device_init)(struct pvr_device *device);
void PVR_PER_ARCH(g9p_device_finish)(struct pvr_device *device);

VkResult PVR_PER_ARCH(g9p_compute_pipeline_compile)(
   struct pvr_device *device,
   const VkComputePipelineCreateInfo *create_info,
   struct pvr_compute_pipeline *pipeline);

VkResult PVR_PER_ARCH(g9p_cmd_dispatch)(struct pvr_cmd_buffer *cmd_buffer,
                                        struct pvr_sub_cmd_compute *sub_cmd,
                                        const uint32_t group_count[static 3]);

void PVR_PER_ARCH(g9p_end_pass)(struct pvr_cmd_buffer *cmd_buffer,
                                struct pvr_sub_cmd_compute *sub_cmd);

#   define pvr_arch_g9p_device_init PVR_PER_ARCH(g9p_device_init)
#   define pvr_arch_g9p_device_finish PVR_PER_ARCH(g9p_device_finish)
#   define pvr_arch_g9p_compute_pipeline_compile \
      PVR_PER_ARCH(g9p_compute_pipeline_compile)
#   define pvr_arch_g9p_cmd_dispatch PVR_PER_ARCH(g9p_cmd_dispatch)
#   define pvr_arch_g9p_end_pass PVR_PER_ARCH(g9p_end_pass)

#endif /* PVR_PER_ARCH */

#endif /* PVR_G9P_H */
