/* SPDX-License-Identifier: MIT */
/*
 * The Apple A7's (G3) compute paths of the driver (pvr_arch_g3_compute.c):
 * the A8's job, stream, CDM and PDS glue with the A7's own USC programs
 * (g3/g3_compute.h). Reached from the A8/A9 compute paths when the device
 * has apple_g3.
 */
#ifndef PVR_G3_COMPUTE_H
#define PVR_G3_COMPUTE_H

#include <stdint.h>
#include <vulkan/vulkan_core.h>

#include "pvr_macros.h"

struct pvr_device;
struct pvr_compute_pipeline;
struct pvr_cmd_buffer;
struct pvr_sub_cmd_compute;

#ifdef PVR_PER_ARCH
VkResult PVR_PER_ARCH(g3_compute_pipeline_compile)(struct pvr_device *, const VkComputePipelineCreateInfo *,
                                                   struct pvr_compute_pipeline *);
VkResult PVR_PER_ARCH(g3_cmd_dispatch)(struct pvr_cmd_buffer *, struct pvr_sub_cmd_compute *, uint64_t,
                                       const uint32_t[3], const uint32_t[3]);
#   define pvr_arch_g3_compute_pipeline_compile PVR_PER_ARCH(g3_compute_pipeline_compile)
#   define pvr_arch_g3_cmd_dispatch PVR_PER_ARCH(g3_cmd_dispatch)
#endif

#endif
