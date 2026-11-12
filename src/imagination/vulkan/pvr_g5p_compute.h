/* SPDX-License-Identifier: MIT */
#ifndef PVR_G5P_COMPUTE_H
#define PVR_G5P_COMPUTE_H
#include <stdint.h>
#include <vulkan/vulkan_core.h>
#include "pvr_macros.h"
struct pvr_device;
struct pvr_compute_pipeline;
struct pvr_cmd_buffer;
struct pvr_sub_cmd_compute;
struct pvr_compute_ctx;
struct vk_sync;
/* Private winsys stream; fits fw_stream[100], not a kernel ABI. */
struct pvr_g5p_compute_stream {
 uint64_t cdm_va, state_va, tail_va;
 uint32_t cdm_size, cdm_end;
 uint32_t halt_code, halt_data;
};
#ifdef PVR_PER_ARCH
VkResult PVR_PER_ARCH(g5p_compute_pipeline_compile)(struct pvr_device *, const VkComputePipelineCreateInfo *, struct pvr_compute_pipeline *);
VkResult PVR_PER_ARCH(g5p_cmd_dispatch)(struct pvr_cmd_buffer *, struct pvr_sub_cmd_compute *, uint64_t, const uint32_t[3], const uint32_t[3]);
VkResult PVR_PER_ARCH(g5p_compute_end_pass)(struct pvr_cmd_buffer *, struct pvr_sub_cmd_compute *);
VkResult PVR_PER_ARCH(g5p_compute_job_submit)(struct pvr_compute_ctx *, struct pvr_sub_cmd_compute *, struct vk_sync *, struct vk_sync *);
VkResult PVR_PER_ARCH(g5p_compute_ctx_create)(struct pvr_device *, struct pvr_compute_ctx **);
void PVR_PER_ARCH(g5p_compute_ctx_destroy)(struct pvr_compute_ctx *);
#define pvr_arch_g5p_compute_pipeline_compile PVR_PER_ARCH(g5p_compute_pipeline_compile)
#define pvr_arch_g5p_cmd_dispatch PVR_PER_ARCH(g5p_cmd_dispatch)
#define pvr_arch_g5p_compute_end_pass PVR_PER_ARCH(g5p_compute_end_pass)
#define pvr_arch_g5p_compute_job_submit PVR_PER_ARCH(g5p_compute_job_submit)
#define pvr_arch_g5p_compute_ctx_create PVR_PER_ARCH(g5p_compute_ctx_create)
#define pvr_arch_g5p_compute_ctx_destroy PVR_PER_ARCH(g5p_compute_ctx_destroy)
#endif
#endif
