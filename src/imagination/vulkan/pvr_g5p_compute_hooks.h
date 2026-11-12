/* SPDX-License-Identifier: MIT */
#ifndef PVR_G5P_COMPUTE_HOOKS_H
#define PVR_G5P_COMPUTE_HOOKS_H
#if defined(PVR_APPLE_G5P)
#include "pvr_g5p_compute.h"
#include "pvr_g5p_query.h"
/* All G5P engines use typed kernel contexts and kernel fault recovery. The
 * generic Rogue reset command is empty and its BVNC compute precondition does
 * not describe A9's native CDM path. Its matching cleanup is a no-op too. */
#define PVR_G5P_HOOK_ctx_reset_cmd_init() \
   if (PVR_HAS_FEATURE(dev_info, apple_g5p)) \
      return VK_SUCCESS;
#define PVR_G5P_HOOK_compute_pipeline() \
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p)) { \
      result = pvr_arch_g5p_compute_pipeline_compile(device, pCreateInfo, compute_pipeline); \
      if (result != VK_SUCCESS) pvr_pipeline_finish(device, &compute_pipeline->base); \
      return result; \
   }
#define PVR_G5P_HOOK_compute_pipeline_destroy() \
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p)) { \
      pvr_bo_suballoc_free(compute_pipeline->g9p.shader_bo); \
      pvr_pipeline_finish(device, &compute_pipeline->base); \
      vk_free2(&device->vk.alloc, allocator, compute_pipeline); \
      return; \
   }
#define PVR_G5P_HOOK_compute_dispatch() \
   if (PVR_HAS_FEATURE(&cmd_buffer->device->pdevice->dev_info, apple_g5p)) { \
      result = pvr_arch_g5p_cmd_dispatch(cmd_buffer, sub_cmd, indirect_addr.addr, base_group, workgroup_size); \
      if (result != VK_SUCCESS) pvr_cmd_buffer_set_error_unwarned(cmd_buffer, result); \
      goto end_cmd_dispatch; \
   }
#define PVR_G5P_HOOK_compute_end_pass() \
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p)) { \
      result = pvr_arch_g5p_compute_end_pass(cmd_buffer, compute_sub_cmd); \
      if (result != VK_SUCCESS) return pvr_cmd_buffer_set_error_unwarned(cmd_buffer, result); \
   } else
#define PVR_G5P_HOOK_compute_barrier() \
   if (PVR_HAS_FEATURE(&cmd_buffer->device->pdevice->dev_info, apple_g5p)) { \
      result = pvr_arch_g5p_compute_end_pass(cmd_buffer, &current_sub_cmd->compute); \
      if (result != VK_SUCCESS) { pvr_cmd_buffer_set_error_unwarned(cmd_buffer, result); return; } \
      break; \
   }
#define PVR_G5P_HOOK_compute_job_submit() \
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p)) \
      return pvr_arch_g5p_compute_job_submit(ctx, sub_cmd, wait, signal_sync);
#define PVR_G5P_HOOK_compute_ctx_create() \
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p)) \
      return pvr_arch_g5p_compute_ctx_create(device, ctx_out);
#define PVR_G5P_HOOK_compute_ctx_destroy() \
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p)) { \
      pvr_arch_g5p_compute_ctx_destroy(ctx); return; \
   }
#define PVR_G5P_HOOK_query_programs_create(device) \
   if (PVR_HAS_FEATURE(&(device)->pdevice->dev_info, apple_g5p)) \
      return pvr_arch_g5p_query_programs_create(device);
#define PVR_G5P_HOOK_query_programs_destroy(device) \
   if (PVR_HAS_FEATURE(&(device)->pdevice->dev_info, apple_g5p)) { \
      pvr_arch_g5p_query_programs_destroy(device); \
      return; \
   }
#define PVR_G5P_HOOK_query_program(cmd_buffer, query_info) \
   if (PVR_HAS_FEATURE(&(cmd_buffer)->device->pdevice->dev_info, apple_g5p)) \
      return pvr_arch_g5p_add_query_program(cmd_buffer, query_info);
#endif
#endif
