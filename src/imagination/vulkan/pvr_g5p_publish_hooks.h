/* SPDX-License-Identifier: MIT */
/* G5P-only bodies hoisted from shared files. Calls replace existing blank lines; no #line directives. */
#ifndef PVR_G5P_PUBLISH_HOOKS_H
#define PVR_G5P_PUBLISH_HOOKS_H
#include "pvr_g5p_hooks.h"
#if defined(PVR_APPLE_G5P)
#include "vk_format.h"
#include "winsys/apple_g5p/pvr_g5p_winsys.h"
#include "pvr_g5p.h"

#define PVR_G5P_HOOK_pub_pvr_arch_cmd_buffer_1661() \
   if (PVR_HAS_FEATURE(dev_info, apple_g5p)) { \
      /* the background, end of tile and stream of a G5P render: pvr_arch_g5p_job_init() */ \
      result = pvr_arch_g5p_job_init(cmd_buffer, sub_cmd, hw_render, job); \
      if (result != VK_SUCCESS) \
         return result; \
   } else

#define PVR_G5P_HOOK_pub_pvr_arch_cmd_buffer_1740() \
   if (PVR_HAS_FEATURE(dev_info, apple_g5p)) { \
      /* the background object: pvr_arch_g5p_job_init() */ \
   } else

#define PVR_G5P_HOOK_pub_pvr_arch_cmd_buffer_3551() \
   /* G5P: clears are background objects too (pvr_arch_g5p_job_init()) */ \
   if (PVR_HAS_FEATURE(&cmd_buffer->device->pdevice->dev_info, apple_g5p)) \
      return;

#define PVR_G5P_HOOK_pub_pvr_arch_cmd_buffer_4253() \
   if (PVR_HAS_FEATURE(&cmd_buffer->device->pdevice->dev_info, apple_g5p)) { \
      /* G5P: clears are background objects too (pvr_arch_g5p_job_init()) */ \
   } else

#define PVR_G5P_HOOK_pub_pvr_arch_cmd_buffer_9258() \
   if (PVR_HAS_FEATURE(&cmd_buffer->device->pdevice->dev_info, apple_g5p)) { \
      /* G5P: the draw's VDM/PPP words and programs (pvr_arch_g5p_gfx.c) */ \
      const struct pvr_g5p_draw g5p_draw = { \
         .count = vertexCount, \
         .instances = instanceCount, \
         .first = firstVertex, \
         .first_instance = firstInstance, \
      }; \
 \
      result = pvr_arch_g5p_cmd_draw(cmd_buffer, &g5p_draw); \
      if (result != VK_SUCCESS) \
         vk_command_buffer_set_error(&cmd_buffer->vk, result); \
      goto end_cmd_draw; \
   }

#define PVR_G5P_HOOK_pub_pvr_arch_cmd_buffer_9338() \
   if (PVR_HAS_FEATURE(&cmd_buffer->device->pdevice->dev_info, apple_g5p)) { \
      /* G5P: indexed draws are refused by name (pvr_arch_g5p_gfx.c) */ \
      vk_command_buffer_set_error(&cmd_buffer->vk, \
                                  pvr_arch_g5p_cmd_draw_indexed(cmd_buffer)); \
      goto end_cmd_draw_indexed; \
   }

#define PVR_G5P_HOOK_pub_pvr_arch_cmd_buffer_10235() \
   if (PVR_HAS_FEATURE(&cmd_buffer->device->pdevice->dev_info, apple_g5p)) { \
      /* G5P: clears and loads are background objects (pvr_arch_g5p_job_init()) */ \
   } else

#define PVR_G5P_HOOK_pub_pvr_arch_device_914() \
   if (PVR_HAS_FEATURE(&pdevice->dev_info, apple_g5p)) { \
      result = pvr_arch_g5p_device_init(device); \
      if (result != VK_SUCCESS) \
         goto err_pvr_free_compute_empty; \
   }

#define PVR_G5P_HOOK_pub_pvr_arch_device_1019() \
   if (PVR_HAS_FEATURE(&pdevice->dev_info, apple_g5p)) \
      pvr_arch_g5p_device_finish(device);

#define PVR_G5P_HOOK_pub_pvr_arch_device_1102() \
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p)) \
      pvr_arch_g5p_device_finish(device);

#define PVR_G5P_HOOK_pub_pvr_arch_job_render_879() \
   if (job->g5p.enabled) { \
      /* G5P: the pass is in the fragment stream; no geometry words */ \
      state->fw_stream_len = 0; \
   } else

#define PVR_G5P_HOOK_pub_pvr_arch_job_render_1390() \
   if (job->g5p.enabled) { \
      /* G5P: the TYPED_V3 pass the apple_g5p winsys submits */ \
      state->fw_stream_len = pvr_arch_g5p_frag_stream(ctx->device, job, state->fw_stream, \
                                                      sizeof(state->fw_stream)); \
   } else

#define PVR_G5P_HOOK_pub_pvr_arch_job_render_1503() \
   if (job->g5p.enabled) { \
      /* G5P: no partial renders yet */ \
      submit_info->fragment_pr.fw_stream_len = 0; \
      submit_info->fragment_pr.wait = wait_frag; \
   } else

#define PVR_G5P_HOOK_pub_pvr_arch_pipeline_1317() \
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p)) { \
      /* G5P: the programs of pvr_arch_g5p_graphics_pipeline_compile() */ \
      pvr_arch_g5p_graphics_pipeline_destroy(device, gfx_pipeline); \
      pvr_pipeline_finish(device, &gfx_pipeline->base); \
      vk_free2(&device->vk.alloc, allocator, gfx_pipeline); \
      return; \
   }

#define PVR_G5P_HOOK_pub_pvr_arch_pipeline_3136() \
   /* G5P: its own compile, the G5P backend from NIR (pvr_arch_g5p_gfx.c) */ \
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p)) \
      return pvr_arch_g5p_graphics_pipeline_compile(device, pCreateInfo, gfx_pipeline, \
                                                    state);

#define PVR_G5P_HOOK_pub_pvr_device_245() \
   /* G5P: memory that may become a dma-buf (exportable, or a WSI image) is a \
    * kernel BO of its own; the winsys's arena slices cannot be exported. */ \
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p) && \
       (vk_find_struct_const(pAllocateInfo->pNext, EXPORT_MEMORY_ALLOCATE_INFO) || \
        vk_find_struct_const(pAllocateInfo->pNext, WSI_MEMORY_ALLOCATE_INFO_MESA))) \
      type = PVR_WINSYS_BO_TYPE_DISPLAY;

#define PVR_G5P_HOOK_pub_pvr_instance_117() \
   /* Apple A9 (G5P): our kernel's typed UAPI (winsys/apple_g5p) */ \
   is_pvr |= !strcmp(version->name, "apple_g5p");

#define PVR_G5P_HOOK_pub_pvr_physical_device_1275() \
   PVR_G5P_HOOK_msaa_limits(pdevice, &supported_properties) \
   /* G5P: no transform feedback is built yet (G5P refuses every XFB pipeline), \
    * so none is advertised: Zink derives its GL/ES version from what is real */ \
   if (PVR_HAS_FEATURE(&pdevice->dev_info, apple_g5p)) { \
      supported_extensions.EXT_transform_feedback = false; \
      supported_features.transformFeedback = false; \
      supported_features.geometryStreams = false; \
   }

#define PVR_G5P_HOOK_winsys(version, render_fd, display_fd, alloc, ws, result) \
   if (strcmp((version)->name, PVR_G5P_DRIVER_NAME) == 0) \
      result = pvr_g5p_winsys_create(render_fd, display_fd, alloc, ws); \
   else
#endif
#endif
