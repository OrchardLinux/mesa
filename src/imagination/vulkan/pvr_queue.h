/*
 * Copyright © 2022 Imagination Technologies Ltd.
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * based in part on radv driver which is:
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef PVR_QUEUE_H
#define PVR_QUEUE_H

#include "pvr_macros.h"
#include "vk_queue.h"

#include "pvr_common.h"
#ifdef PVR_APPLE_G5P
#include "winsys/apple_g5p/g5p_ws_perf.h"
#endif

struct vk_sync;

struct pvr_device;
struct pvr_render_ctx;
struct pvr_compute_ctx;
struct pvr_transfer_ctx;

struct pvr_queue {
   struct vk_queue vk;

   struct pvr_device *device;

   struct pvr_render_ctx *gfx_ctx;
   struct pvr_compute_ctx *compute_ctx;
   struct pvr_compute_ctx *query_ctx;
   struct pvr_transfer_ctx *transfer_ctx;

   struct vk_sync *last_job_signal_sync[PVR_JOB_TYPE_MAX];
   struct vk_sync *next_job_wait_sync[PVR_JOB_TYPE_MAX];
#ifdef PVR_APPLE_G5P
   struct pvr_g5p_inflight {
      struct vk_sync *done;
      bool known;
      uint8_t writes, reads;
      uint64_t w_base[G5P_PERF_MAX_WRITES], w_size[G5P_PERF_MAX_WRITES];
      uint64_t r_base[G5P_PERF_MAX_READS], r_size[G5P_PERF_MAX_READS];
   } g5p_inflight[16];
   uint32_t g5p_inflight_head;
   struct vk_sync *g5p_inflight_overflow;
   struct vk_sync *g5p_sem_wait[PVR_JOB_TYPE_MAX];
#endif
};

VK_DEFINE_HANDLE_CASTS(pvr_queue, vk.base, VkQueue, VK_OBJECT_TYPE_QUEUE)

#ifdef PVR_PER_ARCH
VkResult PVR_PER_ARCH(queues_create)(struct pvr_device *device,
                                     const VkDeviceCreateInfo *pCreateInfo);
#   define pvr_arch_queues_create PVR_PER_ARCH(queues_create)

void PVR_PER_ARCH(queues_destroy)(struct pvr_device *device);
#   define pvr_arch_queues_destroy PVR_PER_ARCH(queues_destroy)

#endif /* PVR_PER_ARCH */

#endif /* PVR_QUEUE_H */
