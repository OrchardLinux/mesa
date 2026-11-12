/*
 * Copyright © 2022 Imagination Technologies Ltd.
 *
 * based in part on radv driver which is:
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * This file implements VkQueue, VkFence, and VkSemaphore
 */

#include "util/os_time.h"
#include "pvr_g9p.h"
#ifdef PVR_APPLE_G5P
#include "winsys/apple_g5p/pvr_g5p_winsys.h"
#endif
#include "pvr_queue.h"
#include "pvr_rt_dataset.h"

#include <assert.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "pvr_cmd_buffer.h"
#include "pvr_device.h"
#include "pvr_entrypoints.h"
#include "pvr_job_compute.h"
#include "pvr_job_context.h"
#include "pvr_job_render.h"
#include "pvr_job_transfer.h"
#include "pvr_limits.h"
#include "pvr_macros.h"
#include "pvr_physical_device.h"
#include "pvr_query.h"
#include "pvr_buffer.h"
#include "pvr_bo.h"
#include "pvr_pipeline.h"

#include "util/macros.h"
#include "util/u_atomic.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_fence.h"
#include "vk_log.h"
#include "vk_object.h"
#include "vk_queue.h"
#include "vk_semaphore.h"
#include "vk_sync.h"
#include "vk_sync_dummy.h"
#include "vk_util.h"

static bool pvr_g5p_env_on(const char *name)
{
   const char *value = getenv(name);
   return value && strcmp(value, "0");
}

static bool pvr_g5p_serial_submits(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = pvr_g5p_env_on("PVR_G5P_SERIAL_SUBMITS");
   return enabled;
}

static bool pvr_g5p_deplog(void)
{
   static int enabled = -1;
   if (enabled < 0) {
      const char *v = getenv("PVR_G5P_DEPLOG");
      enabled = v ? strcmp(v, "0") != 0 : 0;
   }
   return enabled;
}

#ifdef PVR_APPLE_G5P
static bool pvr_g5p_barrier_skip_done(void)
{
   static int on = -1;
   if (on < 0) {
      const char *v = getenv("PVR_G5P_BARRIER_SKIP_DONE");
      on = !(v && !strcmp(v, "0"));
   }
   return on;
}

static bool pvr_g5p_signal_share(void)
{
   static int on = -1;
   if (on < 0) {
      const char *v = getenv("PVR_G5P_SIGNAL_SHARE");
      on = !(v && !strcmp(v, "0"));
   }
   return on;
}
#endif

static bool pvr_g5p_waitonly_local(void)
{
   static int enabled = -1;
   if (enabled < 0) {
      const char *v = getenv("PVR_G5P_WAITONLY_LOCAL");
      enabled = v ? strcmp(v, "0") != 0 : 1;
   }
   return enabled;
}

static uint32_t g5p_dep_submit;
static int g5p_dep_pending(struct pvr_device *d, struct vk_sync *s)
{
   return vk_sync_wait(&d->vk, s, 0, VK_SYNC_WAIT_COMPLETE, 0) == VK_TIMEOUT;
}

static bool pvr_g5p_waitlog(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = pvr_g5p_env_on("PVR_G5P_WAITLOG");
   return enabled;
}

static inline enum pvr_winsys_ctx_priority
pvr_winsys_from_vk_queue_priority(const VkQueueGlobalPriority priority)
{
   switch (priority) {
   case VK_QUEUE_GLOBAL_PRIORITY_LOW:
      return PVR_WINSYS_CTX_PRIORITY_LOW;
   case VK_QUEUE_GLOBAL_PRIORITY_MEDIUM:
      return PVR_WINSYS_CTX_PRIORITY_MEDIUM;
   default:
      UNREACHABLE("Invalid queue global priority.");
   }
}

static VkResult pvr_driver_queue_submit(struct vk_queue *queue,
                                        struct vk_queue_submit *submit);

static VkResult pvr_queue_init(struct pvr_device *device,
                               struct pvr_queue *queue,
                               const VkDeviceQueueCreateInfo *pCreateInfo,
                               uint32_t index_in_family)
{
   struct pvr_transfer_ctx *transfer_ctx = NULL;
   struct pvr_compute_ctx *compute_ctx;
   struct pvr_compute_ctx *query_ctx;
   struct pvr_render_ctx *gfx_ctx;
   VkResult result;

   const VkDeviceQueueGlobalPriorityCreateInfo *gp =
      vk_find_struct_const(pCreateInfo->pNext,
                           DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO);

   const VkQueueGlobalPriority global_priority =
      gp ? gp->globalPriority : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM;

   if (global_priority == VK_QUEUE_GLOBAL_PRIORITY_REALTIME ||
       global_priority == VK_QUEUE_GLOBAL_PRIORITY_HIGH) {
      return VK_ERROR_NOT_PERMITTED;
   }

   *queue = (struct pvr_queue){ 0 };

   result =
      vk_queue_init(&queue->vk, &device->vk, pCreateInfo, index_in_family);
   if (result != VK_SUCCESS)
      return result;

   if (device->ws->features.supports_threaded_submit) {
      result = vk_queue_enable_submit_thread(&queue->vk);
      if (result != VK_SUCCESS)
         goto err_vk_queue_finish;
   }

   /* Apple's firmware has no transfer engine; the kernel refuses the context. */
   if (!PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g9p)) {
      result = pvr_arch_transfer_ctx_create(
         device,
         pvr_winsys_from_vk_queue_priority(global_priority),
         &transfer_ctx);
      if (result != VK_SUCCESS)
         goto err_vk_queue_finish;
   }

   result = pvr_arch_compute_ctx_create(
      device,
      pvr_winsys_from_vk_queue_priority(global_priority),
      &compute_ctx);
   if (result != VK_SUCCESS)
      goto err_transfer_ctx_destroy;

   result = pvr_arch_compute_ctx_create(
      device,
      pvr_winsys_from_vk_queue_priority(global_priority),
      &query_ctx);
   if (result != VK_SUCCESS)
      goto err_compute_ctx_destroy;

   result = pvr_arch_render_ctx_create(
      device,
      pvr_winsys_from_vk_queue_priority(global_priority),
      &gfx_ctx);
   if (result != VK_SUCCESS)
      goto err_query_ctx_destroy;

   queue->device = device;
   queue->gfx_ctx = gfx_ctx;
   queue->compute_ctx = compute_ctx;
   queue->query_ctx = query_ctx;
   queue->transfer_ctx = transfer_ctx;

   queue->vk.driver_submit = pvr_driver_queue_submit;

   return VK_SUCCESS;

err_query_ctx_destroy:
   pvr_arch_compute_ctx_destroy(query_ctx);

err_compute_ctx_destroy:
   pvr_arch_compute_ctx_destroy(compute_ctx);

err_transfer_ctx_destroy:
   if (transfer_ctx)
      pvr_arch_transfer_ctx_destroy(transfer_ctx);

err_vk_queue_finish:
   vk_queue_finish(&queue->vk);

   return result;
}

VkResult pvr_arch_queues_create(struct pvr_device *device,
                                const VkDeviceCreateInfo *pCreateInfo)
{
   VkResult result;

   /* Check requested queue families and queues */
   assert(pCreateInfo->queueCreateInfoCount == 1);
   assert(pCreateInfo->pQueueCreateInfos[0].queueFamilyIndex == 0);
   assert(pCreateInfo->pQueueCreateInfos[0].queueCount <= PVR_MAX_QUEUES);

   const VkDeviceQueueCreateInfo *queue_create =
      &pCreateInfo->pQueueCreateInfos[0];

   device->queues = vk_alloc(&device->vk.alloc,
                             queue_create->queueCount * sizeof(*device->queues),
                             8,
                             VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device->queues)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   device->queue_count = 0;

   for (uint32_t i = 0; i < queue_create->queueCount; i++) {
      result = pvr_queue_init(device, &device->queues[i], queue_create, i);
      if (result != VK_SUCCESS)
         goto err_queues_finish;

      device->queue_count++;
   }

   return VK_SUCCESS;

err_queues_finish:
   pvr_arch_queues_destroy(device);
   return result;
}

static void pvr_queue_finish(struct pvr_queue *queue)
{
#ifdef PVR_APPLE_G5P
   for (uint32_t i = 0; i < ARRAY_SIZE(queue->g5p_inflight); i++) {
      if (queue->g5p_inflight[i].done)
         vk_sync_destroy(&queue->device->vk, queue->g5p_inflight[i].done);
   }
   if (queue->g5p_inflight_overflow)
      vk_sync_destroy(&queue->device->vk, queue->g5p_inflight_overflow);
   for (uint32_t i = 0; i < PVR_JOB_TYPE_MAX; i++) {
      if (queue->g5p_sem_wait[i])
         vk_sync_destroy(&queue->device->vk, queue->g5p_sem_wait[i]);
   }
#endif
   for (uint32_t i = 0; i < ARRAY_SIZE(queue->next_job_wait_sync); i++) {
      if (queue->next_job_wait_sync[i])
         vk_sync_destroy(&queue->device->vk, queue->next_job_wait_sync[i]);
   }

   for (uint32_t i = 0; i < ARRAY_SIZE(queue->last_job_signal_sync); i++) {
      if (queue->last_job_signal_sync[i])
         vk_sync_destroy(&queue->device->vk, queue->last_job_signal_sync[i]);
   }

   pvr_arch_render_ctx_destroy(queue->gfx_ctx);
   pvr_arch_compute_ctx_destroy(queue->query_ctx);
   pvr_arch_compute_ctx_destroy(queue->compute_ctx);
   if (queue->transfer_ctx)
      pvr_arch_transfer_ctx_destroy(queue->transfer_ctx);

   vk_queue_finish(&queue->vk);
}

void pvr_arch_queues_destroy(struct pvr_device *device)
{
   for (uint32_t q_idx = 0; q_idx < device->queue_count; q_idx++)
      pvr_queue_finish(&device->queues[q_idx]);

   vk_free(&device->vk.alloc, device->queues);
}

static void pvr_update_job_syncs(struct pvr_device *device,
                                 struct pvr_queue *queue,
                                 struct vk_sync *new_signal_sync,
                                 enum pvr_job_type submitted_job_type)
{
   if (queue->next_job_wait_sync[submitted_job_type]) {
      vk_sync_destroy(&device->vk,
                      queue->next_job_wait_sync[submitted_job_type]);
      queue->next_job_wait_sync[submitted_job_type] = NULL;
   }

   if (queue->last_job_signal_sync[submitted_job_type]) {
      vk_sync_destroy(&device->vk,
                      queue->last_job_signal_sync[submitted_job_type]);
   }

#ifdef PVR_APPLE_G5P
   queue->g5p_src_done[submitted_job_type] = false;
   if (queue->g5p_sem_wait[submitted_job_type]) {
      vk_sync_destroy(&device->vk, queue->g5p_sem_wait[submitted_job_type]);
      queue->g5p_sem_wait[submitted_job_type] = NULL;
   }
#endif
   queue->last_job_signal_sync[submitted_job_type] = new_signal_sync;
}

#if defined(PVR_APPLE_G5P)
/* A sync that signals when both a and b (either may be NULL) have: a null
 * job's syncobj transfer, no CPU wait (g5p_ws_null_job_submit()) */
static VkResult
g5p_merge_syncs(struct pvr_device *device, struct vk_sync *a, struct vk_sync *b, struct vk_sync **out)
{
   struct vk_sync_wait waits[2];
   uint32_t n = 0;
   VkResult result;

   if (a)
      waits[n++] = (struct vk_sync_wait){ .sync = a, .stage_mask = ~(VkPipelineStageFlags2)0 };
   if (b)
      waits[n++] = (struct vk_sync_wait){ .sync = b, .stage_mask = ~(VkPipelineStageFlags2)0 };
   result = vk_sync_create(&device->vk, &device->pdevice->ws->syncobj_type, 0U, 0UL, out);
   if (result != VK_SUCCESS)
      return result;
   const struct vk_sync_signal signal = { .sync = *out, .stage_mask = ~(VkPipelineStageFlags2)0 };
   result = device->ws->ops->null_job_submit(device->ws, waits, n, (struct vk_sync_signal *)&signal);
   if (result != VK_SUCCESS) {
      vk_sync_destroy(&device->vk, *out);
      *out = NULL;
   }
   return result;
}

/* Retain completions independently of last_job_signal_sync ownership. */
static VkResult
g5p_track_job(struct pvr_queue *queue, struct vk_sync *done,
              const struct pvr_g5p_job *job)
{
   struct pvr_device *device = queue->device;
   struct pvr_g5p_inflight next = {0};
   VkResult result = g5p_merge_syncs(device, done, NULL, &next.done);
   if (result != VK_SUCCESS)
      return result;
   if (job) {
      next.known = !job->reads.unknown &&
                   (job->perf.flags & G5P_PERF_WRITES_KNOWN) &&
                   job->reads.count <= G5P_PERF_MAX_READS &&
                   job->perf.writes <= G5P_PERF_MAX_WRITES;
      if (next.known) {
         next.reads = job->reads.count;
         memcpy(next.r_base, job->reads.va, next.reads * sizeof(uint64_t));
         memcpy(next.r_size, job->reads.size, next.reads * sizeof(uint64_t));
         next.writes = job->perf.writes;
         for (uint32_t i = 0; i < next.writes; i++)
            next.known &= pvr_g5p_winsys_bo_range(device->ws, job->perf.write_va[i],
                                                 &next.w_base[i], &next.w_size[i]);
      }
   }
   struct pvr_g5p_inflight *slot =
      &queue->g5p_inflight[queue->g5p_inflight_head % ARRAY_SIZE(queue->g5p_inflight)];
   if (slot->done && vk_sync_wait(&device->vk, slot->done, 0,
                                  VK_SYNC_WAIT_COMPLETE, 0) != VK_SUCCESS) {
      struct vk_sync *overflow;
      result = g5p_merge_syncs(device, queue->g5p_inflight_overflow, slot->done, &overflow);
      if (result != VK_SUCCESS) {
         vk_sync_destroy(&device->vk, next.done);
         return result;
      }
      if (queue->g5p_inflight_overflow)
         vk_sync_destroy(&device->vk, queue->g5p_inflight_overflow);
      queue->g5p_inflight_overflow = overflow;
   }
   if (slot->done)
      vk_sync_destroy(&device->vk, slot->done);
   *slot = next;
   queue->g5p_inflight_head++;
   return VK_SUCCESS;
}

static bool
g5p_ranges_overlap(uint64_t a, uint64_t as, uint64_t b, uint64_t bs)
{
   return a <= b ? b - a < as : a - b < bs;
}

static VkResult
g5p_transfer_submit(struct pvr_device *device, struct pvr_queue *queue,
                     struct pvr_sub_cmd_transfer *sub_cmd, struct vk_sync **signal)
{
   bool known = true;
   bool hazards[16] = {0};
   struct vk_sync_wait waits[18]; /* ring, overflow, semaphore waits */
   uint32_t count = 0, candidates = 0;
   VkResult result;
   list_for_each_entry (struct pvr_transfer_cmd, cmd, sub_cmd->transfer_cmds, link) {
      /* Resolve each touched surface to a whole BO; no fixed-size transfer set. */
      for (uint32_t src = 0; src <= cmd->source_count; src++) {
         uint64_t base, size;
         const uint64_t addr = src ? cmd->sources[src - 1].surface.dev_addr.addr : cmd->dst.dev_addr.addr;
         if (!pvr_g5p_winsys_bo_range(device->ws, addr, &base, &size)) {
            known = false;
            continue;
         }
         for (uint32_t i = 0; i < ARRAY_SIZE(queue->g5p_inflight); i++) {
            const struct pvr_g5p_inflight *slot = &queue->g5p_inflight[i];
            for (uint32_t w = 0; w < slot->writes; w++)
               hazards[i] |= g5p_ranges_overlap(base, size, slot->w_base[w], slot->w_size[w]);
            if (!src)
               for (uint32_t r = 0; r < slot->reads; r++)
                  hazards[i] |= g5p_ranges_overlap(base, size, slot->r_base[r], slot->r_size[r]);
         }
      }
   }
   if (!known) {
      if (queue->next_job_wait_sync[PVR_JOB_TYPE_TRANSFER])
         waits[count++] = (struct vk_sync_wait){
            .sync = queue->next_job_wait_sync[PVR_JOB_TYPE_TRANSFER],
            .stage_mask = ~(VkPipelineStageFlags2)0 };
   } else {
      for (uint32_t i = 0; i < ARRAY_SIZE(queue->g5p_inflight); i++) {
         const struct pvr_g5p_inflight *slot = &queue->g5p_inflight[i];
         if (!slot->done)
            continue;
         candidates++;
         if (!slot->known || hazards[i])
            waits[count++] = (struct vk_sync_wait){
               .sync = slot->done, .stage_mask = ~(VkPipelineStageFlags2)0 };
      }
   }
   sub_cmd->g5p_waits_skipped = known ? candidates - count : 0;
   /* Overflow covers jobs evicted even before a barrier was recorded. */
   if (known && queue->g5p_inflight_overflow)
      waits[count++] = (struct vk_sync_wait){
         .sync = queue->g5p_inflight_overflow, .stage_mask = ~(VkPipelineStageFlags2)0 };
   if (known && queue->g5p_sem_wait[PVR_JOB_TYPE_TRANSFER])
      waits[count++] = (struct vk_sync_wait){
         .sync = queue->g5p_sem_wait[PVR_JOB_TYPE_TRANSFER], .stage_mask = ~(VkPipelineStageFlags2)0 };
   result = pvr_arch_g5p_transfer_submit(device, sub_cmd, waits, count, *signal);
   if (result != VK_SUCCESS)
      return result;
   if (known && queue->g5p_inflight_overflow) {
      vk_sync_destroy(&device->vk, queue->g5p_inflight_overflow);
      queue->g5p_inflight_overflow = NULL;
   }
   /* Preserve execution-dependency chains without delaying the CPU copy. */
   if (known && queue->next_job_wait_sync[PVR_JOB_TYPE_TRANSFER]) {
      struct vk_sync *merged;
      result = g5p_merge_syncs(device, *signal,
                              queue->next_job_wait_sync[PVR_JOB_TYPE_TRANSFER], &merged);
      if (result != VK_SUCCESS)
         return result;
      vk_sync_destroy(&device->vk, *signal);
      *signal = merged;
   }
   return VK_SUCCESS;
}

#endif

static VkResult
pvr_process_graphics_cmd_for_view(struct pvr_device *device,
                                  struct pvr_queue *queue,
                                  struct pvr_cmd_buffer *cmd_buffer,
                                  struct pvr_sub_cmd_gfx *sub_cmd,
                                  uint32_t view_index)
{
   pvr_dev_addr_t original_ctrl_stream_addr = { 0 };
   struct pvr_render_job *job = &sub_cmd->job;
   struct vk_sync *geom_signal_sync;
   struct vk_sync *frag_signal_sync = NULL;
   VkResult result;

   job->ds.addr = PVR_DEV_ADDR_OFFSET(job->ds.addr, job->ds.layer_size *
      (view_index + job->ds.base_array_layer));
   job->view_state.view_index = view_index;

   result = vk_sync_create(&device->vk,
                           &device->pdevice->ws->syncobj_type,
                           0U,
                           0UL,
                           &geom_signal_sync);
   if (result != VK_SUCCESS)
      return result;

   if (job->run_frag) {
      result = vk_sync_create(&device->vk,
                              &device->pdevice->ws->syncobj_type,
                              0U,
                              0UL,
                              &frag_signal_sync);
      if (result != VK_SUCCESS)
         goto err_destroy_geom_sync;
   }

   /* FIXME: DoShadowLoadOrStore() */

   /* Perform two render submits when using multiple framebuffer layers. The
    * first submit contains just geometry, while the second only terminates
    * (and triggers the fragment render if originally specified). This is needed
    * because the render target cache gets cleared on terminating submits, which
    * could result in missing primitives.
    */
   if (pvr_sub_cmd_gfx_requires_split_submit(sub_cmd)) {
      /* If fragment work shouldn't be run there's no need for a split,
       * and if geometry_terminate is false this kick can't have a fragment
       * stage without another terminating geometry kick.
       */
      assert(job->geometry_terminate && job->run_frag);

      /* First submit must not touch fragment work. */
      job->geometry_terminate = false;
      job->run_frag = false;

      result = pvr_arch_render_job_submit(
         queue->gfx_ctx,
         &sub_cmd->job,
         queue->next_job_wait_sync[PVR_JOB_TYPE_GEOM],
         NULL,
         NULL,
         NULL);

      job->geometry_terminate = true;
      job->run_frag = true;

      if (result != VK_SUCCESS)
         goto err_destroy_frag_sync;

      original_ctrl_stream_addr = job->ctrl_stream_addr;

      /* Second submit contains only a trivial control stream to terminate the
       * geometry work.
       */
      assert(sub_cmd->terminate_ctrl_stream);
      job->ctrl_stream_addr = sub_cmd->terminate_ctrl_stream->vma->dev_addr;
   } else if (job->g9p.enabled) {
      if (getenv("G9P_DEBUG"))
         fprintf(stderr, "g9p: render view %u of mask %#x, end of tile data 0x%x, stream 0x%" PRIx64 "\n",
                 view_index, sub_cmd->view_mask,
                 job->view_state.view[view_index].pds_pixel_event_data_offset,
                 sub_cmd->g9p_view_stream[view_index]);
      /* the view's copy of the stream when its draws read the view index */
      if (sub_cmd->g9p_view_stream[view_index]) {
         original_ctrl_stream_addr = job->ctrl_stream_addr;
         job->ctrl_stream_addr.addr = sub_cmd->g9p_view_stream[view_index];
      }
   } else if (sub_cmd->view_index_wanted) {
      original_ctrl_stream_addr = job->ctrl_stream_addr;
      job->ctrl_stream_addr.addr =
         sub_cmd->multiview_ctrl_stream->vma->dev_addr.addr +
         (view_index * PVR_DW_TO_BYTES(sub_cmd->multiview_ctrl_stream_stride));
   }

   /* G9P_SYNC: is what the render waits for done? */
   if (job->g9p.enabled && getenv("G9P_SYNC")) {
      for (unsigned t = 0; t < 2; t++) {
         struct vk_sync *w = queue->next_job_wait_sync[t ? PVR_JOB_TYPE_FRAG : PVR_JOB_TYPE_GEOM];

         if (w && vk_sync_wait(&device->vk, w, 0, VK_SYNC_WAIT_COMPLETE,
                               os_time_get_absolute_timeout(2000000000ull)) != VK_SUCCESS)
            fprintf(stderr, "g9p: render %ux%u: its %s wait is not signalled\n",
                    job->g9p.width, job->g9p.height, t ? "fragment" : "geometry");
      }
   }

   struct vk_sync *frag_wait = queue->next_job_wait_sync[PVR_JOB_TYPE_FRAG];
#if defined(PVR_APPLE_G5P)
   /* a vk_meta blit's render (pvr_arch_g5p_meta.c) is transfer work: its
    * fragments also wait for what barriers made transfers wait for */
   struct vk_sync *g5p_frag_wait = NULL;
   if (sub_cmd->g5p_transfer && queue->next_job_wait_sync[PVR_JOB_TYPE_TRANSFER]) {
      result = g5p_merge_syncs(device, frag_wait, queue->next_job_wait_sync[PVR_JOB_TYPE_TRANSFER],
                               &g5p_frag_wait);
      if (result != VK_SUCCESS)
         goto err_destroy_frag_sync;
      frag_wait = g5p_frag_wait;
   }
#endif
   result =
      pvr_arch_render_job_submit(queue->gfx_ctx,
                                 &sub_cmd->job,
                                 queue->next_job_wait_sync[PVR_JOB_TYPE_GEOM],
                                 frag_wait,
                                 geom_signal_sync,
                                 frag_signal_sync);
#if defined(PVR_APPLE_G5P)
   if (g5p_frag_wait)
      vk_sync_destroy(&device->vk, g5p_frag_wait);
#endif

   if (original_ctrl_stream_addr.addr > 0)
      job->ctrl_stream_addr = original_ctrl_stream_addr;

   if (result != VK_SUCCESS)
      goto err_destroy_frag_sync;

   /*
    * G9P_PMDUMP=N: every Nth render, the start of its render target's
    * parameter-manager memory (VHEAP table and RTC, the macrotile array of
    * the data set in use) as the previous scene left it: what grows from
    * scene to scene shows where parameter memory is lost
    */
   if (job->g9p.enabled && getenv("G9P_PMDUMP")) {
      static unsigned n;
      struct pvr_rt_dataset *rt = job->view_state.rt_datasets[job->view_state.view_index];
      const unsigned every = MAX2(atoi(getenv("G9P_PMDUMP")), 1);

      if (!(++n % every)) {
         struct pvr_bo *bos[2] = { rt->vheap_rtc_bo, rt->mta_bo };
         static const char *const names[2] = { "vheap", "mta" };

         for (unsigned b = 0; b < 2; b++) {
            const uint32_t *w;

            if (!bos[b])
               continue;
            if (!bos[b]->bo->map) {
               VkResult r = pvr_bo_cpu_map(device, bos[b]);

               if (r != VK_SUCCESS) {
                  fprintf(stderr, "g9p: pm %u %s: no CPU mapping (%d)\n", n, names[b], r);
                  continue;
               }
            }
            w = bos[b]->bo->map;
            fprintf(stderr, "g9p: pm %u rt %p data %u %s:", n, (void *)rt, rt->rt_data_idx,
                    names[b]);
            for (unsigned i = 0; i < 16; i++)
               fprintf(stderr, " %08x", w[i]);
            fprintf(stderr, "\n");
         }
      }
   }

   /* G9P_TRACE: each render as submitted, to find the one a hang stops at */
   if (job->g9p.enabled && getenv("G9P_TRACE")) {
      static unsigned n;

      const VkRect2D *ra = &job->g9p_draws.render_area;
      const uint32_t *t = job->g9p_draws.tiles;

      fprintf(stderr, "g9p: trace %u: %ux%u x%u %s, %u draws, at %.1f ms, area %d,%d %ux%u, "
              "tiles %u,%u-%u,%u, load op %d, pipelines", ++n,
              job->g9p.width, job->g9p.height, job->g9p.samples, job->g9p.load ? "loaded" : "cleared",
              job->g9p_draws.num_draws, os_time_get_nano() / 1e6, ra->offset.x, ra->offset.y,
              ra->extent.width, ra->extent.height, t[0], t[1], t[2], t[3], job->g9p.load_op);
      for (unsigned i = 0; i < MIN2(job->g9p_draws.num_draws, 12); i++)
         fprintf(stderr, " %u", job->g9p_draws.pipelines[i]);
      fprintf(stderr, "\n");
   }

   /*
    * G9P_SYNC: wait for each render and report the draws of one that hangs
    * (0.5 s: renders take milliseconds, and the kernel's own timeout signals
    * the fence of a hung job before a longer wait would notice)
    */
   if (job->g9p.enabled && getenv("G9P_SYNC")) {
      fprintf(stderr, "g9p: device %p: render %ux%u submitted, %u draws\n", (void *)device,
              job->g9p.width, job->g9p.height, job->g9p_draws.num_draws);
      const int64_t t0 = os_time_get_nano();
      const bool geom = vk_sync_wait(&device->vk, geom_signal_sync, 0, VK_SYNC_WAIT_COMPLETE,
                                     os_time_get_absolute_timeout(500000000ull)) == VK_SUCCESS;
      const int64_t t1 = os_time_get_nano();

      if (!geom || (frag_signal_sync &&
                    vk_sync_wait(&device->vk, frag_signal_sync, 0, VK_SYNC_WAIT_COMPLETE,
                                 os_time_get_absolute_timeout(500000000ull)) != VK_SUCCESS)) {
         fprintf(stderr, "g9p: device %p: render %ux%u: %s timed out; %u draws, pipelines:",
                 (void *)device, job->g9p.width, job->g9p.height,
                 geom ? "fragment job" : "geometry job", job->g9p_draws.num_draws);
         for (unsigned i = 0; i < MIN2(job->g9p_draws.num_draws, 16); i++)
            fprintf(stderr, " %u", job->g9p_draws.pipelines[i]);
         fprintf(stderr, "\n");
      } else if (getenv("G9P_TIME")) {
         /* G9P_TIME=1: each render's geometry and fragment time (waited for in turn) */
         const int64_t t2 = os_time_get_nano();

         fprintf(stderr, "g9p: time %s %u draws: geometry %" PRId64 " us, fragment %" PRId64
                         " us, pipelines",
                 job->g9p.load ? "loaded" : "cleared", job->g9p_draws.num_draws,
                 (t1 - t0) / 1000, (t2 - t1) / 1000);
         for (unsigned i = 0; i < MIN2(job->g9p_draws.num_draws, 12); i++)
            fprintf(stderr, " %u", job->g9p_draws.pipelines[i]);
         fprintf(stderr, "\n");
      } else if (getenv("G9P_PROBE")) {
         /* G9P_PROBE=1: eight pixels down the target's diagonal after each render */
         const uint8_t *px = PVR_PER_ARCH(g9p_cpu_addr)(device, job->g9p.target_addr);
         const unsigned pw = util_next_power_of_two(job->g9p.width);
         const unsigned ph = util_next_power_of_two(job->g9p.height);

         fprintf(stderr, "g9p: probe %#" PRIx64 " %s %u draws (pipelines %u %u):",
                 job->g9p.target_addr, job->g9p.load ? "loaded" : "cleared",
                 job->g9p_draws.num_draws, job->g9p_draws.pipelines[0],
                 job->g9p_draws.pipelines[1]);
         for (unsigned i = 0; px && i < 8; i++) {
            const unsigned x = job->g9p.width * (2 * i + 1) / 16;
            const unsigned y = job->g9p.height * (2 * i + 1) / 16;
            unsigned m = 0, bit = 0;

            if (job->g9p.target_linear) {
               m = y * (job->g9p.target_pitch / 4) + x;
            } else {
               for (unsigned b = 0; (1u << b) < pw || (1u << b) < ph; b++) {
                  if ((1u << b) < ph)
                     m |= ((y >> b) & 1) << bit++;
                  if ((1u << b) < pw)
                     m |= ((x >> b) & 1) << bit++;
               }
            }
            fprintf(stderr, " %08x", ((const uint32_t *)px)[m]);
         }
         fprintf(stderr, "\n");
      } else if (getenv("G9P_GRAB")) {
         /* G9P_GRAB=N: the first N renders' targets to /tmp/g9p-grab-K.raw */
         static unsigned grabbed;
         const uint8_t *px = PVR_PER_ARCH(g9p_cpu_addr)(device, job->g9p.target_addr);
         const size_t size =
            job->g9p.target_linear
               ? (size_t)job->g9p.target_pitch * job->g9p.height
               : (size_t)4 * util_next_power_of_two(job->g9p.width) *
                    util_next_power_of_two(job->g9p.height);

         /*
          * G9P_GRAB_LOADED=1: only renders whose background loads the
          * target; G9P_GRAB_SKIP=K: not the first K (of the twiddled ones)
          */
         static unsigned seen;

         if (px && !job->g9p.target_linear && seen++ < (unsigned)atoi(getenv("G9P_GRAB_SKIP") ?: "0"))
            px = NULL;
         if (px && grabbed < (unsigned)atoi(getenv("G9P_GRAB")) &&
             (job->g9p.load || !getenv("G9P_GRAB_LOADED"))) {
            char path[64];
            FILE *f;

            snprintf(path, sizeof(path), "/tmp/g9p-grab-%u.raw", grabbed++);
            f = fopen(path, "wb");
            if (f) {
               fwrite(px, 1, size, f);
               fclose(f);
               fprintf(stderr, "g9p: %s: %ux%u %s pitch %u at %#" PRIx64 ", %s, %u draws\n",
                       path, job->g9p.width, job->g9p.height,
                       job->g9p.target_linear ? "linear" : "twiddled", job->g9p.target_pitch,
                       job->g9p.target_addr, job->g9p.load ? "loaded" : "cleared",
                       job->g9p_draws.num_draws);
            }
         }
      }
   }

#ifdef PVR_APPLE_G5P
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p)) {
      result = g5p_track_job(queue, job->run_frag ? frag_signal_sync : geom_signal_sync,
                             job->run_frag ? &job->g5p : NULL);
      if (result != VK_SUCCESS)
         goto err_destroy_frag_sync;
   }
#endif
   pvr_update_job_syncs(device, queue, geom_signal_sync, PVR_JOB_TYPE_GEOM);

   if (job->run_frag)
      pvr_update_job_syncs(device, queue, frag_signal_sync, PVR_JOB_TYPE_FRAG);

#if defined(PVR_APPLE_G5P)
   /* ... and later transfer-stage barriers wait for its fragments */
   if (sub_cmd->g5p_transfer && job->run_frag) {
      struct vk_sync *transfer_done;
      result = g5p_merge_syncs(device, queue->last_job_signal_sync[PVR_JOB_TYPE_TRANSFER], frag_signal_sync,
                               &transfer_done);
      if (result != VK_SUCCESS)
         return result;
      pvr_update_job_syncs(device, queue, transfer_done, PVR_JOB_TYPE_TRANSFER);
   }
#endif

   /* FIXME: DoShadowLoadOrStore() */

   return VK_SUCCESS;

err_destroy_frag_sync:
   if (frag_signal_sync)
      vk_sync_destroy(&device->vk, frag_signal_sync);
err_destroy_geom_sync:
   vk_sync_destroy(&device->vk, geom_signal_sync);

   return result;
}

static VkResult pvr_process_graphics_cmd(struct pvr_device *device,
                                         struct pvr_queue *queue,
                                         struct pvr_cmd_buffer *cmd_buffer,
                                         struct pvr_sub_cmd_gfx *sub_cmd)
{
   const pvr_dev_addr_t ds_addr = sub_cmd->job.ds.addr;
   PVR_G5P_HOOK(xfb_before_job, device, cmd_buffer, sub_cmd)

   if (util_dynarray_num_elements(&sub_cmd->g9p_realigns, char)) {
      struct vk_sync *wait = queue->next_job_wait_sync[PVR_JOB_TYPE_GEOM];

      if (wait && PVR_PER_ARCH(g9p_realign_needs_wait)(sub_cmd)) {
         const int64_t tw = os_time_get_nano();
         VkResult result = vk_sync_wait(&device->vk, wait, 0, VK_SYNC_WAIT_COMPLETE, UINT64_MAX);

         if (getenv("G9P_WAITLOG"))
            fprintf(stderr, "g9p: wait realign %" PRId64 " us\n", (os_time_get_nano() - tw) / 1000);
         if (result != VK_SUCCESS)
            return result;
      }
      PVR_PER_ARCH(g9p_realign_vertex_bindings)(sub_cmd);
   }

   if (util_dynarray_num_elements(&sub_cmd->g9p_fills, char)) {
      const int64_t tw = os_time_get_nano();

      for (unsigned i = 0; i < ARRAY_SIZE(queue->last_job_signal_sync); i++) {
         struct vk_sync *s = queue->last_job_signal_sync[i];

         if (s) {
            VkResult result = vk_sync_wait(&device->vk, s, 0, VK_SYNC_WAIT_COMPLETE, UINT64_MAX);

            if (result != VK_SUCCESS)
               return result;
         }
      }
      if (getenv("G9P_WAITLOG"))
         fprintf(stderr, "g9p: wait fills %" PRId64 " us\n", (os_time_get_nano() - tw) / 1000);
      PVR_PER_ARCH(g9p_run_fills)(device, sub_cmd);
   }

   sub_cmd->job.g9p.oclqry_base = sub_cmd->query_pool ? sub_cmd->query_pool->result_buffer->dev_addr.addr : 0;

   {
      VkResult result = PVR_PER_ARCH(g9p_xfb_before_job)(device, cmd_buffer, sub_cmd);

      if (result != VK_SUCCESS)
         return result;
   }

   u_foreach_bit (view_idx, sub_cmd->view_mask) {
      VkResult result;

      result = pvr_process_graphics_cmd_for_view(device,
                                                 queue,
                                                 cmd_buffer,
                                                 sub_cmd,
                                                 view_idx);
      sub_cmd->job.ds.addr = ds_addr;

      if (result != VK_SUCCESS)
         return result;
   }
   PVR_G5P_HOOK(xfb_after_job, device, queue, cmd_buffer, sub_cmd)

   return PVR_PER_ARCH(g9p_xfb_after_job)(device, queue, cmd_buffer, sub_cmd);
}

static VkResult pvr_process_compute_cmd(struct pvr_device *device,
                                        struct pvr_queue *queue,
                                        struct pvr_sub_cmd_compute *sub_cmd)
{
   struct vk_sync *sync;
   VkResult result;

   result = vk_sync_create(&device->vk,
                           &device->pdevice->ws->syncobj_type,
                           0U,
                           0UL,
                           &sync);
   if (result != VK_SUCCESS)
      return result;

   result = pvr_arch_compute_job_submit(
      queue->compute_ctx,
      sub_cmd,
      queue->next_job_wait_sync[PVR_JOB_TYPE_COMPUTE],
      sync);
   if (result != VK_SUCCESS) {
      vk_sync_destroy(&device->vk, sync);
      return result;
   }

   /* G9P_SYNC: wait for each compute job too */
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g9p) && getenv("G9P_SYNC") &&
       vk_sync_wait(&device->vk, sync, 0, VK_SYNC_WAIT_COMPLETE,
                    os_time_get_absolute_timeout(3000000000ull)) != VK_SUCCESS)
      fprintf(stderr, "g9p: compute job timed out\n");
   else if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g9p) && getenv("G9P_SYNC"))
      fprintf(stderr, "g9p: compute job done\n");

#ifdef PVR_APPLE_G5P
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p)) {
      result = g5p_track_job(queue, sync, NULL);
      if (result != VK_SUCCESS) {
         vk_sync_destroy(&device->vk, sync);
         return result;
      }
   }
#endif
   pvr_update_job_syncs(device, queue, sync, PVR_JOB_TYPE_COMPUTE);

   return result;
}

static VkResult pvr_process_transfer_cmds(struct pvr_device *device,
                                          struct pvr_queue *queue,
                                          struct pvr_sub_cmd_transfer *sub_cmd)
{
   struct vk_sync *sync;
   VkResult result;

   result = vk_sync_create(&device->vk,
                           &device->pdevice->ws->syncobj_type,
                           0U,
                           0UL,
                           &sync);
   if (result != VK_SUCCESS)
      return result;

#ifdef PVR_APPLE_G5P
   /* G5P: the A9's firmware has no transfer engine either (the apple_g5p
    * winsys refuses transfer jobs); the CPU does them, as on G9P, plus what
    * only the A9's copies know (pvr_g5p_layout.h)
    */
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p))
      result = g5p_transfer_submit(device, queue, sub_cmd, &sync);
   else
#endif
   /* G9P: no GPU transfer path yet; done by the CPU */
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g9p))
      result = pvr_arch_g9p_transfer_submit(
         device,
         sub_cmd,
         queue->next_job_wait_sync[PVR_JOB_TYPE_TRANSFER],
         sync);
   else
      result = pvr_arch_transfer_job_submit(
         queue->transfer_ctx,
         sub_cmd,
         queue->next_job_wait_sync[PVR_JOB_TYPE_TRANSFER],
         sync);
   if (result != VK_SUCCESS) {
      vk_sync_destroy(&device->vk, sync);
      return result;
   }

   pvr_update_job_syncs(device, queue, sync, PVR_JOB_TYPE_TRANSFER);

   return result;
}

/*
 * G9P: a query's reset, availability write or copy on the CPU, once every job
 * submitted before it has retired: the fragment job writes the counters.
 */
static VkResult
g9p_process_query(struct pvr_device *device, struct pvr_queue *queue, const struct pvr_query_info *info)
{
   for (unsigned t = 0; t < ARRAY_SIZE(queue->last_job_signal_sync); t++) {
      struct vk_sync *s[2] = { queue->last_job_signal_sync[t], queue->next_job_wait_sync[t] };

      for (unsigned i = 0; i < 2; i++) {
         if (!s[i])
            continue;
         VkResult result = vk_sync_wait(&device->vk, s[i], 0, VK_SYNC_WAIT_COMPLETE, UINT64_MAX);

         if (result != VK_SUCCESS)
            return result;
      }
   }

   switch (info->type) {
   case PVR_QUERY_TYPE_AVAILABILITY_WRITE: {
      const uint32_t *index = pvr_bo_suballoc_get_map_addr(info->availability_write.index_bo);
      uint32_t *available = pvr_bo_suballoc_get_map_addr(info->availability_write.availability_bo);

      for (uint32_t i = 0; i < info->availability_write.num_query_indices; i++)
         available[index[i]] = 1;
      break;
   }
   case PVR_QUERY_TYPE_RESET_QUERY_POOL: {
      VK_FROM_HANDLE(pvr_query_pool, pool, info->reset_query_pool.query_pool);
      uint32_t *available = pvr_bo_suballoc_get_map_addr(pool->availability_buffer);
      uint32_t *results = pvr_bo_suballoc_get_map_addr(pool->result_buffer);

      memset(available + info->reset_query_pool.first_query, 0,
             info->reset_query_pool.query_count * sizeof(uint32_t));
      memset(results + info->reset_query_pool.first_query, 0,
             info->reset_query_pool.query_count * sizeof(uint32_t));
      break;
   }
   case PVR_QUERY_TYPE_COPY_QUERY_RESULTS: {
      VK_FROM_HANDLE(pvr_query_pool, pool, info->copy_query_results.query_pool);
      VK_FROM_HANDLE(pvr_buffer, buffer, info->copy_query_results.dst_buffer);
      const VkQueryResultFlags flags = info->copy_query_results.flags;
      const uint32_t *available = pvr_bo_suballoc_get_map_addr(pool->availability_buffer);
      const uint32_t *results = pvr_bo_suballoc_get_map_addr(pool->result_buffer);
      uint8_t *dst = PVR_PER_ARCH(g9p_cpu_addr)(device, buffer->dev_addr.addr + info->copy_query_results.dst_offset);

      if (!dst)
         return vk_error(device, VK_ERROR_DEVICE_LOST);
      for (uint32_t i = 0; i < info->copy_query_results.query_count; i++, dst += info->copy_query_results.stride) {
         const uint32_t q = info->copy_query_results.first_query + i;
         const uint64_t v[2] = { results[q], available[q] };
         const bool has_value = v[1] || (flags & VK_QUERY_RESULT_PARTIAL_BIT);

         for (unsigned w = 0; w < 2; w++) {
            if (w == 0 ? !has_value : !(flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT))
               continue;
            if (flags & VK_QUERY_RESULT_64_BIT)
               memcpy(dst + 8 * w, &v[w], 8);
            else {
               const uint32_t low = v[w];
               memcpy(dst + 4 * w, &low, 4);
            }
         }
      }
      break;
   }
   default:
      UNREACHABLE("Invalid query type");
   }
   return VK_SUCCESS;
}

static VkResult pvr_process_query_cmd(struct pvr_device *device,
                                      struct pvr_queue *queue,
                                      struct pvr_sub_cmd_compute *sub_cmd)
{
   struct vk_sync *sync;
   VkResult result;

   if (sub_cmd->g9p_query)
      return g9p_process_query(device, queue, sub_cmd->g9p_query);

   /* TODO: Currently we add barrier event sub commands to handle the sync
    * necessary for the different query types. Would we get any speed
    * up in processing the queue by doing that sync here without using event sub
    * commands?
    */

   result = vk_sync_create(&device->vk,
                           &device->pdevice->ws->syncobj_type,
                           0U,
                           0UL,
                           &sync);
   if (result != VK_SUCCESS)
      return result;

   result =
      pvr_arch_compute_job_submit(queue->query_ctx,
                                  sub_cmd,
                                  queue->next_job_wait_sync[PVR_JOB_TYPE_QUERY],
                                  sync);
   if (result != VK_SUCCESS) {
      vk_sync_destroy(&device->vk, sync);
      return result;
   }

#ifdef PVR_APPLE_G5P
   if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p)) {
      result = g5p_track_job(queue, sync, NULL);
      if (result != VK_SUCCESS) {
         vk_sync_destroy(&device->vk, sync);
         return result;
      }
   }
#endif
   pvr_update_job_syncs(device, queue, sync, PVR_JOB_TYPE_QUERY);

   return result;
}

static VkResult
pvr_process_event_cmd_barrier(struct pvr_device *device,
                              struct pvr_queue *queue,
                              struct pvr_sub_cmd_event_barrier *sub_cmd)
{
   const uint32_t src_mask = sub_cmd->wait_for_stage_mask;
   const uint32_t dst_mask = sub_cmd->wait_at_stage_mask;
   struct vk_sync_wait wait_syncs[PVR_JOB_TYPE_MAX + 1];
   uint32_t src_wait_count = 0;
   VkResult result;

   assert(!(src_mask &
            ~(PVR_PIPELINE_STAGE_ALL_BITS | PVR_PIPELINE_STAGE_QUERY_BIT)));
   assert(!(dst_mask &
            ~(PVR_PIPELINE_STAGE_ALL_BITS | PVR_PIPELINE_STAGE_QUERY_BIT)));

   u_foreach_bit (stage, src_mask) {
      struct vk_sync *s = queue->last_job_signal_sync[stage];
      if (!s)
         continue;
#ifdef PVR_APPLE_G5P
      /* G5P: a source whose work already completed orders nothing; skip it
       * instead of building a syncobj per dst stage around a signalled fence.
       * The cache is per stage and cleared when the sync is replaced. */
      if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p) && pvr_g5p_barrier_skip_done()) {
         if (queue->g5p_src_done[stage])
            continue;
         if (vk_sync_wait(&device->vk, s, 0, VK_SYNC_WAIT_COMPLETE, 0) == VK_SUCCESS) {
            queue->g5p_src_done[stage] = true;
            continue;
         }
      }
#endif
      wait_syncs[src_wait_count++] = (struct vk_sync_wait){
         .sync = s,
         .stage_mask = ~(VkPipelineStageFlags2)0,
         .wait_value = 0,
      };
   }

   /* No previous src jobs that need finishing so no need for a barrier. */
   if (src_wait_count == 0)
      return VK_SUCCESS;

   if (pvr_g5p_deplog() && (dst_mask & PVR_PIPELINE_STAGE_GEOM_BIT)) {
      unsigned pending = 0;
      for (unsigned k = 0; k < src_wait_count; k++) pending += g5p_dep_pending(device, wait_syncs[k].sync);
      fprintf(stderr, "G5P dep: submit=%u barrier src=%#x dst=%#x pending=%u\n", g5p_dep_submit, src_mask, dst_mask, pending);
   }

   u_foreach_bit (stage, dst_mask) {
      uint32_t wait_count = src_wait_count;
      struct vk_sync_signal signal;
      struct vk_sync *signal_sync;

      result = vk_sync_create(&device->vk,
                              &device->pdevice->ws->syncobj_type,
                              0U,
                              0UL,
                              &signal_sync);
      if (result != VK_SUCCESS)
         return result;

      signal = (struct vk_sync_signal){
         .sync = signal_sync,
         .stage_mask = ~(VkPipelineStageFlags2)0,
         .signal_value = 0,
      };

      if (queue->next_job_wait_sync[stage]) {
         wait_syncs[wait_count++] = (struct vk_sync_wait){
            .sync = queue->next_job_wait_sync[stage],
            .stage_mask = ~(VkPipelineStageFlags2)0,
            .wait_value = 0,
         };
      }

      result = device->ws->ops->null_job_submit(device->ws,
                                                wait_syncs,
                                                wait_count,
                                                &signal);
      if (result != VK_SUCCESS) {
         vk_sync_destroy(&device->vk, signal_sync);
         return result;
      }

      if (queue->next_job_wait_sync[stage])
         vk_sync_destroy(&device->vk, queue->next_job_wait_sync[stage]);

      queue->next_job_wait_sync[stage] = signal_sync;
   }

   return VK_SUCCESS;
}

static VkResult
pvr_process_event_cmd_set_or_reset(struct pvr_device *device,
                                   struct pvr_queue *queue,
                                   struct pvr_sub_cmd_event_set_reset *sub_cmd,
                                   const enum pvr_event_state new_event_state)
{
   /* Not PVR_JOB_TYPE_MAX since that also includes
    * PVR_JOB_TYPE_QUERY so no stage in the src mask.
    */
   struct vk_sync_wait waits[PVR_NUM_SYNC_PIPELINE_STAGES];
   struct vk_sync_signal signal;
   struct vk_sync *signal_sync;

   uint32_t wait_count = 0;
   VkResult result;

   assert(!(sub_cmd->wait_for_stage_mask & ~PVR_PIPELINE_STAGE_ALL_BITS));

   u_foreach_bit (stage, sub_cmd->wait_for_stage_mask) {
      if (!queue->last_job_signal_sync[stage])
         continue;

      waits[wait_count++] = (struct vk_sync_wait){
         .sync = queue->last_job_signal_sync[stage],
         .stage_mask = ~(VkPipelineStageFlags2)0,
         .wait_value = 0,
      };
   }

   result = vk_sync_create(&device->vk,
                           &device->pdevice->ws->syncobj_type,
                           0U,
                           0UL,
                           &signal_sync);
   if (result != VK_SUCCESS)
      return result;

   signal = (struct vk_sync_signal){
      .sync = signal_sync,
      .stage_mask = ~(VkPipelineStageFlags2)0,
      .signal_value = 0,
   };

   result =
      device->ws->ops->null_job_submit(device->ws, waits, wait_count, &signal);
   if (result != VK_SUCCESS) {
      vk_sync_destroy(&device->vk, signal_sync);
      return result;
   }

   if (sub_cmd->event->sync)
      vk_sync_destroy(&device->vk, sub_cmd->event->sync);

   sub_cmd->event->sync = signal_sync;
   sub_cmd->event->state = new_event_state;

   return VK_SUCCESS;
}

static inline VkResult
pvr_process_event_cmd_set(struct pvr_device *device,
                          struct pvr_queue *queue,
                          struct pvr_sub_cmd_event_set_reset *sub_cmd)
{
   return pvr_process_event_cmd_set_or_reset(device,
                                             queue,
                                             sub_cmd,
                                             PVR_EVENT_STATE_SET_BY_DEVICE);
}

static inline VkResult
pvr_process_event_cmd_reset(struct pvr_device *device,
                            struct pvr_queue *queue,
                            struct pvr_sub_cmd_event_set_reset *sub_cmd)
{
   return pvr_process_event_cmd_set_or_reset(device,
                                             queue,
                                             sub_cmd,
                                             PVR_EVENT_STATE_RESET_BY_DEVICE);
}

/**
 * \brief Process an event sub command of wait type.
 *
 * This sets up barrier syncobjs to create a dependency from the event syncobjs
 * onto the next job submissions.
 *
 * The barriers are setup by taking into consideration each event's dst stage
 * mask so this is in line with vkCmdWaitEvents2().
 *
 * \param[in] device                       Device to create the syncobjs on.
 * \param[in] sub_cmd                      Sub command to process.
 * \param[in,out] barriers                 Current barriers as input. Barriers
 *                                         for the next jobs as output.
 * \parma[in,out] per_cmd_buffer_syncobjs  Completion syncobjs for the command
 *                                         buffer being processed.
 */
static VkResult
pvr_process_event_cmd_wait(struct pvr_device *device,
                           struct pvr_queue *queue,
                           struct pvr_sub_cmd_event_wait *sub_cmd)
{
   uint32_t dst_mask = 0;
   VkResult result;

   STACK_ARRAY(struct vk_sync_wait, waits, sub_cmd->count + 1);
   if (!waits)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   for (uint32_t i = 0; i < sub_cmd->count; i++)
      dst_mask |= sub_cmd->wait_at_stage_masks[i];

   u_foreach_bit (stage, dst_mask) {
      struct vk_sync_signal signal;
      struct vk_sync *signal_sync;
      uint32_t wait_count = 0;

      for (uint32_t i = 0; i < sub_cmd->count; i++) {
         if (sub_cmd->wait_at_stage_masks[i] & BITFIELD_BIT(stage)) {
            waits[wait_count++] = (struct vk_sync_wait){
               .sync = sub_cmd->events[i]->sync,
               .stage_mask = ~(VkPipelineStageFlags2)0,
               .wait_value = 0,
            };
         }
      }

      if (!wait_count)
         continue;

      if (pvr_g5p_deplog() && stage == PVR_JOB_TYPE_GEOM) {
         unsigned pending = 0;
         for (unsigned k = 0; k < wait_count; k++) pending += g5p_dep_pending(device, waits[k].sync);
         fprintf(stderr, "G5P dep: submit=%u event pending=%u\n", g5p_dep_submit, pending);
      }
      if (queue->next_job_wait_sync[stage]) {
         waits[wait_count++] = (struct vk_sync_wait){
            .sync = queue->next_job_wait_sync[stage],
            .stage_mask = ~(VkPipelineStageFlags2)0,
            .wait_value = 0,
         };
      }

      assert(wait_count <= (sub_cmd->count + 1));

      result = vk_sync_create(&device->vk,
                              &device->pdevice->ws->syncobj_type,
                              0U,
                              0UL,
                              &signal_sync);
      if (result != VK_SUCCESS)
         goto err_free_waits;

      signal = (struct vk_sync_signal){
         .sync = signal_sync,
         .stage_mask = ~(VkPipelineStageFlags2)0,
         .signal_value = 0,
      };

      result = device->ws->ops->null_job_submit(device->ws,
                                                waits,
                                                wait_count,
                                                &signal);
      if (result != VK_SUCCESS) {
         vk_sync_destroy(&device->vk, signal.sync);
         goto err_free_waits;
      }

      if (queue->next_job_wait_sync[stage])
         vk_sync_destroy(&device->vk, queue->next_job_wait_sync[stage]);

      queue->next_job_wait_sync[stage] = signal.sync;
   }

   STACK_ARRAY_FINISH(waits);

   return VK_SUCCESS;

err_free_waits:
   STACK_ARRAY_FINISH(waits);

   return result;
}

static VkResult pvr_process_event_cmd(struct pvr_device *device,
                                      struct pvr_queue *queue,
                                      struct pvr_sub_cmd_event *sub_cmd)
{
   switch (sub_cmd->type) {
   case PVR_EVENT_TYPE_SET:
      return pvr_process_event_cmd_set(device, queue, &sub_cmd->set_reset);
   case PVR_EVENT_TYPE_RESET:
      return pvr_process_event_cmd_reset(device, queue, &sub_cmd->set_reset);
   case PVR_EVENT_TYPE_WAIT:
      return pvr_process_event_cmd_wait(device, queue, &sub_cmd->wait);
   case PVR_EVENT_TYPE_BARRIER:
      return pvr_process_event_cmd_barrier(device, queue, &sub_cmd->barrier);
   default:
      UNREACHABLE("Invalid event sub-command type.");
   };
}

struct pvr_suspended_data {
   struct pvr_rt_dataset **rts;
   bool frag_uses_atomic_ops;
   bool disable_compute_overlap;
   bool get_vis_results;
   const struct pvr_query_pool *query_pool;
};

static VkResult pvr_process_cmd_buffer(struct pvr_device *device,
                                       struct pvr_queue *queue,
                                       struct pvr_cmd_buffer *cmd_buffer,
                                       struct pvr_suspended_data *suspended_data)
{
   VkResult result;

   list_for_each_entry_safe (struct pvr_sub_cmd,
                             sub_cmd,
                             &cmd_buffer->sub_cmds,
                             link) {
      const int64_t g9p_t0 = getenv("G9P_TIME") ? os_time_get_nano() : 0;
      switch (sub_cmd->type) {
      case PVR_SUB_CMD_TYPE_GRAPHICS: {
         /* If the fragment job utilizes queries, for data integrity
          * it needs to wait for the query to be processed.
          */
         if (sub_cmd->gfx.has_query) {
            struct pvr_sub_cmd_event_barrier barrier = {
               .wait_for_stage_mask = PVR_PIPELINE_STAGE_QUERY_BIT,
               .wait_at_stage_mask = PVR_PIPELINE_STAGE_FRAG_BIT,
            };

            result = pvr_process_event_cmd_barrier(device, queue, &barrier);
            if (result != VK_SUCCESS)
               break;
         }

         if (sub_cmd->gfx.wait_on_previous_transfer) {
            struct pvr_sub_cmd_event_barrier barrier = {
               .wait_for_stage_mask = PVR_PIPELINE_STAGE_TRANSFER_BIT,
               .wait_at_stage_mask = PVR_PIPELINE_STAGE_FRAG_BIT,
            };

            result = pvr_process_event_cmd_barrier(device, queue, &barrier);
            if (result != VK_SUCCESS)
               break;
         }

         if (sub_cmd->is_resume) {
            sub_cmd->gfx.job.view_state.rt_datasets = suspended_data->rts;
            sub_cmd->gfx.job.frag_uses_atomic_ops |=
               suspended_data->frag_uses_atomic_ops;
            sub_cmd->gfx.job.disable_compute_overlap |=
               suspended_data->disable_compute_overlap;
            sub_cmd->gfx.job.get_vis_results |= suspended_data->get_vis_results;
            if (!sub_cmd->gfx.query_pool) {
               sub_cmd->gfx.query_pool = suspended_data->query_pool;
            } else if (suspended_data->query_pool &&
                       sub_cmd->gfx.query_pool != suspended_data->query_pool) {
               pvr_finishme(
                  "Emit a barrier between suspending and resuming jobs with different query pools");
            }
         }

         if (sub_cmd->is_suspend) {
            suspended_data->rts = sub_cmd->gfx.job.view_state.rt_datasets;
            suspended_data->frag_uses_atomic_ops =
               sub_cmd->gfx.job.frag_uses_atomic_ops;
            suspended_data->disable_compute_overlap =
               sub_cmd->gfx.job.disable_compute_overlap;
            suspended_data->get_vis_results = sub_cmd->gfx.job.get_vis_results;
            suspended_data->query_pool = sub_cmd->gfx.query_pool;
         } else {
            suspended_data->rts = NULL;
            suspended_data->frag_uses_atomic_ops = false;
            suspended_data->disable_compute_overlap = false;
            suspended_data->get_vis_results = false;
            suspended_data->query_pool = NULL;
         }

         assert(sub_cmd->gfx.job.view_state.rt_datasets);
         result =
            pvr_process_graphics_cmd(device, queue, cmd_buffer, &sub_cmd->gfx);
         break;
      }

      case PVR_SUB_CMD_TYPE_COMPUTE:
         result = pvr_process_compute_cmd(device, queue, &sub_cmd->compute);
         break;

      case PVR_SUB_CMD_TYPE_TRANSFER: {
         const bool serialize_with_frag = sub_cmd->transfer.serialize_with_frag;

         if (serialize_with_frag) {
            struct pvr_sub_cmd_event_barrier barrier = {
               .wait_for_stage_mask = PVR_PIPELINE_STAGE_FRAG_BIT,
               .wait_at_stage_mask = PVR_PIPELINE_STAGE_TRANSFER_BIT,
            };

            result = pvr_process_event_cmd_barrier(device, queue, &barrier);
            if (result != VK_SUCCESS)
               break;
         }

         result = pvr_process_transfer_cmds(device, queue, &sub_cmd->transfer);

         if (serialize_with_frag) {
            struct pvr_sub_cmd_event_barrier barrier = {
               .wait_for_stage_mask = PVR_PIPELINE_STAGE_TRANSFER_BIT,
               .wait_at_stage_mask = PVR_PIPELINE_STAGE_FRAG_BIT,
            };

            if (result != VK_SUCCESS)
               break;

            result = pvr_process_event_cmd_barrier(device, queue, &barrier);
         }

         break;
      }

      case PVR_SUB_CMD_TYPE_QUERY:
         result = pvr_process_query_cmd(device, queue, &sub_cmd->compute);
         break;

      case PVR_SUB_CMD_TYPE_EVENT:
         result = pvr_process_event_cmd(device, queue, &sub_cmd->event);
         break;

      default:
         mesa_loge("Unsupported sub-command type %d", sub_cmd->type);
         result = vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
      }

      if (result != VK_SUCCESS)
         return result;
      if (g9p_t0 && os_time_get_nano() - g9p_t0 > 1000000)
         fprintf(stderr, "g9p: sub-command type %d took %" PRId64 " us\n", sub_cmd->type,
                 (os_time_get_nano() - g9p_t0) / 1000);

      p_atomic_inc(&device->global_cmd_buffer_submit_count);
   }
   PVR_G5P_HOOK(xfb_end_cmd_buffer, device, cmd_buffer)

   return PVR_PER_ARCH(g9p_xfb_end_cmd_buffer)(device, cmd_buffer);
}

static VkResult pvr_clear_last_submits_syncs(struct pvr_queue *queue)
{
   /* Keep the last submission's syncs instead of waiting for them on the CPU.
    * The next jobs and barriers wait for them on the GPU: a barrier's first
    * scope reaches into earlier submissions. pvr_update_job_syncs replaces each
    * sync when its job type signals again; the kernel holds its own fence refs.
    * PVR_G5P_SERIAL_SUBMITS=1 restores the old behaviour. */
   if (PVR_HAS_FEATURE(&queue->device->pdevice->dev_info, apple_g5p) &&
       !pvr_g5p_serial_submits())
      return VK_SUCCESS;

   struct vk_sync_wait waits[PVR_JOB_TYPE_MAX * 2];
   uint32_t wait_count = 0;
   VkResult result;

   /*
    * G9P: keep the last submission's syncs instead of waiting for them on the
    * CPU: the next jobs and barriers wait for them on the GPU (a barrier's
    * first scope reaches into earlier submissions), each is replaced when its
    * job type signals again, and the kernel holds its own fence references.
    * Waiting here made every submission wait for the previous one to finish.
    * G9P_SERIAL_SUBMITS gives the old behaviour.
    */
   if (PVR_HAS_FEATURE(&queue->device->pdevice->dev_info, apple_g9p) &&
       !getenv("G9P_SERIAL_SUBMITS"))
      return VK_SUCCESS;

   for (uint32_t i = 0; i < PVR_JOB_TYPE_MAX; i++) {
      if (queue->next_job_wait_sync[i]) {
         waits[wait_count++] = (struct vk_sync_wait){
            .sync = queue->next_job_wait_sync[i],
            .stage_mask = ~(VkPipelineStageFlags2)0,
            .wait_value = 0,
         };
      }

      if (queue->last_job_signal_sync[i]) {
         waits[wait_count++] = (struct vk_sync_wait){
            .sync = queue->last_job_signal_sync[i],
            .stage_mask = ~(VkPipelineStageFlags2)0,
            .wait_value = 0,
         };
      }
   }

   /* G9P_DUMP: which of the last submit's syncs are still pending */
   if (getenv("G9P_DUMP")) {
      for (uint32_t i = 0; i < PVR_JOB_TYPE_MAX; i++) {
         struct vk_sync *s[2] = { queue->next_job_wait_sync[i],
                                  queue->last_job_signal_sync[i] };

         for (unsigned k = 0; k < 2; k++) {
            if (s[k] && vk_sync_wait(&queue->device->vk, s[k], 0, VK_SYNC_WAIT_COMPLETE,
                                     os_time_get_absolute_timeout(500000000)) !=
                           VK_SUCCESS)
               fprintf(stderr, "g9p: job type %u %s sync still pending\n", i,
                       k ? "last signal" : "next wait");
         }
      }
   }

   result = vk_sync_wait_many(&queue->device->vk,
                              wait_count,
                              waits,
                              VK_SYNC_WAIT_COMPLETE,
                              UINT64_MAX);

   if (result != VK_SUCCESS)
      return vk_error(queue, result);

   for (uint32_t i = 0; i < PVR_JOB_TYPE_MAX; i++) {
      if (queue->next_job_wait_sync[i]) {
         vk_sync_destroy(&queue->device->vk, queue->next_job_wait_sync[i]);
         queue->next_job_wait_sync[i] = NULL;
      }
#ifdef PVR_APPLE_G5P
      if (queue->g5p_sem_wait[i]) {
         vk_sync_destroy(&queue->device->vk, queue->g5p_sem_wait[i]);
         queue->g5p_sem_wait[i] = NULL;
      }
#endif

      if (queue->last_job_signal_sync[i]) {
         vk_sync_destroy(&queue->device->vk, queue->last_job_signal_sync[i]);
         queue->last_job_signal_sync[i] = NULL;
#ifdef PVR_APPLE_G5P
         queue->g5p_src_done[i] = false;
#endif
      }
   }

   return VK_SUCCESS;
}

#ifdef PVR_APPLE_G5P
static VkResult pvr_process_queue_signals_g5p(struct pvr_queue *queue,
                                          struct vk_sync_signal *signals,
                                          uint32_t signal_count,
                                          const struct vk_sync_wait *extra, uint32_t extra_count)
{
   STACK_ARRAY(struct vk_sync_wait, signal_waits, PVR_JOB_TYPE_MAX * 2 + extra_count);
   if (!signal_waits) return vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);
   struct pvr_device *device = queue->device;
   const struct vk_sync_signal *built = NULL;
   enum pvr_pipeline_stage_bits built_src = 0;
   VkResult result;

   for (uint32_t signal_idx = 0; signal_idx < signal_count; signal_idx++) {
      struct vk_sync_signal *signal = &signals[signal_idx];
      const enum pvr_pipeline_stage_bits signal_stage_src =
         pvr_stage_mask_src(&device->pdevice->dev_info, signal->stage_mask);
      uint32_t wait_count = 0;

      /* A signal with the stage mask of one already built waits for the same
       * fences: take them from that signal's payload, one syncobj transfer
       * instead of another merge (Zink signals its batch timeline and the
       * present semaphore together). PVR_G5P_SIGNAL_SHARE=0 builds each. */
      if (built && built_src == signal_stage_src && pvr_g5p_signal_share()) {
         struct vk_sync_wait same = {
            .sync = built->sync,
            .stage_mask = ~(VkPipelineStageFlags2)0,
            .wait_value = built->signal_value,
         };

         result = device->ws->ops->null_job_submit(device->ws, &same, 1, signal);
         if (result != VK_SUCCESS) {
            STACK_ARRAY_FINISH(signal_waits);
            return result;
         }
         continue;
      }

      for (uint32_t i = 0; i < PVR_JOB_TYPE_MAX; i++) {
         /* Include waits left by job types that ran no job, even when the
          * signal stage mask does not include that type. */
         if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p) &&
             queue->next_job_wait_sync[i]) {
            signal_waits[wait_count++] = (struct vk_sync_wait){
               .sync = queue->next_job_wait_sync[i],
               .stage_mask = ~(VkPipelineStageFlags2)0,
               .wait_value = 0,
            };
         }

         /* Exception for query jobs since that's something internal,
          * so the user provided syncs won't ever have it as a source stage.
          */
         if (!(signal_stage_src & BITFIELD_BIT(i)) && i != PVR_JOB_TYPE_QUERY)
            continue;

         if (!queue->last_job_signal_sync[i])
            continue;

         /* a job a barrier already saw retired (pvr_process_event_cmd_barrier())
          * orders nothing; known without asking the kernel again */
         if (queue->g5p_src_done[i] && pvr_g5p_signal_share())
            continue;

         signal_waits[wait_count++] = (struct vk_sync_wait){
            .sync = queue->last_job_signal_sync[i],
            .stage_mask = ~(VkPipelineStageFlags2)0,
            .wait_value = 0,
         };
      }

      for (uint32_t k = 0; k < extra_count; k++)
         signal_waits[wait_count++] = (struct vk_sync_wait){
            .sync = extra[k].sync, .stage_mask = ~(VkPipelineStageFlags2)0, .wait_value = extra[k].wait_value };
      result = device->ws->ops->null_job_submit(device->ws,
                                                signal_waits,
                                                wait_count,
                                                signal);
      if (result != VK_SUCCESS) {
         STACK_ARRAY_FINISH(signal_waits);
         return result;
      }
      if (!built) {
         built = signal;
         built_src = signal_stage_src;
      }
   }

   STACK_ARRAY_FINISH(signal_waits);
   return VK_SUCCESS;
}

#endif

static VkResult pvr_process_queue_signals(struct pvr_queue *queue,
                                          struct vk_sync_signal *signals,
                                          uint32_t signal_count)
{
   struct vk_sync_wait signal_waits[PVR_JOB_TYPE_MAX];
   struct pvr_device *device = queue->device;
   VkResult result;

   for (uint32_t signal_idx = 0; signal_idx < signal_count; signal_idx++) {
      struct vk_sync_signal *signal = &signals[signal_idx];
      const enum pvr_pipeline_stage_bits signal_stage_src =
         pvr_stage_mask_src(&device->pdevice->dev_info, signal->stage_mask);
      uint32_t wait_count = 0;

      for (uint32_t i = 0; i < PVR_JOB_TYPE_MAX; i++) {
         /* Exception for query jobs since that's something internal,
          * so the user provided syncs won't ever have it as a source stage.
          */
         if (!(signal_stage_src & BITFIELD_BIT(i)) && i != PVR_JOB_TYPE_QUERY)
            continue;

         if (!queue->last_job_signal_sync[i])
            continue;

         signal_waits[wait_count++] = (struct vk_sync_wait){
            .sync = queue->last_job_signal_sync[i],
            .stage_mask = ~(VkPipelineStageFlags2)0,
            .wait_value = 0,
         };
      }

      result = device->ws->ops->null_job_submit(device->ws,
                                                signal_waits,
                                                wait_count,
                                                signal);
      if (result != VK_SUCCESS)
         return result;
   }

   return VK_SUCCESS;
}

static VkResult pvr_process_queue_signals_for_device(struct pvr_queue *queue,
   struct vk_sync_signal *signals, uint32_t signal_count,
   const struct vk_sync_wait *extra, uint32_t extra_count)
{
#ifdef PVR_APPLE_G5P
   if (PVR_HAS_FEATURE(&queue->device->pdevice->dev_info, apple_g5p))
      return pvr_process_queue_signals_g5p(queue, signals, signal_count, extra, extra_count);
#endif
   return pvr_process_queue_signals(queue, signals, signal_count);
}

static VkResult pvr_process_queue_waits(struct pvr_queue *queue,
                                        struct vk_sync_wait *waits,
                                        uint32_t wait_count, uint32_t command_buffer_count)
{
   struct pvr_device *device = queue->device;
   VkResult result;

   STACK_ARRAY(struct vk_sync_wait, stage_waits, wait_count + 1);
   if (!stage_waits)
      return vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);

   for (uint32_t i = 0; i < PVR_JOB_TYPE_MAX; i++) {
      struct vk_sync_signal next_job_wait_signal_sync;
      uint32_t stage_wait_count = 0;
      /* a wait the last submit left for this job type (G9P keeps them): still waited for */
      struct vk_sync *prev_wait = NULL;

      for (uint32_t wait_idx = 0; wait_idx < wait_count; wait_idx++) {
         if (!(pvr_stage_mask_dst(&device->pdevice->dev_info, waits[wait_idx].stage_mask) &
               BITFIELD_BIT(i))) {
            continue;
         }

         if (pvr_g5p_deplog() && i == PVR_JOB_TYPE_GEOM)
            fprintf(stderr, "G5P dep: submit=%u qwait dst=%#x pending=%u cbs=%u waits=%u\n",
                    g5p_dep_submit, pvr_stage_mask_dst(&device->pdevice->dev_info, waits[wait_idx].stage_mask),
                    g5p_dep_pending(device, waits[wait_idx].sync), command_buffer_count, wait_count);
         stage_waits[stage_wait_count++] = (struct vk_sync_wait){
            .sync = waits[wait_idx].sync,
            .stage_mask = ~(VkPipelineStageFlags2)0,
            .wait_value = waits[wait_idx].wait_value,
         };
      }

      if (!stage_wait_count)
         continue;

#ifdef PVR_APPLE_G5P
      if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p)) {
         struct vk_sync *sem_wait;
         uint32_t sem_count = stage_wait_count;
         if (queue->g5p_sem_wait[i])
            stage_waits[sem_count++] = (struct vk_sync_wait){
               .sync = queue->g5p_sem_wait[i], .stage_mask = ~(VkPipelineStageFlags2)0 };
         result = vk_sync_create(&device->vk, &device->pdevice->ws->syncobj_type,
                                 0U, 0UL, &sem_wait);
         if (result != VK_SUCCESS)
            goto err_free_waits;
         struct vk_sync_signal sem_signal = {
            .sync = sem_wait, .stage_mask = ~(VkPipelineStageFlags2)0 };
         result = device->ws->ops->null_job_submit(device->ws, stage_waits, sem_count, &sem_signal);
         if (result != VK_SUCCESS) {
            vk_sync_destroy(&device->vk, sem_wait);
            goto err_free_waits;
         }
         if (queue->g5p_sem_wait[i])
            vk_sync_destroy(&device->vk, queue->g5p_sem_wait[i]);
         queue->g5p_sem_wait[i] = sem_wait;
      }
#endif
      if (queue->next_job_wait_sync[i]) {
         prev_wait = queue->next_job_wait_sync[i];
         stage_waits[stage_wait_count++] = (struct vk_sync_wait){
            .sync = prev_wait,
            .stage_mask = ~(VkPipelineStageFlags2)0,
            .wait_value = 0,
         };
      }

      result = vk_sync_create(&device->vk,
                              &device->pdevice->ws->syncobj_type,
                              0U,
                              0UL,
                              &queue->next_job_wait_sync[i]);
      if (result != VK_SUCCESS) {
         queue->next_job_wait_sync[i] = prev_wait;
         goto err_free_waits;
      }

      next_job_wait_signal_sync = (struct vk_sync_signal){
         .sync = queue->next_job_wait_sync[i],
         .stage_mask = ~(VkPipelineStageFlags2)0,
         .signal_value = 0,
      };

      result = device->ws->ops->null_job_submit(device->ws,
                                                stage_waits,
                                                stage_wait_count,
                                                &next_job_wait_signal_sync);
      if (result != VK_SUCCESS)
         goto err_free_waits;
      if (prev_wait)
         vk_sync_destroy(&device->vk, prev_wait);
   }

   STACK_ARRAY_FINISH(stage_waits);

   return VK_SUCCESS;

err_free_waits:
   STACK_ARRAY_FINISH(stage_waits);

   return result;
}

static VkResult pvr_driver_queue_submit_jobs(struct vk_queue *queue,
                                             struct vk_queue_submit *submit)
{
   struct pvr_queue *driver_queue = container_of(queue, struct pvr_queue, vk);
   struct pvr_device *device = driver_queue->device;
   VkResult result;

   const int64_t g9p_tc = os_time_get_nano();
   if (pvr_g5p_deplog()) g5p_dep_submit++;
   /* a semaphore wait's second synchronization scope is its own batch; a batch
    * without command buffers orders only its signals */
   const bool waitonly_local = PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p) && !submit->command_buffer_count && submit->wait_count && pvr_g5p_waitonly_local();
   result = pvr_clear_last_submits_syncs(driver_queue);
   if (getenv("G9P_WAITLOG"))
      fprintf(stderr, "g9p: wait last submit %" PRId64 " us\n", (os_time_get_nano() - g9p_tc) / 1000);
   if (result != VK_SUCCESS)
      return result;

   const int64_t g9p_tw = os_time_get_nano();

   if (submit->wait_count && !waitonly_local) {
      result = pvr_process_queue_waits(driver_queue,
                                       submit->waits,
                                       submit->wait_count, submit->command_buffer_count);
      if (result != VK_SUCCESS)
         return result;
   }
   if (getenv("G9P_TIME") && os_time_get_nano() - g9p_tw > 1000000)
      fprintf(stderr, "g9p: %u queue waits took %" PRId64 " us\n", submit->wait_count,
              (os_time_get_nano() - g9p_tw) / 1000);

   struct pvr_suspended_data suspended_data = {0};

   for (uint32_t i = 0U; i < submit->command_buffer_count; i++) {
      result = pvr_process_cmd_buffer(
         device,
         driver_queue,
         container_of(submit->command_buffers[i], struct pvr_cmd_buffer, vk),
         &suspended_data);
      if (result != VK_SUCCESS)
         return result;
   }

   assert(suspended_data.rts == NULL && "suspended graphics job never resumed");

   result = pvr_process_queue_signals_for_device(driver_queue,
                                      submit->signals,
                                      submit->signal_count,
                                      waitonly_local ? submit->waits : NULL,
                                      waitonly_local ? submit->wait_count : 0);
   if (result != VK_SUCCESS)
      return result;

   return VK_SUCCESS;
}

/*
 * Performs a signal operation of a submission that will not run, unless the
 * payload already has a signal pending (a job the kernel accepted signals it,
 * with an error if its context is lost).
 */
static void pvr_signal_unsubmitted(struct vk_device *device,
                                   struct vk_sync *sync,
                                   uint64_t value)
{
   if ((sync->type->features & VK_SYNC_FEATURE_WAIT_PENDING) &&
       vk_sync_wait(device, sync, value, VK_SYNC_WAIT_PENDING, 0) ==
          VK_SUCCESS)
      return;

   /* a failure is logged by vk_sync_signal(); nothing else can be done */
   VkResult result = vk_sync_signal(device, sync, value);
   (void)result;
}

/*
 * A queue whose GPU context was lost (the kernel ends a hung job's context
 * and refuses its jobs with EIO) stays lost, so that clients see
 * VK_ERROR_DEVICE_LOST from here on instead of retrying
 */
static VkResult pvr_driver_queue_submit(struct vk_queue *queue,
                                        struct vk_queue_submit *submit)
{
   const int64_t g9p_t0 = os_time_get_nano();
   VkResult result = pvr_driver_queue_submit_jobs(queue, submit);

   if (getenv("G9P_TIME"))
      fprintf(stderr, "g9p: submit of %u command buffers took %" PRId64 " us\n",
              submit->command_buffer_count, (os_time_get_nano() - g9p_t0) / 1000);

   /*
    * A failed submission still signals what it promised: a waiter on its semaphores or
    * fence (Zink's flush thread in a compositor) otherwise waited forever after a GPU
    * reset, freezing the desktop; signalled, it sees the device lost and can recover
    */
   if (result != VK_SUCCESS) {
      struct pvr_queue *driver_queue = container_of(queue, struct pvr_queue, vk);

      for (uint32_t i = 0; i < submit->signal_count; i++)
         pvr_signal_unsubmitted(&driver_queue->device->vk,
                                submit->signals[i].sync,
                                submit->signals[i].signal_value);
   }
   if (result == VK_ERROR_DEVICE_LOST)
      return vk_queue_set_lost(queue, "GPU context lost");
   return result;
}

/*
 * vk_common_QueueSubmit2() returns VK_ERROR_DEVICE_LOST on a lost device
 * before it builds a submission, and stops at the first submission that
 * fails, so none of the signal operations of those submissions reach
 * pvr_driver_queue_submit(). Their semaphores and the fence then never get a
 * payload. A wait for a pending signal on one of them never returns:
 * vkGetSemaphoreFdKHR(SYNC_FD) on a binary semaphore waits for its signal to
 * be submitted with no timeout, as does a wait-for-submit timeline wait. Zink
 * exports its batch's semaphores into dma-bufs right after each submit, so a
 * lost context left its flush thread blocked in the kernel and the
 * application unable to exit.
 *
 * The submission is still refused with VK_ERROR_DEVICE_LOST and runs no
 * work, but every signal it promised and no job will perform is performed
 * here, as pvr_driver_queue_submit() does for a submission that fails.
 */
VKAPI_ATTR VkResult VKAPI_CALL
PVR_PER_ARCH(QueueSubmit2)(VkQueue _queue,
                           uint32_t submitCount,
                           const VkSubmitInfo2 *pSubmits,
                           VkFence _fence)
{
   VK_FROM_HANDLE(vk_queue, queue, _queue);
   VK_FROM_HANDLE(vk_fence, fence, _fence);
   struct vk_device *device = queue->base.device;
   VkResult result;

   result = vk_common_QueueSubmit2(_queue, submitCount, pSubmits, _fence);
   if (result == VK_SUCCESS || !vk_device_is_lost_no_report(device))
      return result;

   for (uint32_t i = 0; i < submitCount; i++) {
      for (uint32_t j = 0; j < pSubmits[i].signalSemaphoreInfoCount; j++) {
         const VkSemaphoreSubmitInfo *info =
            &pSubmits[i].pSignalSemaphoreInfos[j];
         VK_FROM_HANDLE(vk_semaphore, semaphore, info->semaphore);
         const uint64_t value =
            semaphore->type == VK_SEMAPHORE_TYPE_TIMELINE ? info->value : 0;

         pvr_signal_unsubmitted(device,
                                vk_semaphore_get_active_sync(semaphore),
                                value);
      }
   }
   if (fence)
      pvr_signal_unsubmitted(device, vk_fence_get_active_sync(fence), 0);

   return result;
}

VkResult PVR_PER_ARCH(QueueBindSparse)(VkQueue _queue,
                                       uint32_t bindInfoCount,
                                       const VkBindSparseInfo *pBindInfo,
                                       VkFence fence)
{
   return VK_SUCCESS;
}
