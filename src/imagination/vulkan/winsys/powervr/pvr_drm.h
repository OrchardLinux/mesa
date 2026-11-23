/*
 * Copyright © 2022 Imagination Technologies Ltd.
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

#ifndef PVR_DRM_H
#define PVR_DRM_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#include "pvr_winsys.h"
#include "vk_drm_syncobj.h"
#include "util/macros.h"
#include "util/rwlock.h"
#include "util/sparse_array.h"

/* Identify job out-fences without inspecting ordinary poll waits. */
struct pvr_drm_job_sync {
   struct vk_drm_syncobj base;
   const char *job_type;
};

struct pvr_drm_winsys_heap {
   struct pvr_winsys_heap base;
};

struct pvr_drm_winsys {
   struct pvr_winsys base;

   /* Packed bvnc */
   uint64_t bvnc;

   bool is_g5p;
   /* pvr_drm_winsys_create(query_only) on the A8: device queries only, no VM context */
   bool query_only;
   uint64_t caps;
   int madvise_warning;
   int64_t retry_log_time;
   /* The kernel sets a job fence's error when it resets (pvr_apple.c);
    * upstream pvr_queue.c does not.
    */
   bool job_fence_errors;
   int loss_reported;
   int lost; /* Atomic and terminal. */
   int unmap_loss_reported;

   /* Protects imported BOs creation/freeing, which includes the bo_map member
    * below.
    */
   struct u_rwlock dmabuf_bo_lock;

   /* This array holds all our 'struct pvr_drm_winsys_bo' allocations. We use
    * this so we can add a refcount to our BOs and check if a particular BO was
    * already allocated in this device using its GEM handle. This is necessary
    * to properly manage BO imports, because the kernel doesn't refcount the
    * underlying BO memory.
    *
    * Specifically, when self-importing (i.e. importing a BO into the same
    * device that created it), the kernel will give us the same BO handle for
    * both BOs and we must only free it once when both references are
    * freed. Otherwise, if we are not self-importing, we get two different BO
    * handles, and we want to free each one individually.
    *
    * The refcount is also useful for being able to maintain BOs across
    * VK object lifetimes, such as pipelines suballocating out of BOs
    * allocated on the device.
    */
   struct util_sparse_array bo_map;

   /* Required heaps */
   struct pvr_drm_winsys_heap general_heap;
   struct pvr_drm_winsys_heap pds_heap;
   struct pvr_drm_winsys_heap usc_heap;
   struct pvr_drm_winsys_heap vis_test_heap;
   struct pvr_drm_winsys_heap transfer_frag_heap;

   /* Optional heaps */
   bool rgn_hdr_heap_present;
   struct pvr_drm_winsys_heap rgn_hdr_heap;

   /* vma's for carveout memory regions */
   struct pvr_winsys_vma *pds_vma;
   struct pvr_winsys_vma *usc_vma;
   struct pvr_winsys_vma *general_vma;

   uint32_t vm_context;
};

#define PVR_DRM_CAP_RETRY_ENOMEM DRM_PVR_CAP_RETRY_ENOMEM
#define PVR_DRM_CAP_BO_MADVISE DRM_PVR_CAP_BO_MADVISE
int pvr_drm_ioctl_retry(struct pvr_drm_winsys *ws, unsigned long request, void *args);

struct drm_pvr_ioctl_submit_jobs_args;
VkResult pvr_drm_submit_jobs(struct pvr_drm_winsys *ws,
                             struct drm_pvr_ioctl_submit_jobs_args *args,
                             const char *job_type);
void pvr_drm_report_loss(struct pvr_drm_winsys *ws, const char *job_type,
                         const char *source, int status);

static inline void pvr_drm_tag_job_sync(struct pvr_drm_winsys *ws,
                                         struct vk_sync *sync, const char *job_type)
{
   if (ws->job_fence_errors && sync && ws->base.job_sync_size &&
       sync->type->size == ws->base.job_sync_size)
      ((struct pvr_drm_job_sync *)sync)->job_type = job_type;
}

/*******************************************
    helper macros
 *******************************************/

#define to_pvr_drm_winsys(ws) container_of((ws), struct pvr_drm_winsys, base)
#define to_pvr_drm_winsys_heap(heap) \
   container_of((heap), struct pvr_drm_winsys_heap, base)

#endif /* PVR_DRM_H */
