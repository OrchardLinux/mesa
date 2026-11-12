/* SPDX-License-Identifier: MIT */
#ifndef G5P_DRM_TRANSPORT_H
#define G5P_DRM_TRANSPORT_H
#include "apple_g5p_drm.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

/* The kernel calls of the apple_g5p winsys (our A9 kernel's typed UAPI,
 * apple_g5p_drm.h): device, VM, context, BOs, PRIME, cache sync, job status
 * and fault reports. The pvr_winsys_ops layer on top of it comes separately.
 * All calls return zero or negative errno. The caller serializes them. */
/* One GEM handle as this file holds it.  PRIME_FD_TO_HANDLE of a dma-buf whose
 * object this file already holds (our own export, imported back) returns that
 * same handle, so several g5p_drm_bos can share it; GEM_CLOSE goes out when the
 * last of them is destroyed. */
struct g5p_drm_handle {
   uint32_t handle, refs;
};
struct g5p_drm {
   int fd;
   uint32_t vm, context, fence;
   uint64_t job;
   int accepted; /* Ambiguous copy-out errors also prevent destructive cleanup. */
   int completed_success;
   struct drm_g5p_get_info info;
   struct drm_g5p_get_exec_info exec;
   struct g5p_drm_handle *handles;
   uint32_t handle_count, handle_cap;
};
struct g5p_drm_bo {
   uint32_t handle, roles;
   uint64_t va, size;
   void *map;
   uint64_t map_size; /* imported dma-bufs: CPU mapping length if != size */
   /* Imported or exported dma-buf: typed jobs pass ACCESS_IMPLICIT|CACHE
    * (dma_resv fences, kernel cache maintenance) and BO_SYNC is skipped. */
   bool shared;
};
int g5p_drm_open(struct g5p_drm *, const char *);
int g5p_drm_init(struct g5p_drm *, int fd);
void g5p_drm_close(struct g5p_drm *);
int g5p_drm_compute_ready(const struct g5p_drm *);
int g5p_drm_context_create(struct g5p_drm *);
/* Render-only kernels: VM + fence syncobj, no compute context. */
int g5p_drm_vm_create(struct g5p_drm *);
int g5p_drm_bo_create(struct g5p_drm *, struct g5p_drm_bo *, uint64_t, uint64_t, uint32_t);
int g5p_drm_bo_create_at_cpu(struct g5p_drm *, struct g5p_drm_bo *, uint64_t, uint64_t, uint32_t, void *);
int g5p_drm_bo_destroy(struct g5p_drm *, struct g5p_drm_bo *);
/* PRIME (DRM core ioctls): import maps a foreign dma-buf at va (READ|WRITE),
 * CPU mapping through the dma-buf itself; export returns a new dma-buf fd. */
int g5p_drm_bo_import(struct g5p_drm *, struct g5p_drm_bo *, int fd, uint64_t va, uint64_t *size);
int g5p_drm_bo_export(struct g5p_drm *, struct g5p_drm_bo *, int *fd);
/* The BO's mmap offset on the DRM fd (BO_MMAP_OFFSET), for a second mapping of it */
int g5p_drm_bo_mmap_offset(struct g5p_drm *, const struct g5p_drm_bo *, uint64_t *offset);
int g5p_drm_bo_sync(struct g5p_drm *, const struct g5p_drm_bo *, uint32_t);
int g5p_drm_bo_sync_range(struct g5p_drm *, const struct g5p_drm_bo *, uint32_t op, uint64_t va_lo, uint64_t va_hi);
int g5p_drm_wait(struct g5p_drm *, int64_t timeout_ns);
/* 0 when no job of this file's render context is queued or running: no
 * accepted job is outstanding and the last job's fence has signalled (the
 * kernel drops its queued count before it signals).  -EBUSY otherwise.  A VM
 * edit (VM_BIND map or unmap) is only issued after this returned 0. */
int g5p_drm_vm_idle(struct g5p_drm *);
int g5p_drm_fault(struct g5p_drm *, struct drm_g5p_get_fault_info *);
int g5p_drm_context_status(struct g5p_drm *, struct drm_g5p_get_context_status *);
/* GET_RENDER_INFO into *info (size set, the rest zeroed first). */
int g5p_drm_render_info(struct g5p_drm *, struct drm_g5p_get_render_info *info);
/* Typed render submission (TYPED_V3): a render context on the transport's VM,
 * kernel render targets per size, and the submit of one pass. */
int g5p_drm_render_context_create(struct g5p_drm *, uint32_t *context);
int g5p_drm_render_context_destroy(struct g5p_drm *, uint32_t context);
int g5p_drm_render_target_create(struct g5p_drm *, uint32_t context, uint32_t width, uint32_t height,
                                 uint32_t samples, uint32_t layers, uint32_t *handle);
int g5p_drm_render_target_destroy(struct g5p_drm *, uint32_t handle);
int g5p_drm_submit_render_typed(struct g5p_drm *, uint32_t context, uint32_t render_target,
                                const struct drm_g5p_render_pass_v3 *pass,
                                const struct drm_g5p_render_resource *res, uint32_t res_count,
                                const struct drm_g5p_sync *syncs, uint32_t sync_count, uint64_t *job_id);
#include "apple_g5p_compute_typed.h"
int g5p_drm_submit_compute_typed(struct g5p_drm *,uint32_t,const struct drm_g5p_compute_pass *,const struct drm_g5p_render_resource *,uint32_t,const struct drm_g5p_sync *,uint32_t,uint64_t *);
#endif
