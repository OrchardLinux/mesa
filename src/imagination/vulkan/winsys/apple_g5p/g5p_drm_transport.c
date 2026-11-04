/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include "g5p_drm_transport.h"
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static int call(int fd, unsigned long cmd, void *arg)
{
   /* Never retry SUBMIT: EINTR/copy-out failure may follow acceptance. */
   return ioctl(fd, cmd, arg) < 0 ? -errno : 0;
}
int g5p_drm_open(struct g5p_drm *ws, const char *node)
{
   int fd = open(node, O_RDWR | O_CLOEXEC);
   if (fd < 0) return -errno;
   return g5p_drm_init(ws, fd);
}
/* Takes fd: closed on failure, and by g5p_drm_close(). */
int g5p_drm_init(struct g5p_drm *ws, int fd)
{
   memset(ws, 0, sizeof(*ws));
   ws->fd = fd;
   ws->info.size = sizeof(ws->info);
   ws->exec.size = sizeof(ws->exec);
   int r = call(ws->fd, DRM_IOCTL_G5P_GET_INFO, &ws->info);
   if (!r) r = call(ws->fd, DRM_IOCTL_G5P_GET_EXEC_INFO, &ws->exec);
   if (!r && (ws->info.version != DRM_G5P_UAPI_VERSION ||
              ws->info.size != sizeof(ws->info) || ws->exec.size != sizeof(ws->exec) ||
              ws->info.gpu_page_shift != 14)) r = -EPROTO;
   if (r) g5p_drm_close(ws);
   return r;
}
void g5p_drm_close(struct g5p_drm *ws)
{
   /* After any possibly accepted submit the kernel owns quarantine/lifetime.
    * No explicit unbind/destroy is attempted, even after an error fence. */
   if (ws->fd >= 0) close(ws->fd);
   ws->fd = -1;
   free(ws->handles);
   ws->handles = NULL;
   ws->handle_count = ws->handle_cap = 0;
}
/* A g5p_drm_bo takes a reference on its GEM handle. */
static int handle_ref(struct g5p_drm *ws, uint32_t handle)
{
   for (uint32_t i = 0; i < ws->handle_count; i++)
      if (ws->handles[i].handle == handle) {
         ws->handles[i].refs++;
         return 0;
      }
   if (ws->handle_count == ws->handle_cap) {
      uint32_t cap = ws->handle_cap ? 2 * ws->handle_cap : 64;
      struct g5p_drm_handle *h = realloc(ws->handles, cap * sizeof(*h));
      if (!h) return -ENOMEM;
      ws->handles = h;
      ws->handle_cap = cap;
   }
   ws->handles[ws->handle_count++] = (struct g5p_drm_handle){.handle = handle, .refs = 1};
   return 0;
}
/* Drop a reference; true when it was the last one and the handle may be closed.
 * A handle without an entry (its reference could not be recorded) is closed. */
static bool handle_unref(struct g5p_drm *ws, uint32_t handle)
{
   for (uint32_t i = 0; i < ws->handle_count; i++)
      if (ws->handles[i].handle == handle) {
         if (--ws->handles[i].refs)
            return false;
         ws->handles[i] = ws->handles[--ws->handle_count];
         return true;
      }
   return true;
}
int g5p_drm_vm_idle(struct g5p_drm *ws)
{
   if (ws->accepted) return -EBUSY;
   if (!ws->job || !ws->fence) return 0;
   /* An absolute timeout of 0 has passed: the kernel only reports the state. */
   struct drm_syncobj_wait wait = {.handles = (uintptr_t)&ws->fence, .count_handles = 1,
      .timeout_nsec = 0, .flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL};
   int r;
   do { r = call(ws->fd, DRM_IOCTL_SYNCOBJ_WAIT, &wait); } while (r == -EINTR);
   return r == -ETIME ? -EBUSY : r;
}
int g5p_drm_compute_ready(const struct g5p_drm *ws)
{
   const uint64_t caps = DRM_G5P_CAP_BO_CPU | DRM_G5P_CAP_VM_PREPARE |
      DRM_G5P_CAP_SYNCOBJ | DRM_G5P_CAP_COMPUTE | DRM_G5P_CAP_CODE_BASE_VERIFIED |
      DRM_G5P_CAP_VM_LIVE | DRM_G5P_CAP_CONTEXT_LIVE | DRM_G5P_CAP_JOB_STATUS |
      DRM_G5P_CAP_BOUNDED_COMPUTE;
   const struct drm_g5p_get_info *i = &ws->info;
   const struct drm_g5p_get_exec_info *e = &ws->exec;
   if ((i->capabilities & caps) != caps) return -EOPNOTSUPP;
   if (i->flags || i->reserved[0] || i->reserved[1] || e->flags || e->pad ||
       e->reserved[0] || e->reserved[1] || e->reserved[2] ||
       i->bvnc[0] != 9 || i->bvnc[1] != 15 || i->bvnc[2] != 6 || i->bvnc[3] != 2)
      return -EPROTO;
   if (e->base_scope < DRM_G5P_BASE_DEVICE || e->base_scope > DRM_G5P_BASE_CONTEXT ||
       !e->usc_base || e->usc_base != e->pds_base || i->code_base != e->usc_base ||
       !e->window_size || e->window_size > (1ULL << 32) ||
       e->usc_offset_bits != 32 || e->pds_offset_bits != 32 ||
       e->allocation_alignment < 16384 ||
       (e->allocation_alignment & (e->allocation_alignment - 1)) ||
       e->usc_base % e->allocation_alignment || e->window_size % e->allocation_alignment ||
       i->va_start < 0x10000 || i->va_start >= i->va_end || i->va_end > (1ULL << 40) ||
       i->va_start % 16384 || i->va_end % 16384 ||
       e->usc_base < i->va_start || e->usc_base >= i->va_end ||
       e->window_size > i->va_end - e->usc_base ||
       i->reserved_start > i->reserved_end || i->reserved_end > (1ULL << 40) ||
       i->reserved_start % 16384 || i->reserved_end % 16384) return -ERANGE;
   return 0;
}
int g5p_drm_context_create(struct g5p_drm *ws)
{
   if (ws->vm || ws->context || ws->accepted) return -EBUSY;
   struct drm_g5p_create_vm vm = {0};
   int r = call(ws->fd, DRM_IOCTL_G5P_CREATE_VM, &vm);
   if (r) return r;
   ws->vm = vm.handle;
   struct drm_g5p_create_context ctx = {.vm = vm.handle, .type = DRM_G5P_CONTEXT_COMPUTE};
   r = call(ws->fd, DRM_IOCTL_G5P_CREATE_CONTEXT, &ctx);
   if (r) return r;
   ws->context = ctx.handle;
   /* CREATE_CONTEXT is bookkeeping. No public live-attach ioctl exists. */
   struct drm_syncobj_create fence = {0};
   r = call(ws->fd, DRM_IOCTL_SYNCOBJ_CREATE, &fence);
   if (!r) ws->fence = fence.handle;
   return r;
}
int g5p_drm_vm_create(struct g5p_drm *ws)
{
   if (ws->vm || ws->accepted) return -EBUSY;
   struct drm_g5p_create_vm vm = {0};
   int r = call(ws->fd, DRM_IOCTL_G5P_CREATE_VM, &vm);
   if (r) return r;
   ws->vm = vm.handle;
   struct drm_syncobj_create fence = {0};
   r = call(ws->fd, DRM_IOCTL_SYNCOBJ_CREATE, &fence);
   if (!r) ws->fence = fence.handle;
   return r;
}
int g5p_drm_bo_destroy(struct g5p_drm *ws, struct g5p_drm_bo *bo)
{
   if (ws->accepted) return -EBUSY;
   if (bo->va) {
      struct drm_g5p_vm_bind bind = {.vm = ws->vm,
         .va = bo->va, .size = bo->size, .op = DRM_G5P_VM_UNMAP};
      int r = call(ws->fd, DRM_IOCTL_G5P_VM_BIND, &bind);
      if (r) return r;
      bo->va = 0;
   }
   if (bo->map) { munmap(bo->map, bo->map_size ? bo->map_size : bo->size); bo->map = NULL; }
   if (bo->handle) {
      if (handle_unref(ws, bo->handle)) {
         struct drm_gem_close close_bo = {.handle = bo->handle};
         int r = call(ws->fd, DRM_IOCTL_GEM_CLOSE, &close_bo);
         if (r) return r;
      }
      bo->handle = 0;
   }
   return 0;
}
int g5p_drm_bo_create(struct g5p_drm *ws, struct g5p_drm_bo *bo,
                      uint64_t va, uint64_t size, uint32_t roles)
{
   return g5p_drm_bo_create_at_cpu(ws, bo, va, size, roles, NULL);
}
/* cpu: map the BO there (MAP_FIXED over the caller's PROT_NONE reservation),
 * so that the parts of one allocation are CPU-contiguous; NULL: anywhere. */
int g5p_drm_bo_create_at_cpu(struct g5p_drm *ws, struct g5p_drm_bo *bo,
                             uint64_t va, uint64_t size, uint32_t roles, void *cpu)
{
   const struct drm_g5p_get_info *i = &ws->info;
   if (ws->accepted) return -EBUSY;
   if (!ws->vm || !size || size % 16384 || va % 16384 || va < i->va_start ||
       va >= i->va_end || size > i->va_end - va || size > i->max_bo_size ||
       size > SIZE_MAX || !roles || (roles & ~7U) ||
       (i->reserved_start < i->reserved_end && va < i->reserved_end &&
        va + size > i->reserved_start)) return -EINVAL;
   memset(bo, 0, sizeof(*bo));
   struct drm_g5p_create_bo create = {.size = size, .flags = roles};
   int r = call(ws->fd, DRM_IOCTL_G5P_CREATE_BO, &create);
   if (r) return r;
   if (handle_ref(ws, create.handle)) {
      struct drm_gem_close close_bo = {.handle = create.handle};
      (void)call(ws->fd, DRM_IOCTL_GEM_CLOSE, &close_bo);
      return -ENOMEM;
   }
   bo->handle = create.handle; bo->size = size; bo->roles = roles;
   struct drm_g5p_bo_mmap_offset off = {.handle = bo->handle};
   r = call(ws->fd, DRM_IOCTL_G5P_BO_MMAP_OFFSET, &off);
   if (r) goto fail;
   if (off.offset > INT64_MAX) { r = -ERANGE; goto fail; }
   bo->map = mmap(cpu, size, PROT_READ | PROT_WRITE, MAP_SHARED | (cpu ? MAP_FIXED : 0), ws->fd,
                  off.offset);
   if (bo->map == MAP_FAILED) { bo->map = NULL; r = -errno; goto fail; }
   struct drm_g5p_vm_bind bind = {.vm = ws->vm, .bo = bo->handle,
      .va = va, .size = size, .op = DRM_G5P_VM_MAP};
   r = call(ws->fd, DRM_IOCTL_G5P_VM_BIND, &bind);
   if (r) goto fail;
   bo->va = va;
   return 0;
fail:
   (void)g5p_drm_bo_destroy(ws, bo);
   return r;
}
int g5p_drm_bo_import(struct g5p_drm *ws, struct g5p_drm_bo *bo, int fd, uint64_t va, uint64_t *size)
{
   const struct drm_g5p_get_info *i = &ws->info;
   if (ws->accepted) return -EBUSY;
   off_t end = lseek(fd, 0, SEEK_END);
   if (end <= 0) return end < 0 ? -errno : -EINVAL;
   lseek(fd, 0, SEEK_SET);
   /* VM_BIND granularity for imported sg-tables is the kernel's call
    * (render-uapi.md); request whole 16 KiB GPU pages covering the buffer. */
   uint64_t bind_size = ((uint64_t)end + 16383) & ~16383ULL;
   if (!ws->vm || va % 16384 || va < i->va_start || va >= i->va_end ||
       bind_size > i->va_end - va) return -EINVAL;
   memset(bo, 0, sizeof(*bo));
   struct drm_prime_handle prime = {.fd = fd};
   int r = call(ws->fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime);
   if (r) return r;
   /* the handle may be one this file already holds (our own export) */
   if (handle_ref(ws, prime.handle)) {
      if (handle_unref(ws, prime.handle)) {
         struct drm_gem_close close_bo = {.handle = prime.handle};
         (void)call(ws->fd, DRM_IOCTL_GEM_CLOSE, &close_bo);
      }
      return -ENOMEM;
   }
   bo->handle = prime.handle; bo->size = bind_size; bo->map_size = end;
   bo->roles = DRM_G5P_BO_READ | DRM_G5P_BO_WRITE;
   bo->map = mmap(NULL, end, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
   if (bo->map == MAP_FAILED) { bo->map = NULL; r = -errno; goto fail; }
   struct drm_g5p_vm_bind bind = {.vm = ws->vm, .bo = bo->handle, .va = va, .size = bind_size,
                                  .op = DRM_G5P_VM_MAP};
   r = call(ws->fd, DRM_IOCTL_G5P_VM_BIND, &bind);
   if (r) goto fail;
   bo->va = va;
   bo->shared = true;
   *size = end;
   return 0;
fail:
   (void)g5p_drm_bo_destroy(ws, bo);
   return r;
}
int g5p_drm_bo_export(struct g5p_drm *ws, struct g5p_drm_bo *bo, int *fd)
{
   struct drm_prime_handle prime = {.handle = bo->handle, .flags = DRM_CLOEXEC | DRM_RDWR};
   int r = call(ws->fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime);
   if (!r) *fd = prime.fd, bo->shared = true;
   return r;
}
/* The kernel's BO_SYNC takes whole GPU pages only: g5p_bo_range_valid() wants offset
 * and size multiples of G5P_GPU_PAGE (16 KiB), in every release8 kernel, with or
 * without its range commit; anything else is EINVAL. */
#define G5P_DRM_SYNC_PAGE 0x4000ull
/* CPU cache maintenance of the GPU VA range [va_lo, va_hi) of one kernel BO (its
 * intersection with the BO, widened to whole GPU pages of the BO: syncing more is
 * always allowed; nothing when they do not meet).  The kernel validates offset/size
 * against the BO and may sync more (release8 by default syncs the whole BO). */
int g5p_drm_bo_sync_range(struct g5p_drm *ws, const struct g5p_drm_bo *bo, uint32_t op, uint64_t va_lo,
                          uint64_t va_hi)
{
   if (op != DRM_G5P_BO_SYNC_TO_DEVICE && op != DRM_G5P_BO_SYNC_TO_CPU) return -EINVAL;
   if (bo->shared) return 0; /* the kernel maintains ACCESS_CACHE BOs */
   uint64_t lo = va_lo > bo->va ? va_lo : bo->va;
   uint64_t hi = va_hi < bo->va + bo->size ? va_hi : bo->va + bo->size;
   if (lo >= hi) return 0;
   if (ws->accepted && (op == DRM_G5P_BO_SYNC_TO_DEVICE || !ws->completed_success)) return -EBUSY;
   uint64_t off = (lo - bo->va) & ~(G5P_DRM_SYNC_PAGE - 1);
   uint64_t end = (hi - bo->va + G5P_DRM_SYNC_PAGE - 1) & ~(G5P_DRM_SYNC_PAGE - 1);
   if (end > bo->size) end = bo->size; /* kernel BO sizes are whole pages: the whole-BO sync always passed */
   struct drm_g5p_bo_sync sync = {.handle = bo->handle, .op = op, .offset = off, .size = end - off};
   return call(ws->fd, DRM_IOCTL_G5P_BO_SYNC, &sync);
}

int g5p_drm_bo_sync(struct g5p_drm *ws, const struct g5p_drm_bo *bo, uint32_t op)
{
   return g5p_drm_bo_sync_range(ws, bo, op, bo->va, bo->va + bo->size);
}
int g5p_drm_wait(struct g5p_drm *ws, int64_t timeout_ns)
{
   if (!ws->job || timeout_ns < 0) return -EINVAL;
   struct timespec now;
   if (clock_gettime(CLOCK_MONOTONIC, &now)) return -errno;
   int64_t ns = (int64_t)now.tv_sec * 1000000000 + now.tv_nsec;
   if (timeout_ns > INT64_MAX - ns) return -ERANGE;
   struct drm_syncobj_wait wait = {.handles = (uintptr_t)&ws->fence,
      .count_handles = 1, .timeout_nsec = ns + timeout_ns,
      .flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL};
   int r;
   do { r = call(ws->fd, DRM_IOCTL_SYNCOBJ_WAIT, &wait); } while (r == -EINTR);
   if (r) return r;
   struct drm_g5p_get_job_status status = {.job_id = ws->job};
   r = call(ws->fd, DRM_IOCTL_G5P_GET_JOB_STATUS, &status);
   if (r) return r;
   if (status.state != DRM_G5P_JOB_COMPLETE || status.error > 0) return -EPROTO;
   if (!status.error) ws->completed_success = 1;
   return status.error;
}
int g5p_drm_fault(struct g5p_drm *ws, struct drm_g5p_get_fault_info *fault)
{
   memset(fault, 0, sizeof(*fault));
   if (!(ws->info.capabilities & DRM_G5P_CAP_FAULT_INFO)) return -EOPNOTSUPP;
   return call(ws->fd, DRM_IOCTL_G5P_GET_FAULT_INFO, fault);
}
int g5p_drm_context_status(struct g5p_drm *ws, struct drm_g5p_get_context_status *status)
{
   memset(status, 0, sizeof(*status));
   if (!ws->context || !(ws->info.capabilities & DRM_G5P_CAP_CONTEXT_STATUS)) return -EOPNOTSUPP;
   status->context = ws->context;
   return call(ws->fd, DRM_IOCTL_G5P_GET_CONTEXT_STATUS, status);
}
_Static_assert(sizeof(struct drm_g5p_get_render_info) == 120, "GET_RENDER_INFO is 120 bytes, as the kernel's");
int g5p_drm_render_info(struct g5p_drm *ws, struct drm_g5p_get_render_info *info)
{
   memset(info, 0, sizeof(*info));
   info->size = sizeof(*info);
   int r = call(ws->fd, DRM_IOCTL_G5P_GET_RENDER_INFO, info);
   return !r && info->size != sizeof(*info) ? -EPROTO : r;
}

/* ---- typed render submission (TYPED_V3) ---- */

/* The layout the kernel's TYPED_V3 checks (the sizes and offsets our A9 ICD's
 * passes were proven with on the device). */
_Static_assert(sizeof(struct drm_g5p_render_resource) == 32, "render resource is 32 bytes");
_Static_assert(sizeof(struct drm_g5p_render_pass) == 360, "TYPED_V1 pass is 360 bytes");
_Static_assert(offsetof(struct drm_g5p_render_pass_ext, dbias_va) == 16, "ext dbias_va @16");
_Static_assert(offsetof(struct drm_g5p_render_pass_ext, bgobj_depth) == 32, "ext bgobj_depth @32");
_Static_assert(offsetof(struct drm_g5p_render_pass_ext, zls) == 48, "ext zls @48");
_Static_assert(sizeof(struct drm_g5p_render_pass_ext) == 128, "the V2 extension is 128 bytes");
_Static_assert(offsetof(struct drm_g5p_render_pass_v3, ext) == 360, "V3 ext @360");
_Static_assert(offsetof(struct drm_g5p_render_pass_v3, ta_program_va) == 488, "V3 ta_program_va @488");
_Static_assert(offsetof(struct drm_g5p_render_pass_v3, ta_program_size) == 496, "V3 ta_program_size @496");
_Static_assert(offsetof(struct drm_g5p_render_pass_v3, vdm_end) == 500, "V3 vdm_end @500");
_Static_assert(offsetof(struct drm_g5p_render_pass_v3, pixel_output_width) == 504, "V3 pixel_output_width @504");
_Static_assert(offsetof(struct drm_g5p_render_pass_v3, reserved) == 512, "V3 reserved @512");
_Static_assert(sizeof(struct drm_g5p_render_pass_v3) == 536, "TYPED_V3 pass is 536 bytes");

int g5p_drm_render_context_create(struct g5p_drm *ws, uint32_t *context)
{
   if (!ws->vm) return -EINVAL;
   struct drm_g5p_create_context ctx = {.vm = ws->vm, .type = DRM_G5P_CONTEXT_RENDER};
   int r = call(ws->fd, DRM_IOCTL_G5P_CREATE_CONTEXT, &ctx);
   if (!r) *context = ctx.handle;
   return r;
}

int g5p_drm_render_context_destroy(struct g5p_drm *ws, uint32_t context)
{
   struct drm_g5p_destroy_context d = {.handle = context};
   return call(ws->fd, DRM_IOCTL_G5P_DESTROY_CONTEXT, &d);
}

int g5p_drm_render_target_create(struct g5p_drm *ws, uint32_t context, uint32_t width, uint32_t height,
                                 uint32_t samples, uint32_t layers, uint32_t *handle)
{
   /* No USER_GEOMETRY: the kernel computes the geometry and owns the parameter buffers. */
   struct drm_g5p_create_render_target rt = {.context = context, .width = width, .height = height,
                                             .samples = samples, .layers = layers};
   int r = call(ws->fd, DRM_IOCTL_G5P_CREATE_RENDER_TARGET, &rt);
   if (!r) *handle = rt.handle;
   return r;
}

int g5p_drm_render_target_destroy(struct g5p_drm *ws, uint32_t handle)
{
   struct drm_g5p_destroy_render_target d = {.handle = handle};
   return call(ws->fd, DRM_IOCTL_G5P_DESTROY_RENDER_TARGET, &d);
}

int g5p_drm_submit_render_typed(struct g5p_drm *ws, uint32_t context, uint32_t render_target,
                                const struct drm_g5p_render_pass_v3 *pass,
                                const struct drm_g5p_render_resource *res, uint32_t res_count,
                                const struct drm_g5p_sync *syncs, uint32_t sync_count, uint64_t *job_id)
{
   if (res_count > DRM_G5P_RENDER_MAX_RESOURCES || sync_count > DRM_G5P_RENDER_MAX_SYNCS) return -E2BIG;
   struct drm_g5p_submit_render_typed s = {
      .context = context, .profile = DRM_G5P_RENDER_PROFILE_TYPED_V3, .render_target = render_target,
      .resources = (uintptr_t)res, .resource_count = res_count, .syncs = (uintptr_t)syncs,
      .sync_count = sync_count, .pass = (uintptr_t)pass, .pass_size = sizeof(*pass)};
   int r = call(ws->fd, DRM_IOCTL_G5P_SUBMIT_RENDER_TYPED, &s);
   if (!r && job_id) *job_id = s.job_id;
   return r;
}
