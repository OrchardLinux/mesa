/* SPDX-License-Identifier: MIT */
/*
 * The apple_g5p winsys: struct pvr_winsys_ops over our A9 kernel's typed UAPI.
 * Only g5p_drm_transport.[ch] issues ioctls; this file is the memory model.
 *
 * Memory. Our kernel edits a context's VM only while the context is idle, and
 * pins at most 768 MiB per VM in BOs of at most 64 MiB (GET_INFO.max_bo_size). So this winsys keeps
 * an arena: a range of GPU VA at the kernel's code base (pds_base ==
 * usc_base), reserved up front for the heap size (PVR_G5P_ARENA_MIB overrides) and committed in
 * READ|WRITE|EXEC chunks as it fills (the first at the first slice, 512 KiB minimum; then demand sized, at
 * least 2 MiB and an eighth of what is committed, so that the chunks reach the heap size), each
 * CPU-mapped at its place in one reservation. Pinned memory is what a process
 * uses, not the maximum (g5p_ws_arena_grow()). Every BO is a slice of it, taken at buffer_create:
 * its CPU and GPU addresses are fixed from then on. Because the whole arena
 * lies inside the 4 GiB code window above that base, a slice is a valid USC or
 * PDS program address whichever heap it is later bound in.
 *
 * Heaps are address bookkeeping only: heap_alloc reserves a range from the
 * heap, and vma_map returns the BO slice's own address plus the offset (the
 * device address pvr_* uses from then on). Several binds of one BO alias one
 * slice; the only VM_BIND after creation is a new chunk's. vma_unmap and heap_free give the
 * reservation back.
 *
 * Display BOs, dma-buf imports and application allocations of 2 MiB and more are their own kernel BOs, bound
 * in a window above the arena (a VM edit: the context must be idle). One larger than the kernel's largest BO is
 * several kernel BOs, contiguous in GPU VA and in one CPU reservation (g5p_ws_own_create()).
 *
 * Not here yet: render, compute and transfer submission (the G5P job streams
 * and their typed records come with the draw hooks), and cache maintenance at
 * submit. Contexts are winsys-side objects until then.
 */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "hwdef/rogue_hw_utils.h"
#include "util/list.h"
#include "util/macros.h"
#include "util/os_misc.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_atomic.h"
#include "util/u_math.h"
#include "util/vma.h"
#include "vk_alloc.h"
#include "vk_drm_syncobj.h"
#include "vk_log.h"
#include "vk_sync.h"
#include <xf86drm.h>

#include "pvr_device_info.h"
#include "pvr_winsys.h"
#include "pvr_winsys_helper.h"

#include "g5p_drm_transport.h"
#include "pvr_g5p_compute.h"
#include "g5p_ws_perf.h"
#include "pvr_g5p_winsys.h"

#define G5P_WS_VDM_GUARD 64u /* Rogue VDM read-ahead padding, vdm.xml */
#define G5P_WS_PAGE 16384ull
#define G5P_WS_CHUNK (64ull << 20)         /* the kernel's largest BO */
#define G5P_WS_MAX_CHUNKS 128               /* chunks grow with the arena (g5p_ws_arena_grow()): ~66 reach 768 MiB */
/* The most one VM pins (the kernel's G5P_MAX_PINNED_BYTES, drivers/gpu/apple-g5p/drm-validate.h; not in the
 * UAPI yet): the heap and the arena are bounded by it. The device-wide budget (render_pinned_mib, 1 GiB) is
 * shared by all clients. */
#define G5P_WS_VM_PIN_MAX (768ull << 20)
#define G5P_WS_ARENA_FALLBACK (128ull << 20) /* the arena's bound when the RAM size is unknown */
#define G5P_WS_ARENA_FIRST (512ull << 10) /* the first chunk; PVR_G5P_ARENA_FIRST_MIB overrides */
#define G5P_WS_CHUNK_FLOOR (2ull << 20)    /* every later chunk: demand, at least this */
#define G5P_WS_CHUNK_GROWTH 8              /* ... and at least 1/8 of the committed arena */
#define G5P_WS_CHUNK_GROWTH_MAX (16ull << 20) /* ... up to this; PVR_G5P_ARENA_CHUNK_MIB overrides (64: before) */
/* VA window for own BOs (display buffers, imports, application data): the kernel maps at most 1 GiB per VM
 * (G5P_MAX_VM_BYTES), so twice that leaves room for fragmentation. VA only: nothing is committed for it. */
#define G5P_WS_DEDICATED_SIZE (2ull << 30)
/* Render programs are 32-bit USC/PDS offsets from the code base, so no code window is larger. The kernel's
 * (GET_RENDER_INFO.code_window) bounds the arena; GET_EXEC_INFO.window_size (0x34000) is the older compute path's
 * and does not bound render programs. */
#define G5P_WS_MAX_CODE_WINDOW (1ull << 32)
/* The pvr_* heaps are address bookkeeping only (see the top of this file): heap_alloc reserves a range, and
 * vma_map then gives the bound BO's own address. So their bookkeeping range is not the arena's size: every bind of
 * a heap reserves its whole size there, own BOs (display buffers, imports: the 2 GiB window) included, and an
 * arena-sized range capped everything bound at once at 64 MiB (zink10's nested Hyprland: vkBindImageMemory failed
 * with the arena and the window still free). heap->size stays the arena's. */
#define G5P_WS_BOOKKEEPING (1ull << 40)
/* How long a VM edit (CREATE_RENDER_TARGET, an own BO's create, import or destroy) waits for the render contexts
 * to idle when the kernel says EBUSY (g5p_render_vm_idle()): its own job timeout. */
#define G5P_WS_BUSY_NS 2000000000ll

/* A refusal at winsys creation, on stderr too: vk_errorf() is silent in release builds, and a device that is not
 * found should say why. */
#define g5p_ws_fail(result, ...)                                                                                   \
   (fprintf(stderr, "apple_g5p winsys: " __VA_ARGS__), fprintf(stderr, "\n"), vk_errorf(NULL, result, __VA_ARGS__))

struct g5p_ws {
   struct pvr_winsys base;
   struct g5p_drm drm;
   struct drm_g5p_get_render_info render; /* the render VM layout (GET_RENDER_INFO) */
   simple_mtx_t lock;

   /* [arena_va, arena_va + arena_size): the arena's reserved GPU VA, and its CPU reservation at arena_cpu. The
    * chunks are committed from its start as the arena grows (g5p_ws_arena_grow()), one unbound page apart so that
    * no slice spans two chunks (the kernel takes a stream or a program inside one listed range); the slices heap
    * holds only committed chunks. */
   uint64_t arena_va, arena_size;
   uint64_t arena_max;      /* the most the chunks may total (the heap size; PVR_G5P_ARENA_MIB) */
   uint64_t heap_size;      /* the RAM share, bounded by G5P_WS_VM_PIN_MAX (pvr_g5p_winsys_heap_size()) */
   uint64_t arena_first;    /* the first chunk's floor (PVR_G5P_ARENA_FIRST_MIB), committed at the first slice */
   uint64_t arena_growth_max; /* the most a chunk grows by the arena's size (G5P_WS_CHUNK_GROWTH_MAX) */
   uint64_t arena_next;     /* the offset of the next chunk */
   uint64_t arena_committed; /* bytes in chunks */
   uint64_t arena_live, arena_peak, arena_allocs, arena_reused; /* slice bytes, high-water, allocation/reuse counts */
   bool arena_log; /* optional allocation accounting; no per-allocation logging by default */
   uint8_t *arena_cpu;
   uint64_t *arena_used;    /* one bit per page of [arena_va, +arena_size): handed out before (memset on reuse) */
   struct g5p_drm_bo chunks[G5P_WS_MAX_CHUNKS];
   unsigned chunk_count;
   int64_t chunk_idle_ns[G5P_WS_MAX_CHUNKS];
   uint32_t chunk_live[G5P_WS_MAX_CHUNKS]; /* slices referencing each chunk; lock held */
   uint64_t arena_reclaimed;
   uint64_t arena_submits;
   uint64_t arena_stats_after; /* bounded idle snapshots as well as120-job reports */
   uint64_t data_live, data_peak;
   unsigned data_count;
   struct util_vma_heap slices;

   /* own kernel BOs: display buffers and imports; every live one is listed in own_bos, because a job names
    * them all as its resources (a swapchain image or an imported buffer the job may write or read) */
   struct util_vma_heap dedicated;
   struct list_head own_bos;
   unsigned own_count;
   bool vma_heaps_ready; /* slices and dedicated are initialized */

   /* One kernel render context per VM, shared by all engine wrappers. */
   struct g5p_ws_context *context;

   struct pvr_winsys_heap general_heap, pds_heap, usc_heap, vis_test_heap, transfer_frag_heap;
   struct g5p_ws_perf perf; /* the jobs' resource lists (g5p_ws_perf.h) */
};

struct g5p_ws_bo {
   struct pvr_winsys_bo base;
   uint64_t va; /* GPU address of the slice or of the own BO */
   uint8_t *cpu;
   uint64_t alloc_size;
   unsigned chunk; /* stable until this slice is destroyed */
   bool dedicated_image; /* VkMemoryDedicatedAllocateInfo.image was supplied */
   bool data_only; /* large application VkDeviceMemory, not code/control streams */
   bool own; /* an own kernel BO (display buffer or import) */
   struct g5p_drm_bo kbo;
   struct list_head link; /* in g5p_ws::own_bos, while own */
   /* An own allocation larger than the kernel's largest BO: this BO is its first part, next_part the following
    * ones (each its own kbo, listed in own_bos, so a job names every part). [span_va, +span_size) is the whole
    * allocation, for every part: hazard tracking takes one allocation as one range (bo_range). */
   struct g5p_ws_bo *next_part;
   uint64_t span_va, span_size;
   bool placed; /* bo->map is a placed mapping (VK_EXT_map_memory_placed), not cpu */
   int refs;
};

static void g5p_ws_own_list(struct g5p_ws *ws, struct g5p_ws_bo *b)
{
   simple_mtx_lock(&ws->lock);
   list_addtail(&b->link, &ws->own_bos);
   ws->own_count++;
   if (b->data_only) {
      ws->data_live += b->alloc_size;
      ws->data_peak = MAX2(ws->data_peak, ws->data_live);
      ws->data_count++;
      if (ws->arena_log)
         fprintf(stderr, "apple_g5p winsys: data allocation pid=%ld fd=%d handle=%u size_kib=%" PRIu64 " live_kib=%" PRIu64 " bos=%u dedicated_image=%u\n",
                 (long)getpid(), ws->drm.fd, b->kbo.handle, b->alloc_size >> 10, ws->data_live >> 10, ws->data_count, b->dedicated_image);
   }
   simple_mtx_unlock(&ws->lock);
}

struct g5p_ws_vma {
   struct pvr_winsys_vma base;
   pvr_dev_addr_t reserved; /* the heap reservation, given back by heap_free */
};

struct g5p_ws_obj { /* free lists, datasets and contexts: winsys-side only for now */
   struct pvr_winsys *ws;
};

static inline struct g5p_ws *to_ws(struct pvr_winsys *ws) { return container_of(ws, struct g5p_ws, base); }

/* 4-sample render targets and the pass's MSAA fields (kernel a9/gpu-msaa):
 * TYPED_V3 and CAP_RENDER_MSAA, which the kernel reports only with its
 * kernel-computed render-target geometry and the V3 dialect on */
bool
pvr_g5p_winsys_msaa(struct pvr_winsys *ws)
{
   const uint64_t need = DRM_G5P_CAP_TYPED_RENDER_V3 | DRM_G5P_CAP_RENDER_MSAA;
   return (to_ws(ws)->drm.info.capabilities & need) == need;
}
static inline struct g5p_ws_bo *to_bo(struct pvr_winsys_bo *bo) { return container_of(bo, struct g5p_ws_bo, base); }
static inline struct g5p_ws_vma *to_vma(struct pvr_winsys_vma *v) { return container_of(v, struct g5p_ws_vma, base); }

uint64_t
pvr_g5p_winsys_heap_size(struct pvr_winsys *base, uint64_t ram_share)
{
   /* the drirc share (heap_memory_percent) when there is one, else the winsys's own (arena init) */
   return ram_share ? MIN2(ram_share, G5P_WS_VM_PIN_MAX) : to_ws(base)->heap_size;
}

bool
pvr_g5p_winsys_bo_range(struct pvr_winsys *base, uint64_t addr,
                         uint64_t *va, uint64_t *size)
{
   struct g5p_ws *ws = to_ws(base);
   bool found = false;

   simple_mtx_lock(&ws->lock);
   list_for_each_entry (struct g5p_ws_bo, bo, &ws->own_bos, link) {
      if (addr >= bo->va && addr - bo->va < bo->kbo.size) {
         *va = bo->span_size ? bo->span_va : bo->va;
         *size = bo->span_size ? bo->span_size : bo->kbo.size;
         found = true;
         break;
      }
   }
   simple_mtx_unlock(&ws->lock);
   return found;
}

bool
pvr_g5p_winsys_vma_cpu_addr(struct pvr_winsys_vma *vma,
                            uint64_t addr,
                            uint64_t size,
                            uint8_t **ptr_out,
                            uint64_t *available_out)
{
   struct g5p_ws_bo *bo;
   uint64_t delta, offset, available;

   if (!vma->bo || addr < vma->dev_addr.addr)
      return false;

   delta = addr - vma->dev_addr.addr;
   if (delta > vma->mapped_size)
      return false;

   bo = to_bo(vma->bo);
   if (!bo->cpu || vma->bo_offset > bo->alloc_size ||
       delta > bo->alloc_size - vma->bo_offset)
      return false;

   offset = vma->bo_offset + delta;
   available = MIN2(vma->mapped_size - delta, bo->alloc_size - offset);
   if (size > available)
      return false;

   *ptr_out = bo->cpu + offset;
   *available_out = available;
   return true;
}

/* ---- device info: the runtime values as the services winsys derives them
 * from the device info (pvr_srv.c, Imagination Technologies); the G5P paths
 * of the driver do not rely on them ---- */

static uint64_t g5p_num_phantoms(const struct pvr_device_info *d)
{
   return DIV_ROUND_UP(PVR_GET_FEATURE_VALUE(d, num_clusters, 1U), 4U);
}

static uint64_t g5p_total_reserved_partition_size(const struct pvr_device_info *d)
{
   const uint32_t tx = PVR_GET_FEATURE_VALUE(d, tile_size_x, 0), ty = PVR_GET_FEATURE_VALUE(d, tile_size_y, 0);
   const uint32_t partitions = PVR_GET_FEATURE_VALUE(d, max_partitions, 0);
   if (tx == 16 && ty == 16)
      return tx * ty * partitions * PVR_GET_FEATURE_VALUE(d, usc_min_output_registers_per_pix, 0);
   return (uint64_t)partitions * 1024U;
}

static uint64_t g5p_reserved_shared_size(const struct pvr_device_info *d)
{
   const uint32_t cs = PVR_GET_FEATURE_VALUE(d, common_store_size_in_dwords, 512U * 4U * 4U);
   return cs - (256U * 4U) - g5p_total_reserved_partition_size(d);
}

static uint64_t g5p_max_coeffs(const struct pvr_device_info *d)
{
   const uint32_t tiles = PVR_GET_FEATURE_VALUE(d, isp_max_tiles_in_flight, 0);
   const uint32_t pixel = DIV_ROUND_UP(tiles, g5p_num_phantoms(d)) * ROGUE_MAX_PIXEL_SHARED_REGISTERS;
   return g5p_reserved_shared_size(d) - (pixel + ROGUE_MAX_VERTEX_SHARED_REGISTERS + 2U * 1024U);
}

static VkResult g5p_ws_device_info_init(struct pvr_winsys *ws, struct pvr_device_info *dev_info,
                                        struct pvr_device_runtime_info *runtime_info)
{
   const struct drm_g5p_get_info *i = &to_ws(ws)->drm.info;
   const uint64_t bvnc = PVR_BVNC_PACK(i->bvnc[0], i->bvnc[1], i->bvnc[2], i->bvnc[3]);

   if (pvr_device_info_init(dev_info, bvnc))
      return g5p_ws_fail(VK_ERROR_INCOMPATIBLE_DRIVER, "no device info for BVNC %u.%u.%u.%u",
                       i->bvnc[0], i->bvnc[1], i->bvnc[2], i->bvnc[3]);
   if (!PVR_HAS_FEATURE(dev_info, apple_g5p))
      return g5p_ws_fail(VK_ERROR_INCOMPATIBLE_DRIVER, "BVNC is not an Apple G5P device");

   memset(runtime_info, 0, sizeof(*runtime_info));
   runtime_info->core_count = 1;
   runtime_info->num_phantoms = g5p_num_phantoms(dev_info);
   /* The kernel builds the parameter manager's lists itself (no free list
    * crosses this UAPI); the sizes only keep pvr_* consistent. */
   runtime_info->min_free_list_size = 50U << ROGUE_BIF_PM_PHYSICAL_PAGE_SHIFT;
   runtime_info->max_free_list_size = 2048U << ROGUE_BIF_PM_PHYSICAL_PAGE_SHIFT;
   runtime_info->reserved_shared_size = g5p_reserved_shared_size(dev_info);
   runtime_info->total_reserved_partition_size = g5p_total_reserved_partition_size(dev_info);
   runtime_info->max_coeffs = g5p_max_coeffs(dev_info);
   runtime_info->cdm_max_local_mem_size_regs =
      MIN2(runtime_info->max_coeffs, ROGUE_MAX_PER_KERNEL_LOCAL_MEM_SIZE_REGS);
   return VK_SUCCESS;
}

/* ---- heaps: bookkeeping ranges over the arena ---- */

static void g5p_ws_heap_init(struct g5p_ws *ws, struct pvr_winsys_heap *heap, uint64_t base, uint64_t size)
{
   memset(heap, 0, sizeof(*heap));
   heap->ws = &ws->base;
   heap->base_addr = PVR_DEV_ADDR(base);
   heap->size = size;
   heap->page_size = G5P_WS_PAGE;
   heap->log2_page_size = util_logbase2(G5P_WS_PAGE);
   util_vma_heap_init(&heap->vma_heap, base, G5P_WS_BOOKKEEPING);
   pthread_mutex_init(&heap->lock, NULL);
}

static void g5p_ws_get_heaps_info(struct pvr_winsys *ws, struct pvr_winsys_heaps *heaps)
{
   struct g5p_ws *w = to_ws(ws);
   heaps->general_heap = &w->general_heap;
   heaps->pds_heap = &w->pds_heap;
   heaps->usc_heap = &w->usc_heap;
   heaps->rgn_hdr_heap = &w->general_heap;
   heaps->vis_test_heap = &w->vis_test_heap;
   heaps->transfer_frag_heap = &w->transfer_frag_heap;
}

static VkResult g5p_ws_heap_alloc(struct pvr_winsys_heap *heap, uint64_t size, uint64_t alignment,
                                  struct pvr_winsys_vma **vma_out)
{
   struct g5p_ws *ws = to_ws(heap->ws);
   struct g5p_ws_vma *v = vk_zalloc(ws->base.alloc, sizeof(*v), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!v)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);
   VkResult r = pvr_winsys_helper_heap_alloc(heap, size, alignment, &v->base);
   if (r != VK_SUCCESS) {
      vk_free(ws->base.alloc, v);
      return g5p_ws_fail(r, "bind of %" PRIu64 " bytes (alignment %" PRIu64 "): the heap's bookkeeping is full",
                         size, alignment);
   }
   v->reserved = v->base.dev_addr;
   *vma_out = &v->base;
   return VK_SUCCESS;
}

static void g5p_ws_heap_free(struct pvr_winsys_vma *vma)
{
   struct g5p_ws_vma *v = to_vma(vma);
   struct g5p_ws *ws = to_ws(vma->heap->ws);
   vma->dev_addr = v->reserved;
   vma->bo = NULL;
   pvr_winsys_helper_heap_free(vma);
   vk_free(ws->base.alloc, v);
}

static VkResult g5p_ws_vma_map(struct pvr_winsys_vma *vma, struct pvr_winsys_bo *bo, uint64_t offset, uint64_t size,
                               pvr_dev_addr_t *dev_addr_out)
{
   struct g5p_ws_bo *b = to_bo(bo);
   if (offset + size > b->alloc_size || offset + size < offset)
      return g5p_ws_fail(VK_ERROR_MEMORY_MAP_FAILED,
                         "bind of %" PRIu64 " bytes at offset %" PRIu64 " past the end of a %s BO of %" PRIu64 " bytes",
                         size, offset, b->base.is_imported ? "imported" : b->own ? "own" : "slice", b->alloc_size);
   p_atomic_inc(&b->refs);
   vma->bo = bo;
   vma->bo_offset = offset;
   vma->mapped_size = size;
   vma->dev_addr = PVR_DEV_ADDR(b->va + offset);
   if (dev_addr_out)
      *dev_addr_out = vma->dev_addr;
   return VK_SUCCESS;
}

static void g5p_ws_buffer_destroy(struct pvr_winsys_bo *bo);

static void g5p_ws_vma_unmap(struct pvr_winsys_vma *vma)
{
   struct pvr_winsys_bo *bo = vma->bo;
   vma->dev_addr = to_vma(vma)->reserved;
   vma->bo = NULL;
   vma->bo_offset = 0;
   vma->mapped_size = 0;
   if (bo)
      g5p_ws_buffer_destroy(bo); /* drops the bind's reference */
}

#define g5p_ws_refuse(result, ...)                                                                                 \
   (fprintf(stderr, "apple_g5p render: " __VA_ARGS__), fprintf(stderr, "\n"), vk_errorf(NULL, result, __VA_ARGS__))

/* ---- buffers ---- */

static bool g5p_ws_wait_3d(void)
{
   static int enabled = -1;
   if (enabled < 0) {
      const char *v = getenv("PVR_G5P_WAIT_3D");
      enabled = v ? strcmp(v, "0") != 0 : 1;
   }
   return enabled;
}

static bool g5p_ws_deplog(void)
{
   static int enabled = -1;
   if (enabled < 0) {
      const char *v = getenv("PVR_G5P_DEPLOG");
      enabled = v ? strcmp(v, "0") != 0 : 0;
   }
   return enabled;
}

static int g5p_ws_dep_pending(struct g5p_ws *ws, struct vk_sync *sync)
{
   if (!sync) return -1;
   uint32_t h = vk_sync_as_drm_syncobj(sync)->syncobj;
   return drmSyncobjWait(ws->base.render_fd, &h, 1, 0, 0, NULL) == -ETIME;
}

static bool g5p_ws_waitlog(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = getenv("PVR_G5P_WAITLOG") != NULL;
   return enabled;
}

/* One wait of a VM edit the kernel refused with EBUSY: every live render context's last job, the lock not held */
static int g5p_ws_wait_jobs(struct g5p_ws *ws, int64_t deadline);

/* A VM edit (own BO create, import, destroy) that the kernel refuses with EBUSY while a render job of this file
 * is queued (g5p_render_vm_idle(): "G5P VM_BIND busy=render-idle" in dmesg): wait for the jobs and try again, for
 * G5P_WS_BUSY_NS at most. The lock must not be held. */
#define G5P_WS_VM_RETRY(ws, r, waits, call)                                                                 \
   do {                                                                                                     \
      const int64_t deadline_ = os_time_get_nano() + G5P_WS_BUSY_NS;                                        \
      while (((r) = (call)) == -EBUSY && os_time_get_nano() < deadline_) {                                  \
         const int64_t start_ = os_time_get_nano();                                                         \
         g5p_ws_wait_jobs((ws), deadline_);                                                                 \
         if (g5p_ws_waitlog())                                                                              \
            fprintf(stderr, "G5P wait: vm-edit %lld us\n",                                                  \
                    (long long)((os_time_get_nano() - start_) / 1000));                                     \
         (waits)++;                                                                                         \
      }                                                                                                     \
   } while (0)
/* ---- the arena: committed on demand ---- */

/* One more chunk, at least `need` and `min_bytes` (lock held), at most a kernel BO (64 MiB)
 * and what is left of PVR_G5P_ARENA_MIB. A fresh kernel
 * BO reads as zero, so its pages are not memset (arena_used). The bind is a VM edit: the kernel refuses it (EBUSY)
 * while the context has queued jobs and no job has retired on its binding yet; then it is retried for up to 2 s. */
static int g5p_ws_arena_grow(struct g5p_ws *ws, uint64_t need, uint64_t alignment, uint64_t min_bytes)
{
   if (ws->chunk_count == G5P_WS_MAX_CHUNKS || ws->arena_committed >= ws->arena_max)
      return -ENOSPC;
   /* Bounded demand, not doubling: a transient large allocation must not double every later chunk. But a
    * 2 MiB floor alone capped the arena at 64 chunks of 2 MiB (STK: ~1 MiB allocations failed at 128 MiB), so a
    * chunk is also at least 1/8 of the committed arena: at most 1/8 committed and not yet used, and no more
    * than G5P_WS_CHUNK_GROWTH_MAX of it (1/8 of a 768 MiB arena was a 64 MiB chunk for one slice). The VA, not
    * extra physical memory, provides alignment. */
   uint64_t bytes = MAX3(need, min_bytes,
                         ws->chunk_count ? MIN2(ws->arena_committed / G5P_WS_CHUNK_GROWTH, ws->arena_growth_max) : 0);
   bytes = MIN3(align64(bytes, G5P_WS_PAGE), G5P_WS_CHUNK, ws->arena_max - ws->arena_committed);
   const uint64_t off = align64(ws->arena_va + ws->arena_next, alignment) - ws->arena_va;
   if (bytes < need || off > ws->arena_size || bytes > ws->arena_size - off)
      return -ENOSPC;
   struct g5p_drm_bo *k = &ws->chunks[ws->chunk_count];
   const int64_t deadline = os_time_get_nano() + 2000000000ll;
   int r;
   while ((r = g5p_drm_bo_create_at_cpu(&ws->drm, k, ws->arena_va + off, bytes,
                                        DRM_G5P_BO_READ | DRM_G5P_BO_WRITE | DRM_G5P_BO_EXEC, ws->arena_cpu + off))) {
      /* a failed create unmapped its MAP_FIXED range: put the reservation back */
      mmap(ws->arena_cpu + off, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
      if (r != -EBUSY || os_time_get_nano() >= deadline)
         break;
      os_time_sleep(1000);
   }
   if (r) {
      fprintf(stderr, "apple_g5p winsys: arena chunk of %" PRIu64 " MiB at +%#" PRIx64 ": %s\n", bytes >> 20, off,
              strerror(-r));
      return r;
   }
   util_vma_heap_free(&ws->slices, ws->arena_va + off, bytes);
   ws->arena_committed += bytes;
   ws->arena_next = off + bytes + G5P_WS_PAGE; /* one unbound page: no slice spans two chunks */
   p_atomic_set(&ws->chunk_count, ws->chunk_count + 1);
   if (ws->arena_log || getenv("PVR_G5P_WS_LOG") || getenv("PVR_G5P_PERF_LOG"))
      fprintf(stderr, "apple_g5p winsys: arena chunk pid=%ld fd=%d handle=%u chunk=%u of %" PRIu64 " KiB at +%#" PRIx64 ": %" PRIu64
              " KiB committed of at most %" PRIu64 " KiB\n", (long)getpid(), ws->drm.fd, k->handle, ws->chunk_count, bytes >> 10, off,
              ws->arena_committed >> 10, ws->arena_max >> 10);
   return 0;
}

/* lock held: requested slice bytes versus physically committed chunk bytes, never a device-pinned estimate */
static void g5p_ws_arena_stats(struct g5p_ws *ws, const char *why)
{
   fprintf(stderr, "apple_g5p winsys: arena stats pid=%ld fd=%d why=%s committed_kib=%" PRIu64 " live_kib=%" PRIu64
           " peak_kib=%" PRIu64 " data_live_kib=%" PRIu64 " data_peak_kib=%" PRIu64 " data_bos=%u reclaimed_kib=%" PRIu64 " allocs=%" PRIu64 " reused_pages=%" PRIu64 " chunks=%u max_kib=%" PRIu64 "\n",
           (long)getpid(), ws->drm.fd, why, ws->arena_committed >> 10, ws->arena_live >> 10, ws->arena_peak >> 10, ws->data_live >> 10, ws->data_peak >> 10, ws->data_count, ws->arena_reclaimed >> 10,
           ws->arena_allocs, ws->arena_reused, ws->chunk_count, ws->arena_max >> 10);
}

/* A slice [va, va + size) is handed out (lock held): zero the pages handed out before; a page never handed out
 * is still the kernel's zero page of a fresh chunk */
static void g5p_ws_arena_take(struct g5p_ws *ws, uint64_t va, uint64_t size)
{
   const uint64_t first = (va - ws->arena_va) / G5P_WS_PAGE, n = size / G5P_WS_PAGE;
   for (uint64_t p = first; p < first + n; p++) {
      uint64_t *w = &ws->arena_used[p / 64];
      const uint64_t bit = 1ull << (p % 64);
      if (*w & bit) {
         memset(ws->arena_cpu + p * G5P_WS_PAGE, 0, G5P_WS_PAGE);
         ws->arena_reused++;
      }
      *w |= bit;
   }
   ws->arena_live += size;
   ws->arena_peak = MAX2(ws->arena_peak, ws->arena_live);
   ws->arena_allocs++;
   if (ws->arena_log && (ws->arena_allocs % 256 == 0))
      g5p_ws_arena_stats(ws, "alloc");
}


/* Final BO references are gone. Never move a live slice or release the first chunk.
 * Two seconds empty avoids allocating/unmapping a transient chunk every frame.
 * VM_UNMAP itself proves the VM idle; EBUSY leaves the chunk intact for a later
 * free/allocation. A failed unmap is never reported as reclaimed memory. Lock held.
 * Only the tail is removed, so all remaining chunk indexes and CPU mappings stay fixed. */
static void g5p_ws_arena_trim(struct g5p_ws *ws)
{
   while (ws->chunk_count > 1) {
      const unsigned i = ws->chunk_count - 1;
      struct g5p_drm_bo *k = &ws->chunks[i];
      if (ws->chunk_live[i] || os_time_get_nano() - ws->chunk_idle_ns[i] < G5P_WS_BUSY_NS)
         break;
      const uint64_t va = k->va, size = k->size;
      if (g5p_drm_bo_destroy(&ws->drm, k))
         break;
      /* All of this chunk's slice range is free. Remove it before reusing its VA. */
      bool removed = util_vma_heap_alloc_addr(&ws->slices, va, size);
      assert(removed);
      (void)removed;
      void *reserved = mmap(ws->arena_cpu + va - ws->arena_va, size, PROT_NONE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
      if (reserved == MAP_FAILED)
         fprintf(stderr, "apple_g5p winsys: empty chunk CPU reservation: %s\n", strerror(errno));
      for (uint64_t page = (va - ws->arena_va) / G5P_WS_PAGE;
           page < (va + size - ws->arena_va) / G5P_WS_PAGE; page++)
         ws->arena_used[page / 64] &= ~(1ull << (page % 64));
      ws->arena_committed -= size;
      ws->arena_reclaimed += size;
      ws->arena_next = va - ws->arena_va;
      p_atomic_set(&ws->chunk_count, i);
      if (ws->arena_log)
         g5p_ws_arena_stats(ws, "trim");
   }
}

static struct g5p_ws_bo *g5p_ws_bo_new(struct g5p_ws *ws, uint64_t size)
{
   struct g5p_ws_bo *b = vk_zalloc(ws->base.alloc, sizeof(*b), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!b)
      return NULL;
   b->base.ws = &ws->base;
   b->base.size = size;
   b->refs = 1;
   return b;
}

/* One kernel BO of an own allocation at va; at cpu (MAP_FIXED over the caller's PROT_NONE reservation) when the
 * allocation has several parts. A failed create unmapped its MAP_FIXED range: the reservation is put back before
 * the next try, so that no other mapping can take the hole. */
static int g5p_ws_own_part_create(struct g5p_ws *ws, struct g5p_drm_bo *k, uint64_t va, uint64_t size, uint8_t *cpu)
{
   const int r = g5p_drm_bo_create_at_cpu(&ws->drm, k, va, size, DRM_G5P_BO_READ | DRM_G5P_BO_WRITE, cpu);
   if (r && cpu)
      mmap(cpu, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
   return r;
}

/* Releases the kernel BOs of an own allocation (each UNMAP a VM edit, waiting for the render jobs like the
 * others) and frees the part records after the first; the caller has taken them off own_bos. The VA goes back to
 * the window only if every part was released. Lock not held. */
static void g5p_ws_own_release(struct g5p_ws *ws, struct g5p_ws_bo *b)
{
   bool released = true;
   for (struct g5p_ws_bo *p = b, *next; p; p = next) {
      next = p->next_part;
      int r = 0;
      unsigned waits = 0;
      if (p->kbo.handle)
         G5P_WS_VM_RETRY(ws, r, waits, g5p_drm_bo_destroy(&ws->drm, &p->kbo));
      if (r) {
         released = false;
         fprintf(stderr, "apple_g5p winsys: own BO at %#" PRIx64 " of %" PRIu64 " bytes not released (%s, %u waits): "
                 "its VA stays reserved\n", p->va, p->kbo.size, strerror(-r), waits);
      }
      if (p != b)
         vk_free(ws->base.alloc, p);
   }
   b->next_part = NULL;
   simple_mtx_lock(&ws->lock);
   if (released) {
      util_vma_heap_free(&ws->dedicated, b->va, b->alloc_size);
      if (b->data_only) {
         ws->data_live -= b->alloc_size;
         ws->data_count--;
      }
   }
   simple_mtx_unlock(&ws->lock);
}

/* An own allocation in the dedicated window (a VM edit: the kernel refuses it while a job of the context is
 * queued or running). One kernel BO, or, above the kernel's largest BO (GET_INFO.max_bo_size, 64 MiB), parts of
 * at most that size at consecutive VAs, CPU-mapped at consecutive places of one reservation: one allocation to
 * the driver, with one GPU and one CPU address (glmark2 refract's 85 MiB allocation was refused whole). */
static VkResult g5p_ws_own_create(struct g5p_ws *ws, uint64_t size, uint64_t alignment, unsigned data, struct g5p_ws_bo *b)
{
   const uint64_t alloc = align64(size, G5P_WS_PAGE);
   const uint64_t part_max = ROUND_DOWN_TO(MIN2(ws->drm.info.max_bo_size, G5P_WS_CHUNK), G5P_WS_PAGE);
   if (!part_max)
      return g5p_ws_fail(VK_ERROR_OUT_OF_DEVICE_MEMORY, "own BO of %" PRIu64 " bytes: no kernel BO size", alloc);
   simple_mtx_lock(&ws->lock);
   uint64_t va = util_vma_heap_alloc(&ws->dedicated, alloc, MAX2(alignment, G5P_WS_PAGE));
   simple_mtx_unlock(&ws->lock);
   if (!va)
      return g5p_ws_fail(VK_ERROR_OUT_OF_DEVICE_MEMORY, "own BO of %" PRIu64 " bytes: the %" PRIu64
                         " MiB window for display buffers, imports and data is full", alloc,
                         (uint64_t)(G5P_WS_DEDICATED_SIZE >> 20));
   b->own = true;
   b->va = va;
   b->alloc_size = alloc;
   b->span_va = va;
   b->span_size = alloc;

   uint8_t *cpu = NULL;
   if (alloc > part_max) {
      cpu = mmap(NULL, alloc, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
      if (cpu == MAP_FAILED) {
         simple_mtx_lock(&ws->lock);
         util_vma_heap_free(&ws->dedicated, va, alloc);
         simple_mtx_unlock(&ws->lock);
         return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
   }

   struct g5p_ws_bo *part = b, **link = &b->next_part;
   uint64_t off = 0;
   int r = 0;
   unsigned waits = 0;
   while (off < alloc) {
      const uint64_t bytes = MIN2(alloc - off, part_max);
      if (part != b) {
         part = g5p_ws_bo_new(ws, bytes);
         if (!part) {
            r = -ENOMEM;
            break;
         }
         part->own = true;
         part->va = va + off;
         part->alloc_size = bytes;
         part->span_va = va;
         part->span_size = alloc;
         *link = part;
         link = &part->next_part;
      }
      G5P_WS_VM_RETRY(ws, r, waits, g5p_ws_own_part_create(ws, &part->kbo, va + off, bytes, cpu ? cpu + off : NULL));
      if (r)
         break;
      off += bytes;
      part = NULL; /* the next one is a new record */
   }
   if (r) {
      /* the parts made so far, then the rest of the reservation (each made part unmapped its own range) */
      g5p_ws_own_release(ws, b);
      if (cpu && off < alloc)
         munmap(cpu + off, alloc - off);
      if (r == -EBUSY)
         return g5p_ws_fail(VK_ERROR_OUT_OF_DEVICE_MEMORY,
                            "own BO of %" PRIu64 " bytes refused: the VM stayed busy (%u waits for the render jobs, "
                            "%lld ms): not out of memory", alloc, waits, G5P_WS_BUSY_NS / 1000000);
      return g5p_ws_fail(VK_ERROR_OUT_OF_DEVICE_MEMORY, "own BO of %" PRIu64 " bytes (part at +%#" PRIx64 "): %s",
                         alloc, off, strerror(-r));
   }
   b->cpu = cpu ? cpu : b->kbo.map;
   b->data_only = data != 0; /* counted by g5p_ws_own_list() */
   b->dedicated_image = data == 2;
   for (struct g5p_ws_bo *p = b->next_part; p; p = p->next_part) {
      p->cpu = p->kbo.map;
      g5p_ws_own_list(ws, p);
   }
   g5p_ws_own_list(ws, b);
   return VK_SUCCESS;
}

static VkResult g5p_ws_buffer_create(struct pvr_winsys *ws_, uint64_t size, uint64_t alignment,
                                     enum pvr_winsys_bo_type type, uint32_t flags, struct pvr_winsys_bo **bo_out)
{
   struct g5p_ws *ws = to_ws(ws_);
   struct g5p_ws_bo *b = g5p_ws_bo_new(ws, size);
   if (!b)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult r = VK_SUCCESS;
   if (type == PVR_WINSYS_BO_TYPE_DISPLAY || type == PVR_G5P_BO_TYPE_DATA || type == PVR_G5P_BO_TYPE_IMAGE_DATA) {
      r = g5p_ws_own_create(ws, size, alignment, type == PVR_G5P_BO_TYPE_IMAGE_DATA ? 2 : type == PVR_G5P_BO_TYPE_DATA, b);
   } else {
      const uint64_t alloc = align64(size, G5P_WS_PAGE), align = MAX2(alignment, G5P_WS_PAGE);
      simple_mtx_lock(&ws->lock);
      g5p_ws_arena_trim(ws);
      uint64_t va = util_vma_heap_alloc(&ws->slices, alloc, align);
      /* the first chunk starts at the code base, as an eager one did; the slice fits at its alignment */
      const bool first_chunk = !ws->chunk_count;
      if (!va && g5p_ws_arena_grow(ws, first_chunk ? alloc + align : alloc, first_chunk ? G5P_WS_PAGE : align,
                                   first_chunk ? ws->arena_first : G5P_WS_CHUNK_FLOOR) == 0)
         va = util_vma_heap_alloc(&ws->slices, alloc, align);
      if (va) {
         for (unsigned i = 0; i < ws->chunk_count; i++)
            if (va >= ws->chunks[i].va && va + alloc <= ws->chunks[i].va + ws->chunks[i].size) {
               b->chunk = i;
               ws->chunk_live[i]++;
               ws->chunk_idle_ns[i] = 0;
               break;
            }
         g5p_ws_arena_take(ws, va, alloc);
      }
      simple_mtx_unlock(&ws->lock);
      if (!va) {
         r = g5p_ws_fail( VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "arena full (%" PRIu64 " bytes asked, %" PRIu64 " of at most %" PRIu64
                       " committed; PVR_G5P_ARENA_MIB)", alloc, ws->arena_committed, ws->arena_max);

      } else {
         b->va = va;
         g5p_ws_perf_slice(&ws->perf, va);
         b->cpu = ws->arena_cpu + (va - ws->arena_va);
         b->alloc_size = alloc;
      }
   }
   if (r != VK_SUCCESS) {
      vk_free(ws->base.alloc, b);
      return r;
   }
   (void)flags;
   *bo_out = &b->base;
   return VK_SUCCESS;
}

static VkResult g5p_ws_buffer_create_from_fd(struct pvr_winsys *ws_, int fd, struct pvr_winsys_bo **bo_out)
{
   struct g5p_ws *ws = to_ws(ws_);
   /* A dma-buf this winsys exported (a GBM buffer allocated through Zink, imported back by EGL, as labwc's first
    * frame does): the kernel gives the same GEM handle back (drm_prime_lookup_buf_handle), so it is the same
    * buffer. Return that BO, one more reference: a second BO would bind the object at a second VA and name the
    * handle twice in a job, which the kernel refuses (g5p_render_resolve, duplicate handle: EINVAL). */
   uint32_t handle;
   if (!drmPrimeFDToHandle(ws->drm.fd, fd, &handle)) {
      struct g5p_ws_bo *found = NULL;
      simple_mtx_lock(&ws->lock);
      list_for_each_entry (struct g5p_ws_bo, o, &ws->own_bos, link)
         if (o->kbo.handle == handle) {
            found = o;
            p_atomic_inc(&o->refs);
            break;
         }
      simple_mtx_unlock(&ws->lock);
      if (found) {
         *bo_out = &found->base;
         return VK_SUCCESS;
      }
   }
   const off_t end = lseek(fd, 0, SEEK_END);
   if (end <= 0)
      return vk_error(NULL, VK_ERROR_INVALID_EXTERNAL_HANDLE);
   lseek(fd, 0, SEEK_SET);
   const uint64_t alloc = align64(end, G5P_WS_PAGE);
   struct g5p_ws_bo *b = g5p_ws_bo_new(ws, end);
   if (!b)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);
   simple_mtx_lock(&ws->lock);
   const uint64_t va = util_vma_heap_alloc(&ws->dedicated, alloc, G5P_WS_PAGE);
   simple_mtx_unlock(&ws->lock);
   uint64_t size = 0;
   int r = -ENOSPC;
   unsigned waits = 0;
   if (va)
      G5P_WS_VM_RETRY(ws, r, waits, g5p_drm_bo_import(&ws->drm, &b->kbo, fd, va, &size));
   if (r) {
      if (va) {
         simple_mtx_lock(&ws->lock);
         util_vma_heap_free(&ws->dedicated, va, alloc);
         simple_mtx_unlock(&ws->lock);
      }
      vk_free(ws->base.alloc, b);
      if (r == -ENOSPC)
         return g5p_ws_fail(VK_ERROR_OUT_OF_DEVICE_MEMORY, "import of %" PRIu64 " bytes: the %" PRIu64
                            " MiB window for display buffers and imports is full", alloc, (uint64_t)(G5P_WS_DEDICATED_SIZE >> 20));
      if (r == -EBUSY)
         return g5p_ws_fail(VK_ERROR_OUT_OF_DEVICE_MEMORY,
                            "import of %" PRIu64 " bytes refused: the VM stayed busy (%u waits for the render jobs, "
                            "%lld ms): not a bad handle", alloc, waits, G5P_WS_BUSY_NS / 1000000);
      return g5p_ws_fail(VK_ERROR_INVALID_EXTERNAL_HANDLE, "import of %" PRIu64 " bytes: %s", alloc, strerror(-r));
   }
   b->own = true;
   b->va = va;
   b->cpu = b->kbo.map;
   b->alloc_size = alloc;
   b->base.is_imported = true;
   g5p_ws_own_list(ws, b);
   *bo_out = &b->base;
   return VK_SUCCESS;
}

static void g5p_ws_buffer_destroy(struct pvr_winsys_bo *bo)
{
   struct g5p_ws_bo *b = to_bo(bo);
   struct g5p_ws *ws = to_ws(bo->ws);
   if (p_atomic_dec_return(&b->refs))
      return;
   simple_mtx_lock(&ws->lock);
   if (b->own) {
      /* every part off the job lists first; the UNMAPs are VM edits that wait for the render jobs */
      for (struct g5p_ws_bo *p = b; p; p = p->next_part) {
         list_del(&p->link);
         ws->own_count--;
      }
      simple_mtx_unlock(&ws->lock);
      g5p_ws_own_release(ws, b);
      vk_free(ws->base.alloc, b);
      return;
   } else {
      util_vma_heap_free(&ws->slices, b->va, b->alloc_size);
      ws->arena_live -= b->alloc_size;
      assert(ws->chunk_live[b->chunk]);
      if (!--ws->chunk_live[b->chunk])
         ws->chunk_idle_ns[b->chunk] = os_time_get_nano();
      g5p_ws_arena_trim(ws);
   }
   simple_mtx_unlock(&ws->lock);
   vk_free(ws->base.alloc, b);
}

static VkResult g5p_ws_buffer_get_fd(struct pvr_winsys_bo *bo, int *const fd_out)
{
   struct g5p_ws_bo *b = to_bo(bo);
   if (!b->own) /* a slice is part of an arena BO: it cannot be a dma-buf of its own */
      return vk_error(NULL, VK_ERROR_FEATURE_NOT_PRESENT);
   if (b->next_part) /* several kernel BOs: one dma-buf cannot name them (exportable memory above max_bo_size) */
      return g5p_ws_refuse(VK_ERROR_FEATURE_NOT_PRESENT, "export of a %" PRIu64 " byte allocation of several kernel BOs",
                           b->alloc_size);
   int r = g5p_drm_bo_export(&to_ws(bo->ws)->drm, &b->kbo, fd_out);
   return r ? vk_errorf(NULL, VK_ERROR_OUT_OF_DEVICE_MEMORY, "apple_g5p: export: %s", strerror(-r)) : VK_SUCCESS;
}

/* A placed mapping (VK_EXT_map_memory_placed, which Zink uses): the BO's pages mapped again at addr, MAP_FIXED over
 * the application's reservation. A slice is its arena chunk at the slice's offset; an own BO is itself. Both views
 * are MAP_SHARED of one GEM object, so they stay coherent with each other. Imports are mapped through their dma-buf
 * fd, which the winsys doesn't keep: placed maps of imports are refused. */
static VkResult g5p_ws_buffer_map(struct pvr_winsys_bo *bo, void *addr)
{
   struct g5p_ws_bo *b = to_bo(bo);
   struct g5p_ws *ws = to_ws(bo->ws);
   if (!addr || addr == (void *)b->cpu) {
      bo->map = b->cpu;
      b->placed = false;
      return VK_SUCCESS;
   }
   const struct g5p_drm_bo *k = NULL;
   uint64_t in_bo = 0;
   if (b->own && b->next_part) {
      /* several kernel BOs: each at its place in addr (MAP_FIXED, over the application's reservation) */
      for (struct g5p_ws_bo *p = b; p; p = p->next_part) {
         uint64_t poff;
         if (g5p_drm_bo_mmap_offset(&ws->drm, &p->kbo, &poff))
            return g5p_ws_refuse(VK_ERROR_MEMORY_MAP_FAILED, "placed map at %p: no mmap offset (part at +%#" PRIx64 ")",
                                 addr, p->va - b->va);
         void *m = mmap((uint8_t *)addr + (p->va - b->va), p->kbo.size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                        ws->drm.fd, poff);
         if (m == MAP_FAILED)
            return g5p_ws_refuse(VK_ERROR_MEMORY_MAP_FAILED, "placed map at %p (part at +%#" PRIx64 "): %s", addr,
                                 p->va - b->va, strerror(errno));
      }
      bo->map = addr;
      b->placed = true;
      return VK_SUCCESS;
   }
   if (b->own) {
      if (b->kbo.shared && b->base.is_imported)
         return g5p_ws_refuse(VK_ERROR_MEMORY_MAP_FAILED, "placed map of an imported dma-buf");
      k = &b->kbo;
   } else {
      for (unsigned i = 0; i < ws->chunk_count && !k; i++)
         if (b->va >= ws->chunks[i].va && b->va + b->alloc_size <= ws->chunks[i].va + ws->chunks[i].size) {
            k = &ws->chunks[i];
            in_bo = b->va - ws->chunks[i].va;
         }
   }
   uint64_t off;
   if (!k || g5p_drm_bo_mmap_offset(&ws->drm, k, &off))
      return g5p_ws_refuse(VK_ERROR_MEMORY_MAP_FAILED, "placed map at %p of %" PRIu64 " bytes: no mmap offset (own %d)", addr, b->alloc_size, b->own);
   void *m = mmap(addr, b->alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, ws->drm.fd, off + in_bo);
   if (m == MAP_FAILED)
      return g5p_ws_refuse(VK_ERROR_MEMORY_MAP_FAILED, "placed map at %p of %" PRIu64 " bytes (own %d): %s", addr, b->alloc_size, b->own, strerror(errno));
   bo->map = m;
   b->placed = true;
   return VK_SUCCESS;
}

static VkResult g5p_ws_buffer_unmap(struct pvr_winsys_bo *bo, bool reserve)
{
   struct g5p_ws_bo *b = to_bo(bo);
   if (b->placed) {
      /* reserve (VK_MEMORY_UNMAP_RESERVE_BIT_EXT): keep the range as an inaccessible reservation */
      if (reserve)
         mmap(bo->map, b->alloc_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
      else
         munmap(bo->map, b->alloc_size);
      b->placed = false;
   }
   bo->map = NULL;
   return VK_SUCCESS;
}

/* ---- the objects the kernel builds itself, and the queues ---- */

static VkResult g5p_ws_obj_new(struct g5p_ws *ws, void **out)
{
   struct g5p_ws_obj *o = vk_zalloc(ws->base.alloc, sizeof(*o), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!o)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);
   o->ws = &ws->base;
   *out = o;
   return VK_SUCCESS;
}

static void g5p_ws_obj_free(void *obj)
{
   struct g5p_ws_obj *o = obj;
   vk_free(o->ws->alloc, o);
}

static VkResult g5p_ws_free_list_create(struct pvr_winsys *ws, struct pvr_winsys_vma *free_list_vma,
                                        uint32_t initial_num_pages, uint32_t max_num_pages, uint32_t grow_num_pages,
                                        uint32_t grow_threshold, struct pvr_winsys_free_list *parent_free_list,
                                        struct pvr_winsys_free_list **const free_list_out)
{
   (void)free_list_vma, (void)initial_num_pages, (void)max_num_pages, (void)grow_num_pages, (void)grow_threshold,
      (void)parent_free_list;
   return g5p_ws_obj_new(to_ws(ws), (void **)free_list_out);
}

static void g5p_ws_free_list_destroy(struct pvr_winsys_free_list *free_list) { g5p_ws_obj_free(free_list); }

/* A render target dataset: the size of the kernel render target its renders
 * use. The kernel target (CREATE_RENDER_TARGET: the parameter-buffer geometry,
 * the kernel computes it) belongs to a render context, so it is created at the
 * first submit, on that context. */
/* RT geometry sharing adapted from speed-lane a3417e169f9. */
struct g5p_ws_rt {
   struct pvr_winsys_rt_dataset base;
   uint32_t width, height, samples, layers;
   uint32_t handle, context; /* 0: not created */
   struct g5p_ws_rt *shared; /* dataset reference to a context-owned geometry key */
   unsigned users; /* pool key only: number of live datasets, plus a teardown hold */
   bool in_pool;
   struct list_head pool_link;
};

/* A render context: the kernel's DRM_G5P_CONTEXT_RENDER, created at the first
 * submit (a device that renders nothing creates nothing in the kernel). */
struct g5p_ws_context {
   unsigned users; /* compute and graphics wrappers, under ws->lock */
   uint32_t context; /* 0: not created */
   /* a syncobj holding the fence of the context's last job (0: none yet): the
    * kernel creates a render target only on an idle context, so a new size
    * waits on it (g5p_ws_rt_create_idle()) */
   uint32_t last;
   struct list_head targets; /* kernel datasets, keyed by dimensions/samples/layers */
   uint64_t rt_creates, rt_reuses;
   bool reported_render_enoent; /* ws->lock: one range report per kernel context */
};

/* A wrapper is a reference, not another kernel context/RT-pool owner. */
struct g5p_ws_render_ctx {
   struct pvr_winsys_render_ctx base;
   struct g5p_ws_context *shared;
};

/* A private snapshot holds its fence alive while the winsys lock is dropped. */
static int g5p_ws_wait_jobs(struct g5p_ws *ws, int64_t deadline)
{
   uint32_t snapshot = 0;
   int r = 0;
   simple_mtx_lock(&ws->lock);
   if (ws->context && ws->context->last) {
      if (drmSyncobjCreate(ws->base.render_fd, 0, &snapshot))
         r = -errno;
      else if (drmSyncobjTransfer(ws->base.render_fd, snapshot, 0, ws->context->last, 0, 0))
         r = -errno;
   }
   simple_mtx_unlock(&ws->lock);
   if (snapshot) {
      if (!r) {
         do {
            r = drmSyncobjWait(ws->base.render_fd, &snapshot, 1, deadline,
                               DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL) ? -errno : 0;
         } while (r == -EINTR && os_time_get_nano() < deadline);
      }
      drmSyncobjDestroy(ws->base.render_fd, snapshot);
   }
   /* A first-claim contention can leave the last fence already signalled.
    * Yield even then rather than spinning on admission. */
   os_time_sleep(1000);
   return r;
}

/* Called by both engine wrappers. No kernel allocation until the first submit. */
static struct g5p_ws_context *g5p_ws_context_get(struct g5p_ws *ws)
{
   simple_mtx_lock(&ws->lock);
   struct g5p_ws_context *c = ws->context;
   if (!c) {
      c = vk_zalloc(ws->base.alloc, sizeof(*c), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
      if (c) {
         list_inithead(&c->targets);
         ws->context = c;
      }
   }
   if (c) c->users++;
   simple_mtx_unlock(&ws->lock);
   return c;
}

/* Lock held. Allocate tracking before accepting GPU work; an unused fence is
 * signaled, so a failed first CREATE_CONTEXT cannot leave idle waits blocked. */
static int g5p_ws_context_prepare(struct g5p_ws *ws, struct g5p_ws_context *c)
{
   if (!c->last && drmSyncobjCreate(ws->base.render_fd, DRM_SYNCOBJ_CREATE_SIGNALED, &c->last))
      return -errno;
   return c->context ? 0 : g5p_drm_render_context_create(&ws->drm, &c->context);
}

/* Last wrapper only. Keep ws->lock throughout final teardown so a concurrent
 * get cannot create a second kernel context while this one still owns the VM.
 * Kernel typed-context close drains accepted jobs; failed target destroys are
 * therefore left to that close. Dataset references can safely outlive a pool. */
static void g5p_ws_context_put(struct g5p_ws *ws, struct g5p_ws_context *c)
{
   simple_mtx_lock(&ws->lock);
   assert(c == ws->context && c->users);
   if (--c->users) {
      simple_mtx_unlock(&ws->lock);
      return;
   }
   if (c->last)
      drmSyncobjWait(ws->base.render_fd, &c->last, 1, os_time_get_nano() + G5P_WS_BUSY_NS,
                     DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);
   list_for_each_entry_safe (struct g5p_ws_rt, key, &c->targets, pool_link) {
      list_del(&key->pool_link);
      key->in_pool = false;
      if (key->handle)
         (void)g5p_drm_render_target_destroy(&ws->drm, key->handle);
      key->handle = key->context = 0;
      if (!key->users) vk_free(ws->base.alloc, key);
   }
   if (ws->arena_log)
      fprintf(stderr, "apple_g5p winsys: rt stats pid=%ld context=%u creates=%" PRIu64 " reuses=%" PRIu64 "\n",
              (long)getpid(), c->context, c->rt_creates, c->rt_reuses);
   if (c->context) {
      int r = g5p_drm_render_context_destroy(&ws->drm, c->context);
      if (r) mesa_loge("apple_g5p: final context destroy %u: %s", c->context, strerror(-r));
   }
   if (c->last) drmSyncobjDestroy(ws->base.render_fd, c->last);
   ws->context = NULL;
   vk_free(ws->base.alloc, c);
   simple_mtx_unlock(&ws->lock);
}

static VkResult g5p_ws_rt_dataset_create(struct pvr_winsys *ws, const struct pvr_winsys_rt_dataset_create_info *info,
                                         const struct pvr_device_info *dev_info,
                                         struct pvr_winsys_rt_dataset **const out)
{
   (void)dev_info;
   struct g5p_ws_rt *rt = vk_zalloc(ws->alloc, sizeof(*rt), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!rt)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);
   rt->base.ws = ws;
   rt->width = info->width;
   rt->height = info->height;
   rt->samples = info->samples;
   rt->layers = info->layers;
   *out = &rt->base;
   return VK_SUCCESS;
}

/* Destroy is a VM edit too (g5p_destroy_render_target): another job of the context can still be queued,
 * even if this target's last use has completed. Call without ws->lock; waits acquire it to snapshot fences. */
static int g5p_ws_rt_destroy_idle(struct g5p_ws *ws, struct g5p_ws_rt *rt)
{
   int r;
   unsigned waits = 0;
   G5P_WS_VM_RETRY(ws, r, waits, g5p_drm_render_target_destroy(&ws->drm, rt->handle));
   if (r)
      fprintf(stderr, "apple_g5p render: DESTROY_RENDER_TARGET %u (%ux%u, context %u): %s after %u waits; "
              "target remains live until context teardown\n", rt->handle, rt->width, rt->height,
              rt->context, strerror(-r), waits);
   else if (waits && getenv("PVR_G5P_WS_LOG"))
      fprintf(stderr, "apple_g5p render: DESTROY_RENDER_TARGET %u waited for the context to idle (%u waits): destroyed\n",
              rt->handle, waits);
   return r;
}

/* A key owns geometry scratch only, never attachment pixels (kernel g5p_rt_create
 * and g5p_ru_rt_fill). Jobs on one context are serialized by the kernel render
 * worker; the same key already serves successive renders of a single framebuffer.
 * Share it across identical live framebuffer datasets too. Never across contexts. */
static int g5p_ws_rt_create_idle(struct g5p_ws *, struct g5p_ws_context *, struct g5p_ws_rt *);

/* ws lock held. A context teardown may orphan a key before its last dataset dies. */
static void g5p_ws_rt_drop(struct g5p_ws *ws, struct g5p_ws_rt *rt)
{
   struct g5p_ws_rt *key = rt->shared;
   if (key) {
      assert(key->users);
      if (!--key->users && !key->in_pool)
         vk_free(ws->base.alloc, key);
   }
   rt->shared = NULL;
   rt->handle = rt->context = 0;
}

/* Lock held on entry and return. Queue submits for this context are externally
 * serialized. Dataset destruction only drops users under the same ws lock. */
static int g5p_ws_rt_acquire(struct g5p_ws *ws, struct g5p_ws_context *c, struct g5p_ws_rt *rt)
{
   if (rt->shared) {
      if (rt->shared->context == c->context && rt->shared->handle)
         return 0;
      g5p_ws_rt_drop(ws, rt);
   } else if (rt->handle) {
      if (rt->context == c->context)
         return 0;
      simple_mtx_unlock(&ws->lock);
      int r = g5p_ws_rt_destroy_idle(ws, rt);
      simple_mtx_lock(&ws->lock);
      if (r)
         return r;
      rt->handle = 0;
   }
   const char *cache = getenv("PVR_G5P_RT_CACHE");
   if (cache && !strcmp(cache, "0")) {
      int r = g5p_ws_rt_create_idle(ws, c, rt);
      if (!r) { rt->context = c->context; c->rt_creates++; }
      return r;
   }
   list_for_each_entry (struct g5p_ws_rt, key, &c->targets, pool_link)
      if (key->width == rt->width && key->height == rt->height &&
          key->samples == rt->samples && key->layers == rt->layers) {
         key->users++;
         rt->shared = key;
         rt->handle = key->handle;
         rt->context = key->context;
         c->rt_reuses++;
         return 0;
      }
   /* A miss discards unused geometries before growing the pool. Resizing a
    * window never fills the kernel cap with dead sizes. Failed destroys stay
    * tracked; no key with a live dataset is evicted. */
   list_for_each_entry_safe (struct g5p_ws_rt, key, &c->targets, pool_link) {
      if (key->users)
         continue;
      list_del(&key->pool_link);
      key->in_pool = false;
      simple_mtx_unlock(&ws->lock);
      int r = g5p_ws_rt_destroy_idle(ws, key);
      simple_mtx_lock(&ws->lock);
      if (r) {
         key->in_pool = true;
         list_addtail(&key->pool_link, &c->targets);
         return r;
      }
      vk_free(ws->base.alloc, key);
   }
   struct g5p_ws_rt *key = vk_zalloc(ws->base.alloc, sizeof(*key), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!key)
      return -ENOMEM;
   key->base.ws = &ws->base;
   key->width = rt->width; key->height = rt->height;
   key->samples = rt->samples; key->layers = rt->layers;
   int r = g5p_ws_rt_create_idle(ws, c, key);
   if (r) { vk_free(ws->base.alloc, key); return r; }
   key->context = c->context;
   key->users = 1;
   key->in_pool = true;
   list_addtail(&key->pool_link, &c->targets);
   rt->shared = key;
   rt->handle = key->handle;
   rt->context = key->context;
   c->rt_creates++;
   return 0;
}

static void g5p_ws_rt_dataset_destroy(struct pvr_winsys_rt_dataset *d)
{
   struct g5p_ws_rt *rt = container_of(d, struct g5p_ws_rt, base);
   struct g5p_ws *ws = to_ws(d->ws);
   simple_mtx_lock(&ws->lock);
   if (rt->shared) {
      g5p_ws_rt_drop(ws, rt);
      simple_mtx_unlock(&ws->lock);
   } else {
      simple_mtx_unlock(&ws->lock);
      if (rt->handle)
         g5p_ws_rt_destroy_idle(ws, rt);
   }
   vk_free(d->ws->alloc, rt);
}

static VkResult g5p_ws_render_ctx_create(struct pvr_winsys *ws, struct pvr_winsys_render_ctx_create_info *info,
                                         const struct pvr_device_info *dev_info,
                                         struct pvr_winsys_render_ctx **const out)
{
   (void)info; (void)dev_info;
   struct g5p_ws_render_ctx *c = vk_zalloc(ws->alloc, sizeof(*c), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!c) return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);
   c->base.ws = ws;
   c->shared = g5p_ws_context_get(to_ws(ws));
   if (!c->shared) { vk_free(ws->alloc, c); return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY); }
   *out = &c->base;
   return VK_SUCCESS;
}

static void g5p_ws_render_ctx_destroy(struct pvr_winsys_render_ctx *base)
{
   struct g5p_ws_render_ctx *c = container_of(base, struct g5p_ws_render_ctx, base);
   g5p_ws_context_put(to_ws(base->ws), c->shared);
   vk_free(base->ws->alloc, c);
}

/* CREATE_RENDER_TARGET on the context, with the ws lock held. The kernel
 * edits a render context's VM for a new target only while no job of it is
 * queued (g5p_render_vm_idle(): EBUSY otherwise, "G5P VM_BIND
 * busy=render-idle" in dmesg). A compositor's new framebuffer size comes while
 * its previous frame is in flight, so on EBUSY this waits for the context's
 * last job (c->last) with the lock dropped, and tries again: 2 s at most,
 * the kernel's own job timeout. Other errors, and an EBUSY that outlasts the
 * wait (a recovery, a bad context), are returned as they are. */
static int g5p_ws_rt_create_idle(struct g5p_ws *ws, struct g5p_ws_context *c, struct g5p_ws_rt *rt)
{
   const int64_t deadline = os_time_get_nano() + 2000000000ll;
   unsigned waits = 0;
   int r;

   while ((r = g5p_drm_render_target_create(&ws->drm, c->context, rt->width, rt->height, MAX2(rt->samples, 1),
                                            MAX2(rt->layers, 1), &rt->handle)) == -EBUSY &&
          os_time_get_nano() < deadline) {
      const uint32_t last = c->last;
      simple_mtx_unlock(&ws->lock);
      if (last)
         drmSyncobjWait(ws->base.render_fd, (uint32_t *)&last, 1, deadline, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);
      else
         os_time_sleep(1000);
      /* the kernel drops its queued count before it signals: a short pause after the fence */
      if (++waits > 1)
         os_time_sleep(1000);
      simple_mtx_lock(&ws->lock);
   }
   if (waits && getenv("PVR_G5P_WS_LOG"))
      fprintf(stderr, "apple_g5p render: CREATE_RENDER_TARGET %ux%u waited for the context to idle (%u waits): %s\n",
              rt->width, rt->height, waits, r ? strerror(-r) : "created");
   return r;
}

/* PVR_G5P_WS_LOG: n dwords at a base-relative offset of the arena */
static void g5p_ws_dump(const struct g5p_ws *ws, const char *what, uint32_t off, unsigned n)
{
   fprintf(stderr, "apple_g5p render: %-14s @%#07x:", what, off);
   for (unsigned i = 0; i < n && off + 4 * (i + 1) <= ws->arena_size; i++)
      fprintf(stderr, " %08x", *(const uint32_t *)(ws->arena_cpu + off + 4 * i));
   fprintf(stderr, "\n");
}

/* What a DOUTD data block (source address, control) moves: its bsize dwords
 * (control bits 0..11) at the source, when the source is in the arena */
static void g5p_ws_dump_doutd(const struct g5p_ws *ws, const char *what, uint32_t data_off)
{
   if (data_off + 12 > ws->arena_size)
      return;
   const uint32_t *d = (const uint32_t *)(ws->arena_cpu + data_off);
   const uint64_t src = (uint64_t)d[1] << 32 | d[0];
   if (src >= ws->arena_va && src - ws->arena_va < ws->arena_size)
      g5p_ws_dump(ws, what, (uint32_t)(src - ws->arena_va), d[2] & 0xfff);
}

/* A submit refusal, on stderr too (vk_errorf() is silent in release builds) */

/* Match the kernel's single-range containment rule without adding addresses:
 * malformed or wrapping intervals cannot appear covered. */
static int
g5p_ws_cover(const struct drm_g5p_render_resource *res, unsigned n,
             uint64_t va, uint64_t size)
{
   if (!size || va > UINT64_MAX - size)
      return -1;
   for (unsigned i = 0; i < n; i++)
      if (va >= res[i].va && va - res[i].va <= res[i].size &&
          size <= res[i].size - (va - res[i].va))
         return i;
   return -1;
}

/* No CPU/GPU memory reads on the failure path. The kernel rejected the job;
 * only the immutable submission snapshot is safe and useful to report. */
static void
g5p_ws_render_enoent(struct g5p_ws *ws, const struct g5p_ws_context *c,
                     const struct g5p_ws_rt *rt, const struct drm_g5p_render_pass_v3 *pass,
                     const struct drm_g5p_render_resource *res, unsigned n,
                     const struct drm_g5p_sync *syncs, unsigned nsyncs)
{
   fprintf(stderr, "apple_g5p render ENOENT: pid=%ld drm_fd=%d sync_fd=%d dataset=%p handle=%u context=%u shared=%p shared_handle=%u shared_context=%u shared_users=%u pooled=%u size=%ux%u samples=%u layers=%u last_sync=%u\n",
           (long)getpid(), ws->drm.fd, ws->base.render_fd, (const void *)rt, rt->handle, rt->context,
           (void *)rt->shared, rt->shared ? rt->shared->handle : 0, rt->shared ? rt->shared->context : 0,
           rt->shared ? rt->shared->users : 0, rt->shared && rt->shared->in_pool,
           rt->width, rt->height, rt->samples, rt->layers, c->last);
   const uint64_t bytes = (uint64_t)pass->base.vdm_size + G5P_WS_VDM_GUARD;
   fprintf(stderr, "apple_g5p render ENOENT: context=%u rt=%u stream=%#" PRIx64
           "+%" PRIu64 " (commands=%u guard=%u) range=%d program=%#" PRIx64 "+%u range=%d\n",
           c->context, rt->handle, (uint64_t)pass->base.vdm_va, bytes,
           pass->base.vdm_size, G5P_WS_VDM_GUARD,
           g5p_ws_cover(res, n, pass->base.vdm_va, bytes),
           (uint64_t)pass->ta_program_va, pass->ta_program_size,
           g5p_ws_cover(res, n, pass->ta_program_va, pass->ta_program_size));
   for (unsigned i = 0; i < 2; i++) {
      const uint64_t va[] = {ws->render.pds_base + pass->base.context_pds[i][2],
                             ws->render.pds_base + pass->base.context_pds[i][5],
                             ws->render.pds_base + pass->base.resume_pds[i][2]};
      for (unsigned j = 0; j < ARRAY_SIZE(va); j++)
         fprintf(stderr, "apple_g5p render ENOENT: task[%u][%u]=%#" PRIx64 "+16 range=%d\n",
                 i, j, va[j], g5p_ws_cover(res, n, va[j], 16));
   }
   for (unsigned i = 0; i < n; i++) {
      struct g5p_drm_bo bo = {.handle = res[i].handle};
      uint64_t offset;
      const int lookup = g5p_drm_bo_mmap_offset(&ws->drm, &bo, &offset);
      fprintf(stderr, "apple_g5p render ENOENT: range[%u] handle=%u va=%#" PRIx64
              " size=%#" PRIx64 " offset=%#" PRIx64 " access=%#x lookup=%d\n", i, res[i].handle,
              (uint64_t)res[i].va, (uint64_t)res[i].size, (uint64_t)res[i].offset, res[i].kind, lookup);
   }
   for (unsigned i = 0; i < nsyncs; i++) {
      int fd = -1;
      const int ret = drmSyncobjHandleToFD(ws->base.render_fd, syncs[i].handle, &fd);
      const int lookup = ret ? -errno : 0;
      if (!ret) close(fd);
      fprintf(stderr, "apple_g5p render ENOENT: sync[%u] handle=%u point=%" PRIu64 " flags=%#x lookup=%d\n",
              i, syncs[i].handle, (uint64_t)syncs[i].point, syncs[i].flags, lookup);
   }
}

/* ws->lock must remain held from this snapshot through the submit ioctl.
 * Even a BO the job never uses may be listed by conservative read tracking;
 * its owner may legally free it on another thread. Empty arena chunks can
 * likewise be trimmed. Rebuild after any backpressure wait drops the lock. */
static int
g5p_ws_render_resources(struct g5p_ws *ws, const struct g5p_ws_perf_job *pj,
                         struct drm_g5p_render_resource *res,
                         unsigned *count, unsigned *nchunks)
{
   unsigned nres = 0;
   ws->perf.last_omitted = 0;
   if (ws->chunk_count + ws->own_count > DRM_G5P_RENDER_MAX_RESOURCES)
      return -ENOMEM;
   for (unsigned i = 0; i < ws->chunk_count; i++) {
      uint64_t off, len;
      if (g5p_ws_perf_chunk(&ws->perf, ws->chunks[i].va, ws->chunks[i].size, &off, &len))
         res[nres++] = (struct drm_g5p_render_resource){
            .handle = ws->chunks[i].handle, .offset = off, .va = ws->chunks[i].va + off, .size = len,
            .kind = DRM_G5P_RENDER_ACCESS_READ | DRM_G5P_RENDER_ACCESS_WRITE | DRM_G5P_RENDER_ACCESS_CACHE};
   }
   for (unsigned i = 0; i < nres; i++)
      g5p_ws_perf_resource(&ws->perf, &res[i], "arena");
   *nchunks = nres;
   list_for_each_entry (struct g5p_ws_bo, b, &ws->own_bos, link) {
      /* 0: the job neither writes nor reads it (g5p_ws_perf.h): not listed, no maintenance, no fence */
      const uint32_t kind = g5p_ws_perf_own_kind(&ws->perf, pj, b->va, b->kbo.size, b->kbo.shared);
      if (kind) {
         res[nres++] = (struct drm_g5p_render_resource){
            .handle = b->kbo.handle, .va = b->va, .size = b->kbo.size, .kind = kind};
         g5p_ws_perf_resource(&ws->perf, &res[nres - 1], b->data_only
                                ? (b->dedicated_image ? "image-data" : "data")
                                : (b->kbo.shared ? "shared" : "display"));
      }
   }
   *count = nres;
   return 0;
}

/* One G5P render: the fragment stream holds the TYPED_V3 pass
 * (pvr_arch_g5p_frag_stream); the job reads and writes inside the arena, every
 * chunk listed RW with the kernel's cache maintenance. Own BOs are listed by
 * the pass's read/write tracking, conservatively when tracking is incomplete. */
static VkResult g5p_ws_render_submit(const struct pvr_winsys_render_ctx *ctx,
                                     const struct pvr_winsys_render_submit_info *info,
                                     const struct pvr_device_info *dev_info, struct vk_sync *geom,
                                     struct vk_sync *frag)
{
   (void)dev_info;
   struct g5p_ws_context *c = container_of(ctx, struct g5p_ws_render_ctx, base)->shared;
   struct g5p_ws *ws = to_ws(ctx->ws);
   struct g5p_ws_rt *rt = container_of(info->rt_dataset, struct g5p_ws_rt, base);
   struct drm_g5p_render_pass_v3 pass;
   int r;

   if (info->fragment.fw_stream_len < sizeof(pass) || info->geometry.fw_stream_len)
      return g5p_ws_refuse(VK_ERROR_FEATURE_NOT_PRESENT,
                           "a job without a G5P pass (fragment stream %u bytes, geometry %u): not a G5P render",
                           info->fragment.fw_stream_len, info->geometry.fw_stream_len);
   if (!info->has_fragment_job)
      return g5p_ws_refuse(VK_ERROR_FEATURE_NOT_PRESENT, "a geometry-only job (transform feedback alone is step 8)");
   memcpy(&pass, info->fragment.fw_stream, sizeof(pass));
   if (pass.base.samples > 1 && !pvr_g5p_winsys_msaa(ctx->ws))
      return g5p_ws_refuse(VK_ERROR_FEATURE_NOT_PRESENT,
                           "a %u-sample pass on a kernel without DRM_G5P_CAP_RENDER_MSAA (and TYPED_V3)",
                           pass.base.samples);

   simple_mtx_lock(&ws->lock);
   if ((r = g5p_ws_context_prepare(ws, c))) {
      simple_mtx_unlock(&ws->lock);
      return g5p_ws_refuse(VK_ERROR_INITIALIZATION_FAILED, "CREATE_CONTEXT (render): %s", strerror(-r));
   }
   r = g5p_ws_rt_acquire(ws, c, rt);
   if (r) {
      simple_mtx_unlock(&ws->lock);
      return g5p_ws_refuse(VK_ERROR_OUT_OF_DEVICE_MEMORY, "CREATE_RENDER_TARGET %ux%u: %s",
                           rt->width, rt->height, strerror(-r));
   }
   const uint32_t rt_handle = rt->handle;
   simple_mtx_unlock(&ws->lock);

   /* the arena's chunks, then every own BO (display buffers, imports): the job may write a swapchain image or
    * read an imported buffer. A shared one (imported or exported dma-buf) is IMPLICIT|CACHE: the kernel fences
    * it in its dma_resv, which is what a compositor waits on, and does its cache maintenance. Every own BO is
    * named, not just the job's own: correct, at the cost of fencing buffers the job doesn't touch. */
   struct drm_g5p_render_resource res[DRM_G5P_RENDER_MAX_RESOURCES];
   unsigned nres = 0;
   /* the speed lane's narrowing (g5p_ws_perf.h): each chunk from the arena's lowest slice, and each own BO
    * WRITE only when the job writes it */
   struct g5p_ws_perf_job pj;
   g5p_ws_perf_job(&ws->perf, &pass, info->fragment.fw_stream + sizeof(pass),
                   info->fragment.fw_stream_len - sizeof(pass), &pj);
   unsigned nchunks = 0;
   simple_mtx_lock(&ws->lock);
   r = g5p_ws_render_resources(ws, &pj, res, &nres, &nchunks);
   if (r) {
      simple_mtx_unlock(&ws->lock);
      return g5p_ws_refuse(VK_ERROR_OUT_OF_DEVICE_MEMORY, "too many render resources");
   }

   /* waits (the two stages' waits), then the signal: the kernel takes one per job, so the fragment stage's is
    * signalled and the geometry stage's gets the same fence afterwards (a G5P job is one TA + 3D pass) */
   struct drm_g5p_sync syncs[DRM_G5P_RENDER_MAX_SYNCS];
   unsigned n = 0;
   /* F5b: the fragment stage's wait gates only the 3D: the kernel drops this context's own
    * earlier jobs from it (3D runs in queue order); PVR_G5P_WAIT_3D=0 sends it as a plain input */
   const bool wait_3d = (ws->drm.info.capabilities & DRM_G5P_CAP_SYNC_WAIT_3D) && g5p_ws_wait_3d();
   if (info->geometry.wait)
      syncs[n++] = (struct drm_g5p_sync){.handle = vk_sync_as_drm_syncobj(info->geometry.wait)->syncobj};
   if (info->fragment.wait && info->fragment.wait != info->geometry.wait)
      syncs[n++] = (struct drm_g5p_sync){.handle = vk_sync_as_drm_syncobj(info->fragment.wait)->syncobj,
                                         .flags = wait_3d ? DRM_G5P_SYNC_WAIT_3D : 0};
   struct vk_sync *signal = frag ? frag : geom;
   if (signal)
      syncs[n++] = (struct drm_g5p_sync){.handle = vk_sync_as_drm_syncobj(signal)->syncobj,
                                         .flags = DRM_G5P_SYNC_SIGNAL};

   if (getenv("PVR_G5P_WS_LOG"))
      fprintf(stderr, "apple_g5p render: %ux%u rt %u ctx %u vdm %#" PRIx64 "+%u ta %#" PRIx64 " bgnd %08x %08x %08x %08x "
              "eot %08x %08x %08x, %u syncs, %u resources\n", rt->width, rt->height, rt_handle, c->context,
              (uint64_t)pass.base.vdm_va, pass.base.vdm_size, (uint64_t)pass.ta_program_va, pass.base.background[0],
              pass.base.background[1], pass.base.background[4], pass.base.background[6], pass.base.eot[0],
              pass.base.eot[1], pass.base.eot[2], n, nres);
   if (getenv("PVR_G5P_WS_LOG")) {
      /* the words the pass points at, through the arena's CPU mapping (the step-6 comparison reads these) */
      const uint32_t bg0 = pass.base.background[0], bg_tex = pass.base.background[1], bg_uni = pass.base.background[4];
      const uint64_t ta = pass.ta_program_va - ws->arena_va;
      const uint32_t ta_doutu = ta + 8 <= ws->arena_size ? *(const uint32_t *)(ws->arena_cpu + ta) : 0;
      g5p_ws_dump(ws, "bgnd kick data", bg0, 4);
      g5p_ws_dump(ws, "bgnd kick code", bg0 + 0x20, 2);
      g5p_ws_dump(ws, "bgnd dma code", bg_tex, 2);
      g5p_ws_dump(ws, "bgnd dma data", bg_uni, 4);
      g5p_ws_dump(ws, "eot data", pass.base.eot[0], 8);
      g5p_ws_dump(ws, "eot code", pass.base.eot[1], 3);
      /* what the two DOUTDs move: the clear colours and the PBE block (the MRT comparison reads these) */
      g5p_ws_dump_doutd(ws, "bgnd dma src", bg_uni);
      g5p_ws_dump_doutd(ws, "eot dma src", pass.base.eot[0]);
      g5p_ws_dump(ws, "ta kick data", (uint32_t)ta, 4);
      g5p_ws_dump(ws, "ta kick code", (uint32_t)ta + 0x20, 2);
      g5p_ws_dump(ws, "ta usc", ta_doutu & ~0xfu, 17);
      g5p_ws_dump(ws, "ta ctx", pass.base.ta_pds[1] & ~0xfu, 4);
      g5p_ws_dump(ws, "ta ctx code", pass.base.ta_pds[2], 2);
      /* the control stream, and the PPP block of its tail (pvr's terminate: PPP_STATE, STREAM_TERMINATE) */
      const uint64_t vdm = pass.base.vdm_va - ws->arena_va;
      if (pass.base.vdm_size >= 12 && vdm + pass.base.vdm_size <= ws->arena_size) {
         const uint32_t *t = (const uint32_t *)(ws->arena_cpu + vdm + pass.base.vdm_size - 12);
         const uint64_t ppp = ((uint64_t)(t[0] & 0xff) << 32 | t[1]) - ws->arena_va;
         g5p_ws_dump(ws, "vdm", (uint32_t)vdm, pass.base.vdm_size / 4);
         g5p_ws_dump(ws, "ppp terminate", (uint32_t)ppp, (t[0] >> 8) & 0xff);
      }
   }
   uint64_t job_id = 0;
   const uint64_t t0 = g5p_ws_perf_now();
   uint32_t local_signal = 0;
   if (!signal && drmSyncobjCreate(ws->base.render_fd, 0, &local_signal)) {
      simple_mtx_unlock(&ws->lock);
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   uint32_t sig = signal ? vk_sync_as_drm_syncobj(signal)->syncobj : local_signal;
   if (local_signal)
      syncs[n++] = (struct drm_g5p_sync){.handle = sig, .flags = DRM_G5P_SYNC_SIGNAL};
   const int64_t deadline = os_time_get_nano() + G5P_WS_BUSY_NS;
   unsigned busy_retries = 0;
   const char *operation = "SUBMIT_RENDER_TYPED";
   bool accepted = false;
   for (;;) {
      r = g5p_drm_submit_render_typed(&ws->drm, c->context, rt_handle, &pass, res, nres, syncs, n, &job_id);
      /* release9d rejects queued>=max_queued_jobs before acceptance. Retry
       * only EBUSY: EINTR/EFAULT may follow acceptance and are never replayed.
       * Another queue can publish a dependency while we wait on a retained
       * snapshot of the last accepted fence, so do not hold the winsys lock. */
      if (r != -EBUSY || busy_retries >= 2048 || os_time_get_nano() >= deadline)
         break;
      if (!busy_retries)
         mesa_logw("apple_g5p render: backpressure context=%u rt=%u max_queued=%u; waiting before retry",
                   c->context, rt_handle, ws->render.max_queued_jobs);
      busy_retries++;
      simple_mtx_unlock(&ws->lock);
      const int64_t wait_start = os_time_get_nano();
      r = g5p_ws_wait_jobs(ws, deadline);
      if (g5p_ws_waitlog())
         fprintf(stderr, "G5P wait: backpressure %lld us\n",
                 (long long)((os_time_get_nano() - wait_start) / 1000));
      simple_mtx_lock(&ws->lock);
      if (r) {
         operation = "backpressure fence wait";
         break;
      }
      if (os_time_get_nano() >= deadline) {
         r = -EBUSY;
         break;
      }
      r = g5p_ws_render_resources(ws, &pj, res, &nres, &nchunks);
      if (r) {
         operation = "resource snapshot after backpressure";
         break;
      }
   }
   if (r == -ENOENT && !c->reported_render_enoent) {
      c->reported_render_enoent = true;
      g5p_ws_render_enoent(ws, c, rt, &pass, res, nres, syncs, n);
   }
   if (!r) {
      accepted = true;
      operation = "SYNCOBJ_TRANSFER last";
      if (drmSyncobjTransfer(ws->base.render_fd, c->last, 0, sig, 0, 0)) r = -errno;
      if (g5p_ws_deplog())
         fprintf(stderr, "G5P dep: job=%" PRIu64 " geom=%d frag=%d wait3d=%u\n",
                 job_id, g5p_ws_dep_pending(ws, info->geometry.wait),
                 g5p_ws_dep_pending(ws, info->fragment.wait), wait_3d);
   }
   if (!r && frag && geom) {
      operation = "SYNCOBJ_TRANSFER geometry";
      if (drmSyncobjTransfer(ws->base.render_fd,
          vk_sync_as_drm_syncobj(geom)->syncobj, 0, sig, 0, 0)) r = -errno;
   }
   if (busy_retries)
      mesa_logw("apple_g5p render: backpressure context=%u retries=%u accepted=%u job=%" PRIu64 " result=%d",
                c->context, busy_retries, accepted, job_id, r);
   if (ws->perf.log >= 3)
      fprintf(stderr, "G5P perf submit pid=%ld context=%u job=%" PRIu64 " start_ns=%" PRIu64
              " elapsed_us=%" PRIu64 " width=%u height=%u flags=%#x vdm=%u retries=%u result=%d\n",
              (long)getpid(), c->context, job_id, t0, (g5p_ws_perf_now() - t0) / 1000,
              rt->width, rt->height, pass.base.flags, pass.base.vdm_size, busy_retries, r);
   if (local_signal) drmSyncobjDestroy(ws->base.render_fd, local_signal);
   g5p_ws_perf_account(&ws->perf, res, nres, nchunks, t0);
   simple_mtx_unlock(&ws->lock);
   if (ws->arena_log) {
      simple_mtx_lock(&ws->lock);
      if (++ws->arena_submits % 120 == 0 || t0 >= ws->arena_stats_after) {
         ws->arena_stats_after = t0 + 5000000000ull;
         g5p_ws_arena_stats(ws, "submit");
         fprintf(stderr, "apple_g5p winsys: rt stats pid=%ld context=%u creates=%" PRIu64 " reuses=%" PRIu64 "\n",
                 (long)getpid(), c->context, c->rt_creates, c->rt_reuses);
      }
      simple_mtx_unlock(&ws->lock);
   }
   if (r)
      return g5p_ws_refuse(r == -ENOMEM || r == -EBUSY ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_ERROR_DEVICE_LOST,
                           "%s: %s context=%u rt=%u accepted=%u job=%" PRIu64 " retries=%u",
                           operation, strerror(-r), c->context, rt_handle, accepted, job_id, busy_retries);

   return VK_SUCCESS;
}

#include "g5p_ws_compute.inc"

static VkResult g5p_ws_transfer_ctx_create(struct pvr_winsys *ws,
                                           const struct pvr_winsys_transfer_ctx_create_info *info,
                                           struct pvr_winsys_transfer_ctx **const out)
{
   (void)info;
   return g5p_ws_obj_new(to_ws(ws), (void **)out);
}

static void g5p_ws_transfer_ctx_destroy(struct pvr_winsys_transfer_ctx *ctx) { g5p_ws_obj_free(ctx); }

static VkResult g5p_ws_transfer_submit(const struct pvr_winsys_transfer_ctx *ctx,
                                       const struct pvr_winsys_transfer_submit_info *info,
                                       const struct pvr_device_info *dev_info, struct vk_sync *sync)
{
   (void)ctx, (void)info, (void)dev_info, (void)sync;
   return vk_errorf(NULL, VK_ERROR_FEATURE_NOT_PRESENT, "apple_g5p: transfers run on the CPU");
}

/* No GPU work: the signal sync takes the waits' fences through syncobj
 * transfers, as the powervr winsys's null job does (pvr_drm_job_null.c,
 * Imagination Technologies). */
static VkResult g5p_ws_null_job_submit(struct pvr_winsys *ws, struct vk_sync_wait *waits, uint32_t wait_count,
                                       struct vk_sync_signal *signal)
{
   const int fd = ws->render_fd;
   const uint32_t dst = vk_sync_as_drm_syncobj(signal->sync)->syncobj;
   uint32_t tmp;
   int ret;

   if (wait_count == 1 && waits[0].sync)
      ret = drmSyncobjTransfer(fd, dst, signal->signal_value, vk_sync_as_drm_syncobj(waits[0].sync)->syncobj,
                               waits[0].wait_value, 0);
   else if ((ret = drmSyncobjCreate(fd, wait_count ? 0 : DRM_SYNCOBJ_CREATE_SIGNALED, &tmp)) == 0) {
      for (uint32_t i = 0; i < wait_count && !ret; i++)
         if (waits[i].sync)
            ret = drmSyncobjTransfer(fd, tmp, i + 1, vk_sync_as_drm_syncobj(waits[i].sync)->syncobj,
                                     waits[i].wait_value, 0);
      if (!ret)
         ret = drmSyncobjTransfer(fd, dst, signal->signal_value, tmp, wait_count, 0);
      drmSyncobjDestroy(fd, tmp);
   }
   return ret ? vk_errorf(NULL, VK_ERROR_OUT_OF_DEVICE_MEMORY, "apple_g5p: null job syncobj transfer: %s",
                          strerror(errno))
              : VK_SUCCESS;
}

/* ---- creation ---- */

static void g5p_ws_destroy(struct pvr_winsys *ws_)
{
   struct g5p_ws *ws = to_ws(ws_);
   struct pvr_winsys_heap *heaps[] = {&ws->general_heap, &ws->pds_heap, &ws->usc_heap, &ws->vis_test_heap,
                                      &ws->transfer_frag_heap};
   for (unsigned i = 0; i < ARRAY_SIZE(heaps); i++)
      if (heaps[i]->ws && !pvr_winsys_helper_winsys_heap_finish(heaps[i]))
         mesa_logw("apple_g5p: heap %u still has reservations at destroy", i);
   if (ws->arena_log)
      g5p_ws_arena_stats(ws, "destroy");
   for (unsigned i = 0; i < ws->chunk_count; i++)
      (void)g5p_drm_bo_destroy(&ws->drm, &ws->chunks[i]);
   if (ws->arena_cpu)
      munmap(ws->arena_cpu, ws->arena_size);
   free(ws->arena_used);
   if (ws->vma_heaps_ready) {
      util_vma_heap_finish(&ws->slices);
      util_vma_heap_finish(&ws->dedicated);
   }
   g5p_drm_close(&ws->drm); /* its own dup of render_fd; pvr_winsys_destroy() closes the two fds */
   simple_mtx_destroy(&ws->lock);
   vk_free(ws->base.alloc, ws);
}

static const struct pvr_winsys_ops g5p_ws_ops = {
   .destroy = g5p_ws_destroy,
   .device_info_init = g5p_ws_device_info_init,
   .get_heaps_info = g5p_ws_get_heaps_info,
   .buffer_create = g5p_ws_buffer_create,
   .buffer_create_from_fd = g5p_ws_buffer_create_from_fd,
   .buffer_destroy = g5p_ws_buffer_destroy,
   .buffer_get_fd = g5p_ws_buffer_get_fd,
   .buffer_map = g5p_ws_buffer_map,
   .buffer_unmap = g5p_ws_buffer_unmap,
   .heap_alloc = g5p_ws_heap_alloc,
   .heap_free = g5p_ws_heap_free,
   .vma_map = g5p_ws_vma_map,
   .vma_unmap = g5p_ws_vma_unmap,
   .free_list_create = g5p_ws_free_list_create,
   .free_list_destroy = g5p_ws_free_list_destroy,
   .render_target_dataset_create = g5p_ws_rt_dataset_create,
   .render_target_dataset_destroy = g5p_ws_rt_dataset_destroy,
   .render_ctx_create = g5p_ws_render_ctx_create,
   .render_ctx_destroy = g5p_ws_render_ctx_destroy,
   .render_submit = g5p_ws_render_submit,
   .compute_ctx_create = g5p_ws_compute_ctx_create,
   .compute_ctx_destroy = g5p_ws_compute_ctx_destroy,
   .compute_submit = g5p_ws_compute_submit,
   .transfer_ctx_create = g5p_ws_transfer_ctx_create,
   .transfer_ctx_destroy = g5p_ws_transfer_ctx_destroy,
   .transfer_submit = g5p_ws_transfer_submit,
   .null_job_submit = g5p_ws_null_job_submit,
};

/* Reserve the arena: one CPU reservation at the kernel's code base; its chunks
 * (one READ|WRITE|EXEC BO each, mapped into it) are committed as slices need them. */
static VkResult g5p_ws_arena_init(struct g5p_ws *ws)
{
   const char *e = getenv("PVR_G5P_ARENA_MIB"), *f = getenv("PVR_G5P_ARENA_FIRST_MIB");
   /* The heap (a share of RAM, bounded by what one VM pins): the arena's bound, not its commitment. Zink
    * suballocates everything under 2 MiB here, so a fixed 128 MiB failed STK's ~1 MiB allocations long before
    * the heap was used. */
   const uint64_t ram_share = os_get_gpu_heap_size(OS_GPU_HEAP_SIZE_HEURISTIC, NULL);
   ws->heap_size = ram_share ? MIN2(ram_share, G5P_WS_VM_PIN_MAX) : G5P_WS_ARENA_FALLBACK;
   uint64_t max = e ? strtoull(e, NULL, 10) << 20 : ws->heap_size;
   max = align64(MAX2(max, G5P_WS_PAGE), G5P_WS_PAGE);
   uint64_t first = f ? strtoull(f, NULL, 10) << 20 : G5P_WS_ARENA_FIRST;
   first = MIN2(align64(MAX2(first, G5P_WS_PAGE), G5P_WS_PAGE), max);
   /* the VA: every chunk plus one page after each */
   const uint64_t size = 2 * max + G5P_WS_MAX_CHUNKS * G5P_WS_PAGE;
   if (max > G5P_WS_VM_PIN_MAX || size > ws->render.code_window)
      return g5p_ws_fail(VK_ERROR_INITIALIZATION_FAILED, "arena of %" PRIu64 " MiB is too large",
                       max >> 20);

   const char *growth = getenv("PVR_G5P_ARENA_CHUNK_MIB");
   ws->arena_growth_max = growth ? strtoull(growth, NULL, 10) << 20 : G5P_WS_CHUNK_GROWTH_MAX;
   ws->arena_log = getenv("PVR_G5P_ARENA_LOG") != NULL;
   ws->arena_va = ws->drm.exec.usc_base;
   ws->arena_size = size;
   ws->arena_max = max;
   ws->arena_cpu = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
   if (ws->arena_cpu == MAP_FAILED) {
      ws->arena_cpu = NULL;
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   ws->arena_used = calloc(DIV_ROUND_UP(size / G5P_WS_PAGE, 64), sizeof(uint64_t));
   if (!ws->arena_used)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);
   /* the slices heap starts empty; every chunk adds its range */
   util_vma_heap_init(&ws->slices, ws->arena_va, size);
   util_vma_heap_alloc_addr(&ws->slices, ws->arena_va, size);
   /* No chunk yet: the first slice commits it (g5p_ws_buffer_create()). The winsys exists from
    * vkEnumeratePhysicalDevices on, and a process may keep an instance it never makes a device with
    * (Hyprland holds one beside its renderer's): an eager 2 MiB first chunk was committed memory nothing
    * used. A device's own first slices are a few programs and one-page tables, so the first chunk's floor
    * is 512 KiB; later chunks keep the 2 MiB floor. */
   ws->arena_first = first;
   g5p_ws_perf_init(&ws->perf, ws->arena_va, size, ws->arena_cpu);
   if (ws->arena_log)
      g5p_ws_arena_stats(ws, "init");
   /* above the code window: display BOs and imports are never programs */
   util_vma_heap_init(&ws->dedicated, ws->arena_va + ws->render.code_window, G5P_WS_DEDICATED_SIZE);
   ws->vma_heaps_ready = true;

   /* Every heap names the whole arena: an address pvr_* derives from a heap
    * base is an offset from the code base, which the hardware uses for both
    * USC and PDS programs. */
   g5p_ws_heap_init(ws, &ws->general_heap, ws->arena_va, size);
   g5p_ws_heap_init(ws, &ws->pds_heap, ws->arena_va, size);
   g5p_ws_heap_init(ws, &ws->usc_heap, ws->arena_va, size);
   g5p_ws_heap_init(ws, &ws->vis_test_heap, ws->arena_va, size);
   g5p_ws_heap_init(ws, &ws->transfer_frag_heap, ws->arena_va, size);
   return VK_SUCCESS;
}

VkResult pvr_g5p_winsys_create(int render_fd, int display_fd, const VkAllocationCallbacks *alloc,
                               struct pvr_winsys **ws_out)
{
   struct g5p_ws *ws = vk_zalloc(alloc, sizeof(*ws), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!ws)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);
   simple_mtx_init(&ws->lock, mtx_plain);
   list_inithead(&ws->own_bos);
   ws->base.ops = &g5p_ws_ops;
   ws->base.alloc = alloc;
   ws->base.render_fd = render_fd;
   ws->base.display_fd = display_fd;
   ws->base.page_size = G5P_WS_PAGE;
   ws->base.log2_page_size = util_logbase2(G5P_WS_PAGE);

   /* The transport owns a dup of render_fd (the same open file, so the same
    * GEM and syncobj handles); the caller keeps and closes both fds. */
   const int fd = fcntl(render_fd, F_DUPFD_CLOEXEC, 0);
   int r = fd < 0 ? -errno : g5p_drm_init(&ws->drm, fd); /* closes fd itself on failure */
   const bool opened = !r;
   if (opened)
      r = g5p_drm_render_info(&ws->drm, &ws->render);
   if (opened && !r)
      r = g5p_drm_vm_create(&ws->drm);
   if (r) {
      if (opened)
         g5p_drm_close(&ws->drm);
      simple_mtx_destroy(&ws->lock);
      vk_free(alloc, ws);
      return g5p_ws_fail(VK_ERROR_INCOMPATIBLE_DRIVER, "device, render info or VM: %s", strerror(-r));
   }
   /* The arena sits at the code base and the dedicated window right above the code window; both must lie in the
    * user VA range, outside the kernel's private window and below the highest address the TA/3D read. */
   const struct drm_g5p_get_render_info *ri = &ws->render;
   const uint64_t base = ws->drm.exec.usc_base, top = base + ri->code_window + G5P_WS_DEDICATED_SIZE;
   const char *bad = NULL;
   if (base != ws->drm.exec.pds_base || base != ri->usc_base || base != ri->pds_base)
      bad = "the USC and PDS bases of GET_EXEC_INFO and GET_RENDER_INFO differ";
   else if (!ri->code_window || ri->code_window > G5P_WS_MAX_CODE_WINDOW || ri->code_window % G5P_WS_PAGE)
      bad = "GET_RENDER_INFO.code_window is not a page multiple up to 4 GiB";
   else if (base < ri->user_va_min || base < ws->drm.info.va_start || top > ws->drm.info.va_end)
      bad = "the code and dedicated windows leave the user VA range";
   else if (ri->hw_va_limit && top > ri->hw_va_limit)
      bad = "the dedicated window ends above GET_RENDER_INFO.hw_va_limit";
   else if (ri->private_va_end > base && ri->private_va_start < top)
      bad = "the code or dedicated window overlaps the kernel's private window";
   if (bad) {
      fprintf(stderr, "apple_g5p winsys: base %#" PRIx64 " code window %#" PRIx64 " user VA [%#" PRIx64 ", %#" PRIx64
              ") private [%#" PRIx64 ", %#" PRIx64 ") hw_va_limit %#" PRIx64 "\n", base, (uint64_t)ri->code_window,
              (uint64_t)MAX2(ri->user_va_min, ws->drm.info.va_start), (uint64_t)ws->drm.info.va_end,
              (uint64_t)ri->private_va_start, (uint64_t)ri->private_va_end, (uint64_t)ri->hw_va_limit);
      g5p_ws_destroy(&ws->base);
      return g5p_ws_fail(VK_ERROR_INCOMPATIBLE_DRIVER, "%s", bad);
   }

   /* Old kernels report 64 KiB through this same ABI. Bound host uploads
    * even if a future kernel offers a larger range. */
   if (ri->max_vdm_bytes < DRM_G5P_RENDER_MAX_VDM_BYTES || (ri->max_vdm_bytes & 3)) {
      g5p_ws_destroy(&ws->base);
      return g5p_ws_fail(VK_ERROR_INCOMPATIBLE_DRIVER, "invalid max_vdm_bytes");
   }
   ws->base.max_vdm_bytes = MIN2(ri->max_vdm_bytes, 1u << 20);
   if (getenv("PVR_G5P_WS_LOG"))
      fprintf(stderr, "apple_g5p winsys: max_vdm_bytes=%u\n", ws->base.max_vdm_bytes);

   ws->base.syncobj_type = vk_drm_syncobj_get_type(render_fd);
   ws->base.sync_types[0] = &ws->base.syncobj_type;
   ws->base.sync_types[1] = NULL;

   VkResult result = g5p_ws_arena_init(ws);
   if (result != VK_SUCCESS) {
      g5p_ws_destroy(&ws->base);
      return result;
   }
   if (getenv("PVR_G5P_WS_LOG"))
      fprintf(stderr, "apple_g5p winsys: arena [%#" PRIx64 ", +%" PRIu64 " MiB) in %u BOs, code window %#" PRIx64
              ", dedicated [%#" PRIx64 ", +2 GiB), render profiles %#x\n", ws->arena_va, ws->arena_size >> 20,
              ws->chunk_count, (uint64_t)ri->code_window, (uint64_t)(ws->arena_va + ri->code_window), ri->profiles);
   *ws_out = &ws->base;
   return VK_SUCCESS;
}
