/* SPDX-License-Identifier: MIT */
/*
 * The apple_g5p winsys: struct pvr_winsys_ops over our A9 kernel's typed UAPI.
 * Only g5p_drm_transport.[ch] issues ioctls; this file is the memory model.
 *
 * Memory. Our kernel edits a context's VM only while the context is idle, and
 * pins at most 256 MiB per VM in BOs of at most 64 MiB. So this winsys keeps
 * an arena: a range of GPU VA at the kernel's code base (pds_base ==
 * usc_base), reserved up front for PVR_G5P_ARENA_MIB and committed in
 * READ|WRITE|EXEC chunks as it fills (2 MiB minimum, demand sized), each
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
 * Display BOs and dma-buf imports are their own kernel BOs, bound in a window
 * above the arena (a VM edit: the context must be idle).
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
#include "g5p_ws_perf.h"
#include "pvr_g5p_winsys.h"

#define G5P_WS_PAGE 16384ull
#define G5P_WS_CHUNK (64ull << 20)         /* the kernel's largest BO */
#define G5P_WS_MAX_CHUNKS 64                /* at least 2 MiB per chunk; 128 MiB default, 256 MiB kernel cap */
#define G5P_WS_ARENA_DEFAULT (128ull << 20) /* the most the arena grows to; PVR_G5P_ARENA_MIB overrides */
#define G5P_WS_ARENA_FIRST (2ull << 20)    /* initial demand floor; PVR_G5P_ARENA_FIRST_MIB overrides */
#define G5P_WS_DEDICATED_SIZE (1ull << 30) /* VA window for display BOs and imports */
/* Render programs are 32-bit USC/PDS offsets from the code base, so no code window is larger. The kernel's
 * (GET_RENDER_INFO.code_window) bounds the arena; GET_EXEC_INFO.window_size (0x34000) is the older compute path's
 * and does not bound render programs. */
#define G5P_WS_MAX_CODE_WINDOW (1ull << 32)
/* The pvr_* heaps are address bookkeeping only (see the top of this file): heap_alloc reserves a range, and
 * vma_map then gives the bound BO's own address. So their bookkeeping range is not the arena's size: every bind of
 * a heap reserves its whole size there, own BOs (display buffers, imports: the 1 GiB window) included, and an
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
   uint64_t arena_max;      /* the most the chunks may total (PVR_G5P_ARENA_MIB) */
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
   struct util_vma_heap slices;

   /* own kernel BOs: display buffers and imports; every live one is listed in own_bos, because a job names
    * them all as its resources (a swapchain image or an imported buffer the job may write or read) */
   struct util_vma_heap dedicated;
   struct list_head own_bos;
   unsigned own_count;
   bool vma_heaps_ready; /* slices and dedicated are initialized */

   /* every live render context (g5p_ws_render_ctx::link): a VM edit waits for all of their last jobs */
   struct list_head render_ctxs;

   struct pvr_winsys_heap general_heap, pds_heap, usc_heap, vis_test_heap, transfer_frag_heap;
   struct g5p_ws_perf perf; /* the jobs' resource lists (g5p_ws_perf.h) */
};

struct g5p_ws_bo {
   struct pvr_winsys_bo base;
   uint64_t va; /* GPU address of the slice or of the own BO */
   uint8_t *cpu;
   uint64_t alloc_size;
   unsigned chunk; /* stable until this slice is destroyed */
   bool own; /* an own kernel BO (display buffer or import) */
   struct g5p_drm_bo kbo;
   struct list_head link; /* in g5p_ws::own_bos, while own */
   bool placed; /* bo->map is a placed mapping (VK_EXT_map_memory_placed), not cpu */
   int refs;
};

static void g5p_ws_own_list(struct g5p_ws *ws, struct g5p_ws_bo *b)
{
   simple_mtx_lock(&ws->lock);
   list_addtail(&b->link, &ws->own_bos);
   ws->own_count++;
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

/* One wait of a VM edit the kernel refused with EBUSY: every live render context's last job, the lock not held */
static void g5p_ws_wait_jobs(struct g5p_ws *ws, int64_t deadline);

/* A VM edit (own BO create, import, destroy) that the kernel refuses with EBUSY while a render job of this file
 * is queued (g5p_render_vm_idle(): "G5P VM_BIND busy=render-idle" in dmesg): wait for the jobs and try again, for
 * G5P_WS_BUSY_NS at most. The lock must not be held. */
#define G5P_WS_VM_RETRY(ws, r, waits, call)                                                                     \
   do {                                                                                                         \
      const int64_t deadline_ = os_time_get_nano() + G5P_WS_BUSY_NS;                                           \
      while (((r) = (call)) == -EBUSY && os_time_get_nano() < deadline_) {                                      \
         g5p_ws_wait_jobs((ws), deadline_);                                                                     \
         (waits)++;                                                                                             \
      }                                                                                                         \
   } while (0)
/* ---- the arena: committed on demand ---- */

/* One more chunk, at least `need` bytes (lock held), at most a kernel BO (64 MiB)
 * and what is left of PVR_G5P_ARENA_MIB. A fresh kernel
 * BO reads as zero, so its pages are not memset (arena_used). The bind is a VM edit: the kernel refuses it (EBUSY)
 * while the context has queued jobs and no job has retired on its binding yet; then it is retried for up to 2 s. */
static int g5p_ws_arena_grow(struct g5p_ws *ws, uint64_t need, uint64_t alignment)
{
   if (ws->chunk_count == G5P_WS_MAX_CHUNKS || ws->arena_committed >= ws->arena_max)
      return -ENOSPC;
   /* Bounded demand, not geometric commitment: a transient large allocation must not
    * double every later chunk. The VA, not extra physical memory, provides alignment. */
   uint64_t bytes = MAX2(need, 2ull << 20);
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
   if (getenv("PVR_G5P_WS_LOG") || getenv("PVR_G5P_PERF_LOG"))
      fprintf(stderr, "apple_g5p winsys: arena chunk %u of %" PRIu64 " KiB at +%#" PRIx64 ": %" PRIu64
              " KiB committed of at most %" PRIu64 " KiB\n", ws->chunk_count, bytes >> 10, off,
              ws->arena_committed >> 10, ws->arena_max >> 10);
   return 0;
}

/* lock held: requested slice bytes versus physically committed chunk bytes, never a device-pinned estimate */
static void g5p_ws_arena_stats(struct g5p_ws *ws, const char *why)
{
   fprintf(stderr, "apple_g5p winsys: arena stats pid=%ld fd=%d why=%s committed_kib=%" PRIu64 " live_kib=%" PRIu64
           " peak_kib=%" PRIu64 " reclaimed_kib=%" PRIu64 " allocs=%" PRIu64 " reused_pages=%" PRIu64 " chunks=%u max_kib=%" PRIu64 "\n",
           (long)getpid(), ws->drm.fd, why, ws->arena_committed >> 10, ws->arena_live >> 10, ws->arena_peak >> 10, ws->arena_reclaimed >> 10,
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

/* An own kernel BO in the dedicated window (a VM edit: the kernel refuses it
 * while a job of the context is queued or running). */
static VkResult g5p_ws_own_create(struct g5p_ws *ws, uint64_t size, struct g5p_ws_bo *b)
{
   const uint64_t alloc = align64(size, G5P_WS_PAGE);
   simple_mtx_lock(&ws->lock);
   uint64_t va = util_vma_heap_alloc(&ws->dedicated, alloc, G5P_WS_PAGE);
   simple_mtx_unlock(&ws->lock);
   if (!va)
      return g5p_ws_fail(VK_ERROR_OUT_OF_DEVICE_MEMORY, "own BO of %" PRIu64 " bytes: the %" PRIu64
                         " MiB window for display buffers and imports is full", alloc, (uint64_t)(G5P_WS_DEDICATED_SIZE >> 20));
   int r;
   unsigned waits = 0;
   G5P_WS_VM_RETRY(ws, r, waits,
                   g5p_drm_bo_create(&ws->drm, &b->kbo, va, alloc, DRM_G5P_BO_READ | DRM_G5P_BO_WRITE));
   if (r) {
      simple_mtx_lock(&ws->lock);
      util_vma_heap_free(&ws->dedicated, va, alloc);
      simple_mtx_unlock(&ws->lock);
      if (r == -EBUSY)
         return g5p_ws_fail(VK_ERROR_OUT_OF_DEVICE_MEMORY,
                            "own BO of %" PRIu64 " bytes refused: the VM stayed busy (%u waits for the render jobs, "
                            "%lld ms): not out of memory", alloc, waits, G5P_WS_BUSY_NS / 1000000);
      return g5p_ws_fail(VK_ERROR_OUT_OF_DEVICE_MEMORY, "own BO of %" PRIu64 " bytes: %s", alloc, strerror(-r));
   }
   b->own = true;
   b->va = va;
   b->cpu = b->kbo.map;
   b->alloc_size = alloc;
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
   if (type == PVR_WINSYS_BO_TYPE_DISPLAY) {
      r = g5p_ws_own_create(ws, size, b);
   } else {
      const uint64_t alloc = align64(size, G5P_WS_PAGE), align = MAX2(alignment, G5P_WS_PAGE);
      simple_mtx_lock(&ws->lock);
      g5p_ws_arena_trim(ws);
      uint64_t va = util_vma_heap_alloc(&ws->slices, alloc, align);
      if (!va && g5p_ws_arena_grow(ws, alloc, align) == 0)
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
      const uint64_t va = b->va;
      list_del(&b->link);
      ws->own_count--;
      simple_mtx_unlock(&ws->lock);
      /* the UNMAP is a VM edit: it waits for the render jobs like the others (the BO is off the job lists) */
      int r;
      unsigned waits = 0;
      G5P_WS_VM_RETRY(ws, r, waits, g5p_drm_bo_destroy(&ws->drm, &b->kbo));
      simple_mtx_lock(&ws->lock);
      if (r)
         fprintf(stderr, "apple_g5p winsys: own BO at %#" PRIx64 " of %" PRIu64 " bytes not released (%s, %u waits): "
                 "its VA stays reserved\n", va, b->alloc_size, strerror(-r), waits);
      else
         util_vma_heap_free(&ws->dedicated, va, b->alloc_size);
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
struct g5p_ws_rt {
   struct pvr_winsys_rt_dataset base;
   uint32_t width, height, samples, layers;
   uint32_t handle, context; /* 0: not created */
};

/* A render context: the kernel's DRM_G5P_CONTEXT_RENDER, created at the first
 * submit (a device that renders nothing creates nothing in the kernel). */
struct g5p_ws_render_ctx {
   struct pvr_winsys_render_ctx base;
   uint32_t context; /* 0: not created */
   /* a syncobj holding the fence of the context's last job (0: none yet): the
    * kernel creates a render target only on an idle context, so a new size
    * waits on it (g5p_ws_rt_create_idle()) */
   uint32_t last;
   struct g5p_ws_rt *idle_rt; /* one completed dataset, same context and geometry only */
   uint64_t rt_creates, rt_reuses;
   struct list_head link; /* in g5p_ws::render_ctxs */
};

static void g5p_ws_wait_jobs(struct g5p_ws *ws, int64_t deadline)
{
   uint32_t syncs[16];
   unsigned n = 0;

   simple_mtx_lock(&ws->lock);
   list_for_each_entry (struct g5p_ws_render_ctx, c, &ws->render_ctxs, link)
      if (c->last && n < ARRAY_SIZE(syncs))
         syncs[n++] = c->last;
   simple_mtx_unlock(&ws->lock);
   if (n)
      drmSyncobjWait(ws->base.render_fd, syncs, n, deadline, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);
   /* the kernel drops its queued count before it signals, and a job may have been queued since: a short pause */
   os_time_sleep(1000);
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

static void g5p_ws_rt_dataset_destroy(struct pvr_winsys_rt_dataset *d)
{
   struct g5p_ws_rt *rt = container_of(d, struct g5p_ws_rt, base);
   struct g5p_ws *ws = to_ws(d->ws);
   /* Kernel g5p_rt_create stores context + dimensions/sample geometry, no image
    * or framebuffer attachment. Keep one idle dataset per context for reuse.
    * Never share a live dataset across contexts, or keep an unbounded size cache. */
   struct g5p_ws_rt *old = NULL;
   bool cached = false;
   const char *cache = getenv("PVR_G5P_RT_CACHE");
   simple_mtx_lock(&ws->lock);
   if (rt->handle && (!cache || strcmp(cache, "0")))
      list_for_each_entry (struct g5p_ws_render_ctx, c, &ws->render_ctxs, link) {
         if (c->context == rt->context) {
            old = c->idle_rt;
            c->idle_rt = rt;
            cached = true;
            break;
         }
      }
   simple_mtx_unlock(&ws->lock);
   if (old) {
      g5p_ws_rt_destroy_idle(ws, old);
      vk_free(d->ws->alloc, old);
   }
   if (!cached) {
      if (rt->handle)
         g5p_ws_rt_destroy_idle(ws, rt);
      vk_free(d->ws->alloc, rt);
   }
}

static VkResult g5p_ws_render_ctx_create(struct pvr_winsys *ws, struct pvr_winsys_render_ctx_create_info *info,
                                         const struct pvr_device_info *dev_info,
                                         struct pvr_winsys_render_ctx **const out)
{
   (void)info, (void)dev_info;
   struct g5p_ws_render_ctx *c = vk_zalloc(ws->alloc, sizeof(*c), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!c)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);
   c->base.ws = ws;
   simple_mtx_lock(&to_ws(ws)->lock);
   list_addtail(&c->link, &to_ws(ws)->render_ctxs);
   simple_mtx_unlock(&to_ws(ws)->lock);
   *out = &c->base;
   return VK_SUCCESS;
}

static void g5p_ws_render_ctx_destroy(struct pvr_winsys_render_ctx *ctx)
{
   struct g5p_ws_render_ctx *c = container_of(ctx, struct g5p_ws_render_ctx, base);
   simple_mtx_lock(&to_ws(ctx->ws)->lock);
   list_del(&c->link);
   simple_mtx_unlock(&to_ws(ctx->ws)->lock);
   if (c->idle_rt) {
      g5p_ws_rt_destroy_idle(to_ws(ctx->ws), c->idle_rt);
      vk_free(ctx->ws->alloc, c->idle_rt);
   }
   if (to_ws(ctx->ws)->arena_log)
      fprintf(stderr, "apple_g5p winsys: rt stats pid=%ld context=%u creates=%" PRIu64 " reuses=%" PRIu64 "\n",
              (long)getpid(), c->context, c->rt_creates, c->rt_reuses);
   if (c->context)
      g5p_drm_render_context_destroy(&to_ws(ctx->ws)->drm, c->context);
   if (c->last)
      drmSyncobjDestroy(ctx->ws->render_fd, c->last);
   vk_free(ctx->ws->alloc, c);
}

/* CREATE_RENDER_TARGET on the context, with the ws lock held. The kernel
 * edits a render context's VM for a new target only while no job of it is
 * queued (g5p_render_vm_idle(): EBUSY otherwise, "G5P VM_BIND
 * busy=render-idle" in dmesg). A compositor's new framebuffer size comes while
 * its previous frame is in flight, so on EBUSY this waits for the context's
 * last job (c->last) with the lock dropped, and tries again: 2 s at most,
 * the kernel's own job timeout. Other errors, and an EBUSY that outlasts the
 * wait (a recovery, a bad context), are returned as they are. */
static int g5p_ws_rt_create_idle(struct g5p_ws *ws, struct g5p_ws_render_ctx *c, struct g5p_ws_rt *rt)
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

/* One G5P render: the fragment stream holds the TYPED_V3 pass
 * (pvr_arch_g5p_frag_stream); the job reads and writes inside the arena, every
 * chunk listed RW with the kernel's cache maintenance. Display BOs and imports
 * are not listed yet, so a pass that reaches one is refused by the kernel. */
static VkResult g5p_ws_render_submit(const struct pvr_winsys_render_ctx *ctx,
                                     const struct pvr_winsys_render_submit_info *info,
                                     const struct pvr_device_info *dev_info, struct vk_sync *geom,
                                     struct vk_sync *frag)
{
   (void)dev_info;
   struct g5p_ws_render_ctx *c = container_of((struct pvr_winsys_render_ctx *)ctx, struct g5p_ws_render_ctx, base);
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
   if (!c->context && (r = g5p_drm_render_context_create(&ws->drm, &c->context))) {
      simple_mtx_unlock(&ws->lock);
      return g5p_ws_refuse(VK_ERROR_INITIALIZATION_FAILED, "CREATE_CONTEXT (render): %s", strerror(-r));
   }
   if (rt->handle && rt->context != c->context) {
      simple_mtx_unlock(&ws->lock);
      r = g5p_ws_rt_destroy_idle(ws, rt);
      if (r)
         return g5p_ws_refuse(VK_ERROR_DEVICE_LOST,
                              "DESTROY_RENDER_TARGET %u before context reassignment: %s", rt->handle, strerror(-r));
      simple_mtx_lock(&ws->lock);
      rt->handle = 0;
   }
   if (!rt->handle && c->idle_rt) {
      struct g5p_ws_rt *idle = c->idle_rt;
      c->idle_rt = NULL;
      if (idle->width == rt->width && idle->height == rt->height &&
          idle->samples == rt->samples && idle->layers == rt->layers) {
         rt->handle = idle->handle;
         rt->context = idle->context;
         c->rt_reuses++;
      } else {
         /* Evict before creating another geometry: the kernel has a 32-target cap. */
         simple_mtx_unlock(&ws->lock);
         r = g5p_ws_rt_destroy_idle(ws, idle);
         simple_mtx_lock(&ws->lock);
         if (r) {
            c->idle_rt = idle;
            simple_mtx_unlock(&ws->lock);
            return g5p_ws_refuse(VK_ERROR_DEVICE_LOST, "cached target eviction: %s", strerror(-r));
         }
      }
      vk_free(ws->base.alloc, idle);
   }
   if (!rt->handle) {
      r = g5p_ws_rt_create_idle(ws, c, rt);
      if (r) {
         simple_mtx_unlock(&ws->lock);
         return g5p_ws_refuse(VK_ERROR_OUT_OF_DEVICE_MEMORY, "CREATE_RENDER_TARGET %ux%u: %s", rt->width,
                              rt->height, strerror(-r));
      }
      rt->context = c->context;
      c->rt_creates++;
   }
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
   simple_mtx_lock(&ws->lock);
   for (unsigned i = 0; i < ws->chunk_count; i++) {
      uint64_t off, len;
      if (g5p_ws_perf_chunk(&ws->perf, ws->chunks[i].va, ws->chunks[i].size, &off, &len))
         res[nres++] = (struct drm_g5p_render_resource){
            .handle = ws->chunks[i].handle, .offset = off, .va = ws->chunks[i].va + off, .size = len,
            .kind = DRM_G5P_RENDER_ACCESS_READ | DRM_G5P_RENDER_ACCESS_WRITE | DRM_G5P_RENDER_ACCESS_CACHE};
   }
   const unsigned nchunks = nres;
   if (ws->chunk_count + ws->own_count > ARRAY_SIZE(res)) {
      simple_mtx_unlock(&ws->lock);
      return g5p_ws_refuse(VK_ERROR_OUT_OF_DEVICE_MEMORY, "%u own BOs: more than a job's %u resources",
                           ws->own_count, (unsigned)ARRAY_SIZE(res) - ws->chunk_count);
   }
   list_for_each_entry (struct g5p_ws_bo, b, &ws->own_bos, link) {
      /* 0: the job neither writes nor reads it (g5p_ws_perf.h): not listed, no maintenance, no fence */
      const uint32_t kind = g5p_ws_perf_own_kind(&ws->perf, &pj, b->va, b->kbo.size, b->kbo.shared);
      if (kind)
         res[nres++] = (struct drm_g5p_render_resource){
            .handle = b->kbo.handle, .va = b->va, .size = b->kbo.size, .kind = kind};
   }
   simple_mtx_unlock(&ws->lock);

   /* waits (the two stages' waits), then the signal: the kernel takes one per job, so the fragment stage's is
    * signalled and the geometry stage's gets the same fence afterwards (a G5P job is one TA + 3D pass) */
   struct drm_g5p_sync syncs[DRM_G5P_RENDER_MAX_SYNCS];
   unsigned n = 0;
   struct vk_sync *waits[] = {info->geometry.wait, info->fragment.wait};
   for (unsigned i = 0; i < ARRAY_SIZE(waits); i++)
      if (waits[i])
         syncs[n++] = (struct drm_g5p_sync){.handle = vk_sync_as_drm_syncobj(waits[i])->syncobj};
   struct vk_sync *signal = frag ? frag : geom;
   if (signal)
      syncs[n++] = (struct drm_g5p_sync){.handle = vk_sync_as_drm_syncobj(signal)->syncobj,
                                         .flags = DRM_G5P_SYNC_SIGNAL};

   if (getenv("PVR_G5P_WS_LOG"))
      fprintf(stderr, "apple_g5p render: %ux%u rt %u ctx %u vdm %#" PRIx64 "+%u ta %#" PRIx64 " bgnd %08x %08x %08x %08x "
              "eot %08x %08x %08x, %u syncs, %u resources\n", rt->width, rt->height, rt->handle, c->context,
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
   r = g5p_drm_submit_render_typed(&ws->drm, c->context, rt->handle, &pass, res, nres, syncs, n,
                                   &job_id);
   g5p_ws_perf_account(&ws->perf, res, nres, nchunks, t0);
   if (r)
      return g5p_ws_refuse(r == -ENOMEM ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_ERROR_DEVICE_LOST,
                           "SUBMIT_RENDER_TYPED: %s (the kernel's reason is in dmesg)", strerror(-r));
   /* the job's fence into the geometry stage's syncobj too (SYNCOBJ_TRANSFER, DRM core) */
   if (frag && geom &&
       drmSyncobjTransfer(ws->base.render_fd, vk_sync_as_drm_syncobj(geom)->syncobj, 0,
                          vk_sync_as_drm_syncobj(frag)->syncobj, 0, 0))
      return g5p_ws_refuse(VK_ERROR_DEVICE_LOST, "the geometry signal (SYNCOBJ_TRANSFER): %s", strerror(errno));
   /* the job's fence into the context's last-job syncobj, for the next new render target. A job without a
    * signal leaves the old one; g5p_ws_rt_create_idle() then also retries on a timer. */
   simple_mtx_lock(&ws->lock);
   if (signal && (c->last || !drmSyncobjCreate(ws->base.render_fd, 0, &c->last)))
      drmSyncobjTransfer(ws->base.render_fd, c->last, 0, vk_sync_as_drm_syncobj(signal)->syncobj, 0, 0);
   simple_mtx_unlock(&ws->lock);
   return VK_SUCCESS;
}

static VkResult g5p_ws_compute_ctx_create(struct pvr_winsys *ws, const struct pvr_winsys_compute_ctx_create_info *info,
                                          const struct pvr_device_info *dev_info,
                                          struct pvr_winsys_compute_ctx **const out)
{
   (void)info, (void)dev_info;
   return g5p_ws_obj_new(to_ws(ws), (void **)out);
}

static void g5p_ws_compute_ctx_destroy(struct pvr_winsys_compute_ctx *ctx) { g5p_ws_obj_free(ctx); }

static VkResult g5p_ws_compute_submit(const struct pvr_winsys_compute_ctx *ctx,
                                      const struct pvr_winsys_compute_submit_info *info,
                                      const struct pvr_device_info *dev_info, struct vk_sync *sync)
{
   (void)ctx, (void)info, (void)dev_info, (void)sync;
   return vk_errorf(NULL, VK_ERROR_FEATURE_NOT_PRESENT, "apple_g5p: no compute yet");
}

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

/* Commit the arena: one CPU reservation, then one READ|WRITE|EXEC BO per
 * chunk, mapped into it and bound at the kernel's code base. */
static VkResult g5p_ws_arena_init(struct g5p_ws *ws)
{
   const char *e = getenv("PVR_G5P_ARENA_MIB"), *f = getenv("PVR_G5P_ARENA_FIRST_MIB");
   uint64_t max = e ? strtoull(e, NULL, 10) << 20 : G5P_WS_ARENA_DEFAULT;
   max = align64(MAX2(max, G5P_WS_PAGE), G5P_WS_PAGE);
   uint64_t first = f ? strtoull(f, NULL, 10) << 20 : G5P_WS_ARENA_FIRST;
   first = MIN2(align64(MAX2(first, G5P_WS_PAGE), G5P_WS_PAGE), max);
   /* the VA: every chunk plus one page after each */
   const uint64_t size = 2 * max + G5P_WS_MAX_CHUNKS * G5P_WS_PAGE;
   if (max > (256ull << 20) || size > ws->render.code_window)
      return g5p_ws_fail(VK_ERROR_INITIALIZATION_FAILED, "arena of %" PRIu64 " MiB is too large",
                       max >> 20);

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
   int r = g5p_ws_arena_grow(ws, first, G5P_WS_PAGE);
   if (r)
      return g5p_ws_fail(VK_ERROR_INITIALIZATION_FAILED, "the arena's first chunk at %#" PRIx64 ": %s",
                       ws->arena_va, strerror(-r));
   g5p_ws_perf_init(&ws->perf, ws->arena_va, size, ws->arena_cpu);
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
   list_inithead(&ws->render_ctxs);
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
              ", dedicated [%#" PRIx64 ", +1 GiB), render profiles %#x\n", ws->arena_va, ws->arena_size >> 20,
              ws->chunk_count, (uint64_t)ri->code_window, (uint64_t)(ws->arena_va + ri->code_window), ri->profiles);
   *ws_out = &ws->base;
   return VK_SUCCESS;
}
