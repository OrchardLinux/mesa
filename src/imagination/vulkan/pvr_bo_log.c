/* SPDX-License-Identifier: MIT */
#include "pvr_bo_log.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if defined(__linux__)
#   include <sys/prctl.h>
#endif

#include "pvr_bo.h"
#include "pvr_buffer.h"
#include "pvr_debug.h"
#include "pvr_device.h"
#include "pvr_image.h"
#include "util/hash_table.h"
#include "util/list.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_math.h"
#include "util/log.h"
#include "util/u_debug.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_util.h"
#include "winsys/pvr_winsys.h"

struct pvr_bo_log_entry {
   const char *kind;
   const char *tag;
   const char *heap;
   uint64_t size;
   bool imported;
   bool display;
   char thread[17];
};

/* Keep object records separately so padding totals include every live binding,
 * while each allocation's live line shows only its four largest objects.
 * Descriptions never dereference Vulkan objects at exit.
 */
struct pvr_bo_log_binding {
   const void *memory;
   uint64_t size;
   uint64_t tight;
   bool image;
   char description[512];
};

struct pvr_bo_log {
   struct list_head link;
   simple_mtx_t mutex;
   struct pvr_device *device;
   uint64_t last_summary_ns;
   struct hash_table *entries;
   struct hash_table *bindings;
   uint64_t min_size;
   uint64_t app_live;
   uint64_t app_count;
   uint64_t import_live;
   uint64_t display_live;
   uint64_t drv_live;
   uint64_t drv_count;
   uint64_t peak;
   uint64_t failed;
   uint64_t tracking_failed;
};

static struct list_head logs = { &logs, &logs };
static simple_mtx_t logs_mutex = SIMPLE_MTX_INITIALIZER;
DEBUG_GET_ONCE_NUM_OPTION(pvr_bo_log_min_kib, "PVR_BO_LOG_MIN_KIB", 64)
DEBUG_GET_ONCE_NUM_OPTION(pvr_bo_log_period_s, "PVR_BO_LOG_PERIOD_S", 0)

static const char *thread_name(char buf[17])
{
   buf[0] = 0;
#if defined(PR_GET_NAME)
   if (prctl(PR_GET_NAME, buf) != 0)
      buf[0] = 0;
   buf[16] = 0;
#endif
   for (char *c = buf; *c; c++) {
      if (*c == ' ')
         *c = '_';
   }
   return buf[0] ? buf : "?";
}

static void
dedicated_word(const VkMemoryAllocateInfo *info, char *buf, size_t n)
{
   const VkMemoryDedicatedAllocateInfo *ded =
      vk_find_struct_const(info->pNext, MEMORY_DEDICATED_ALLOCATE_INFO);

   if (ded && ded->image != VK_NULL_HANDLE)
      snprintf(buf, n, "image:%#" PRIx64, (uint64_t)(uintptr_t)ded->image);
   else if (ded && ded->buffer != VK_NULL_HANDLE)
      snprintf(buf, n, "buffer:%#" PRIx64, (uint64_t)(uintptr_t)ded->buffer);
   else
      snprintf(buf, n, "none");
}

static const char *heap_name(const struct pvr_device *device,
                             const struct pvr_winsys_heap *heap)
{
   if (heap == device->heaps.general_heap)
      return "general";
   if (heap == device->heaps.pds_heap)
      return "pds";
   if (heap == device->heaps.rgn_hdr_heap)
      return "rgn_hdr";
   if (heap == device->heaps.transfer_frag_heap)
      return "transfer_frag";
   if (heap == device->heaps.usc_heap)
      return "usc";
   if (heap == device->heaps.vis_test_heap)
      return "vis_test";
   return "unknown";
}

static uint64_t live_bytes(const struct pvr_bo_log *log)
{
   return log->app_live - log->import_live + log->drv_live;
}

struct pvr_bo_log_tag {
   const char *kind;
   const char *tag;
   const char *heap;
   uint64_t size;
   uint64_t count;
};

static int compare_tags(const void *a, const void *b)
{
   const struct pvr_bo_log_tag *ta = a, *tb = b;
   return (ta->size < tb->size) - (ta->size > tb->size);
}

static void summary_locked(struct pvr_bo_log *log, const char *why, bool tables)
{
   const uint64_t created = live_bytes(log) - log->display_live;
   const uint64_t ws_created = pvr_winsys_counted_bytes(0);

   mesa_logi("pvr bo summary why=%s pid=%ld app_live_kib=%" PRIu64
           " app_count=%" PRIu64 " import_kib=%" PRIu64 " drv_live_kib=%" PRIu64
           " drv_count=%" PRIu64 " created_live_kib=%" PRIu64
           " ws_created_kib=%" PRIu64 " ws_display_kib=%" PRIu64
           " untagged_kib=%" PRId64 " peak_kib=%" PRIu64 " failed=%" PRIu64
           " tracking_failed=%" PRIu64 "\n",
           why,
           (long)getpid(),
           log->app_live >> 10,
           log->app_count,
           log->import_live >> 10,
           log->drv_live >> 10,
           log->drv_count,
           created >> 10,
           ws_created >> 10,
           pvr_winsys_counted_bytes(1) >> 10,
           ((int64_t)ws_created - (int64_t)created) / 1024,
           log->peak >> 10,
           log->failed,
           log->tracking_failed);
   if (!tables) {

      return;
   }

   const struct {
      const char *name;
      struct pvr_suballocator *allocator;
   } suballocators[] = {
      { "general", &log->device->suballoc_general },
      { "pds", &log->device->suballoc_pds },
      { "transfer", &log->device->suballoc_transfer },
      { "usc", &log->device->suballoc_usc },
      { "vis_test", &log->device->suballoc_vis_test },
   };
   for (unsigned i = 0; i < ARRAY_SIZE(suballocators); i++) {
      const struct pvr_suballocator *a = suballocators[i].allocator;
      mesa_logi("pvr bo summary-suballoc name=%s bos=%u bo_kib=%" PRIu64
                " live_kib=%" PRIu64 " peak_bo_kib=%" PRIu64,
                suballocators[i].name,
                p_atomic_read(&a->live_bos),
                p_atomic_read(&a->live_bo_bytes) >> 10,
                p_atomic_read(&a->live_bytes) >> 10,
                p_atomic_read(&a->peak_bo_bytes) >> 10);
   }

   uint64_t images = 0, total = 0, tight = 0;
   hash_table_foreach (log->bindings, h) {
      const struct pvr_bo_log_binding *b = h->data;
      if (b->image) {
         images++;
         total += b->size;
         tight += b->tight;
      }
   }
   mesa_logi("pvr bo summary-pad images=%" PRIu64 " total_kib=%" PRIu64
           " tight_kib=%" PRIu64 " pad_kib=%" PRId64 "\n",
           images,
           total >> 10,
           tight >> 10,
           ((int64_t)total - (int64_t)tight) / 1024);

   struct pvr_bo_log_tag *tags =
      calloc(MAX2(log->entries->entries, 1), sizeof(*tags));
   unsigned num_tags = 0;
   const struct pvr_bo_log_entry *largest[40] = { 0 };
   const void *keys[40] = { 0 };

   hash_table_foreach (log->entries, h) {
      const struct pvr_bo_log_entry *e = h->data;
      if (tags) {
         unsigned t;
         for (t = 0; t < num_tags; t++) {
            if (!strcmp(tags[t].kind, e->kind) &&
                !strcmp(tags[t].tag, e->tag) && !strcmp(tags[t].heap, e->heap))
               break;
         }
         if (t == num_tags) {
            tags[t].kind = e->kind;
            tags[t].tag = e->tag;
            tags[t].heap = e->heap;
            num_tags++;
         }
         tags[t].size += e->size;
         tags[t].count++;
      }
      for (unsigned i = 0; i < ARRAY_SIZE(largest); i++) {
         if (!largest[i] || e->size > largest[i]->size) {
            for (unsigned j = ARRAY_SIZE(largest) - 1; j > i; j--) {
               largest[j] = largest[j - 1];
               keys[j] = keys[j - 1];
            }
            largest[i] = e;
            keys[i] = h->key;
            break;
         }
      }
   }
   if (tags) {
      qsort(tags, num_tags, sizeof(*tags), compare_tags);
      for (unsigned i = 0; i < num_tags; i++) {
         mesa_logi("pvr bo summary-tag kind=%s tag=%s heap=%s count=%" PRIu64
                 " kib=%" PRIu64 "\n",
                 tags[i].kind,
                 tags[i].tag,
                 tags[i].heap,
                 tags[i].count,
                 tags[i].size >> 10);
      }
      free(tags);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(largest) && largest[i]; i++) {
      const struct pvr_bo_log_entry *e = largest[i];
      const struct pvr_bo_log_binding *bound[4] = { 0 };
      unsigned count = 0;
      hash_table_foreach (log->bindings, h) {
         const struct pvr_bo_log_binding *b = h->data;
         if (b->memory != keys[i])
            continue;
         count++;
         for (unsigned j = 0; j < ARRAY_SIZE(bound); j++) {
            if (!bound[j] || b->size > bound[j]->size) {
               for (unsigned k = ARRAY_SIZE(bound) - 1; k > j; k--)
                  bound[k] = bound[k - 1];
               bound[j] = b;
               break;
            }
         }
      }
      char descriptions[4096] = { 0 };
      size_t used = 0;
      for (unsigned j = 0; j < ARRAY_SIZE(bound) && bound[j]; j++) {
         int n = snprintf(descriptions + used, sizeof(descriptions) - used,
                          "%s%s", j ? ";" : "", bound[j]->description);
         used += MIN2(MAX2(n, 0), sizeof(descriptions) - used - 1);
      }
      if (count > ARRAY_SIZE(bound))
         snprintf(descriptions + used, sizeof(descriptions) - used,
                  ";+%u", count - (unsigned)ARRAY_SIZE(bound));
      mesa_logi("pvr bo live kind=%s size_kib=%" PRIu64
                " tag=%s heap=%s thread=%s bound=%s",
                e->kind, e->size >> 10, e->tag, e->heap, e->thread, descriptions);
   }

}

static void summary_period_locked(struct pvr_bo_log *log)
{
   if (!PVR_IS_DEBUG_SET(BO_LOG))
      return;

   const int64_t period_s = debug_get_option_pvr_bo_log_period_s();
   if (period_s <= 0)
      return;

   const uint64_t now = os_time_get_nano();
   if ((now - log->last_summary_ns) / 1000000000ULL >= (uint64_t)period_s) {
      summary_locked(log, "period", true);
      log->last_summary_ns = now;
   }
}

void pvr_bo_log_summary(struct pvr_device *device, const char *why)
{
   struct pvr_bo_log *log = device->bo_log;
   simple_mtx_lock(&log->mutex);
   summary_locked(log, why, true);
   simple_mtx_unlock(&log->mutex);
}

void pvr_bo_log_device_init(struct pvr_device *device)
{
   if (!PVR_IS_DEBUG_SET(BO_LOG))
      return;

   struct pvr_bo_log *log = calloc(1, sizeof(*log));
   if (!log)
      return;
   log->entries = _mesa_pointer_hash_table_create(NULL);
   log->bindings = _mesa_pointer_hash_table_create(NULL);
   if (!log->entries || !log->bindings) {
      _mesa_hash_table_destroy(log->entries, NULL);
      _mesa_hash_table_destroy(log->bindings, NULL);
      free(log);
      return;
   }
   log->device = device;
   log->last_summary_ns = os_time_get_nano();
   log->min_size = MAX2(0, debug_get_option_pvr_bo_log_min_kib());
   log->min_size = MIN2(log->min_size, UINT64_MAX / 1024) * 1024;
   simple_mtx_init(&log->mutex, mtx_plain);
   simple_mtx_lock(&logs_mutex);
   list_addtail(&log->link, &logs);
   device->bo_log = log;
   simple_mtx_unlock(&logs_mutex);
}

static void free_record(struct hash_entry *entry)
{
   free(entry->data);
}

void pvr_bo_log_device_finish(struct pvr_device *device)
{
   struct pvr_bo_log *log = device->bo_log;
   simple_mtx_lock(&logs_mutex);
   list_del(&log->link);
   device->bo_log = NULL;
   simple_mtx_unlock(&logs_mutex);
   _mesa_hash_table_destroy(log->bindings, free_record);
   _mesa_hash_table_destroy(log->entries, free_record);
   simple_mtx_destroy(&log->mutex);
   free(log);
}

static void record_alloc(struct pvr_bo_log *log,
                         const void *key,
                         const struct pvr_bo_log_entry *entry,
                         const char *detail,
                         VkResult result)
{
   simple_mtx_lock(&log->mutex);
   if (result == VK_SUCCESS) {
      struct pvr_bo_log_entry *copy = malloc(sizeof(*copy));
      if (copy) {
         *copy = *entry;
         if (!_mesa_hash_table_insert(log->entries, key, copy)) {
            free(copy);
            log->tracking_failed++;
         }
      } else {
         log->tracking_failed++;
      }
      if (!strcmp(entry->kind, "app")) {
         log->app_live += entry->size;
         log->app_count++;
         if (entry->imported)
            log->import_live += entry->size;
         else if (entry->display)
            log->display_live += entry->size;
      } else {
         log->drv_live += entry->size;
         log->drv_count++;
      }
   } else {
      log->failed++;
   }
   if (entry->size >= log->min_size || result != VK_SUCCESS) {
      mesa_logi("pvr bo %s kind=%s %s=%p size_kib=%" PRIu64
              " tag=%s heap=%s %s result=%s live_kib=%" PRIu64
              " thread=%s pid=%ld\n",
              result == VK_SUCCESS ? "alloc" : "alloc-failed",
              entry->kind,
              !strcmp(entry->kind, "app") ? "mem" : "bo",
              key,
              entry->size >> 10,
              entry->tag,
              entry->heap,
              detail,
              vk_Result_to_str(result),
              live_bytes(log) >> 10,
              entry->thread,
              (long)getpid());

   }
   uint64_t old_peak = log->peak;
   log->peak = MAX2(log->peak, live_bytes(log));
   if ((log->peak >> 26) > (old_peak >> 26))
      summary_locked(log, "peak", false);
   if (result != VK_SUCCESS && log->failed <= 3)
      summary_locked(log, "alloc-failed", true);
   summary_period_locked(log);
   simple_mtx_unlock(&log->mutex);
}

void pvr_bo_log_driver_alloc(struct pvr_device *device,
                             const char *tag,
                             struct pvr_winsys_heap *heap,
                             uint64_t size,
                             uint64_t flags,
                             const struct pvr_bo *bo,
                             VkResult result)
{
   struct pvr_bo_log_entry entry = {
      .kind = "drv",
      .tag = tag,
      .heap = heap_name(device, heap),
      .size = bo ? bo->bo->size : size,
   };
   char detail[48];
   snprintf(detail, sizeof(detail), "flags=%#" PRIx64, flags);
   const char *name = thread_name(entry.thread);
   if (name != entry.thread)
      snprintf(entry.thread, sizeof(entry.thread), "%s", name);
   record_alloc(device->bo_log, bo, &entry, detail, result);
}

void pvr_bo_log_memory_alloc(struct pvr_device *device,
                             const VkMemoryAllocateInfo *info,
                             const struct pvr_device_memory *mem,
                             VkResult result)
{
   const VkImportMemoryFdInfoKHR *imp =
      vk_find_struct_const(info->pNext, IMPORT_MEMORY_FD_INFO_KHR);
   const VkMemoryDedicatedAllocateInfo *ded =
      vk_find_struct_const(info->pNext, MEMORY_DEDICATED_ALLOCATE_INFO);
   const bool wsi =
      vk_find_struct_const(info->pNext, WSI_MEMORY_ALLOCATE_INFO_MESA) != NULL;
   struct pvr_bo_log_entry entry = {
      .kind = "app",
      .tag = "app",
      .heap = "general",
      .size = mem ? mem->bo->size : info->allocationSize,
      .imported = imp && imp->handleType,
      .display = wsi && device->ws->display_fd >= 0,
   };
   if (wsi)
      entry.tag = "wsi";
   else if (entry.imported)
      entry.tag = "import";
   else if (vk_find_struct_const(info->pNext, EXPORT_MEMORY_ALLOCATE_INFO))
      entry.tag = "export";
   else if (ded && ded->image != VK_NULL_HANDLE)
      entry.tag = "dedicated-image";
   else if (ded && ded->buffer != VK_NULL_HANDLE)
      entry.tag = "dedicated-buffer";

   char dedicated[48], detail[96];
   dedicated_word(info, dedicated, sizeof(dedicated));
   snprintf(detail,
            sizeof(detail),
            "type=%u dedicated=%s display=%u",
            info->memoryTypeIndex,
            dedicated,
            entry.display);
   const char *name = thread_name(entry.thread);
   if (name != entry.thread)
      snprintf(entry.thread, sizeof(entry.thread), "%s", name);
   record_alloc(device->bo_log, mem, &entry, detail, result);
}

void pvr_bo_log_free(struct pvr_device *device, const void *key)
{
   struct pvr_bo_log *log = device->bo_log;
   simple_mtx_lock(&log->mutex);
   struct hash_entry *h = _mesa_hash_table_search(log->entries, key);
   if (h) {
      struct pvr_bo_log_entry *e = h->data;
      if (!strcmp(e->kind, "app")) {
         log->app_live -= e->size;
         log->app_count--;
         if (e->imported)
            log->import_live -= e->size;
         else if (e->display)
            log->display_live -= e->size;
      } else {
         log->drv_live -= e->size;
         log->drv_count--;
      }
      if (e->size >= log->min_size) {
         mesa_logi("pvr bo free kind=%s %s=%p size_kib=%" PRIu64
                 " live_kib=%" PRIu64 "\n",
                 e->kind,
                 !strcmp(e->kind, "app") ? "mem" : "bo",
                 key,
                 e->size >> 10,
                 live_bytes(log) >> 10);

      }
      hash_table_foreach (log->bindings, b) {
         struct pvr_bo_log_binding *binding = b->data;
         if (binding->memory == key) {
            free(binding);
            _mesa_hash_table_remove(log->bindings, b);
         }
      }
      free(e);
      _mesa_hash_table_remove(log->entries, h);
   }
   summary_period_locked(log);
   simple_mtx_unlock(&log->mutex);
}

static void record_binding(struct pvr_bo_log *log,
                           const void *object,
                           const struct pvr_bo_log_binding *binding)
{
   struct hash_entry *old = _mesa_hash_table_search(log->bindings, object);
   if (old) {
      free(old->data);
      _mesa_hash_table_remove(log->bindings, old);
   }
   struct pvr_bo_log_binding *copy = malloc(sizeof(*copy));
   if (copy) {
      *copy = *binding;
      if (!_mesa_hash_table_insert(log->bindings, object, copy)) {
         free(copy);
         log->tracking_failed++;
      }
   } else {
      log->tracking_failed++;
   }
}

void pvr_bo_log_unbind(struct pvr_device *device, const void *object)
{
   struct pvr_bo_log *log = device->bo_log;
   simple_mtx_lock(&log->mutex);
   struct hash_entry *h = _mesa_hash_table_search(log->bindings, object);
   if (h) {
      free(h->data);
      _mesa_hash_table_remove(log->bindings, h);
   }
   simple_mtx_unlock(&log->mutex);
}

static const char *layout_name(enum pvr_memlayout layout)
{
   switch (layout) {
   case PVR_MEMLAYOUT_LINEAR:
      return "LINEAR";
   case PVR_MEMLAYOUT_TWIDDLED:
      return "TWIDDLED";
   case PVR_MEMLAYOUT_3DTWIDDLED:
      return "3DTWIDDLED";
   default:
      return "UNKNOWN";
   }
}

void pvr_bo_log_image(struct pvr_device *device,
                      const struct pvr_image *image,
                      const struct pvr_device_memory *mem,
                      uint64_t offset)
{
   struct pvr_bo_log *log = device->bo_log;
   struct pvr_bo_log_binding binding = {
      .memory = mem,
      .size = image->total_size,
      .image = true,
   };
   const uint32_t bw = vk_format_get_blockwidth(image->vk.format);
   const uint32_t bh = vk_format_get_blockheight(image->vk.format);
   const uint32_t bs = vk_format_get_blocksize(image->vk.format);
   for (unsigned l = 0; l < image->vk.mip_levels; l++) {
      binding.tight +=
         (uint64_t)DIV_ROUND_UP(u_minify(image->vk.extent.width, l), bw) *
         DIV_ROUND_UP(u_minify(image->vk.extent.height, l), bh) * bs *
         u_minify(image->vk.extent.depth, l) * image->vk.array_layers *
         image->vk.samples;
   }
   snprintf(binding.description,
            sizeof(binding.description),
            "image=%p off=%" PRIu64
            " fmt=%s extent=%ux%ux%u levels=%u layers=%u"
            " samples=%u usage=%#" PRIx64 " flags=%#" PRIx64
            " tiling=%s layout=%s phys=%ux%u"
            " size_kib=%" PRIu64 " tight_kib=%" PRIu64 " pad_kib=%" PRId64,
            (const void *)image,
            offset,
            vk_Format_to_str(image->vk.format),
            image->vk.extent.width,
            image->vk.extent.height,
            image->vk.extent.depth,
            image->vk.mip_levels,
            image->vk.array_layers,
            image->vk.samples,
            (uint64_t)image->vk.usage,
            image->vk.create_flags,
            image->vk.tiling == VK_IMAGE_TILING_OPTIMAL ? "optimal" : "linear",
            layout_name(image->memlayout),
            image->planes[0].physical_extent.width,
            image->planes[0].physical_extent.height,
            binding.size >> 10,
            binding.tight >> 10,
            ((int64_t)binding.size - (int64_t)binding.tight) / 1024);
   simple_mtx_lock(&log->mutex);
   record_binding(log, image, &binding);
   if (binding.size >= log->min_size) {
      mesa_logi("pvr bo bind-image mem=%p %s\n",
              (const void *)mem,
              binding.description);

   }
   simple_mtx_unlock(&log->mutex);
}

void pvr_bo_log_buffer(struct pvr_device *device,
                       const struct pvr_buffer *buffer,
                       const struct pvr_device_memory *mem,
                       uint64_t offset)
{
   struct pvr_bo_log *log = device->bo_log;
   struct pvr_bo_log_binding binding = {
      .memory = mem,
      .size = buffer->vk.size,
   };
   /* Match GetBufferMemoryRequirements2, including its read-ahead padding. */
   uint64_t req = buffer->vk.size;
   if (req % device->ws->page_size == 0 ||
       req % device->ws->page_size >
          device->ws->page_size - PVR_BUFFER_MEMORY_PADDING_SIZE)
      req += PVR_BUFFER_MEMORY_PADDING_SIZE;
   req = align64(req, buffer->alignment);
   snprintf(binding.description,
            sizeof(binding.description),
            "buffer=%p off=%" PRIu64 " size_kib=%" PRIu64 " req_kib=%" PRIu64
            " usage=%#" PRIx64,
            (const void *)buffer,
            offset,
            binding.size >> 10,
            req >> 10,
            (uint64_t)buffer->vk.usage);
   simple_mtx_lock(&log->mutex);
   record_binding(log, buffer, &binding);
   if (binding.size >= log->min_size) {
      mesa_logi("pvr bo bind-buffer mem=%p %s\n",
              (const void *)mem,
              binding.description);

   }
   simple_mtx_unlock(&log->mutex);
}
