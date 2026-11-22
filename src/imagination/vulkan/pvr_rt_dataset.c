/*
 * Copyright © 2023 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#include "pvr_rt_dataset.h"

#include "pvr_device.h"
#include "pvr_free_list.h"
#include "pvr_g5p.h"
#include "pvr_physical_device.h"

static void pvr_rt_rgn_headers_data_fini(struct pvr_rt_dataset *rt_dataset)
{
   for (uint32_t i = 0; i < ARRAY_SIZE(rt_dataset->rt_datas); i++)
      rt_dataset->rt_datas[i].rgn_headers_dev_addr = PVR_DEV_ADDR_INVALID;

   pvr_bo_free(rt_dataset->device, rt_dataset->rgn_headers_bo);
   rt_dataset->rgn_headers_bo = NULL;
}

void pvr_rt_mlist_data_fini(struct pvr_rt_dataset *rt_dataset)
{
   for (uint32_t i = 0; i < ARRAY_SIZE(rt_dataset->rt_datas); i++)
      rt_dataset->rt_datas[i].mlist_dev_addr = PVR_DEV_ADDR_INVALID;

   pvr_bo_free(rt_dataset->device, rt_dataset->mlist_bo);
   rt_dataset->mlist_bo = NULL;
}

void pvr_rt_mta_data_fini(struct pvr_rt_dataset *rt_dataset)
{
   if (rt_dataset->mta_bo == NULL)
      return;

   for (uint32_t i = 0; i < ARRAY_SIZE(rt_dataset->rt_datas); i++)
      rt_dataset->rt_datas[i].mta_dev_addr = PVR_DEV_ADDR_INVALID;

   pvr_bo_free(rt_dataset->device, rt_dataset->mta_bo);
   rt_dataset->mta_bo = NULL;
}

void pvr_rt_datas_fini(struct pvr_rt_dataset *rt_dataset)
{
   pvr_rt_rgn_headers_data_fini(rt_dataset);
   pvr_rt_mlist_data_fini(rt_dataset);
   pvr_rt_mta_data_fini(rt_dataset);
}

void pvr_rt_tpc_data_fini(struct pvr_rt_dataset *rt_dataset)
{
   pvr_bo_free(rt_dataset->device, rt_dataset->tpc_bo);
   rt_dataset->tpc_bo = NULL;
}

void pvr_rt_vheap_rtc_data_fini(struct pvr_rt_dataset *rt_dataset)
{
   rt_dataset->rtc_dev_addr = PVR_DEV_ADDR_INVALID;

   pvr_bo_free(rt_dataset->device, rt_dataset->vheap_rtc_bo);
   rt_dataset->vheap_rtc_bo = NULL;
}

void pvr_render_target_dataset_destroy(struct pvr_rt_dataset *rt_dataset)
{
   struct pvr_device *device = rt_dataset->device;

   device->ws->ops->render_target_dataset_destroy(rt_dataset->ws_rt_dataset);

   pvr_rt_datas_fini(rt_dataset);
   pvr_rt_tpc_data_fini(rt_dataset);
   pvr_rt_vheap_rtc_data_fini(rt_dataset);

   /* G9P: the device's list, shared by every dataset */
   if (rt_dataset->local_free_list != device->g9p.free_list &&
       !(device->g5p && rt_dataset->local_free_list == device->g5p->free_list))
      pvr_free_list_destroy(rt_dataset->local_free_list);

   vk_free(&device->vk.alloc, rt_dataset);
}

/* Cache demand is passes per command buffer, not distinct sizes: each pass
 * takes its own dataset. For example, glmark2 terrain has four 256x256 passes
 * and three 2048x1536 passes, among about 24-29 passes including mip renders.
 * Zink returns a batch's datasets together when it resets the finished command
 * buffer (pvr_cmd_buffer_free_sub_cmd -> pvr_rstate_entry_remove), before
 * recording the next batch. Keep enough for a batch, with room for a second
 * burst. Matching remains exact in width, height, samples and layers.
 *
 * A9 datasets share the device's local free list and own five small BOs,
 * rounded to at least 16 KiB each. Only region headers scale with size
 * (3072 tiles * 5 bytes at 2048x1536), so roughly 0.1-0.2 MiB per dataset
 * makes the count limit bind first (about 13 MiB for 64). Rogue datasets own
 * a local free list too, so bound their total BO bytes as well. Keep a single
 * oversized dataset to avoid repeated allocation for a large target.
 * Only released datasets are cached; rendering one or two sizes does not
 * itself fill the cache.
 */
#define PVR_RT_DATASET_CACHE_MAX 64
#define PVR_RT_DATASET_CACHE_BYTES (32ull << 20)
/* The A8 (1 GB shared by the CPU and GPU): an A8 dataset owns the region-header ring (16 frames of the target's
 * headers, 128 KiB at 640x1136) besides the small BOs, so 8 MiB holds some 30 full-screen datasets. */
#define PVR_RT_DATASET_CACHE_BYTES_A8 (8ull << 20)

struct pvr_rt_dataset *
pvr_render_target_dataset_cache_take(struct pvr_device *device,
                                     uint32_t width,
                                     uint32_t height,
                                     uint32_t samples,
                                     uint32_t layers)
{
   struct pvr_rt_dataset *found = NULL;

   simple_mtx_lock(&device->rt_cache_mtx);
   list_for_each_entry (struct pvr_rt_dataset,
                        rt_dataset,
                        &device->rt_dataset_cache,
                        cache_link) {
      if (rt_dataset->width == width && rt_dataset->height == height &&
          rt_dataset->samples == samples && rt_dataset->layers == layers) {
         list_del(&rt_dataset->cache_link);
         device->rt_dataset_cache_count--;
         device->rt_dataset_cache_bytes -= rt_dataset->cache_bytes;
         found = rt_dataset;
         break;
      }
   }
   simple_mtx_unlock(&device->rt_cache_mtx);

   return found;
}

/* The caller is done with the dataset: its render state is being destroyed,
 * which Vulkan allows only once no pending command buffer uses it.
 */
void pvr_render_target_dataset_release(struct pvr_rt_dataset *rt_dataset)
{
   struct pvr_device *device = rt_dataset->device;
   struct list_head evicted;

   list_inithead(&evicted);

   simple_mtx_lock(&device->rt_cache_mtx);
   list_add(&rt_dataset->cache_link, &device->rt_dataset_cache);
   device->rt_dataset_cache_count++;
   device->rt_dataset_cache_bytes += rt_dataset->cache_bytes;
   while (device->rt_dataset_cache_count > PVR_RT_DATASET_CACHE_MAX ||
          (device->rt_dataset_cache_count > 1 &&
           device->rt_dataset_cache_bytes > (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g4p)
                                                ? PVR_RT_DATASET_CACHE_BYTES_A8
                                                : PVR_RT_DATASET_CACHE_BYTES))) {
      struct pvr_rt_dataset *evict =
         list_last_entry(&device->rt_dataset_cache,
                         struct pvr_rt_dataset,
                         cache_link);
      list_del(&evict->cache_link);
      device->rt_dataset_cache_count--;
      device->rt_dataset_cache_bytes -= evict->cache_bytes;
      list_addtail(&evict->cache_link, &evicted);
   }
   simple_mtx_unlock(&device->rt_cache_mtx);

   list_for_each_entry_safe (struct pvr_rt_dataset, evict, &evicted, cache_link) {
      list_del(&evict->cache_link);
      pvr_render_target_dataset_destroy(evict);
   }
}

void pvr_render_target_dataset_cache_finish(struct pvr_device *device)
{
   list_for_each_entry_safe (struct pvr_rt_dataset,
                             rt_dataset,
                             &device->rt_dataset_cache,
                             cache_link) {
      list_del(&rt_dataset->cache_link);
      pvr_render_target_dataset_destroy(rt_dataset);
   }
   device->rt_dataset_cache_count = 0;
   device->rt_dataset_cache_bytes = 0;
   simple_mtx_destroy(&device->rt_cache_mtx);
}
