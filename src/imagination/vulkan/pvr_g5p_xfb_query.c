/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): vkGetQueryPoolResults() of a transform feedback
 * stream query pool (pvr_g5p_xfb.h). Compiled only with
 * -Dimagination-g5p-experimental=true.
 *
 * The counts are the CPU resolve's (pvr_arch_g5p_xfb.c), two uint64_t per
 * query in the pool's result buffer, the availability in its availability
 * buffer; they are final once the query's End has run on the CPU, which
 * happens when the command buffer is submitted.
 */

#include "util/os_time.h"
#include "vk_log.h"

#include "pvr_bo.h"
#include "pvr_device.h"
#include "pvr_physical_device.h"
#include "pvr_g5p_xfb.h"
#include "pvr_g5p_xfb_core.h"
#include "pvr_query.h"

bool
pvr_g5p_xfb_query_pool(const struct pvr_device *device, const struct pvr_query_pool *pool)
{
   return PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g5p) &&
          pool->query_type == VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT;
}

VkResult
pvr_g5p_xfb_query_results(struct pvr_device *device, const struct pvr_query_pool *pool, uint32_t first,
                          uint32_t count, void *data, VkDeviceSize stride, VkQueryResultFlags flags)
{
   if (pool->result_elem_stride < 2 * sizeof(uint64_t))
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   struct g5p_xfb_query_pool view = {
      .count = pool->query_count,
      .stride = pool->result_elem_stride,
      .data = pvr_bo_suballoc_get_map_addr(pool->result_buffer),
      .available = pvr_bo_suballoc_get_map_addr(pool->availability_buffer),
   };
   /* WAIT: the End runs at submission (on the queue's thread, if it has
    * one); 10 s is far beyond any submission */
   if (flags & VK_QUERY_RESULT_WAIT_BIT) {
      const uint64_t deadline = os_time_get_absolute_timeout(10000000000ull);
      for (uint32_t q = first; q < first + count && q < view.count; q++) {
         while (!p_atomic_read(&view.available[q])) {
            if (os_time_get_nano() > (int64_t)deadline)
               return vk_error(device, VK_ERROR_DEVICE_LOST);
            os_time_sleep(100);
         }
      }
   }
   return g5p_xfb_query_results(&view, first, count, data, stride, flags) ? VK_SUCCESS : VK_NOT_READY;
}
