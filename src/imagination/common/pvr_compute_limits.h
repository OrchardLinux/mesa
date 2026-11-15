/* SPDX-License-Identifier: MIT */
#ifndef PVR_COMPUTE_LIMITS_H
#define PVR_COMPUTE_LIMITS_H

#include "hwdef/rogue_hw_utils.h"

/* Check residency, independently of the API's workgroup size limits.
 * max_lanes reports the largest resident workgroup for this allocation.
 */
static inline bool
pvr_compute_wg_residency_ok(const struct pvr_device_info *dev_info,
                          unsigned regs,
                          unsigned bank1_regs,
                          unsigned wg_size,
                          bool has_barriers,
                          unsigned *max_lanes)
{
   if (PVR_HAS_FEATURE(dev_info, apple_compute_dispatch)) {
      const struct pvr_apple_compute_dispatch_limits *limits =
         &dev_info->features.apple_compute_dispatch;
      const unsigned t = ALIGN_POT(MAX2(regs, 1), limits->register_granularity);
      const unsigned f = ALIGN_POT(bank1_regs ? bank1_regs : regs,
                                   limits->register_granularity);
      const unsigned a = (limits->bank0_scale * t + limits->bank0_rounding_bias) &
                         ~(limits->bank0_granularity - 1);
      const unsigned b = (limits->bank1_scale * f + limits->bank1_rounding_bias) &
                         ~(limits->bank1_granularity - 1);
      unsigned max = MIN2(a ? limits->task_size * ((limits->bank0_budget - a) / a) :
                             limits->max_workgroup_size,
                         b ? limits->bank1_budget / b - limits->task_size :
                             limits->max_workgroup_size);

      /* Apple reserves one task in each bank. This matches Metal's
       * maxTotalThreadsPerThreadgroup, including for barriered workgroups.
       */
      max = MIN2(max, limits->max_workgroup_size);
      if (max > limits->task_size)
         max &= ~(limits->task_size - 1);
      *max_lanes = max;
      return wg_size <= max;
   }

   UNREACHABLE("compute residency requires Apple dispatch limits");
}

#endif /* PVR_COMPUTE_LIMITS_H */
