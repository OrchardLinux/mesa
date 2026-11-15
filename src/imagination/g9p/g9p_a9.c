/* SPDX-License-Identifier: MIT */
#include "g9p_compile.h"

/* AGXMetalA9 ComputeProgramVariant -> +0xb08, returned by
 * -[AGXA9FamilyComputePipeline maxTotalThreadsPerThreadgroup]. The allocation
 * uses bank 0's DOUTU temps and bank 1's kernel-record register field. Both
 * banks keep one task (32 lanes) free. With t = 64 and f = 8 this gives 384,
 * matching k_hi48 on iOS.
 */
unsigned
g9p_a9_pipeline_max_lanes(const struct g9p_shader *shader)
{
   const unsigned t = 8 * DIV_ROUND_UP(MAX2(shader->num_regs, 1), 8);
   const unsigned f = 8 * DIV_ROUND_UP(shader->a9_bank1_regs ? shader->a9_bank1_regs : shader->num_regs, 8);
   const unsigned a = (4 * t + 28) & ~31u;
   const unsigned b = (64 * f + 448) & ~511u;
   unsigned max = MIN2(a ? 32 * ((3329 - a) / a) : 512,
                       b ? 393216 / b - 32 : 512);

   max = MIN2(max, 512);
   if (max > 32)
      max &= ~31u;

   return max;
}
