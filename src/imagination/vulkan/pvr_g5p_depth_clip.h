/* SPDX-License-Identifier: MIT
 *
 * Apple's A9 GPU (G5P): the PPP clip mode of a draw's depth clip and clamp
 * state, or the reason the A9 path cannot draw it. Standalone (no Mesa
 * headers), so a host test can include it as it is.
 *
 * AGXMetalA9 -[AGXA9FamilyRenderContext setDepthClipMode:] writes
 * TA_STATE_PPP_CTRL bits 10:11 as 0x400 << mode: Metal's clip mode writes 1
 * (ROGUE_TA_CLIP_MODE_FRONT_REAR), its clamp mode 2 (FRONT_REAR_DEPTH),
 * which keeps geometry beyond the near and far planes and clamps its depth
 * to [0, 1]. On the A9: Vulkan's explicit
 * complementary clip/clamp pairs 12/12 exact; GL depth clamp exact except
 * custom depth ranges, which the A9 clamps to [0, 1], not to the range.
 * So mode 2 is exactly:
 *  - depth clip off, clamp off: Vulkan clamps z_f to [0, 1] with
 *    depthClampZeroOne enabled (Zink enables it), and leaves depth outside
 *    the viewport range undefined without it;
 *  - depth clip off, clamp on: Vulkan's clamp to [min, max] of the
 *    viewport's depth range, when that range is [0, 1] (either order). Zink
 *    sends this pair for GL_DEPTH_CLAMP.
 * Depth bias with mode 2, and clamp with clipping on, have no hardware run.
 */
#ifndef PVR_G5P_DEPTH_CLIP_H
#define PVR_G5P_DEPTH_CLIP_H

#include <stdbool.h>
#include <stddef.h>

#define PVR_G5P_CLIP_MODE_CLIP 1u  /* ROGUE_TA_CLIP_MODE_FRONT_REAR */
#define PVR_G5P_CLIP_MODE_CLAMP 2u /* ROGUE_TA_CLIP_MODE_FRONT_REAR_DEPTH */

/* clip: the effective depth clip enable (vk_rasterization_state_depth_clip_enable());
 * has_range false: the pipeline's static check, before the draw knows its
 * viewport; dbias: a depth bias with a non-zero factor applies. */
static inline const char *
pvr_g5p_depth_clip_mode(bool clip, bool clamp, bool has_range, float min_depth, float max_depth, bool dbias,
                        unsigned *mode)
{
   *mode = PVR_G5P_CLIP_MODE_CLIP;
   if (clip && !clamp)
      return NULL;
   if (clip)
      return "depth clamp with depth clipping enabled is not qualified";
   if (dbias)
      return "depth bias with depth clipping disabled is not qualified";
   if (clamp && has_range) {
      const float lo = min_depth < max_depth ? min_depth : max_depth;
      const float hi = min_depth < max_depth ? max_depth : min_depth;
      if (lo != 0.0f || hi != 1.0f)
         return "depth clamp to a viewport depth range other than [0, 1] (the A9 clamps to [0, 1])";
   }
   *mode = PVR_G5P_CLIP_MODE_CLAMP;
   return NULL;
}

#endif
