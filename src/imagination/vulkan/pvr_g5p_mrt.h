/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): several colour targets (MRT) on the G5P paths.
 * Built only with -Dimagination-g5p-experimental=true (PVR_APPLE_G5P).
 *
 * The render pass side of 1, 2 and 4 targets (clear and end-of-tile
 * programs per target count, the PBE block, the pixel output width) is in
 * pvr_arch_g5p.c; the fragment program's output plan is the backend's
 * (src/imagination/g5p/g5p_mrt_out.h); the blend and write-mask rules with
 * several targets are pvr_g5p_blend.h's. These are the rules around them:
 * what a pipeline, a pass and a draw with several targets may be. They are
 * the rules our A9 driver ran on the device: Apple's own 2- and 4-target
 * pass (one 8-bit UNORM RGBA format for every target, CLEAR or DONT_CARE
 * over the whole render area), one emitpix that writes every target.
 */

#ifndef PVR_G5P_MRT_H
#define PVR_G5P_MRT_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

#include "pvr_macros.h"

struct pvr_cmd_buffer;
struct pvr_device;
struct pvr_renderpass_hwsetup_render;

/* the most colour targets a G5P pipeline or pass has (Apple's 4-target pass) */
#define PVR_G5P_MRT_MAX_COLORS 4

/* The formats of Apple's 2- and 4-target pass */
static inline bool
pvr_g5p_mrt_format_ok(VkFormat format)
{
   return format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_R8G8B8A8_UNORM;
}

#ifdef PVR_PER_ARCH

/* Pipeline creation with n > 1 colour attachments: 2 or 4 of them, every
 * one in the same format of Apple's pass. formats: the subpass's colour
 * attachment formats in order (the caller reads them from the render pass
 * object or the rendering state: pvr_* hands the runtime only the aspects
 * of a render pass object). */
VkResult PVR_PER_ARCH(g5p_mrt_pipeline_check)(struct pvr_device *device, unsigned n, const VkFormat *formats);

/* A render with several colour targets (g5p_job_init): the rules the job
 * itself doesn't check. VK_SUCCESS for one target. */
VkResult PVR_PER_ARCH(g5p_mrt_pass_check)(struct pvr_cmd_buffer *cmd_buffer,
                                          const struct pvr_renderpass_hwsetup_render *hw_render);

/* A draw: one emitpix writes every target, so the program's target count
 * must be the pass's (the kernel can't see the fragment program).
 * pipeline_targets: the pipeline's colour target count (the backend key's
 * convention, 0 for one target, is accepted too). */
VkResult PVR_PER_ARCH(g5p_mrt_draw_check)(struct pvr_cmd_buffer *cmd_buffer, unsigned pipeline_targets);

#   define pvr_arch_g5p_mrt_pipeline_check PVR_PER_ARCH(g5p_mrt_pipeline_check)
#   define pvr_arch_g5p_mrt_pass_check PVR_PER_ARCH(g5p_mrt_pass_check)
#   define pvr_arch_g5p_mrt_draw_check PVR_PER_ARCH(g5p_mrt_draw_check)

#endif /* PVR_PER_ARCH */

#endif /* PVR_G5P_MRT_H */
