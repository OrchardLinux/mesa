/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): the rules for several colour targets (pvr_g5p_mrt.h).
 * Compiled only with -Dimagination-g5p-experimental=true.
 *
 * Ported from our A9 driver (pipeline creation, the pass and the draw of a
 * 2- or 4-target render), which ran these shapes on the device: Apple's own
 * pass for 2 and 4 targets, whose programs and words pvr_arch_g5p.c builds
 * for every target count. Anything outside that shape is refused by name
 * rather than guessed.
 */

#include <inttypes.h>
#include <stdio.h>

#include "vk_enum_to_str.h"
#include "vk_log.h"

#include "g5p_mrt_out.h"
#include "pvr_cmd_buffer.h"
#include "pvr_device.h"
#include "pvr_g5p.h"
#include "pvr_g5p_formats.h"
#include "pvr_g5p_mrt.h"
#include "pvr_hw_pass.h"
#include "pvr_image.h"

/* A refusal, by name, on stderr too (vk_errorf() is silent in release builds) */
#define G5P_MRT_REFUSE(obj, ...)                                                                                   \
   (fprintf(stderr, "G5P refuses: " __VA_ARGS__), fprintf(stderr, "\n"),                                          \
    vk_errorf((obj), VK_ERROR_FEATURE_NOT_PRESENT, "G5P: " __VA_ARGS__))

/* Two 32-bit integer targets of one format (dEQP's ShaderExecutor: a mat2xN
 * output's columns, modf's two results): the raw32 output's four words each,
 * eight in one emitpix, the width of Apple's 4-target pass; each target's PBE
 * reads its four words from output half 8t (pvr_arch_g5p.c). Both halves are
 * device-proven apart (raw32 from half 0, 8-bit RGBA from half 8), and the
 * pair ran on the device too (docs/gpu/DEVICE-READY-int-mrt.md): rendered by
 * default. 0: not such a pair, 1: a pair. */
static int
g5p_mrt_raw32_pair(unsigned n, VkFormat f0, VkFormat ft)
{
   return n == 2 && pvr_g5p_raw32_kind(f0) && ft == f0;
}

/* Four 32-bit integer targets: 16 output words, twice what one emitpix carries */
#define G5P_MRT_RAW32_WIDE "32-bit integer targets take four output words each and one emitpix carries at most eight " \
                           "(two targets)"

#define G5P_MRT_RAW32_PROBE "two 32-bit integer targets render only behind PVR_G5P_FORMATS=probe (not yet run on " \
                            "the device)"

VkResult
PVR_PER_ARCH(g5p_mrt_pipeline_check)(struct pvr_device *device, unsigned n, const VkFormat *formats)
{
   struct g5p_mrt_out_plan plan;

   if (n <= 1)
      return VK_SUCCESS;
   /* 1, 2 or 4: the counts with Apple's pass and a kernel pixel output width; 3 runs as Apple's 4-target pass
    * whose fourth target stores to the null target (pvr_arch_g5p.c, pvr_g5p_nir.c) */
   enum g5p_mrt_refusal r = g5p_mrt_out_plan_make(n == 3 ? 4 : n, &plan);
   if (r != G5P_MRT_OK)
      return G5P_MRT_REFUSE(device, "%u colour attachments: %s", n, g5p_mrt_refusal_str(r));
   for (unsigned i = 0; i < n; i++) {
      const int pair = g5p_mrt_raw32_pair(n, formats[0], formats[i]);
      if (pair < 0)
         return G5P_MRT_REFUSE(device, "colour attachment %u of %u is %s: %s", i, n, vk_Format_to_str(formats[i]),
                               G5P_MRT_RAW32_PROBE);
      if (pair)
         continue;
      if (n > 2 && pvr_g5p_raw32_kind(formats[i]))
         return G5P_MRT_REFUSE(device, "%u colour attachments of %s: %s", n, vk_Format_to_str(formats[i]),
                               G5P_MRT_RAW32_WIDE);
      if (!pvr_g5p_mrt_format_ok(formats[i]) || formats[i] != formats[0])
         return G5P_MRT_REFUSE(device, "colour attachment %u of %u is %s, attachment 0 %s: %s", i, n,
                               vk_Format_to_str(formats[i]), vk_Format_to_str(formats[0]),
                               g5p_mrt_refusal_str(G5P_MRT_MIXED_FORMATS));
   }
   return VK_SUCCESS;
}

static uint64_t
g5p_mrt_target_va(const struct pvr_image_view *iview, uint64_t *bytes)
{
   const struct pvr_image *image = pvr_image_view_get_image(iview);
   const struct pvr_image_plane *plane = pvr_single_plane_const(image);
   const struct pvr_mip_level *level = &plane->mip_levels[iview->vk.base_mip_level];

   *bytes = (uint64_t)level->pitch * iview->vk.extent.height;
   return image->dev_addr.addr + plane->offset + level->offset + iview->vk.base_array_layer * plane->layer_size;
}

VkResult
PVR_PER_ARCH(g5p_mrt_pass_check)(struct pvr_cmd_buffer *cmd_buffer,
                                 const struct pvr_renderpass_hwsetup_render *hw_render)
{
   const struct pvr_render_pass_info *info = &cmd_buffer->state.render_pass_info;
   const unsigned n = hw_render->eot_surface_count;
   const struct pvr_image_view *iview[G5P_MRT_MAX_TARGETS];
   uint64_t va[G5P_MRT_MAX_TARGETS], bytes[G5P_MRT_MAX_TARGETS];

   if (n <= 1)
      return VK_SUCCESS;
   /* three: Apple's 4-target pass with a padding fourth target (pvr_arch_g5p.c) */
   if (n > G5P_MRT_MAX_TARGETS)
      return G5P_MRT_REFUSE(cmd_buffer, "%u colour targets: %s", n, g5p_mrt_refusal_str(G5P_MRT_TOO_MANY_TARGETS));

   for (unsigned t = 0; t < n; t++) {
      const uint32_t idx = hw_render->eot_surfaces[t].attachment_idx;
      if (idx >= info->attachment_count || !info->attachments[idx])
         return G5P_MRT_REFUSE(cmd_buffer, "colour target %u of %u has no image view", t, n);
      iview[t] = info->attachments[idx];
      const struct pvr_image *image = pvr_image_view_get_image(iview[t]);

      const int pair = g5p_mrt_raw32_pair(n, iview[0]->vk.format, iview[t]->vk.format);
      if (pair < 0)
         return G5P_MRT_REFUSE(cmd_buffer, "colour target %u of %u is %s: %s", t, n,
                               vk_Format_to_str(iview[t]->vk.format), G5P_MRT_RAW32_PROBE);
      if (!pair && n > 2 && pvr_g5p_raw32_kind(iview[t]->vk.format))
         return G5P_MRT_REFUSE(cmd_buffer, "%u colour targets of %s: %s", n, vk_Format_to_str(iview[t]->vk.format),
                               G5P_MRT_RAW32_WIDE);
      if (!pair && (!pvr_g5p_mrt_format_ok(iview[t]->vk.format) || iview[t]->vk.format != iview[0]->vk.format))
         return G5P_MRT_REFUSE(cmd_buffer, "colour target %u of %u is %s, target 0 %s: %s", t, n,
                               vk_Format_to_str(iview[t]->vk.format), vk_Format_to_str(iview[0]->vk.format),
                               g5p_mrt_refusal_str(G5P_MRT_MIXED_FORMATS));
      if (iview[t]->vk.base_mip_level || iview[t]->vk.layer_count != 1 || image->vk.samples != VK_SAMPLE_COUNT_1_BIT)
         return G5P_MRT_REFUSE(cmd_buffer, "colour target %u of %u: mip %u, %u layers, %u samples (one level, one layer, "
                               "one sample)", t, n, iview[t]->vk.base_mip_level, iview[t]->vk.layer_count,
                               image->vk.samples);
      if (iview[t]->vk.extent.width != iview[0]->vk.extent.width ||
          iview[t]->vk.extent.height != iview[0]->vk.extent.height)
         return G5P_MRT_REFUSE(cmd_buffer, "colour target %u is %ux%u, target 0 %ux%u (one size per pass)", t,
                               iview[t]->vk.extent.width, iview[t]->vk.extent.height, iview[0]->vk.extent.width,
                               iview[0]->vk.extent.height);
      va[t] = g5p_mrt_target_va(iview[t], &bytes[t]);
   }

   /* The background programs clear or leave every target; a LOAD would need
    * an N-target LOAD background (none exists) */
   for (unsigned i = 0; i < hw_render->color_init_count; i++)
      if (hw_render->color_init[i].op == VK_ATTACHMENT_LOAD_OP_LOAD)
         return G5P_MRT_REFUSE(cmd_buffer, "LOAD of a colour attachment with %u colour targets (no N-target LOAD "
                               "background)", n);

   /* The clear and end-of-tile programs cover whole tiles of every target: a
    * partial render area would need that LOAD background too */
   const VkRect2D *area = &info->render_area;
   if (area->offset.x || area->offset.y || area->extent.width != iview[0]->vk.extent.width ||
       area->extent.height != iview[0]->vk.extent.height)
      return G5P_MRT_REFUSE(cmd_buffer, "%u colour targets in a partial render area (%d,%d %ux%u of %ux%u)", n,
                            area->offset.x, area->offset.y, area->extent.width, area->extent.height,
                            iview[0]->vk.extent.width, iview[0]->vk.extent.height);

   /* One end-of-tile store per target: aliased targets would be written twice */
   for (unsigned a = 0; a < n; a++)
      for (unsigned c = a + 1; c < n; c++)
         if (va[a] < va[c] + bytes[c] && va[c] < va[a] + bytes[a])
            return G5P_MRT_REFUSE(cmd_buffer, "colour targets %u and %u overlap in memory", a, c);

   return VK_SUCCESS;
}

VkResult
PVR_PER_ARCH(g5p_mrt_draw_check)(struct pvr_cmd_buffer *cmd_buffer, unsigned pipeline_targets)
{
   const struct pvr_render_pass_info *info = &cmd_buffer->state.render_pass_info;
   const unsigned pn = MAX2(pipeline_targets, 1);

   if (!info->pass && !info->dr_info)
      return G5P_MRT_REFUSE(cmd_buffer, "a draw outside a render pass");
   /* the render of the current subpass (dynamic rendering has one) */
   const struct pvr_renderpass_hwsetup_render *hw_render =
      pvr_arch_pass_info_get_hw_render(info, info->current_hw_subpass);
   /* a 3-target pass runs, and its pipelines compile, as four targets */
   const unsigned rn = hw_render->eot_surface_count == 3 ? 4 : hw_render->eot_surface_count;
   if ((rn > 1 || pn > 1) && pn != rn)
      return G5P_MRT_REFUSE(cmd_buffer, "the pipeline writes %u colour targets, the pass has %u", pn, rn);
   return VK_SUCCESS;
}
