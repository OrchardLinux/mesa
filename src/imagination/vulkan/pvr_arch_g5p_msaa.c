/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): the shape of a 4x render, checked against the
 * render pass before any word is written (the words: pvr_g5p_msaa.c). What
 * the captures show is built; the rest is refused by name. Built only with
 * -Dimagination-g5p-experimental=true.
 */

#include <stdio.h>

#include "vk_format.h"
#include "vk_log.h"

#include "pvr_cmd_buffer.h"
#include "pvr_g5p_msaa.h"
#include "pvr_hw_pass.h"
#include "pvr_image.h"

#define G5P_MSAA_REFUSE(cmd_buffer, ...)                                                                           \
   (fprintf(stderr, "G5P refuses: " __VA_ARGS__), fprintf(stderr, "\n"),                                          \
    vk_errorf((cmd_buffer), VK_ERROR_FEATURE_NOT_PRESENT, "G5P: " __VA_ARGS__))

/*
 * The captured 4x pass: one colour target of 4-byte samples, cleared or
 * don't-care, either stored with its samples (msaa4-store) or resolved by
 * the PBE into a one-sample target at the end of each tile (msaa4-resolve),
 * a depth/stencil attachment on chip only. Upstream's hardware setup lists
 * the resolve as an end-of-tile surface with need_resolve and resolve_type
 * PVR_RESOLVE_TYPE_PBE.
 */
VkResult
PVR_PER_ARCH(g5p_msaa_admit)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_renderpass_hwsetup_render *hw_render)
{
   const struct pvr_render_pass_info *info = &cmd_buffer->state.render_pass_info;

   if (hw_render->sample_count <= 1)
      return VK_SUCCESS;
   if (hw_render->sample_count != 4)
      return G5P_MSAA_REFUSE(cmd_buffer, "%u samples (1 and 4 are built)", hw_render->sample_count);
   if (hw_render->eot_surface_count != 1)
      return G5P_MSAA_REFUSE(cmd_buffer, "4x with %u end-of-tile surfaces (one target, stored or resolved: a "
                             "STORE with a resolve needs the CPU resolve after the job)",
                             hw_render->eot_surface_count);
   for (unsigned i = 0; i < hw_render->color_init_count; i++)
      if (hw_render->color_init[i].op == VK_ATTACHMENT_LOAD_OP_LOAD)
         return G5P_MSAA_REFUSE(cmd_buffer, "LOAD of a 4-sample attachment (no multisample background loader)");

   const struct pvr_renderpass_hwsetup_eot_surface *eot = &hw_render->eot_surfaces[0];
   const struct pvr_image_view *iview = info->attachments[eot->need_resolve ? eot->src_attachment_idx
                                                                            : eot->attachment_idx];
   const struct pvr_image *image = pvr_image_view_get_image(iview);
   if (image->vk.samples != VK_SAMPLE_COUNT_4_BIT || vk_format_get_blocksize(iview->vk.format) != 4 ||
       vk_format_is_depth_or_stencil(iview->vk.format))
      return G5P_MSAA_REFUSE(cmd_buffer, "a 4x colour target %s of %u samples (4-byte samples are built)",
                             vk_Format_to_str(iview->vk.format), image->vk.samples);
   if (iview->vk.base_mip_level || iview->vk.base_array_layer)
      return G5P_MSAA_REFUSE(cmd_buffer, "a 4-sample view of level %u layer %u", iview->vk.base_mip_level,
                             iview->vk.base_array_layer);
   if (eot->need_resolve) {
      const struct pvr_image_view *r = info->attachments[eot->attachment_idx];
      const struct pvr_image *ri = pvr_image_view_get_image(r);
      if (eot->resolve_type != PVR_RESOLVE_TYPE_PBE)
         return G5P_MSAA_REFUSE(cmd_buffer, "a 4x resolve outside the PBE (transfer resolves are lane g's)");
      if (ri->vk.samples != VK_SAMPLE_COUNT_1_BIT || ri->memlayout != PVR_MEMLAYOUT_LINEAR ||
          r->vk.format != iview->vk.format || r->vk.extent.width != iview->vk.extent.width ||
          r->vk.extent.height != iview->vk.extent.height || r->vk.base_mip_level)
         return G5P_MSAA_REFUSE(cmd_buffer, "a resolve view that is not a %ux%u one-sample linear %s view",
                                iview->vk.extent.width, iview->vk.extent.height, vk_Format_to_str(iview->vk.format));
   }
   return VK_SUCCESS;
}
