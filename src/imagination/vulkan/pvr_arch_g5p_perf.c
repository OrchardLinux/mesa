/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): the desktop speed work (pvr_g5p_perf.h,
 * docs/gpu/orchard-port-perf.md). Compiled only with
 * -Dimagination-g5p-experimental=true.
 */

#include <string.h>

#include "vulkan/vulkan_core.h"

#include "pvr_cmd_buffer.h"
#include "pvr_g5p_perf.h"
#include "pvr_hw_pass.h"
#include "pvr_image.h"

static void
note_write(struct g5p_perf_tail *t, const struct pvr_image_view *iview)
{
   if (!iview) {
      /* an attachment we can't place: the winsys must assume any own BO is written */
      t->flags &= ~G5P_PERF_WRITES_KNOWN;
      return;
   }
   const struct pvr_image *image = pvr_image_view_get_image(iview);
   const uint64_t va = image->dev_addr.addr;

   for (unsigned i = 0; i < t->writes; i++)
      if (t->write_va[i] == va)
         return;
   if (t->writes == G5P_PERF_MAX_WRITES) {
      t->flags &= ~G5P_PERF_WRITES_KNOWN;
      return;
   }
   t->write_va[t->writes++] = va;
}

/*
 * What the render writes outside the arena can only be its attachments: the
 * G5P backend compiles no fragment memory stores (pco_g5p_general.c
 * store_refusal(): a store is a transform feedback store or refused), so the
 * one other writer is transform feedback, whose buffers this does not track:
 * a command buffer with transform feedback state leaves the writes unknown.
 * The base address of an attachment's image marks the BO it lies in, so a
 * view of any level or layer marks the right BO; a 4x render's resolve
 * target is its EOT surface's view (pvr_arch_g5p_job_init()).
 */
void
PVR_PER_ARCH(g5p_perf_job)(const struct pvr_cmd_buffer *cmd_buffer,
                           const struct pvr_renderpass_hwsetup_render *hw_render,
                           struct g5p_perf_tail *t)
{
   const struct pvr_render_pass_info *info = &cmd_buffer->state.render_pass_info;

   memset(t, 0, sizeof(*t));
   t->magic = G5P_PERF_TAIL_MAGIC;
   t->flags = G5P_PERF_WRITES_KNOWN;
   if (cmd_buffer->state.g5p_xfb)
      t->flags &= ~G5P_PERF_WRITES_KNOWN;

   for (unsigned s = 0; s < hw_render->eot_surface_count; s++) {
      const uint32_t idx = hw_render->eot_surfaces[s].attachment_idx;
      note_write(t, idx < info->attachment_count ? info->attachments[idx] : NULL);
   }
   /* a pass that clears nothing: a draw-less job of it changes no pixel (counted by the winsys's log; the next
    * slice skips it) */
   bool clears = hw_render->depth_init == VK_ATTACHMENT_LOAD_OP_CLEAR ||
                 hw_render->stencil_init == VK_ATTACHMENT_LOAD_OP_CLEAR;
   for (unsigned i = 0; i < hw_render->color_init_count; i++)
      clears |= hw_render->color_init[i].op == VK_ATTACHMENT_LOAD_OP_CLEAR;
   if (!clears)
      t->flags |= G5P_PERF_NO_CLEAR;
   if (hw_render->ds_attach_idx != VK_ATTACHMENT_UNUSED)
      note_write(t, hw_render->ds_attach_idx < info->attachment_count ? info->attachments[hw_render->ds_attach_idx]
                                                                      : NULL);
}
