/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): blending (pvr_g5p_blend.h). The pipeline's blend
 * state becomes the G5P backend's fragment program key, and the program
 * decides its draws' ISP pass type. Compiled only with
 * -Dimagination-g5p-experimental=true.
 */

#include "vk_graphics_state.h"

#include "pco_g5p_general.h"
#include "pvr_csb.h"
#include "pvr_g5p_blend.h"

/*
 * Blending and the write mask compile into the fragment program (the
 * backend's epilogues; it refuses factor sets without one). Logic ops and
 * dynamic blend enables, equations, write masks and colour-write enables
 * have no encoding. With several colour targets, one emit writes them all
 * and the blend epilogue owns the output halves of target 1, so no target
 * may blend or mask then (no Apple N-target blend capture). `targets`: the
 * subpass's colour attachments (pvr_* hands the runtime only their aspects
 * for a render pass object, so the caller counts them). Returns the
 * refusal, or NULL with *key's blend, factor, op, write-mask and
 * target-count fields set (the others are left as they are).
 */
const char *
PVR_PER_ARCH(g5p_blend_fs_key)(const struct vk_graphics_pipeline_state *state, unsigned targets,
                               struct pco_g5p_fs_key *key)
{
   const struct vk_color_blend_state *cb = state->cb;

   key->blend = false;
   key->write_mask = 0xf;
   key->color_targets = targets > 1 ? targets : 0;
   if (!cb)
      return NULL;
   if (cb->logic_op_enable)
      return "logic op";
   if (cb->attachment_count != targets)
      return "colour blend attachments other than the render pass's colour attachments";
   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_ENABLES) ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS) ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_WRITE_MASKS) ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES))
      return "dynamic blend enables, equations, write masks or colour-write enables";
   for (unsigned i = 0; targets > 1 && i < cb->attachment_count; i++) {
      const struct vk_color_blend_attachment_state *a = &cb->attachments[i];

      if (a->blend_enable || a->write_mask != 0xf || !(cb->color_write_enables & BITFIELD_BIT(i)))
         return "blend or a write mask other than 0xf with several colour targets";
   }
   if (!cb->attachment_count)
      return NULL;

   const struct vk_color_blend_attachment_state *a = &cb->attachments[0];

   key->write_mask = (cb->color_write_enables & 1) ? a->write_mask : 0;
   key->blend = a->blend_enable;
   if (a->blend_enable) {
      key->src_rgb = a->src_color_blend_factor;
      key->dst_rgb = a->dst_color_blend_factor;
      key->src_a = a->src_alpha_blend_factor;
      key->dst_a = a->dst_alpha_blend_factor;
      key->op_rgb = a->color_blend_op;
      key->op_a = a->alpha_blend_op;
   }
   return NULL;
}

/*
 * Apple's pass rule (AGXMetalA9 getShaderPassInfo; labwc114-passtype-
 * tile-read.md §1, hyprland-gpu.md §10.14 and §11.6):
 * - a program that reads the destination (blend_mode != 0: the
 *   premultiplied, straight-over or general epilogue's tile read) runs
 *   TRANSLUCENT, as Apple's f_blend_premult. An OPAQUE object covering a tile
 *   replaces the tags under it, the LOAD or clear background included, so
 *   the tile read would return stale tile memory;
 * - a program that discards runs FAST_PUNCH_THROUGH, or PUNCH_THROUGH when
 *   it also reads the tile (a discard in an OPAQUE pass faulted the 3D
 *   phase);
 * - a program that writes no colour (write mask 0, Apple's f_mask_none) has
 *   tag writes disabled and upass 0, so it only updates depth/stencil;
 * - rasterizer discard: no tag write (the Rogue rule).
 * The pass type is the same with a depth/stencil attachment; that ISP form
 * takes upass and the tag write as below.
 */
void
PVR_PER_ARCH(g5p_blend_isp)(unsigned blend_mode, bool shader_discard, uint8_t write_mask,
                            bool rasterizer_discard, struct pvr_g5p_blend_isp *isp)
{
   const bool no_colour = write_mask == 0;

   if (shader_discard)
      isp->pass_type = blend_mode ? ROGUE_TA_PASSTYPE_PUNCH_THROUGH : ROGUE_TA_PASSTYPE_FAST_PUNCH_THROUGH;
   else
      isp->pass_type = blend_mode ? ROGUE_TA_PASSTYPE_TRANSLUCENT : ROGUE_TA_PASSTYPE_OPAQUE;
   isp->upass = no_colour && !rasterizer_discard ? 0 : 15;
   isp->tag_write_disable = rasterizer_discard || no_colour;
}
