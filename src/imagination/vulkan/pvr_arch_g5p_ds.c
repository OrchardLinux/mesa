/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): depth/stencil attachments in the driver's objects.
 * The words are pvr_g5p_ds.c's; this file reads them from the render pass,
 * pvr_render_job::ds (upstream pvr_sub_cmd_gfx_job_init()'s load/store
 * decisions) and the dynamic state. Built only with
 * -Dimagination-g5p-experimental=true.
 */

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_log.h"

#include "util/u_dynarray.h"
#include "util/u_math.h"

#include "pvr_cmd_buffer.h"
#include "pvr_csb.h"
#include "pvr_g5p_ds.h"
#include "pvr_g5p_tex_encode.h"
#include "pvr_hw_pass.h"
#include "pvr_image.h"
#include "pvr_job_render.h"

/* A refusal, by name, on stderr too (vk_errorf() is silent in release builds) */
#define G5P_DS_REFUSE(cmd_buffer, ...)                                                                             \
   (fprintf(stderr, "G5P refuses: " __VA_ARGS__), fprintf(stderr, "\n"),                                          \
    vk_errorf((cmd_buffer), VK_ERROR_FEATURE_NOT_PRESENT, "G5P: " __VA_ARGS__))

VkResult
PVR_PER_ARCH(g5p_ds_admit)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_renderpass_hwsetup_render *hw_render)
{
   const struct pvr_render_pass_info *info = &cmd_buffer->state.render_pass_info;

   if (hw_render->ds_attach_idx == VK_ATTACHMENT_UNUSED)
      return VK_SUCCESS;
   const struct pvr_image_view *iview = info->attachments[hw_render->ds_attach_idx];
   const struct pvr_image *image = pvr_image_view_get_image(iview);
   const struct pvr_g5p_ds_format *f = pvr_g5p_ds_format(iview->vk.format);
   struct pvr_g5p_ds_plane plane;

   if (!f)
      return G5P_DS_REFUSE(cmd_buffer, "depth/stencil format %s (the ZLS stores D16, D32F, S8, D32F_S8)",
                           vk_Format_to_str(iview->vk.format));
   struct pvr_g5p_ds_chain chain;
   const bool sampled_chain = pvr_g5p_ds_sampled_chain(image, &chain);
   if (sampled_chain && f->stencil && (iview->vk.extent.width < 32 || iview->vk.extent.height < 32))
      return G5P_DS_REFUSE(cmd_buffer, "small combined depth/stencil chain attachment needs Apple's temporary ZLS plane");
   if (iview->vk.base_mip_level || iview->vk.base_array_layer)
      return G5P_DS_REFUSE(cmd_buffer, "a depth/stencil view of level %u layer %u (level 0 layer 0 are built)",
                           iview->vk.base_mip_level, iview->vk.base_array_layer);
   /* upstream's own test (pvr_arch_hw_pass.c): a resolve mode; the index is set only with one */
   if (hw_render->depth_resolve_mode || hw_render->stencil_resolve_mode)
      return G5P_DS_REFUSE(cmd_buffer, "a depth/stencil resolve (no capture)");
   if (hw_render->eot_surface_count > 1)
      return G5P_DS_REFUSE(cmd_buffer, "depth/stencil beside %u colour targets (no capture of the combination)",
                           hw_render->eot_surface_count);
   if (hw_render->eot_surface_count) {
      const struct pvr_image_view *c = info->attachments[hw_render->eot_surfaces[0].attachment_idx];
      if (c->vk.extent.width != iview->vk.extent.width || c->vk.extent.height != iview->vk.extent.height)
         return G5P_DS_REFUSE(cmd_buffer, "a depth/stencil attachment of %ux%u beside a %ux%u colour target "
                              "(ZLS_PIXELS describe one surface)", iview->vk.extent.width,
                              iview->vk.extent.height, c->vk.extent.width, c->vk.extent.height);
   }
   if (!pvr_g5p_ds_layout(iview->vk.format, image->vk.extent.width, image->vk.extent.height,
                          image->vk.samples, &plane) ||
       image->total_size < (sampled_chain ? chain.size : plane.size))
      return G5P_DS_REFUSE(cmd_buffer, "a depth/stencil image of %" PRIu64 " bytes, the ZLS planes take %" PRIu64
                           " (pvr_g5p_ds_layout())", (uint64_t)image->total_size, plane.size);
   if (image->vk.samples == VK_SAMPLE_COUNT_4_BIT &&
       (hw_render->depth_init == VK_ATTACHMENT_LOAD_OP_LOAD || hw_render->stencil_init == VK_ATTACHMENT_LOAD_OP_LOAD ||
        hw_render->depth_store || hw_render->stencil_store))
      return G5P_DS_REFUSE(cmd_buffer, "a load or store of a 4-sample depth/stencil attachment "
                           "(no multisample ZLS capture: it stays on chip)");
   return VK_SUCCESS;
}

void
PVR_PER_ARCH(g5p_ds_pass)(const struct pvr_render_job *job, struct pvr_g5p_zls_words *out)
{
   if (!job->ds.iview || !(job->has_depth_attachment || job->has_stencil_attachment)) {
      pvr_g5p_ds_zls_none(out);
      return;
   }
   const struct pvr_image *image = pvr_image_view_get_image(job->ds.iview);
   struct pvr_g5p_ds_chain chain;
   bool sampled_chain = pvr_g5p_ds_sampled_chain(image, &chain);
   const struct pvr_g5p_ds_target t = {
      .format = job->ds.iview->vk.format,
      .va = job->ds.addr.addr,
      .texture_chain = sampled_chain,
      .stencil_va = sampled_chain ? image->dev_addr.addr + pvr_single_plane_const(image)->offset + chain.stencil_base : 0,
      .width = job->ds.iview->vk.extent.width,
      .height = job->ds.iview->vk.extent.height,
      .samples = image->vk.samples,
      .depth = job->has_depth_attachment,
      .stencil = job->has_stencil_attachment,
      .zload = job->ds.load.d,
      .zstore = job->ds.store.d,
      .sload = job->ds.load.s,
      .sstore = job->ds.store.s,
      .depth_clear = job->ds_clear_value.depth,
      .stencil_clear = (uint8_t)job->ds_clear_value.stencil,
   };
   /* g5p_ds_admit() refused what pvr_g5p_ds_zls() cannot build */
   const char *why = pvr_g5p_ds_zls(&t, out);
   assert(!why);
   if (why)
      pvr_g5p_ds_zls_none(out);
}

void
PVR_PER_ARCH(g5p_ds_isp)(const struct vk_dynamic_graphics_state *dyn, VkImageAspectFlags aspects,
                         uint32_t samples, bool lines, uint32_t passtype, bool scissor, bool dbias, bool no_colour,
                         uint32_t w[5])
{
   struct vk_depth_stencil_state ds = dyn->ds;
   uint32_t lw = util_unsigned_fixed(dyn->rs.line.width, 4);

   if (aspects)
      vk_optimize_depth_stencil_state(&ds, aspects, true);
   struct pvr_g5p_isp_state s = {
      .lines = lines,
      .passtype = passtype,
      .line_width = MIN2(lw ? lw - 1 : 0, 255),
      .ds_attachment = aspects != 0,
      .depth_test = (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) && ds.depth.test_enable,
      .depth_write = ds.depth.write_enable,
      .depth_compare = ds.depth.compare_op,
      .stencil_test = (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) && ds.stencil.test_enable,
      .cull_front = dyn->rs.cull_mode & VK_CULL_MODE_FRONT_BIT,
      .cull_back = dyn->rs.cull_mode & VK_CULL_MODE_BACK_BIT,
      .front_ccw = dyn->rs.front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .scissor = scissor,
      .dbias = dbias,
      .no_colour = no_colour,
      .samples = samples,
   };
   const struct vk_stencil_test_face_state *face[2] = { &ds.stencil.front, &ds.stencil.back };
   struct pvr_g5p_stencil_face *out[2] = { &s.front, &s.back };
   for (unsigned i = 0; i < 2; i++)
      *out[i] = (struct pvr_g5p_stencil_face){
         .fail = face[i]->op.fail,
         .pass = face[i]->op.pass,
         .depth_fail = face[i]->op.depth_fail,
         .compare = face[i]->op.compare,
         .compare_mask = face[i]->compare_mask,
         .write_mask = face[i]->write_mask,
         .reference = face[i]->reference,
      };
   pvr_g5p_ds_isp(&s, w);
}

VkImageAspectFlags
PVR_PER_ARCH(g5p_ds_aspects)(const struct pvr_cmd_buffer *cmd_buffer)
{
   const struct pvr_render_pass_info *info = &cmd_buffer->state.render_pass_info;
   const struct pvr_sub_cmd *sub_cmd = cmd_buffer->state.current_sub_cmd;

   if (!sub_cmd || sub_cmd->type != PVR_SUB_CMD_TYPE_GRAPHICS || !info->rstate)
      return 0;
   const struct pvr_renderpass_hwsetup_render *hw_render =
      pvr_arch_pass_info_get_hw_render(info, sub_cmd->gfx.hw_render_idx);
   if (hw_render->ds_attach_idx == VK_ATTACHMENT_UNUSED)
      return 0;
   const VkFormat f = info->attachments[hw_render->ds_attach_idx]->vk.format;
   return (vk_format_has_depth(f) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
          (vk_format_has_stencil(f) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
}

VkResult
PVR_PER_ARCH(g5p_ds_bias)(struct pvr_cmd_buffer *cmd_buffer, const struct vk_dynamic_graphics_state *dyn,
                          uint32_t *index)
{
   const struct pvr_depth_bias_state bias = {
      .constant_factor = dyn->rs.depth_bias.constant_factor,
      .slope_factor = dyn->rs.depth_bias.slope_factor,
      .clamp = dyn->rs.depth_bias.clamp,
   };
   const unsigned n = util_dynarray_num_elements(&cmd_buffer->depth_bias_array, struct pvr_depth_bias_state);

   /* the previous draw's entry when the factors did not change */
   if (n && !memcmp(util_dynarray_top_ptr(&cmd_buffer->depth_bias_array, struct pvr_depth_bias_state), &bias,
                    sizeof(bias))) {
      *index = n - 1;
      return VK_SUCCESS;
   }
   struct pvr_depth_bias_state *slot = util_dynarray_grow(&cmd_buffer->depth_bias_array, struct pvr_depth_bias_state, 1);
   if (!slot)
      return vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
   *slot = bias;
   *index = n;
   return VK_SUCCESS;
}

/* The texture keys of a ZLS plane: depth reads as R32_SFLOAT (format code
 * 0x221, the captured shadow-sampling scene's), stencil as the ordinary
 * R8_UINT byte format (code 0x100, pvr_g5p_format_table.h). Their in-place
 * texel addresses must agree with the ZLS order, checked by texture_plane().
 * Both use Vulkan's depth/stencil selectors (D or S, 0, 0, 1), composed with
 * the view's swizzle. */
#define G5P_DS_TEX_KEY_DEPTH \
   PVR_G5P_TEX_KEY(0x221, 0, PVR_G5P_TEX_SEL_ZERO, PVR_G5P_TEX_SEL_ZERO, PVR_G5P_TEX_SEL_ONE)
#define G5P_DS_TEX_KEY_STENCIL \
   PVR_G5P_TEX_KEY(0x100, 0, PVR_G5P_TEX_SEL_ZERO, PVR_G5P_TEX_SEL_ZERO, PVR_G5P_TEX_SEL_ONE)

const char *
PVR_PER_ARCH(g5p_ds_view_state)(const struct pvr_image_view *iview, uint64_t q[3])
{
   const struct pvr_image *image = pvr_image_view_get_image(iview);
   const struct pvr_image_plane *plane = pvr_single_plane_const(image);
   uint64_t va;
   VkFormat tf;

   struct pvr_g5p_ds_chain chain;
   const bool sampled_chain = pvr_g5p_ds_sampled_chain(image, &chain);
   if (!sampled_chain && (image->vk.mip_levels != 1 || image->vk.array_layers != 1 || iview->vk.base_mip_level))
      return "a depth/stencil mip/layer view without sampled-chain storage";
   if (sampled_chain && iview->vk.aspects != VK_IMAGE_ASPECT_DEPTH_BIT)
      return "a stencil aspect of sampled-chain storage (no stencil texture-state witness)";
   if (image->vk.samples != VK_SAMPLE_COUNT_1_BIT)
      return "a sampled multisample depth/stencil image";
   if (iview->vk.view_type != VK_IMAGE_VIEW_TYPE_2D && iview->vk.view_type != VK_IMAGE_VIEW_TYPE_2D_ARRAY && (!sampled_chain || iview->vk.view_type != VK_IMAGE_VIEW_TYPE_CUBE))
      return "a depth/stencil view that is not 2D/array/cube";
   if (sampled_chain) {
      va=image->dev_addr.addr+plane->offset+iview->vk.base_array_layer*chain.layer;
      tf=image->vk.format == VK_FORMAT_D16_UNORM ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
   } else {
      const char *why = pvr_g5p_ds_texture_plane(image->vk.format, iview->vk.aspects,
         image->vk.extent.width, image->vk.extent.height, image->dev_addr.addr+plane->offset, &va, &tf);
      if (why) return why;
   }
   if (va % 64)
      return "a ZLS plane off a 64-byte base";
   const struct pvr_g5p_tex_twiddled_view v = {
      .key = pvr_g5p_tex_swizzle_key(tf == VK_FORMAT_R32_SFLOAT ? G5P_DS_TEX_KEY_DEPTH : tf == VK_FORMAT_R16_UNORM ? PVR_G5P_TEX_KEY(9,0,5,5,4) : G5P_DS_TEX_KEY_STENCIL,
                                     &iview->vk.swizzle),
      .layout = iview->vk.view_type == VK_IMAGE_VIEW_TYPE_2D ? PVR_G5P_TEX_LAYOUT_2D :
                iview->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE ? PVR_G5P_TEX_LAYOUT_CUBE : PVR_G5P_TEX_LAYOUT_2D_ARRAY,
      .width = image->vk.extent.width,
      .height = image->vk.extent.height,
      .depth = sampled_chain ? iview->vk.layer_count : 1,
      .base_level = iview->vk.base_mip_level,
      .level_count = iview->vk.level_count,
      .image_has_mips = image->vk.mip_levels > 1,
      .big_layer = sampled_chain && chain.layer >= PVR_G5P_TEX_BIG_LAYER,
      .va = va,
   };
   pvr_g5p_tex_twiddled(q, &v);
   return NULL;
}
