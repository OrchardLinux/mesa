/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): image blits as fragment work through the common
 * Vulkan runtime's vk_meta, in the shape of Asahi's hk_cmd_meta.c.
 * Built only with -Dimagination-g5p-experimental=true.
 *
 * G5P's transfer sub-commands run on the CPU (pvr_arch_g5p_layout.c). A blit
 * between colour images the A9 can sample and render is instead a render
 * pass of one rectangle: vk_meta_blit_image2() draws it with its own
 * pipeline, push descriptors and push constants, and the application's
 * graphics state is put back afterwards. The render sub-commands it creates
 * are marked as transfer work (pvr_sub_cmd_gfx::g5p_transfer), so barriers
 * naming the transfer stages order them (pvr_arch_queue.c).
 *
 * PVR_G5P_GPU_BLIT=0 keeps every blit on the CPU (for A/B comparison).
 */

#include <stdlib.h>
#include <string.h>

#include "vk_format.h"
#include "vk_meta.h"
#include "vk_util.h"
#include "vk_enum_to_str.h"
#include "util/u_debug.h"

#include "pvr_buffer.h"
#include "pvr_cmd_buffer.h"
#include "pvr_descriptor_set.h"
#include "pvr_device.h"
#include "pvr_g5p.h"
#include "pvr_image.h"
#include "pvr_g5p_meta.h"


DEBUG_GET_ONCE_NUM_OPTION(g5p_perf_log, "PVR_G5P_PERF_LOG", 0)

static VkResult
g5p_meta_bind_map_buffer(struct vk_command_buffer *vk_cmd, struct vk_meta_device *meta, VkBuffer _buffer,
                         void **map_out)
{
   struct pvr_cmd_buffer *cmd_buffer = container_of(vk_cmd, struct pvr_cmd_buffer, vk);
   VK_FROM_HANDLE(pvr_buffer, buffer, _buffer);
   struct pvr_suballoc_bo *bo;

   VkResult result = pvr_arch_cmd_buffer_alloc_mem(cmd_buffer, cmd_buffer->device->heaps.general_heap,
                                                   buffer->vk.size, &bo);
   if (result != VK_SUCCESS)
      return result;
   buffer->dev_addr = bo->dev_addr;
   *map_out = pvr_bo_suballoc_get_map_addr(bo);
   return VK_SUCCESS;
}

VkResult
PVR_PER_ARCH(g5p_meta_init)(struct pvr_device *device)
{
   struct vk_meta_device *meta = &device->g5p->meta;
   VkResult result = vk_meta_device_init(&device->vk, meta);

   if (result != VK_SUCCESS)
      return result;
   /* One layer per draw: G5P renders layer 0 of a single-layer view and
    * lowers the rectangle shader's gl_Layer away (pvr_g5p_nir.c). */
   meta->use_layered_rendering = false;
   meta->use_gs_for_layer = false;
   meta->use_stencil_export = false;
   meta->use_rect_list_pipeline = true;
   meta->cmd_bind_map_buffer = g5p_meta_bind_map_buffer;
   meta->max_bind_map_buffer_size_B = 64 * 1024;
   device->g5p->meta_ready = true;
   return VK_SUCCESS;
}

void
PVR_PER_ARCH(g5p_meta_finish)(struct pvr_device *device)
{
   if (device->g5p->meta_ready)
      vk_meta_device_finish(&device->vk, &device->g5p->meta);
   device->g5p->meta_ready = false;
}

/* The colour formats a meta blit reads and writes: the 8-bit RGBA orders the
 * compositor and WebGL exchange, which G5P samples and renders natively. */
static bool
g5p_meta_format(VkFormat f)
{
   switch (f) {
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_SRGB:
      return true;
   default:
      return false;
   }
}

static bool
g5p_meta_image(const struct pvr_image *image, VkImageUsageFlags needed)
{
   /* twiddled or linear (Firefox's WebGL buffer and compositor texture are
    * linear, shareable images): both are sampled and rendered by G5P */
   return image->vk.image_type == VK_IMAGE_TYPE_2D && image->vk.samples == VK_SAMPLE_COUNT_1_BIT &&
          image->vk.mip_levels == 1 && image->vk.array_layers == 1 &&
          (image->memlayout == PVR_MEMLAYOUT_TWIDDLED || image->memlayout == PVR_MEMLAYOUT_LINEAR) &&
          (image->vk.usage & needed) && g5p_meta_format(image->vk.format);
}

static bool
g5p_meta_gpu_blit_enabled(void)
{
   static int enabled = -1;

   if (enabled < 0)
      enabled = debug_get_bool_option("PVR_G5P_GPU_BLIT", true);
   return enabled;
}

static bool
g5p_meta_blit_eligible(const struct pvr_cmd_buffer *cmd_buffer, const VkBlitImageInfo2 *info)
{
   VK_FROM_HANDLE(pvr_image, src, info->srcImage);
   VK_FROM_HANDLE(pvr_image, dst, info->dstImage);

   if (!cmd_buffer->device->g5p || !cmd_buffer->device->g5p->meta_ready || !g5p_meta_gpu_blit_enabled())
      return false;
   /* a meta draw inside an active occlusion query would be counted */
   if (cmd_buffer->state.vis_test_enabled || cmd_buffer->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
      return false;
   if (!g5p_meta_image(src, VK_IMAGE_USAGE_SAMPLED_BIT) || !g5p_meta_image(dst, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))
      return false;
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageBlit2 *r = &info->pRegions[i];

      if (r->srcSubresource.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT ||
          r->dstSubresource.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT ||
          vk_image_subresource_layer_count(&src->vk, &r->srcSubresource) != 1 ||
          vk_image_subresource_layer_count(&dst->vk, &r->dstSubresource) != 1 ||
          r->srcOffsets[0].z != 0 || r->srcOffsets[1].z != 1 || r->dstOffsets[0].z != 0 ||
          r->dstOffsets[1].z != 1)
         return false;
   }
   return true;
}

/* vkCmdBlitImage2 on the GPU, or false to keep it a CPU transfer */
bool
PVR_PER_ARCH(g5p_meta_blit)(struct pvr_cmd_buffer *cmd_buffer, const VkBlitImageInfo2 *info)
{
   if (!g5p_meta_blit_eligible(cmd_buffer, info))
      return false;

   struct g5p_meta_save *save = malloc(sizeof(*save));
   if (!save) {
      vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return true;
   }

   /* End whatever sub-command is open, so that every sub-command after the
    * mark is one of the blit's render passes. */
   VkResult result = pvr_arch_cmd_buffer_end_sub_cmd(cmd_buffer);
   if (result != VK_SUCCESS) {
      free(save);
      return true;
   }
   struct list_head *mark = cmd_buffer->sub_cmds.prev;

   g5p_meta_begin(cmd_buffer, save);
   vk_meta_blit_image2(&cmd_buffer->vk, &cmd_buffer->device->g5p->meta, info);
   g5p_meta_end(cmd_buffer, save);
   free(save);

   bool rendered = false;
   for (struct list_head *l = mark->next; l != &cmd_buffer->sub_cmds; l = l->next) {
      struct pvr_sub_cmd *sub_cmd = list_entry(l, struct pvr_sub_cmd, link);

      if (sub_cmd->type == PVR_SUB_CMD_TYPE_GRAPHICS) {
         sub_cmd->gfx.g5p_transfer = true;
         rendered = true;
      }
   }
   /* Later transfer-stage barriers wait for these renders as well. */
   if (rendered) {
      for (uint32_t i = 0; i < ARRAY_SIZE(cmd_buffer->state.barriers_needed); i++)
         cmd_buffer->state.barriers_needed[i] |= PVR_PIPELINE_STAGE_TRANSFER_BIT;
   }
   if (debug_get_option_g5p_perf_log() >= 3) {
      VK_FROM_HANDLE(pvr_image, src, info->srcImage);
      VK_FROM_HANDLE(pvr_image, dst, info->dstImage);
      fprintf(stderr, "G5P perf gpu-blit src=%s %ux%u dst=%s %ux%u regions=%u filter=%d rendered=%d\n",
              vk_Format_to_str(src->vk.format), src->vk.extent.width, src->vk.extent.height,
              vk_Format_to_str(dst->vk.format), dst->vk.extent.width, dst->vk.extent.height,
              info->regionCount, info->filter, rendered);
   }
   return true;
}
