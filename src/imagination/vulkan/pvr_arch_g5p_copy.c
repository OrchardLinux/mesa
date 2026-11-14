/* SPDX-License-Identifier: MIT */
/* Stage 1 buffer transfers: exact-extent linear RGBA32UI renders. */
#include "nir_builder.h"
#include "util/u_debug.h"
#include "util/u_atomic.h"
#include "vk_meta.h"
#include "pvr_buffer.h"
#include "pvr_device.h"
#include "pvr_g5p.h"
#include "pvr_g5p_meta.h"
#include "pvr_image.h"
#include "pvr_physical_device.h"

DEBUG_GET_ONCE_BOOL_OPTION(g5p_gpu_copy, "PVR_G5P_GPU_COPY", false)
DEBUG_GET_ONCE_NUM_OPTION(g5p_perf_log, "PVR_G5P_PERF_LOG", 0)
#define COPY_FORMAT VK_FORMAT_R32G32B32A32_UINT
#define COPY_ROW_BYTES (2048 * 16)
#define COPY_MAX_BYTES (64ull * 1024 * 1024)

struct copy_push { uint32_t width, src_elem0, fill; };
struct copy_key { uint32_t type, format, fill, scalar_src; };

static nir_def *
copy_push(nir_builder *b, unsigned offset)
{
   return nir_load_push_constant(b, 1, 32, nir_imm_int(b, offset),
                                 .base = 0, .range = sizeof(struct copy_push));
}

static VkResult
copy_pipeline(struct pvr_cmd_buffer *cmd, bool fill, bool scalar_src,
              VkPipelineLayout *layout, VkPipeline *pipeline)
{
   struct vk_device *dev = &cmd->device->vk;
   struct vk_meta_device *meta = &cmd->device->g5p->meta;
   const struct copy_key key = { VK_META_OBJECT_KEY_DRIVER_OFFSET + 1,
                                 COPY_FORMAT, fill, scalar_src };
   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo set = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
      .bindingCount = fill ? 0 : 1, .pBindings = &binding,
   };
   const VkPushConstantRange push = {
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .size = sizeof(struct copy_push),
   };
   VkResult result = vk_meta_get_pipeline_layout(dev, meta, &set, &push,
                                                 &key, sizeof(key), layout);
   if (result != VK_SUCCESS)
      return result;
   *pipeline = vk_meta_lookup_pipeline(meta, &key, sizeof(key));
   if (*pipeline)
      return VK_SUCCESS;

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, NULL,
                                                  "g5p-buffer-copy");
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out,
                                           glsl_uvec4_type(), "out");
   out->data.location = FRAG_RESULT_DATA0;
   nir_def *value;
   if (fill) {
      nir_def *word = copy_push(&b, offsetof(struct copy_push, fill));
      value = nir_vec4(&b, word, word, word, word);
   } else {
      nir_variable *tex = nir_variable_create(b.shader, nir_var_uniform,
         glsl_sampler_type(GLSL_SAMPLER_DIM_BUF, false, false, GLSL_TYPE_UINT), "src");
      tex->data.descriptor_set = 0;
      tex->data.binding = 0;
      nir_def *xy = nir_f2u32(&b, nir_channels(&b, nir_load_frag_coord(&b), 3));
      nir_def *idx = nir_iadd(&b, nir_imul(&b, nir_channel(&b, xy, 1),
         copy_push(&b, offsetof(struct copy_push, width))), nir_channel(&b, xy, 0));
      if (scalar_src)
         idx = nir_imul_imm(&b, idx, 4);
      idx = nir_iadd(&b, idx, copy_push(&b, offsetof(struct copy_push, src_elem0)));
      nir_deref_instr *deref = nir_build_deref_var(&b, tex);
      if (scalar_src) {
         /* Vulkan buffer offsets may be only four-byte aligned. Assemble
          * four raw words without imposing extra destination eligibility. */
         nir_def *words[4];
         for (unsigned i = 0; i < 4; i++)
            words[i] = nir_channel(&b, nir_txf(&b, nir_iadd_imm(&b, idx, i),
                                               .texture_deref = deref), 0);
         value = nir_vec(&b, words, 4);
      } else {
         value = nir_txf(&b, idx, .texture_deref = deref);
      }
   }
   nir_store_var(&b, out, value, 0xf);
   const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
      .nir = b.shader,
   };
   const VkPipelineShaderStageCreateInfo fs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = &nir_info, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .pName = "main",
   };
   const VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
   };
   const VkPipelineDynamicStateCreateInfo dyn = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
   };
   const VkGraphicsPipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 1, .pStages = &fs, .layout = *layout,
      .pDepthStencilState = &ds, .pDynamicState = &dyn,
   };
   const struct vk_meta_rendering_info render = {
      .samples = 1, .color_attachment_count = 1,
      .color_attachment_formats = { COPY_FORMAT },
      .color_attachment_write_masks = { 0xf },
   };
   result = vk_meta_create_graphics_pipeline(dev, meta, &info, &render,
                                             &key, sizeof(key), pipeline);
   ralloc_free(b.shader);
   return result;
}

static bool
copy_eligible(struct pvr_cmd_buffer *cmd)
{
   return debug_get_option_g5p_gpu_copy() && cmd->device->g5p &&
          cmd->device->g5p->meta_ready && !cmd->state.vis_test_enabled &&
          cmd->vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY;
}

static bool
copy_region_eligible(struct pvr_buffer *dst, VkDeviceSize off, VkDeviceSize size)
{
   return !((dst->dev_addr.addr + off) % 64) && !(size % 16) && size <= COPY_MAX_BYTES;
}

static VkResult
copy_segment(struct pvr_cmd_buffer *cmd, VkBuffer src, VkDeviceSize src_off,
             struct pvr_buffer *dst, VkDeviceSize dst_off, uint32_t width,
             uint32_t rows, bool fill, uint32_t data)
{
   struct vk_meta_device *meta = &cmd->device->g5p->meta;
   const struct vk_device_dispatch_table *disp = &cmd->device->vk.dispatch_table;
   VkCommandBuffer cb = pvr_cmd_buffer_to_handle(cmd);
   const uint32_t size = width * rows * 16;
   const uint32_t pitch = ALIGN_POT(width * 16, 64);
   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = COPY_FORMAT,
      .extent = { width, rows, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
   };
   VkImage image_handle;
   VkResult result = vk_meta_create_image(&cmd->vk, meta, &image_info, &image_handle);
   if (result != VK_SUCCESS)
      return result;
   VK_FROM_HANDLE(pvr_image, image, image_handle);
   image->dev_addr = PVR_DEV_ADDR_OFFSET(dst->dev_addr, dst_off);
   image->memlayout = PVR_MEMLAYOUT_LINEAR;
   image->planes[0].offset = 0;
   image->planes[0].physical_extent = image_info.extent;
   image->planes[0].layer_size = size;
   image->planes[0].size = size;
   image->planes[0].mip_levels[0] = (struct pvr_mip_level) {
      .offset = 0, .size = size, .pitch = pitch, .height_pitch = rows,
   };
   image->total_size = size;
   const VkImageViewUsageCreateInfo usage = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
   };
   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &usage,
      .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
      .image = image_handle, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = COPY_FORMAT,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   };
   VkImageView view;
   result = vk_meta_create_image_view(&cmd->vk, meta, &view_info, &view);
   if (result != VK_SUCCESS)
      return result;
   bool scalar_src = src_off % 16 != 0;
   VkPipelineLayout layout;
   VkPipeline pipeline;
   result = copy_pipeline(cmd, fill, scalar_src, &layout, &pipeline);
   if (result != VK_SUCCESS)
      return result;
   struct copy_push push = { .width = width, .fill = data };
   disp->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   if (!fill) {
      uint32_t alignment = cmd->device->pdevice->vk.properties.minTexelBufferOffsetAlignment;
      VkDeviceSize base = src_off / alignment * alignment;
      uint32_t texel_size = scalar_src ? 4 : 16;
      const VkBufferViewCreateInfo src_info = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = src,
         .format = scalar_src ? VK_FORMAT_R32_UINT : COPY_FORMAT,
         .offset = base, .range = src_off - base + size,
      };
      VkBufferView src_view;
      result = vk_meta_create_buffer_view(&cmd->vk, meta, &src_info, &src_view);
      if (result != VK_SUCCESS)
         return result;
      push.src_elem0 = (src_off - base) / texel_size;
      const VkWriteDescriptorSet write = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0,
         .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
         .pTexelBufferView = &src_view,
      };
      disp->CmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &write);
   }
   disp->CmdPushConstants(cb, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
   const VkRenderingAttachmentInfo attachment = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   const VkRenderingInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea.extent = { width, rows },
      .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &attachment,
   };
   disp->CmdBeginRendering(cb, &rendering);
   const struct vk_meta_rect rect = { .x1 = width, .y1 = rows };
   vk_meta_draw_rects(&cmd->vk, meta, 1, &rect);
   disp->CmdEndRendering(cb);
   return VK_SUCCESS;
}

static VkResult
copy_region(struct pvr_cmd_buffer *cmd, VkBuffer src, VkDeviceSize src_off,
            struct pvr_buffer *dst, VkDeviceSize dst_off, VkDeviceSize size,
            bool fill, uint32_t data, unsigned *rects)
{
   while (size) {
      uint64_t available = size;
      if (!fill) {
         uint32_t align = cmd->device->pdevice->vk.properties.minTexelBufferOffsetAlignment;
         uint32_t texel_size = src_off % 16 ? 4 : 16;
         available = MIN2(available,
            (uint64_t)cmd->device->pdevice->vk.properties.maxTexelBufferElements * texel_size - src_off % align);
      }
      uint32_t rows = available / COPY_ROW_BYTES;
      uint32_t width = rows ? 2048 : MIN2(size, available) / 16;
      if (!rows)
         rows = 1;
      uint32_t bytes = width * rows * 16;
      VkResult result = copy_segment(cmd, src, src_off, dst, dst_off, width, rows, fill, data);
      if (result != VK_SUCCESS)
         return result;
      (*rects)++;
      size -= bytes;
      src_off += bytes;
      dst_off += bytes;
   }
   return VK_SUCCESS;
}

static bool
copy_run(struct pvr_cmd_buffer *cmd, const VkCopyBufferInfo2 *info, bool fill, uint32_t data)
{
   /* Resolve every shader variant before recording any region: a named
    * pipeline refusal must leave the whole call on the CPU path. */
   for (unsigned i = 0; i < info->regionCount; i++) {
      VkPipelineLayout layout;
      VkPipeline pipeline;
      VkResult result = copy_pipeline(cmd, fill, info->pRegions[i].srcOffset % 16 != 0,
                                      &layout, &pipeline);
      if (result != VK_SUCCESS) {
         if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
            vk_command_buffer_set_error(&cmd->vk, result);
            return true;
         }
         static int warned;
         if (!p_atomic_xchg(&warned, 1)) {
            fprintf(stderr, "G5P refuses: GPU buffer copy RGBA32UI pipeline (%d); keeping CPU path\n", result);
         }
         return false;
      }
   }
   struct g5p_meta_save *save = malloc(sizeof(*save));
   if (!save) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return true;
   }
   VkResult result = pvr_arch_cmd_buffer_end_sub_cmd(cmd);
   if (result != VK_SUCCESS) {
      free(save);
      return true;
   }
   struct list_head *mark = cmd->sub_cmds.prev;
   g5p_meta_begin(cmd, save);
   VK_FROM_HANDLE(pvr_buffer, dst, info->dstBuffer);
   uint64_t bytes = 0;
   unsigned rects = 0;
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkBufferCopy2 *r = &info->pRegions[i];
      result = copy_region(cmd, info->srcBuffer, r->srcOffset, dst, r->dstOffset,
                           r->size, fill, data, &rects);
      bytes += r->size;
      if (result != VK_SUCCESS)
         break;
   }
   g5p_meta_end(cmd, save);
   free(save);
   for (struct list_head *l = mark->next; l != &cmd->sub_cmds; l = l->next) {
      struct pvr_sub_cmd *sub = list_entry(l, struct pvr_sub_cmd, link);
      if (sub->type == PVR_SUB_CMD_TYPE_GRAPHICS)
         sub->gfx.g5p_transfer = true;
   }
   if (rects) {
      for (unsigned i = 0; i < ARRAY_SIZE(cmd->state.barriers_needed); i++)
         cmd->state.barriers_needed[i] |= PVR_PIPELINE_STAGE_TRANSFER_BIT;
   }
   if (result != VK_SUCCESS)
      vk_command_buffer_set_error(&cmd->vk, result);
   else
      pvr_arch_g5p_transfer_summary(true, 0, 0, info->regionCount, 0, 0);
   if (debug_get_option_g5p_perf_log() >= 3)
      fprintf(stderr, "G5P perf gpu-copy dsts=1 rects=%u bytes=%" PRIu64 " T=RGBA32UI\n", rects, bytes);
   return true;
}

bool
PVR_PER_ARCH(g5p_meta_copy_buffer)(struct pvr_cmd_buffer *cmd, const VkCopyBufferInfo2 *info)
{
   if (!copy_eligible(cmd))
      return false;
   VK_FROM_HANDLE(pvr_buffer, dst, info->dstBuffer);
   uint64_t total = 0;
   for (unsigned i = 0; i < info->regionCount; i++) {
      const VkBufferCopy2 *r = &info->pRegions[i];
      if (!copy_region_eligible(dst, r->dstOffset, r->size) || r->size > COPY_MAX_BYTES - total)
         return false;
      total += r->size;
   }
   return copy_run(cmd, info, false, 0);
}

bool
PVR_PER_ARCH(g5p_meta_fill_buffer)(struct pvr_cmd_buffer *cmd, VkBuffer buffer,
                                 VkDeviceSize off, VkDeviceSize size, uint32_t data)
{
   VK_FROM_HANDLE(pvr_buffer, dst, buffer);
   size = vk_buffer_range(&dst->vk, off, size) & ~3ull;
   if (!copy_eligible(cmd) || !copy_region_eligible(dst, off, size))
      return false;
   const VkBufferCopy2 region = { .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2, .dstOffset = off, .size = size };
   const VkCopyBufferInfo2 info = { .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
      .dstBuffer = buffer, .regionCount = 1, .pRegions = &region };
   return copy_run(cmd, &info, true, data);
}

bool
PVR_PER_ARCH(g5p_meta_update_buffer)(struct pvr_cmd_buffer *cmd, VkBuffer buffer,
                                   VkDeviceSize off, VkDeviceSize size, const void *data)
{
   VK_FROM_HANDLE(pvr_buffer, dst, buffer);
   if (!copy_eligible(cmd) || !copy_region_eligible(dst, off, size))
      return false;
   struct vk_meta_device *meta = &cmd->device->g5p->meta;
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
      .usage = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
   };
   VkBuffer src;
   void *map;
   VkResult result = vk_meta_create_buffer(&cmd->vk, meta, &buffer_info, &src);
   if (result == VK_SUCCESS)
      result = meta->cmd_bind_map_buffer(&cmd->vk, meta, src, &map);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return true;
   }
   memcpy(map, data, size);
   const VkBufferCopy2 region = { .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2, .dstOffset = off, .size = size };
   const VkCopyBufferInfo2 info = { .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
      .srcBuffer = src, .dstBuffer = buffer, .regionCount = 1, .pRegions = &region };
   return copy_run(cmd, &info, false, 0);
}
