/* SPDX-License-Identifier: MIT */
/*
 * Transfers (copies, blits, fills, buffer updates) on Apple's G9P, done by
 * the CPU: the A10's memory is all CPU-visible, and its GPU transfer path
 * (the TQ, Rogue USC shaders) does not exist here yet. A transfer sub-command
 * waits for its dependencies, runs its commands on the mapped memory and
 * signals its sync from the CPU.
 *
 * Surfaces are linear (row stride in texels) or twiddled: Morton order with
 * the y bit first over the power-of-two padded size, interleaving while both
 * dimensions have bits (the A10 notes' RENDER.md "Textures with our
 * encodings"). Copies convert formats through util_format when they differ
 * and scale with nearest or linear sampling; MSAA resolves and YUV are not
 * handled.
 *
 * The same copies serve VK_EXT_host_image_copy: an upload goes straight from
 * the application's memory into the image, without a staging buffer and a
 * second copy (Zink's glTexSubImage, a compositor's window contents).
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "vk_format.h"
#include "vk_sync.h"

#include "util/format/u_format.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"

#include "pvr_cmd_buffer.h"
#include "pvr_device.h"
#include "pvr_entrypoints.h"
#include "pvr_g9p.h"
#include "pvr_image.h"
#include "pvr_job_transfer.h"
#include "pvr_winsys.h"

/* The CPU address of a device address in a bound buffer or image, or NULL */
uint8_t *
PVR_PER_ARCH(g9p_cpu_addr)(struct pvr_device *device, uint64_t addr)
{
   uint8_t *ptr = NULL;

   simple_mtx_lock(&device->g9p_vmas.lock);
   util_dynarray_foreach (&device->g9p_vmas.vmas, struct pvr_winsys_vma *, v) {
      struct pvr_winsys_vma *vma = *v;

      if (addr < vma->dev_addr.addr || addr >= vma->dev_addr.addr + vma->size)
         continue;
      if (!vma->bo->map && device->ws->ops->buffer_map(vma->bo, NULL) != VK_SUCCESS)
         break;
      ptr = (uint8_t *)vma->bo->map + vma->bo_offset + (addr - vma->dev_addr.addr);
      break;
   }
   simple_mtx_unlock(&device->g9p_vmas.lock);
   return ptr;
}

struct surface {
   uint8_t *base;
   enum pipe_format format;
   unsigned bpp, stride, width, height;
   /* 1 or 4; four samples are a twiddled surface of twice the size (2x2 per pixel) */
   unsigned samples;
   bool twiddled;
};

static bool
surface_init(struct pvr_device *device, const struct pvr_transfer_cmd_surface *s,
             struct surface *out)
{
   out->base = PVR_PER_ARCH(g9p_cpu_addr)(device, s->dev_addr.addr);
   out->format = vk_format_to_pipe_format(s->vk_format);
   out->bpp = vk_format_get_blocksize(s->vk_format);
   out->stride = s->stride ? s->stride : s->width;
   out->width = s->width;
   out->height = s->height;
   out->twiddled = s->mem_layout == PVR_MEMLAYOUT_TWIDDLED;
   out->samples = s->sample_count == 4 ? 4 : 1;
   if (out->samples == 4) {
      out->width *= 2;
      out->height *= 2;
   }
   if (!out->base)
      mesa_loge("G9P transfer: no CPU mapping for %#" PRIx64, s->dev_addr.addr);
   return out->base != NULL;
}

static unsigned
twiddle(unsigned x, unsigned y, unsigned w, unsigned h)
{
   const unsigned pw = util_next_power_of_two(MAX2(w, 1));
   const unsigned ph = util_next_power_of_two(MAX2(h, 1));
   unsigned m = 0, bit = 0;

   for (unsigned i = 0; (1u << i) < pw || (1u << i) < ph; i++) {
      if ((1u << i) < ph)
         m |= ((y >> i) & 1) << bit++;
      if ((1u << i) < pw)
         m |= ((x >> i) & 1) << bit++;
   }
   return m;
}

/*
 * A texel's byte offset is a column part plus a row part: in twiddled
 * surfaces the x and y bits of the index are interleaved, disjoint
 */
static size_t
column_offset(const struct surface *s, unsigned x)
{
   if (s->twiddled)
      return (size_t)twiddle(x, 0, s->width, s->height) * s->bpp;
   return (size_t)x * s->bpp;
}

static size_t
row_offset(const struct surface *s, unsigned y)
{
   if (s->twiddled)
      return (size_t)twiddle(0, y, s->width, s->height) * s->bpp;
   return (size_t)y * s->stride * s->bpp;
}

static void
fill(const struct pvr_transfer_cmd *cmd, const struct surface *dst)
{
   const unsigned ms = dst->samples == 4 ? 2 : 1;
   const VkRect2D scaled = { { cmd->scissor.offset.x * ms, cmd->scissor.offset.y * ms },
                             { cmd->scissor.extent.width * ms, cmd->scissor.extent.height * ms } };
   const VkRect2D *r = &scaled;
   uint8_t packed[16] = { 0 };

   if (util_format_is_pure_integer(dst->format))
      util_format_pack_rgba(dst->format, packed, (const uint32_t[4]){ cmd->clear_color[0].ui,
                                                                       cmd->clear_color[1].ui,
                                                                       cmd->clear_color[2].ui,
                                                                       cmd->clear_color[3].ui },
                            1);
   else
      util_format_pack_rgba(dst->format, packed, (const float[4]){ cmd->clear_color[0].f,
                                                                    cmd->clear_color[1].f,
                                                                    cmd->clear_color[2].f,
                                                                    cmd->clear_color[3].f },
                            1);
   size_t *cols = malloc(r->extent.width * sizeof(*cols));

   if (!cols)
      return;
   for (unsigned x = 0; x < r->extent.width; x++)
      cols[x] = column_offset(dst, r->offset.x + x);
   for (unsigned y = 0; y < r->extent.height; y++) {
      uint8_t *row = dst->base + row_offset(dst, r->offset.y + y);

      for (unsigned x = 0; x < r->extent.width; x++)
         memcpy(row + cols[x], packed, dst->bpp);
   }
   free(cols);
}

/*
 * D32_SFLOAT_S8_UINT is two planes in one allocation, as the ZLS has it
 * (g9p_ds_stencil_offset()): D32 first, S8 after it. A surface's view of one
 * plane, unchanged for a surface that holds only that aspect.
 */
static struct surface
ds_plane(const struct surface *s, bool depth)
{
   struct surface p = *s;

   if (s->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) {
      p.bpp = depth ? 4 : 1;
      if (!depth)
         p.base += (size_t)(s->twiddled ? util_next_power_of_two(s->width) : s->stride) *
                   (s->twiddled ? util_next_power_of_two(s->height) : s->height) * 4;
   }
   return p;
}

/* One aspect: the format pack functions have no depth/stencil conversions */
static void
copy_ds_aspect(const struct pvr_rect_mapping *m, const struct surface *src,
               const struct surface *dst, bool depth)
{
   const struct surface sp = ds_plane(src, depth), dp = ds_plane(dst, depth);

   for (unsigned y = 0; y < m->dst_rect.extent.height; y++) {
      uint8_t *d = dp.base + row_offset(&dp, m->dst_rect.offset.y + y);
      const uint8_t *s = sp.base + row_offset(&sp, m->src_rect.offset.y + y);

      for (unsigned x = 0; x < m->dst_rect.extent.width; x++)
         memcpy(d + column_offset(&dp, m->dst_rect.offset.x + x),
                s + column_offset(&sp, m->src_rect.offset.x + x), dp.bpp);
   }
}

static void
copy(const struct pvr_rect_mapping *m, const struct surface *src, const struct surface *dst)
{
   const bool same = src->format == dst->format ||
                     (src->bpp == dst->bpp && util_format_is_compressed(src->format));
   const bool integer = util_format_is_pure_integer(src->format);
   const unsigned dw = m->dst_rect.extent.width, dh = m->dst_rect.extent.height;
   const unsigned sw = m->src_rect.extent.width, sh = m->src_rect.extent.height;

   /* linear to linear, unscaled and unflipped: whole rows */
   if (same && !src->twiddled && !dst->twiddled && sw == dw && sh == dh && !m->flip_x &&
       !m->flip_y) {
      for (unsigned y = 0; y < dh; y++)
         memcpy(dst->base + row_offset(dst, m->dst_rect.offset.y + y) +
                   column_offset(dst, m->dst_rect.offset.x),
                src->base + row_offset(src, m->src_rect.offset.y + y) +
                   column_offset(src, m->src_rect.offset.x),
                (size_t)dw * dst->bpp);
      return;
   }

   /* each destination column's source and destination offsets */
   size_t *cols = malloc(2 * dw * sizeof(*cols));

   if (!cols)
      return;
   for (unsigned x = 0; x < dw; x++) {
      /* nearest: the source texel under the destination texel's centre */
      unsigned sx = sw == dw ? x : (2 * x + 1) * sw / (2 * dw);

      if (m->flip_x)
         sx = sw - 1 - sx;
      cols[2 * x] = column_offset(src, m->src_rect.offset.x + sx);
      cols[2 * x + 1] = column_offset(dst, m->dst_rect.offset.x + x);
   }
   for (unsigned y = 0; y < dh; y++) {
      unsigned sy = sh == dh ? y : (2 * y + 1) * sh / (2 * dh);

      if (m->flip_y)
         sy = sh - 1 - sy;

      const uint8_t *src_row = src->base + row_offset(src, m->src_rect.offset.y + sy);
      uint8_t *dst_row = dst->base + row_offset(dst, m->dst_rect.offset.y + y);

      if (same && dst->bpp == 4) {
         for (unsigned x = 0; x < dw; x++)
            *(uint32_t *)(dst_row + cols[2 * x + 1]) = *(const uint32_t *)(src_row + cols[2 * x]);
         continue;
      }
      for (unsigned x = 0; x < dw; x++) {
         const uint8_t *from = src_row + cols[2 * x];
         uint8_t *to = dst_row + cols[2 * x + 1];

         if (same) {
            memcpy(to, from, dst->bpp);
         } else if (integer) {
            uint32_t v[4];

            util_format_unpack_rgba(src->format, v, from, 1);
            util_format_pack_rgba(dst->format, to, v, 1);
         } else {
            float v[4];

            util_format_unpack_rgba(src->format, v, from, 1);
            util_format_pack_rgba(dst->format, to, v, 1);
         }
      }
   }
   free(cols);
}

/* A scaled blit with linear filtering; texels past the source's edges clamp */
static void
copy_linear(const struct pvr_rect_mapping *m, const struct surface *src, const struct surface *dst)
{
   const unsigned dw = m->dst_rect.extent.width, dh = m->dst_rect.extent.height;
   const unsigned sw = m->src_rect.extent.width, sh = m->src_rect.extent.height;

   for (unsigned y = 0; y < dh; y++) {
      float v = (y + 0.5f) * sh / dh;

      if (m->flip_y)
         v = sh - v;
      v += m->src_rect.offset.y - 0.5f;

      const int y0 = floorf(v);
      const float fy = v - y0;
      const unsigned rows[2] = { CLAMP(y0, 0, (int)src->height - 1),
                                 CLAMP(y0 + 1, 0, (int)src->height - 1) };
      uint8_t *dst_row = dst->base + row_offset(dst, m->dst_rect.offset.y + y);

      for (unsigned x = 0; x < dw; x++) {
         float u = (x + 0.5f) * sw / dw;

         if (m->flip_x)
            u = sw - u;
         u += m->src_rect.offset.x - 0.5f;

         const int x0 = floorf(u);
         const float fx = u - x0;
         const unsigned cols[2] = { CLAMP(x0, 0, (int)src->width - 1),
                                    CLAMP(x0 + 1, 0, (int)src->width - 1) };
         float sum[4] = { 0 };

         for (unsigned j = 0; j < 2; j++) {
            for (unsigned i = 0; i < 2; i++) {
               const float weight = (j ? fy : 1 - fy) * (i ? fx : 1 - fx);
               float texel[4];

               util_format_unpack_rgba(src->format, texel,
                                       src->base + row_offset(src, rows[j]) +
                                          column_offset(src, cols[i]),
                                       1);
               for (unsigned c = 0; c < 4; c++)
                  sum[c] += weight * texel[c];
            }
         }
         util_format_pack_rgba(dst->format, dst_row + column_offset(dst, m->dst_rect.offset.x + x),
                               sum, 1);
      }
   }
}

/* A rectangle in a four-sample surface's texels (2x2 per pixel) */
static VkRect2D
ms_rect(const VkRect2D *r)
{
   return (VkRect2D){ { r->offset.x * 2, r->offset.y * 2 }, { r->extent.width * 2, r->extent.height * 2 } };
}

/*
 * Four samples into one: the average, or sample 0 for integers and depth/stencil. Nearest
 * for a scaled mapping, as a resolve has none in Vulkan.
 */
static void
resolve(const struct pvr_rect_mapping *m, const struct surface *src, const struct surface *dst)
{
   const bool first = util_format_is_pure_integer(src->format) || util_format_is_depth_or_stencil(src->format);

   for (unsigned y = 0; y < m->dst_rect.extent.height; y++) {
      const unsigned sy = m->src_rect.offset.y + y * m->src_rect.extent.height / m->dst_rect.extent.height;
      uint8_t *row = dst->base + row_offset(dst, m->dst_rect.offset.y + y);

      for (unsigned x = 0; x < m->dst_rect.extent.width; x++) {
         const unsigned sx = m->src_rect.offset.x + x * m->src_rect.extent.width / m->dst_rect.extent.width;
         uint8_t *d = row + column_offset(dst, m->dst_rect.offset.x + x);

         if (first) {
            memcpy(d, src->base + row_offset(src, 2 * sy) + column_offset(src, 2 * sx), dst->bpp);
            continue;
         }

         float sum[4] = { 0 };

         for (unsigned s = 0; s < 4; s++) {
            float texel[4];

            util_format_unpack_rgba(src->format, texel,
                                    src->base + row_offset(src, 2 * sy + (s & 1)) +
                                       column_offset(src, 2 * sx + (s >> 1)),
                                    1);
            for (unsigned c = 0; c < 4; c++)
               sum[c] += texel[c] / 4;
         }
         util_format_pack_rgba(dst->format, d, sum, 1);
      }
   }
}

/* A mapping between surfaces whichever their sample counts */
static void
copy_samples(const struct pvr_rect_mapping *m, const struct surface *src, const struct surface *dst)
{
   if (src->samples == 4 && dst->samples == 1) {
      resolve(m, src, dst);
   } else if (src->samples == dst->samples && src->samples == 4) {
      struct pvr_rect_mapping scaled = *m;

      scaled.src_rect = ms_rect(&m->src_rect);
      scaled.dst_rect = ms_rect(&m->dst_rect);
      copy(&scaled, src, dst);
   } else if (dst->samples == 4) {
      /* one sample into all four */
      struct pvr_rect_mapping scaled = *m;

      scaled.dst_rect = ms_rect(&m->dst_rect);
      copy(&scaled, src, dst);
   } else {
      copy(m, src, dst);
   }
}

static void
run(struct pvr_device *device, const struct pvr_transfer_cmd *cmd)
{
   struct surface dst;

   if (!surface_init(device, &cmd->dst, &dst))
      return;
   if (cmd->flags & PVR_TRANSFER_CMD_FLAGS_FILL) {
      fill(cmd, &dst);
      return;
   }
   for (unsigned i = 0; i < cmd->source_count; i++) {
      const struct pvr_transfer_cmd_source *source = &cmd->sources[i];
      struct surface src;

      if (!surface_init(device, &source->surface, &src))
         continue;
      if (cmd->flags & PVR_TRANSFER_CMD_FLAGS_DSMERGE) {
         for (unsigned k = 0; k < source->mapping_count; k++)
            copy_ds_aspect(&source->mappings[k], &src, &dst,
                           cmd->flags & PVR_TRANSFER_CMD_FLAGS_PICKD);
         continue;
      }
      /* Both aspects of D32F_S8: each plane by itself */
      if (src.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT &&
          dst.format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT && source->mapping_count) {
         for (unsigned k = 0; k < source->mapping_count; k++) {
            for (unsigned depth = 0; depth < 2; depth++) {
               struct surface sp = ds_plane(&src, depth), dp = ds_plane(&dst, depth);

               sp.format = dp.format = depth ? PIPE_FORMAT_Z32_FLOAT : PIPE_FORMAT_S8_UINT;
               copy_samples(&source->mappings[k], &sp, &dp);
            }
         }
         continue;
      }
      if (!source->mapping_count) {
         /* the whole destination scissor, 1:1 */
         const struct pvr_rect_mapping m = { .src_rect = cmd->scissor,
                                             .dst_rect = cmd->scissor };

         copy_samples(&m, &src, &dst);
      }
      for (unsigned k = 0; k < source->mapping_count; k++) {
         const struct pvr_rect_mapping *m = &source->mappings[k];

         if (source->filter == PVR_FILTER_LINEAR && !util_format_is_pure_integer(src.format) &&
             (m->src_rect.extent.width != m->dst_rect.extent.width ||
              m->src_rect.extent.height != m->dst_rect.extent.height))
            copy_linear(m, &src, &dst);
         else
            copy_samples(m, &src, &dst);
      }
   }
}

VkResult
PVR_PER_ARCH(g9p_transfer_submit)(struct pvr_device *device,
                                  struct pvr_sub_cmd_transfer *sub_cmd,
                                  struct vk_sync *wait_sync, struct vk_sync *signal_sync)
{
   VkResult result;

   if (wait_sync) {
      result = vk_sync_wait(&device->vk, wait_sync, 0, VK_SYNC_WAIT_COMPLETE, UINT64_MAX);
      if (result != VK_SUCCESS)
         return result;
   }
   list_for_each_entry (struct pvr_transfer_cmd, cmd, sub_cmd->transfer_cmds, link)
      run(device, cmd);
   return vk_sync_signal(&device->vk, signal_sync, 0);
}

/* --- VK_EXT_host_image_copy -------------------------------------------- */

/* Layer @layer of the subresource @sub of @image, at its mip level's size */
static bool
host_image_surface(struct pvr_device *device, const struct pvr_image *image,
                   const VkImageSubresourceLayers *sub, uint32_t layer, struct surface *out)
{
   const VkImageSubresource s = { sub->aspectMask, sub->mipLevel, sub->baseArrayLayer + layer };
   VkSubresourceLayout info;

   if (image->memlayout == PVR_MEMLAYOUT_3DTWIDDLED)
      return false;
   pvr_get_image_subresource_layout(image, &s, &info);
   out->base = PVR_PER_ARCH(g9p_cpu_addr)(device, image->dev_addr.addr + info.offset);
   out->format = vk_format_to_pipe_format(image->vk.format);
   out->bpp = vk_format_get_blocksize(image->vk.format);
   out->stride = info.rowPitch / out->bpp;
   out->width = MAX2(image->vk.extent.width >> sub->mipLevel, 1);
   out->height = MAX2(image->vk.extent.height >> sub->mipLevel, 1);
   out->twiddled = image->memlayout == PVR_MEMLAYOUT_TWIDDLED;
   return out->base != NULL;
}

/* Host memory holding @extent texels of @image's format at @row_length/@image_height */
static void
host_memory_surface(const struct pvr_image *image, const void *ptr, uint32_t row_length,
                    uint32_t image_height, const VkExtent3D *extent, struct surface *out)
{
   out->base = (uint8_t *)ptr;
   out->format = vk_format_to_pipe_format(image->vk.format);
   out->bpp = vk_format_get_blocksize(image->vk.format);
   out->stride = row_length ? row_length : extent->width;
   out->width = out->stride;
   out->height = image_height ? image_height : extent->height;
   out->twiddled = false;
}

static VkResult
host_copy(struct pvr_device *device, const struct pvr_image *image, bool to_image,
          VkHostImageCopyFlags flags, const void *ptr, uint32_t row_length,
          uint32_t image_height, const VkImageSubresourceLayers *sub, const VkOffset3D *offset,
          const VkExtent3D *extent)
{
   const uint32_t layers = vk_image_subresource_layer_count(&image->vk, sub);

   for (uint32_t l = 0; l < layers; l++) {
      struct surface img, mem;

      if (!host_image_surface(device, image, sub, l, &img))
         return vk_error(device, VK_ERROR_MEMORY_MAP_FAILED);
      if (flags & VK_HOST_IMAGE_COPY_MEMCPY_BIT) {
         /* the subresource as it is laid out, back to back per layer */
         const size_t size = (size_t)img.stride * (img.twiddled ? util_next_power_of_two(img.height)
                                                                : img.height) * img.bpp;
         uint8_t *host = (uint8_t *)ptr + l * size;

         if (to_image)
            memcpy(img.base, host, size);
         else
            memcpy(host, img.base, size);
         continue;
      }
      /* The memory holds one aspect, packed */
      if (image->vk.format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
         const bool depth = sub->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT;
         const unsigned size = depth ? 4 : 1;

         if (!depth && (image->vk.mip_levels > 1 || image->vk.array_layers > 1))
            continue; /* stencil planes only of single level images */

         const size_t row = (size_t)(row_length ? row_length : extent->width) * size;
         const size_t slice = row * (image_height ? image_height : extent->height);
         const struct surface plane = ds_plane(&img, depth);
         uint8_t *host = (uint8_t *)ptr + l * slice;

         for (uint32_t y = 0; y < extent->height; y++) {
            const size_t ro = row_offset(&plane, offset->y + y);

            for (uint32_t x = 0; x < extent->width; x++) {
               uint8_t *texel = plane.base + ro + column_offset(&plane, offset->x + x);
               uint8_t *m = host + y * row + (size_t)x * size;

               memcpy(to_image ? texel : m, to_image ? m : texel, size);
            }
         }
         continue;
      }
      host_memory_surface(image, ptr, row_length, image_height, extent, &mem);
      mem.base += (size_t)l * mem.stride * mem.height * mem.bpp;

      const VkRect2D at_image = { { offset->x, offset->y }, { extent->width, extent->height } };
      const VkRect2D at_memory = { { 0, 0 }, { extent->width, extent->height } };
      const struct pvr_rect_mapping m = { .src_rect = to_image ? at_memory : at_image,
                                          .dst_rect = to_image ? at_image : at_memory };

      copy(&m, to_image ? &mem : &img, to_image ? &img : &mem);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
PVR_PER_ARCH(CopyMemoryToImage)(VkDevice _device, const VkCopyMemoryToImageInfo *info)
{
   VK_FROM_HANDLE(pvr_device, device, _device);
   VK_FROM_HANDLE(pvr_image, image, info->dstImage);

   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkMemoryToImageCopy *r = &info->pRegions[i];
      VkResult result = host_copy(device, image, true, info->flags, r->pHostPointer,
                                  r->memoryRowLength, r->memoryImageHeight, &r->imageSubresource,
                                  &r->imageOffset, &r->imageExtent);

      if (result != VK_SUCCESS)
         return result;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
PVR_PER_ARCH(CopyImageToMemory)(VkDevice _device, const VkCopyImageToMemoryInfo *info)
{
   VK_FROM_HANDLE(pvr_device, device, _device);
   VK_FROM_HANDLE(pvr_image, image, info->srcImage);

   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageToMemoryCopy *r = &info->pRegions[i];
      VkResult result = host_copy(device, image, false, info->flags, r->pHostPointer,
                                  r->memoryRowLength, r->memoryImageHeight, &r->imageSubresource,
                                  &r->imageOffset, &r->imageExtent);

      if (result != VK_SUCCESS)
         return result;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
PVR_PER_ARCH(CopyImageToImage)(VkDevice _device, const VkCopyImageToImageInfo *info)
{
   VK_FROM_HANDLE(pvr_device, device, _device);
   VK_FROM_HANDLE(pvr_image, src_image, info->srcImage);
   VK_FROM_HANDLE(pvr_image, dst_image, info->dstImage);

   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageCopy2 *r = &info->pRegions[i];
      const uint32_t layers = vk_image_subresource_layer_count(&src_image->vk, &r->srcSubresource);

      for (uint32_t l = 0; l < layers; l++) {
         struct surface src, dst;

         if (!host_image_surface(device, src_image, &r->srcSubresource, l, &src) ||
             !host_image_surface(device, dst_image, &r->dstSubresource, l, &dst))
            return vk_error(device, VK_ERROR_MEMORY_MAP_FAILED);

         const struct pvr_rect_mapping m = {
            .src_rect = { { r->srcOffset.x, r->srcOffset.y }, { r->extent.width, r->extent.height } },
            .dst_rect = { { r->dstOffset.x, r->dstOffset.y }, { r->extent.width, r->extent.height } },
         };

         copy(&m, &src, &dst);
      }
   }
   return VK_SUCCESS;
}

/* Layouts change nothing in G9P memory */
VKAPI_ATTR VkResult VKAPI_CALL
PVR_PER_ARCH(TransitionImageLayout)(VkDevice device, uint32_t transitionCount,
                                    const VkHostImageLayoutTransitionInfo *pTransitions)
{
   return VK_SUCCESS;
}
