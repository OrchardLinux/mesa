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
#include "vk_enum_to_str.h"
#include "vk_sync.h"
#include "vk_meta.h"
#include "vk_internal_exts.h"
#include "util/ralloc.h"
#include "compiler/nir/nir_builder.h"
#include "util/bitset.h"

#include "util/format/u_format.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#include "util/os_time.h"

#include "pvr_buffer.h"
#include "pvr_cmd_buffer.h"
#include "pvr_physical_device.h"
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
   /* D32F_S8: from the depth to the stencil, 0 when unknown */
   uint64_t stencil_offset;
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
   out->stencil_offset = s->g9p_stencil_offset;
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
      if (!depth && s->stencil_offset)
         p.base += s->stencil_offset;
      else if (!depth)
         p.base += (size_t)(s->twiddled ? util_next_power_of_two(s->width) : s->stride) *
                   (s->twiddled ? util_next_power_of_two(s->height) : s->height) * 4;
   }
   return p;
}

/* One aspect: the format pack functions have no depth/stencil conversions */
static void copy_samples(const struct pvr_rect_mapping *m, const struct surface *src, const struct surface *dst);

static void
copy_ds_aspect(const struct pvr_rect_mapping *m, const struct surface *src,
               const struct surface *dst, bool depth)
{
   const struct surface sp = ds_plane(src, depth), dp = ds_plane(dst, depth);

   /* samples to resolve or scale: the plane's own format through copy_samples() */
   if (src->samples != 1 || dst->samples != 1) {
      struct surface a = sp, b = dp;

      if (src->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT)
         a.format = b.format = depth ? PIPE_FORMAT_Z32_FLOAT : PIPE_FORMAT_S8_UINT;
      copy_samples(m, &a, &b);
      return;
   }
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

   /*
    * RGBA8 <-> BGRA8 (unorm or both sRGB), linear to linear, unscaled: whole rows through a
    * cached buffer with red and blue swapped. Firefox copies its 1024x1024 WebGL frame into a
    * BGRA image every frame: through the unpack/pack below that took 28 ms.
    */
   {
      const enum pipe_format a = src->format, b = dst->format;
      const bool swap = (a == PIPE_FORMAT_R8G8B8A8_UNORM && b == PIPE_FORMAT_B8G8R8A8_UNORM) ||
                        (a == PIPE_FORMAT_B8G8R8A8_UNORM && b == PIPE_FORMAT_R8G8B8A8_UNORM) ||
                        (a == PIPE_FORMAT_R8G8B8A8_SRGB && b == PIPE_FORMAT_B8G8R8A8_SRGB) ||
                        (a == PIPE_FORMAT_B8G8R8A8_SRGB && b == PIPE_FORMAT_R8G8B8A8_SRGB) ||
                        (a == PIPE_FORMAT_R8G8B8X8_UNORM && b == PIPE_FORMAT_B8G8R8X8_UNORM) ||
                        (a == PIPE_FORMAT_B8G8R8X8_UNORM && b == PIPE_FORMAT_R8G8B8X8_UNORM) ||
                        (a == PIPE_FORMAT_R8G8B8A8_UNORM && b == PIPE_FORMAT_B8G8R8X8_UNORM) ||
                        (a == PIPE_FORMAT_B8G8R8A8_UNORM && b == PIPE_FORMAT_R8G8B8X8_UNORM);

      if (swap && !src->twiddled && !dst->twiddled && sw == dw && sh == dh && !m->flip_x &&
          !m->flip_y && src->samples == 1 && dst->samples == 1) {
         uint32_t *tmp = malloc((size_t)dw * 4);

         if (tmp) {
            for (unsigned y = 0; y < dh; y++) {
               memcpy(tmp, src->base + row_offset(src, m->src_rect.offset.y + y) +
                              column_offset(src, m->src_rect.offset.x),
                      (size_t)dw * 4);
               for (unsigned x = 0; x < dw; x++) {
                  const uint32_t v = tmp[x];

                  tmp[x] = (v & 0xff00ff00u) | ((v & 0xffu) << 16) | ((v >> 16) & 0xffu);
               }
               memcpy(dst->base + row_offset(dst, m->dst_rect.offset.y + y) +
                         column_offset(dst, m->dst_rect.offset.x),
                      tmp, (size_t)dw * 4);
            }
            free(tmp);
            return;
         }
      }
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
   /* G9P_TIME: how long each CPU transfer waited for the GPU and copied, and how much */
   const bool timing = getenv("G9P_TIME") != NULL;
   const int64_t t0 = timing ? os_time_get_nano() : 0;

   if (wait_sync) {
      result = vk_sync_wait(&device->vk, wait_sync, 0, VK_SYNC_WAIT_COMPLETE, UINT64_MAX);
      if (result != VK_SUCCESS)
         return result;
   }
   const int64_t t1 = timing ? os_time_get_nano() : 0;
   uint64_t texels = 0;
   unsigned n = 0;

   list_for_each_entry (struct pvr_transfer_cmd, cmd, sub_cmd->transfer_cmds, link) {
      run(device, cmd);
      if (timing) {
         n++;
         unsigned mapped = 0;

         for (unsigned i = 0; i < cmd->source_count; i++)
            for (unsigned m = 0; m < cmd->sources[i].mapping_count; m++, mapped++)
               texels += (uint64_t)cmd->sources[i].mappings[m].dst_rect.extent.width *
                         cmd->sources[i].mappings[m].dst_rect.extent.height;
         if (!mapped)
            texels += (uint64_t)cmd->scissor.extent.width * cmd->scissor.extent.height;
      }
   }
   if (timing) {
      const struct pvr_transfer_cmd *c0 =
         list_first_entry(sub_cmd->transfer_cmds, struct pvr_transfer_cmd, link);

      fprintf(stderr, "g9p: transfer %u cmds, %" PRIu64 " texels: waited %" PRId64 " us, copied %" PRId64 " us;"
              " dst %s %ux%u layout %d, src %s layout %d stride %u\n",
              n, texels, (t1 - t0) / 1000, (os_time_get_nano() - t1) / 1000,
              vk_Format_to_str(c0->dst.vk_format), c0->dst.width, c0->dst.height, c0->dst.mem_layout,
              c0->source_count ? vk_Format_to_str(c0->sources[0].surface.vk_format) : "-",
              c0->source_count ? c0->sources[0].surface.mem_layout : -1,
              c0->source_count ? c0->sources[0].surface.stride : 0);
   }
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

/* --- GPU blits and copies (vk_meta) -------------------------------------- */

/*
 * Image to image blits and same-format copies drawn by the GPU: Mesa's vk_meta draws a
 * rectangle sampling the source into the destination, through this driver's own
 * pipelines. The CPU path above reads the source through an uncached mapping: a
 * 1024x1024 copy (Firefox's WebGL frame, every frame) took 25-75 ms there, and waited
 * for the GPU before it. Depth/stencil, compressed and multisampled images, 3D and
 * array images stay on the CPU path.
 */

static VkResult
g9p_meta_bind_map_buffer(struct vk_command_buffer *cmd, struct vk_meta_device *meta,
                         VkBuffer _buffer, void **map_out)
{
   struct pvr_cmd_buffer *cmd_buffer = container_of(cmd, struct pvr_cmd_buffer, vk);
   VK_FROM_HANDLE(pvr_buffer, buffer, _buffer);
   struct pvr_suballoc_bo *bo;
   void *zeros = calloc(1, buffer->vk.size);
   VkResult result;

   if (!zeros)
      return vk_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY);
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, zeros, buffer->vk.size, &bo);
   free(zeros);
   if (result != VK_SUCCESS)
      return result;
   buffer->dev_addr = bo->dev_addr;
   *map_out = pvr_bo_suballoc_get_map_addr(bo);
   return VK_SUCCESS;
}

VkResult
PVR_PER_ARCH(g9p_meta_init)(struct pvr_device *device)
{
   VkResult result = vk_meta_device_init(&device->vk, &device->meta);

   if (result != VK_SUCCESS)
      return result;
   device->meta.cmd_bind_map_buffer = g9p_meta_bind_map_buffer;
   /* the rectangle vertex shader comes with the topology (vk_meta_create_graphics_pipeline()) */
   device->meta.use_rect_list_pipeline = true;
   device->meta.max_bind_map_buffer_size_B = 64 * 1024;
   device->meta_ready = true;
   return VK_SUCCESS;
}

void
PVR_PER_ARCH(g9p_meta_finish)(struct pvr_device *device)
{
   if (device->meta_ready)
      vk_meta_device_finish(&device->vk, &device->meta);
   device->meta_ready = false;
}

struct g9p_meta_save {
   struct vk_dynamic_graphics_state dyn;
   const struct pvr_graphics_pipeline *pipeline;
   struct pvr_vertex_binding vertex_binding0;
   struct pvr_descriptor_state desc;
   struct pvr_push_constants push[2];
};

static void
g9p_meta_begin(struct pvr_cmd_buffer *cmd_buffer, struct g9p_meta_save *save)
{
   struct pvr_cmd_buffer_state *state = &cmd_buffer->state;

   save->dyn = cmd_buffer->vk.dynamic_graphics_state;
   save->pipeline = state->gfx_pipeline;
   save->vertex_binding0 = state->vertex_bindings[0];
   save->desc = state->gfx_desc_state;
   save->push[0] = state->push_consts[PVR_STAGE_ALLOCATION_VERTEX_GEOMETRY];
   save->push[1] = state->push_consts[PVR_STAGE_ALLOCATION_FRAGMENT];
}

static void
g9p_meta_end(struct pvr_cmd_buffer *cmd_buffer, const struct g9p_meta_save *save)
{
   struct pvr_cmd_buffer_state *state = &cmd_buffer->state;

   cmd_buffer->vk.dynamic_graphics_state = save->dyn;
   BITSET_SET_RANGE(cmd_buffer->vk.dynamic_graphics_state.dirty, 0, MESA_VK_DYNAMIC_GRAPHICS_STATE_ENUM_MAX - 1);
   state->gfx_pipeline = save->pipeline;
   state->vertex_bindings[0] = save->vertex_binding0;
   state->gfx_desc_state = save->desc;
   state->gfx_desc_state.dirty_sets = ~0u;
   state->gfx_desc_state.push_set_dirty = state->gfx_desc_state.push_set != NULL;
   state->push_consts[PVR_STAGE_ALLOCATION_VERTEX_GEOMETRY] = save->push[0];
   state->push_consts[PVR_STAGE_ALLOCATION_FRAGMENT] = save->push[1];
   state->push_consts[PVR_STAGE_ALLOCATION_VERTEX_GEOMETRY].dirty = true;
   state->push_consts[PVR_STAGE_ALLOCATION_FRAGMENT].dirty = true;
   state->dirty.gfx_pipeline_binding = true;
   state->dirty.gfx_desc_dirty = true;
   state->dirty.vertex_bindings = true;
   state->dirty.vertex_descriptors = true;
   state->dirty.fragment_descriptors = true;
}

/* A colour image the GPU can sample (src) or draw into (dst), one layer, one sample */
static bool
g9p_meta_image_ok(struct pvr_device *device, const struct pvr_image *image, VkFormat format,
                  bool dst)
{
   VkFormatProperties3 props3 = { .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3 };
   VkFormatProperties2 props = { .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &props3 };
   const VkFormatFeatureFlags2 need = dst ? VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT
                                          : VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT;

   /*
    * On by default: the hang with Firefox's blits was the kernel's interleaving of two
    * processes' submissions (fixed there); the aquarium went from 34 to 37-40 fps.
    * G9P_NO_GPU_BLIT: the CPU path.
    */
   if (getenv("G9P_NO_GPU_BLIT"))
      return false;
   if (image->vk.image_type != VK_IMAGE_TYPE_2D || image->vk.samples != VK_SAMPLE_COUNT_1_BIT ||
       image->vk.array_layers != 1 || vk_format_is_depth_or_stencil(format) ||
       vk_format_is_compressed(format) || image->vk.format != format)
      return false;
   if (!(image->vk.usage & (dst ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : VK_IMAGE_USAGE_SAMPLED_BIT)))
      return false;
   device->pdevice->vk.dispatch_table.GetPhysicalDeviceFormatProperties2(
      vk_physical_device_to_handle(&device->pdevice->vk), format, &props);
   return (image->memlayout == PVR_MEMLAYOUT_LINEAR ? props3.linearTilingFeatures
                                                    : props3.optimalTilingFeatures) & need;
}

bool
PVR_PER_ARCH(g9p_meta_blit)(struct pvr_cmd_buffer *cmd_buffer, const VkBlitImageInfo2 *info)
{
   struct pvr_device *device = cmd_buffer->device;
   VK_FROM_HANDLE(pvr_image, src, info->srcImage);
   VK_FROM_HANDLE(pvr_image, dst, info->dstImage);
   struct g9p_meta_save save;

   if (!device->meta_ready || !g9p_meta_image_ok(device, src, src->vk.format, false) ||
       !g9p_meta_image_ok(device, dst, dst->vk.format, true) ||
       (info->filter == VK_FILTER_LINEAR && util_format_is_pure_integer(vk_format_to_pipe_format(src->vk.format))))
      return false;
   /*
    * Scaled with linear filtering: 11 dEQP fbo.blit cases off by a filter step on the GPU.
    * Into a mip level above 0: dEQP basic_copyteximage2d/copytexsubimage2d 2D cases wrong.
    */
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageBlit2 *r = &info->pRegions[i];

      if (r->dstSubresource.mipLevel)
         return false;
      if (info->filter != VK_FILTER_LINEAR)
         continue;

      if (abs(r->srcOffsets[1].x - r->srcOffsets[0].x) != abs(r->dstOffsets[1].x - r->dstOffsets[0].x) ||
          abs(r->srcOffsets[1].y - r->srcOffsets[0].y) != abs(r->dstOffsets[1].y - r->dstOffsets[0].y))
         return false;
   }
   g9p_meta_begin(cmd_buffer, &save);
   vk_meta_blit_image2(&cmd_buffer->vk, &device->meta, info);
   g9p_meta_end(cmd_buffer, &save);
   return true;
}

bool
PVR_PER_ARCH(g9p_meta_copy)(struct pvr_cmd_buffer *cmd_buffer, const VkCopyImageInfo2 *info)
{
   struct pvr_device *device = cmd_buffer->device;
   VK_FROM_HANDLE(pvr_image, src, info->srcImage);
   VK_FROM_HANDLE(pvr_image, dst, info->dstImage);
   VkImageBlit2 regions[16];

   /* a same-format copy is an unscaled blit; nearest keeps the texels as they are */
   /* unorm and float formats only: their texels come through sampling unchanged */
   if (src->vk.format != dst->vk.format || info->regionCount > ARRAY_SIZE(regions) ||
       (!util_format_is_float(vk_format_to_pipe_format(src->vk.format)) &&
        !vk_format_is_unorm(src->vk.format)))
      return false;
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageCopy2 *r = &info->pRegions[i];

      regions[i] = (VkImageBlit2){
         .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
         .srcSubresource = r->srcSubresource,
         .srcOffsets = { r->srcOffset, { r->srcOffset.x + (int32_t)r->extent.width,
                                         r->srcOffset.y + (int32_t)r->extent.height, 1 } },
         .dstSubresource = r->dstSubresource,
         .dstOffsets = { r->dstOffset, { r->dstOffset.x + (int32_t)r->extent.width,
                                         r->dstOffset.y + (int32_t)r->extent.height, 1 } },
      };
   }
   const VkBlitImageInfo2 blit = {
      .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
      .srcImage = info->srcImage,
      .srcImageLayout = info->srcImageLayout,
      .dstImage = info->dstImage,
      .dstImageLayout = info->dstImageLayout,
      .regionCount = info->regionCount,
      .pRegions = regions,
      .filter = VK_FILTER_NEAREST,
   };

   return PVR_PER_ARCH(g9p_meta_blit)(cmd_buffer, &blit);
}

/*
 * Buffer to image uploads drawn by the GPU: one point per texel, the vertex shader fetching
 * the texel as its attribute and handing it to the fragment shader unchanged (flat, raw
 * bits through a same-size uint view). The CPU path writes the image in place, so it waits
 * for every GPU job before it: Firefox's per-frame data uploads (2048x32 RGBA32UI and the
 * like) waited 23 s of a 45 s aquarium run. vk_meta's own upload reads the buffer in the
 * fragment shader, which G9P cannot do yet.
 */

static VkFormat
g9p_upload_format(unsigned bpp)
{
   return bpp == 4 ? VK_FORMAT_R32_UINT : bpp == 8 ? VK_FORMAT_R32G32_UINT
          : bpp == 16 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_UNDEFINED;
}

struct g9p_upload_key {
   char name[16];
   uint32_t bpp;
};

struct g9p_upload_push {
   float x_scale, y_scale;
   uint32_t x0, y;
};

static nir_shader *
g9p_upload_vs(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, NULL, "g9p-upload-vs");
   nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in, glsl_uvec4_type(), "texel");
   nir_variable *pos = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   nir_variable *psz = nir_variable_create(b.shader, nir_var_shader_out, glsl_float_type(), "gl_PointSize");
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out, glsl_uvec4_type(), "texel_out");

   in->data.location = VERT_ATTRIB_GENERIC0;
   pos->data.location = VARYING_SLOT_POS;
   psz->data.location = VARYING_SLOT_PSIZ;
   out->data.location = VARYING_SLOT_VAR0;
   out->data.interpolation = INTERP_MODE_FLAT;

   nir_def *pc = nir_load_push_constant(&b, 4, 32, nir_imm_int(&b, 0), .base = 0, .range = 16);
   nir_def *x = nir_iadd(&b, nir_channel(&b, pc, 2), nir_load_vertex_id(&b));
   nir_def *fx = nir_fadd_imm(&b, nir_fmul(&b, nir_fadd_imm(&b, nir_u2f32(&b, x), 0.5f),
                                           nir_channel(&b, pc, 0)), -1.0f);
   nir_def *fy = nir_fadd_imm(&b, nir_fmul(&b, nir_fadd_imm(&b, nir_u2f32(&b, nir_channel(&b, pc, 3)), 0.5f),
                                           nir_channel(&b, pc, 1)), -1.0f);

   nir_store_var(&b, pos, nir_vec4(&b, fx, fy, nir_imm_float(&b, 0), nir_imm_float(&b, 1)), 0xf);
   nir_store_var(&b, psz, nir_imm_float(&b, 1.0f), 0x1);
   nir_store_var(&b, out, nir_load_var(&b, in), 0xf);
   return b.shader;
}

static nir_shader *
g9p_upload_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, NULL, "g9p-upload-fs");
   nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in, glsl_uvec4_type(), "texel_in");
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out, glsl_uvec4_type(), "color");

   in->data.location = VARYING_SLOT_VAR0;
   in->data.interpolation = INTERP_MODE_FLAT;
   out->data.location = FRAG_RESULT_DATA0;
   nir_store_var(&b, out, nir_load_var(&b, in), 0xf);
   return b.shader;
}

static VkResult
g9p_upload_pipeline(struct pvr_device *device, unsigned bpp, VkPipelineLayout *layout_out,
                    VkPipeline *pipeline_out)
{
   struct g9p_upload_key key = { .name = "g9p-upload", .bpp = bpp };
   const VkPushConstantRange push = { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(struct g9p_upload_push) };
   VkResult result = vk_meta_get_pipeline_layout(&device->vk, &device->meta, NULL, &push, &key,
                                                 sizeof(key), layout_out);

   if (result != VK_SUCCESS)
      return result;
   *pipeline_out = vk_meta_lookup_pipeline(&device->meta, &key, sizeof(key));
   if (*pipeline_out)
      return VK_SUCCESS;

   VkPipelineShaderStageNirCreateInfoMESA vs_nir = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA, .nir = g9p_upload_vs() };
   VkPipelineShaderStageNirCreateInfoMESA fs_nir = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA, .nir = g9p_upload_fs() };
   const VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &vs_nir,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = &fs_nir,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .pName = "main" },
   };
   const VkVertexInputBindingDescription vb = { 0, bpp, VK_VERTEX_INPUT_RATE_VERTEX };
   const VkVertexInputAttributeDescription va = { 0, 0, g9p_upload_format(bpp), 0 };
   const VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vb,
      .vertexAttributeDescriptionCount = 1, .pVertexAttributeDescriptions = &va,
   };
   const VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
   };
   const VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .scissorCount = 1,
   };
   const VkDynamicState dyn_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
   const VkPipelineDynamicStateCreateInfo dyn = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = ARRAY_SIZE(dyn_states), .pDynamicStates = dyn_states,
   };
   const VkGraphicsPipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vp,
      .pDynamicState = &dyn, .layout = *layout_out,
   };
   const struct vk_meta_rendering_info render = {
      .samples = 1, .color_attachment_count = 1,
      .color_attachment_formats = { g9p_upload_format(bpp) },
      .color_attachment_write_masks = { 0xf },
   };

   result = vk_meta_create_graphics_pipeline(&device->vk, &device->meta, &info, &render, &key,
                                             sizeof(key), pipeline_out);
   ralloc_free(vs_nir.nir);
   ralloc_free(fs_nir.nir);
   return result;
}

bool
PVR_PER_ARCH(g9p_meta_upload)(struct pvr_cmd_buffer *cmd_buffer, const VkCopyBufferToImageInfo2 *info)
{
   struct pvr_device *device = cmd_buffer->device;
   VK_FROM_HANDLE(pvr_image, dst, info->dstImage);
   const struct vk_device_dispatch_table *disp = &device->vk.dispatch_table;
   VkCommandBuffer cmd = pvr_cmd_buffer_to_handle(cmd_buffer);
   const unsigned bpp = vk_format_get_blocksize(dst->vk.format);
   VkPipelineLayout layout;
   VkPipeline pipeline;
   struct g9p_meta_save save;

   /*
    * Formats of 32-bit channels only: their texels go through the uint view unchanged.
    * Packed and narrower ones (RGBA4, RGB5A1, RGBA8 snorm) failed dEQP basic_teximage2d.
    */
   const struct util_format_description *desc =
      util_format_description(vk_format_to_pipe_format(dst->vk.format));

   if (!desc || desc->layout != UTIL_FORMAT_LAYOUT_PLAIN || desc->channel[0].size != 32 ||
       (desc->nr_channels > 1 && desc->channel[desc->nr_channels - 1].size != 32))
      return false;
   if (!device->meta_ready || !getenv("G9P_GPU_UPLOAD") || !g9p_upload_format(bpp) ||
       dst->vk.image_type != VK_IMAGE_TYPE_2D || dst->vk.samples != VK_SAMPLE_COUNT_1_BIT ||
       vk_format_is_depth_or_stencil(dst->vk.format) || vk_format_is_compressed(dst->vk.format) ||
       !(dst->vk.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))
      return false;
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkBufferImageCopy2 *r = &info->pRegions[i];

      /* mip levels above 0: dEQP basic_teximage2d rgb32* wrong there (as the blits) */
      if (r->imageSubresource.mipLevel ||
          vk_image_subresource_layer_count(&dst->vk, &r->imageSubresource) != 1 ||
          r->imageSubresource.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT || r->imageExtent.depth != 1 ||
          (r->bufferOffset % bpp))
         return false;
   }
   if (g9p_upload_pipeline(device, bpp, &layout, &pipeline) != VK_SUCCESS)
      return false;
   if (getenv("G9P_UPLOAD_LOG"))
      fprintf(stderr, "g9p: GPU upload into %s, %u regions, first %ux%u row %u offset %" PRIu64 "\n",
              vk_Format_to_str(dst->vk.format), info->regionCount, info->pRegions[0].imageExtent.width,
              info->pRegions[0].imageExtent.height, info->pRegions[0].bufferRowLength,
              (uint64_t)info->pRegions[0].bufferOffset);

   g9p_meta_begin(cmd_buffer, &save);
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkBufferImageCopy2 *r = &info->pRegions[i];
      const uint32_t mip = r->imageSubresource.mipLevel;
      const uint32_t w = u_minify(dst->vk.extent.width, mip), h = u_minify(dst->vk.extent.height, mip);
      const uint32_t row = r->bufferRowLength ? r->bufferRowLength : r->imageExtent.width;
      const VkImageViewUsageCreateInfo view_usage = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
         .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      };
      const VkImageViewCreateInfo view_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .pNext = &view_usage,
         .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
         .image = info->dstImage,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = g9p_upload_format(bpp),
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, r->imageSubresource.baseArrayLayer, 1 },
      };
      VkImageView view;

      if (vk_meta_create_image_view(&cmd_buffer->vk, &device->meta, &view_info, &view) != VK_SUCCESS)
         break;

      const VkRenderingAttachmentInfo att = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = view,
         .imageLayout = info->dstImageLayout,
         .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
      const VkRect2D area = { { r->imageOffset.x, r->imageOffset.y },
                              { r->imageExtent.width, r->imageExtent.height } };
      const VkRenderingInfo rendering = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = area, .layerCount = 1,
         .colorAttachmentCount = 1, .pColorAttachments = &att,
      };
      const VkViewport viewport = { 0, 0, (float)w, (float)h, 0, 1 };

      disp->CmdBeginRendering(cmd, &rendering);
      disp->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      disp->CmdSetViewport(cmd, 0, 1, &viewport);
      disp->CmdSetScissor(cmd, 0, 1, &area);
      for (uint32_t y = 0; y < r->imageExtent.height; y++) {
         const struct g9p_upload_push pc = { 2.0f / w, 2.0f / h, (uint32_t)r->imageOffset.x,
                                             (uint32_t)r->imageOffset.y + y };
         const VkDeviceSize offset = r->bufferOffset + (VkDeviceSize)y * row * bpp;

         disp->CmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
         disp->CmdBindVertexBuffers(cmd, 0, 1, &info->srcBuffer, &offset);
         disp->CmdDraw(cmd, r->imageExtent.width, 1, 0, 0);
      }
      disp->CmdEndRendering(cmd);
   }
   g9p_meta_end(cmd_buffer, &save);
   return true;
}
