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
 * and scale with nearest sampling; MSAA resolves and YUV are not handled.
 */

#include <stdlib.h>
#include <string.h>

#include "vk_format.h"
#include "vk_sync.h"

#include "util/format/u_format.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"

#include "pvr_cmd_buffer.h"
#include "pvr_device.h"
#include "pvr_g9p.h"
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
   const VkRect2D *r = &cmd->scissor;
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
      unsigned sx = sw == dw ? x : x * sw / dw;

      if (m->flip_x)
         sx = sw - 1 - sx;
      cols[2 * x] = column_offset(src, m->src_rect.offset.x + sx);
      cols[2 * x + 1] = column_offset(dst, m->dst_rect.offset.x + x);
   }
   for (unsigned y = 0; y < dh; y++) {
      unsigned sy = sh == dh ? y : y * sh / dh;

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
      if (!source->mapping_count) {
         /* the whole destination scissor, 1:1 */
         const struct pvr_rect_mapping m = { .src_rect = cmd->scissor,
                                             .dst_rect = cmd->scissor };

         copy(&m, &src, &dst);
      }
      for (unsigned k = 0; k < source->mapping_count; k++)
         copy(&source->mappings[k], &src, &dst);
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
