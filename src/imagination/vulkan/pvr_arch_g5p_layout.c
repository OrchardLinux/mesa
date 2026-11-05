/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): the CPU copies that know the A9's layouts
 * (pvr_g5p_layout.h). Built only with -Dimagination-g5p-experimental=true.
 *
 * G5P runs its transfer sub-commands on the CPU through the winsys's permanent
 * BO mappings. Application mappings may be placed or unmapped while an
 * asynchronous queue submission is executing, so they are never used here.
 * A blit from a block-compressed image is not an element
 * copy: upstream hands it over in texels on a surface of 4x4 blocks
 * (pvr_setup_transfer_surface() converts to blocks only for raw copies), and
 * each texel has to be decoded from its block (util's ETC2/EAC decoder, as our
 * ICD's copies did). A sub-command with such a blit runs here: the same
 * copies, nearest scaling, flips, util_format conversion and fills as G9P's,
 * in the shape of his code, plus the decode. Every other sub-command still
 * goes to G9P's code unchanged.
 */

#include <assert.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "vk_format.h"
#include "vk_sync.h"

#include "util/format/u_format.h"
#include "util/log.h"
#include "util/texcompress_etc.h"
#include "util/u_math.h"

#include "pvr_cmd_buffer.h"
#include "pvr_device.h"
#include "pvr_g5p_layout.h"
#include "pvr_g9p.h"
#include "pvr_image.h"
#include "pvr_job_transfer.h"
#include "winsys/apple_g5p/pvr_g5p_winsys.h"

struct g5p_surface {
   uint8_t *base;
   uint64_t available;
   enum pipe_format format;
   unsigned bpp, stride, width, height;
   bool twiddled, blocks; /* blocks: a compressed surface addressed in texels */
};

static bool
g5p_surface_init(struct pvr_device *device, const struct pvr_transfer_cmd_surface *s, struct g5p_surface *out)
{
   out->base = NULL;
   out->available = 0;
   simple_mtx_lock(&device->g9p_vmas.lock);
   util_dynarray_foreach (&device->g9p_vmas.vmas, struct pvr_winsys_vma *, v) {
      if (pvr_g5p_winsys_vma_cpu_addr(*v, s->dev_addr.addr, 1,
                                      &out->base, &out->available))
         break;
   }
   simple_mtx_unlock(&device->g9p_vmas.lock);
   out->format = vk_format_to_pipe_format(s->vk_format);
   out->bpp = vk_format_get_blocksize(s->vk_format);
   out->stride = s->stride ? s->stride : s->width;
   out->width = s->width;
   out->height = s->height;
   out->twiddled = s->mem_layout == PVR_MEMLAYOUT_TWIDDLED;
   out->blocks = util_format_is_compressed(out->format);
   if (!out->base)
      mesa_loge("G5P transfer: no stable CPU mapping for %#" PRIx64, s->dev_addr.addr);
   return out->base != NULL;
}

/* 2D Morton order over the padded size, y first (as G9P's twiddle()) */
static unsigned
g5p_twiddle2d(unsigned x, unsigned y, unsigned w, unsigned h)
{
   const unsigned pw = util_next_power_of_two(MAX2(w, 1)), ph = util_next_power_of_two(MAX2(h, 1));
   unsigned m = 0, bit = 0;

   for (unsigned i = 0; (1u << i) < pw || (1u << i) < ph; i++) {
      if ((1u << i) < ph)
         m |= ((y >> i) & 1) << bit++;
      if ((1u << i) < pw)
         m |= ((x >> i) & 1) << bit++;
   }
   return m;
}

/* The byte offset of element (x, y): a texel, or a block of a compressed surface */
static size_t
g5p_offset(const struct g5p_surface *s, unsigned x, unsigned y)
{
   if (s->twiddled)
      return (size_t)g5p_twiddle2d(x, y, s->width, s->height) * s->bpp;
   return ((size_t)y * s->stride + x) * s->bpp;
}

static uint8_t *
g5p_element(struct g5p_surface *s, unsigned x, unsigned y)
{
   if (x >= s->width || y >= s->height)
      return NULL;

   const size_t offset = g5p_offset(s, x, y);

   if (offset > s->available || s->bpp > s->available - offset)
      return NULL;
   return s->base + offset;
}

static bool
g5p_fill(const struct pvr_transfer_cmd *cmd, struct g5p_surface *dst)
{
   const VkRect2D *r = &cmd->scissor;
   uint8_t packed[16] = { 0 };

   if (util_format_is_pure_integer(dst->format))
      util_format_pack_rgba(dst->format, packed,
                            (const uint32_t[4]){ cmd->clear_color[0].ui, cmd->clear_color[1].ui,
                                                 cmd->clear_color[2].ui, cmd->clear_color[3].ui },
                            1);
   else
      util_format_pack_rgba(dst->format, packed,
                            (const float[4]){ cmd->clear_color[0].f, cmd->clear_color[1].f, cmd->clear_color[2].f,
                                              cmd->clear_color[3].f },
                            1);
   for (unsigned y = 0; y < r->extent.height; y++) {
      for (unsigned x = 0; x < r->extent.width; x++) {
         uint8_t *to = g5p_element(dst, r->offset.x + x, r->offset.y + y);

         if (!to)
            return false;
         memcpy(to, packed, dst->bpp);
      }
   }
   return true;
}

/* The uncompressed format util's ETC2/EAC decoder (_mesa_unpack_etc2_format())
 * writes for a compressed one: util_format has no unpack of ETC2/EAC (as our
 * ICD's g5p_transfer.c) */
static enum pipe_format
g5p_etc2_decoded(enum pipe_format p)
{
   switch (p) {
   case PIPE_FORMAT_ETC2_RGB8:
   case PIPE_FORMAT_ETC2_RGBA8:
   case PIPE_FORMAT_ETC2_RGB8A1:
      return PIPE_FORMAT_R8G8B8A8_UNORM;
   case PIPE_FORMAT_ETC2_SRGB8:
   case PIPE_FORMAT_ETC2_SRGBA8:
   case PIPE_FORMAT_ETC2_SRGB8A1:
      return PIPE_FORMAT_R8G8B8A8_SRGB;
   case PIPE_FORMAT_ETC2_R11_UNORM: return PIPE_FORMAT_R16_UNORM;
   case PIPE_FORMAT_ETC2_RG11_UNORM: return PIPE_FORMAT_R16G16_UNORM;
   case PIPE_FORMAT_ETC2_R11_SNORM: return PIPE_FORMAT_R16_SNORM;
   case PIPE_FORMAT_ETC2_RG11_SNORM: return PIPE_FORMAT_R16G16_SNORM;
   default: return PIPE_FORMAT_NONE;
   }
}

/* A 4x4 block as 16 RGBA floats, row by row */
static void
g5p_decode_block(enum pipe_format format, const uint8_t *p, unsigned bpp, float block[16][4])
{
   const enum pipe_format dec = g5p_etc2_decoded(format);

   if (dec != PIPE_FORMAT_NONE) {
      const unsigned dbpp = util_format_get_blocksize(dec);
      uint8_t texels[16 * 4 * 2];

      _mesa_unpack_etc2_format(texels, 4 * dbpp, p, bpp, 4, 4, format, false);
      util_format_unpack_rgba(dec, block, texels, 16);
      return;
   }
   if (util_format_unpack_description(format)->unpack_rgba_rect)
      util_format_unpack_rgba_rect(format, block, 4 * 4 * sizeof(float), p, bpp, 4, 4);
   else
      memset(block, 0, 16 * 4 * sizeof(float));
}

/* Texel (x, y) of a compressed surface addressed in texels, as RGBA floats;
 * the last block decoded is kept */
static bool
g5p_block_texel(const struct g5p_surface *s, unsigned x, unsigned y, float out[4], const uint8_t **last,
                float block[16][4])
{
   const unsigned bw = util_format_get_blockwidth(s->format), bh = util_format_get_blockheight(s->format);
   struct g5p_surface b = *s;

   /* the blocks' surface: twiddled over the level's extent in blocks */
   b.width = DIV_ROUND_UP(s->width, bw);
   b.height = DIV_ROUND_UP(s->height, bh);
   b.stride = DIV_ROUND_UP(s->stride, bw);

   const uint8_t *p = g5p_element(&b, x / bw, y / bh);

   assert(bw == 4 && bh == 4);
   if (!p)
      return false;
   if (p != *last) {
      g5p_decode_block(s->format, p, s->bpp, block);
      *last = p;
   }
   memcpy(out, block[(y % bh) * bw + x % bw], 4 * sizeof(float));
   return true;
}

static bool
g5p_normalize_axis(int32_t *offset, uint32_t *extent, bool *flip)
{
   const int64_t signed_extent = (int32_t)*extent;
   if (signed_extent < 0) {
      const int64_t start = (int64_t)*offset + signed_extent;
      if (start < 0 || start > INT32_MAX)
         return false;
      *offset = start;
      *extent = -signed_extent;
      *flip = !*flip;
   }
   return *offset >= 0;
}

static bool
g5p_copy(const struct pvr_rect_mapping *mapping, struct g5p_surface *src, struct g5p_surface *dst)
{
   /* Rogue's blit builder keeps signed source deltas in VkExtent2D's
    * unsigned fields, while destination reversal is usually in flip_x/y.
    * CPU addressing needs increasing rectangles and the relative reversal
    * of source and destination. Reversing both axes twice cancels out. */
   struct pvr_rect_mapping normalized = *mapping;
   if (!g5p_normalize_axis(&normalized.src_rect.offset.x, &normalized.src_rect.extent.width, &normalized.flip_x) ||
       !g5p_normalize_axis(&normalized.src_rect.offset.y, &normalized.src_rect.extent.height, &normalized.flip_y) ||
       !g5p_normalize_axis(&normalized.dst_rect.offset.x, &normalized.dst_rect.extent.width, &normalized.flip_x) ||
       !g5p_normalize_axis(&normalized.dst_rect.offset.y, &normalized.dst_rect.extent.height, &normalized.flip_y))
      return false;
   const struct pvr_rect_mapping *m = &normalized;
   const bool decode = src->blocks && !dst->blocks;
   const bool same = !decode && (src->format == dst->format ||
                                 (src->bpp == dst->bpp && util_format_is_compressed(src->format)));
   const bool integer = util_format_is_pure_integer(src->format);
   const unsigned dw = m->dst_rect.extent.width, dh = m->dst_rect.extent.height;
   const unsigned sw = m->src_rect.extent.width, sh = m->src_rect.extent.height;
   const uint8_t *last = NULL;
   float block[16][4];

   if (!dw || !dh || !sw || !sh)
      return true;

   for (unsigned y = 0; y < dh; y++) {
      unsigned sy = sh == dh ? y : (uint64_t)y * sh / dh;

      if (m->flip_y)
         sy = sh - 1 - sy;
      for (unsigned x = 0; x < dw; x++) {
         unsigned sx = sw == dw ? x : (uint64_t)x * sw / dw;

         if (m->flip_x)
            sx = sw - 1 - sx;

         uint8_t *to = g5p_element(dst, m->dst_rect.offset.x + x, m->dst_rect.offset.y + y);

         if (!to)
            return false;

         if (decode) {
            float v[4];

            if (!g5p_block_texel(src, m->src_rect.offset.x + sx,
                                 m->src_rect.offset.y + sy, v, &last, block))
               return false;
            util_format_pack_rgba(dst->format, to, v, 1);
            continue;
         }

         const uint8_t *from = g5p_element(src, m->src_rect.offset.x + sx,
                                          m->src_rect.offset.y + sy);

         if (!from)
            return false;

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
   return true;
}

static bool
g5p_run(struct pvr_device *device, const struct pvr_transfer_cmd *cmd)
{
   struct g5p_surface dst;

   if (!g5p_surface_init(device, &cmd->dst, &dst))
      return false;
   if (cmd->flags & PVR_TRANSFER_CMD_FLAGS_FILL) {
      return g5p_fill(cmd, &dst);
   }
   for (unsigned i = 0; i < cmd->source_count; i++) {
      const struct pvr_transfer_cmd_source *source = &cmd->sources[i];
      struct g5p_surface src;

      if (!g5p_surface_init(device, &source->surface, &src))
         return false;
      if (!source->mapping_count) {
         /* the whole destination scissor, 1:1 */
         const struct pvr_rect_mapping m = { .src_rect = cmd->scissor, .dst_rect = cmd->scissor };

         if (!g5p_copy(&m, &src, &dst))
            return false;
      }
      for (unsigned k = 0; k < source->mapping_count; k++)
         if (!g5p_copy(&source->mappings[k], &src, &dst))
            return false;
   }
   return true;
}

VkResult
PVR_PER_ARCH(g5p_transfer_submit)(struct pvr_device *device, struct pvr_sub_cmd_transfer *sub_cmd,
                                  struct vk_sync *wait_sync, struct vk_sync *signal_sync)
{
   VkResult result;

   if (wait_sync) {
      result = vk_sync_wait(&device->vk, wait_sync, 0, VK_SYNC_WAIT_COMPLETE, UINT64_MAX);
      if (result != VK_SUCCESS)
         return result;
   }
   list_for_each_entry (struct pvr_transfer_cmd, cmd, sub_cmd->transfer_cmds, link) {
      if (!g5p_run(device, cmd))
         return vk_errorf(device, VK_ERROR_DEVICE_LOST,
                          "G5P transfer surface exceeds its stable BO mapping");
   }
   return vk_sync_signal(&device->vk, signal_sync, 0);
}
