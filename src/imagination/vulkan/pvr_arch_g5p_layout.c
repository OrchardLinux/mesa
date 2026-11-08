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
#include "util/os_time.h"
#include <unistd.h>

#include "pvr_cmd_buffer.h"
#include "pvr_device.h"
#include "pvr_g5p_layout.h"
#include "pvr_g5p_ds.h"
#include "pvr_g9p.h"
#include "pvr_image.h"
#include "pvr_job_transfer.h"
#include "winsys/apple_g5p/pvr_g5p_winsys.h"

struct g5p_surface {
   uint8_t *base;
   uint64_t available;
   enum pipe_format format;
   unsigned bpp, cpp, component_offset, stride, width, height;
   unsigned twiddle_width, twiddle_height;
   bool twiddled, blocks; /* blocks: a compressed surface addressed in texels */
   const struct pvr_image *image;
   VkImageAspectFlags aspects;
   uint64_t image_offset;
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
   out->cpp = out->bpp;
   out->component_offset = 0;
   out->stride = s->stride ? s->stride : s->width;
   out->width = s->width;
   out->height = s->height;
   out->twiddle_width = s->width;
   out->twiddle_height = s->height;
   out->twiddled = s->mem_layout == PVR_MEMLAYOUT_TWIDDLED;
   out->blocks = util_format_is_compressed(out->format);
   out->image = s->g5p_image;
   out->aspects = s->g5p_aspect;
   out->image_offset = s->g5p_image ? s->dev_addr.addr - s->g5p_image->dev_addr.addr - pvr_single_plane_const(s->g5p_image)->offset : 0;
   if (!out->base)
      mesa_loge("G5P transfer: no stable CPU mapping for %#" PRIx64, s->dev_addr.addr);
   return out->base != NULL;
}

/* Spread the low 16 coordinate bits into even bit positions. G5P uses
 * rectangular Morton order, with Y first, then the remaining major-axis bits.
 * Keep this identical to the former bit-by-bit address calculation. */
static unsigned
g5p_spread(unsigned v)
{
   v = (v | (v << 8)) & 0x00ff00ff;
   v = (v | (v << 4)) & 0x0f0f0f0f;
   v = (v | (v << 2)) & 0x33333333;
   return (v | (v << 1)) & 0x55555555;
}

static unsigned
g5p_twiddle2d(unsigned x, unsigned y, unsigned w, unsigned h)
{
   const unsigned bits = util_logbase2_ceil(MAX2(MIN2(w, h), 1));
   const unsigned mask = (1u << bits) - 1;
   return (g5p_spread(x & mask) << 1) | g5p_spread(y & mask) |
          (((x | y) & ~mask) << bits);
}

/* The byte offset of element (x, y): a texel, or a block of a compressed surface */
static size_t
g5p_offset(const struct g5p_surface *s, unsigned x, unsigned y)
{
   if (s->twiddled)
      return (size_t)g5p_twiddle2d(x, y, s->twiddle_width, s->twiddle_height) * s->bpp;
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
   return s->base + offset + s->component_offset;
}

/* Transfers describe a packed Rogue DS surface, but G5P's single-level
 * optimal DS image consists of separate ZLS planes over a padded square.
 * Keep the image/aspect contract rather than guessing from a raw-copy alias.
 * Buffers retain their explicit row pitch and tightly packed aspect bytes. */
static bool
g5p_ds_plane(const struct g5p_surface *s, VkImageAspectFlagBits aspect,
             const struct pvr_g5p_ds_format *f, struct g5p_surface *p)
{
   *p = *s;
   const bool stencil = aspect == VK_IMAGE_ASPECT_STENCIL_BIT;
   p->format = stencil ? PIPE_FORMAT_S8_UINT :
               f->depth_bytes == 2 ? PIPE_FORMAT_Z16_UNORM : PIPE_FORMAT_Z32_FLOAT;
   p->cpp = stencil ? 1 : f->depth_bytes;
   if (!p->cpp)
      return false;
   if (!s->image) {
      /* The buffer can have a raw integer alias or a depth/stencil format. */
      return s->bpp == p->cpp;
   }
   const struct pvr_image *image = s->image;
   const struct pvr_g5p_ds_format *own = pvr_g5p_ds_format(image->vk.format);
   if (!own || (stencil ? !own->stencil : !own->depth) || image->vk.samples != 1)
      return false;
   p->format = stencil ? PIPE_FORMAT_S8_UINT :
               own->depth_bytes == 2 ? PIPE_FORMAT_Z16_UNORM : PIPE_FORMAT_Z32_FLOAT;
   p->cpp = stencil ? 1 : own->depth_bytes;
   struct pvr_g5p_ds_chain chain;
   if (pvr_g5p_ds_sampled_chain(image, &chain)) {
      unsigned layer=s->image_offset/chain.layer, level=0;
      uint64_t at=s->image_offset%chain.layer;
      while(level<image->vk.mip_levels && chain.level[level]!=at)level++;
      if(layer>=image->vk.array_layers || level==image->vk.mip_levels)return false;
      uint64_t target=stencil ? chain.stencil_base+layer*chain.stencil_layer+chain.stencil_level[level] : s->image_offset;
      if(target<s->image_offset || target-s->image_offset>p->available)return false;
      uint64_t delta=target-s->image_offset;
      p->base+=delta;p->available-=delta;p->bpp=p->cpp;p->component_offset=0;
      p->twiddled=true;
      return true;
   }
   if (image->vk.tiling == VK_IMAGE_TILING_OPTIMAL && image->vk.mip_levels == 1) {
      struct pvr_g5p_ds_plane zls;
      if (!pvr_g5p_ds_layout(image->vk.format, s->width, s->height, 1, &zls))
         return false;
      uint64_t off = stencil ? zls.stencil_offset : 0;
      if (off > p->available)
         return false;
      p->base += off;
      p->available -= off;
      p->bpp = p->cpp;
      p->twiddled = true;
      /* The allocation pads to a square, but ZLS stores rectangular
       * Morton order over independently 32-pixel-aligned sides (R17's
       * device-proven unequal-side sampling). Match that order in CPU
       * clears/copies too; the unused allocation tail remains padding. */
      p->twiddle_width = util_next_power_of_two(align(s->width, 32));
      p->twiddle_height = util_next_power_of_two(align(s->height, 32));
   } else {
      /* Multi-level/linear images retain upstream's packed storage. */
      p->bpp = vk_format_get_blocksize(image->vk.format);
      p->component_offset = stencil && own->depth ? 4 : 0;
   }
   return true;
}

static bool
g5p_fill(const struct pvr_transfer_cmd *cmd, struct g5p_surface *dst, VkImageAspectFlags aspect)
{
   const VkRect2D *r = &cmd->scissor;
   uint8_t packed[16] = { 0 };

   if (aspect) {
      for (unsigned y = 0; y < r->extent.height; y++) {
         for (unsigned x = 0; x < r->extent.width; x++) {
            uint8_t *to = g5p_element(dst, r->offset.x + x, r->offset.y + y);
            if (!to) return false;
            if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT)
               util_format_pack_z_float(dst->format, to, &cmd->clear_color[0].f, 1);
            else {
               uint8_t value = cmd->clear_color[1].ui;
               util_format_pack_s_8uint(dst->format, to, &value, 1);
            }
         }
      }
      return true;
   }
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
         memcpy(to, packed, dst->cpp);
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
   b.twiddle_width = b.width;
   b.twiddle_height = b.height;

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

/* Texel (x, y) of the source as floats: a decoded block texel or an unpacked element */
static bool
g5p_texel(struct g5p_surface *s, unsigned x, unsigned y, float out[4], const uint8_t **last, float block[16][4])
{
   if (s->blocks)
      return g5p_block_texel(s, x, y, out, last, block);

   const uint8_t *from = g5p_element(s, x, y);

   if (!from)
      return false;
   util_format_unpack_rgba(s->format, out, from, 1);
   return true;
}

/* VK_FILTER_LINEAR blit sample (Vulkan's vkCmdBlitImage rules): destination pixel centre i + 0.5 maps to
 * source coordinate u = src0 + (i + 0.5) * scale (mirrored inside the source rect for a flip), the
 * 2x2 texels around u - 0.5 are weighted bilinearly, and the sampler clamps to the edge of the
 * source level (not of the region). A 2:1 reduction is the 2x2 box average, which is what
 * glGenerateMipmap through Zink's blit expects. */
static bool
g5p_linear_sample(struct g5p_surface *src, const struct pvr_rect_mapping *m, unsigned x, unsigned y, float out[4],
                  const uint8_t **last, float block[16][4])
{
   const double sw = m->src_rect.extent.width, sh = m->src_rect.extent.height;
   const double dw = m->dst_rect.extent.width, dh = m->dst_rect.extent.height;
   double fu = (x + 0.5) * sw / dw, fv = (y + 0.5) * sh / dh;

   if (m->flip_x)
      fu = sw - fu;
   if (m->flip_y)
      fv = sh - fv;
   fu += m->src_rect.offset.x - 0.5;
   fv += m->src_rect.offset.y - 0.5;

   const double u0f = floor(fu), v0f = floor(fv);
   const float a = fu - u0f, b = fv - v0f;
   const int64_t u0 = u0f, v0 = v0f;
   const int64_t maxu = (int64_t)src->width - 1, maxv = (int64_t)src->height - 1;
   const unsigned us[2] = { CLAMP(u0, 0, maxu), CLAMP(u0 + 1, 0, maxu) };
   const unsigned vs[2] = { CLAMP(v0, 0, maxv), CLAMP(v0 + 1, 0, maxv) };
   const float w[4] = { (1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b };

   if (maxu < 0 || maxv < 0)
      return false;
   for (unsigned c = 0; c < 4; c++)
      out[c] = 0.0f;
   for (unsigned t = 0; t < 4; t++) {
      float v[4];

      if (!g5p_texel(src, us[t & 1], vs[t >> 1], v, last, block))
         return false;
      for (unsigned c = 0; c < 4; c++)
         out[c] += w[t] * v[c];
   }
   return true;
}

/* Whether util's unpack/pack round trip from one format to the other only
 * moves whole bytes: four 8-bit array channels of one type in the linear
 * colour space on both sides, every destination channel taken from a source
 * channel. UNORM and pure integers only: an SNORM -128 becomes -127 through
 * float, and sRGB encoding is not assumed to round-trip. map[] gives, for
 * each destination byte, the source byte. */
static bool
g5p_byte_permutation(enum pipe_format from, enum pipe_format to, uint8_t map[4])
{
   const struct util_format_description *f[2] = { util_format_description(from), util_format_description(to) };

   if (!f[0] || !f[1])
      return false;

   const struct util_format_channel_description *type = &f[0]->channel[0];

   if (type->type == UTIL_FORMAT_TYPE_UNSIGNED ? !type->normalized && !type->pure_integer :
       type->type != UTIL_FORMAT_TYPE_SIGNED || !type->pure_integer)
      return false;
   for (unsigned i = 0; i < 2; i++) {
      if (f[i]->layout != UTIL_FORMAT_LAYOUT_PLAIN || !f[i]->is_array || f[i]->block.bits != 32 ||
          f[i]->block.width != 1 || f[i]->block.height != 1 || f[i]->block.depth != 1 ||
          f[i]->nr_channels != 4 || f[i]->colorspace != UTIL_FORMAT_COLORSPACE_RGB)
         return false;
      for (unsigned c = 0; c < 4; c++) {
         const struct util_format_channel_description *ch = &f[i]->channel[c];

         if (ch->size != 8 || ch->shift % 8 || ch->type != type->type ||
             ch->normalized != type->normalized || ch->pure_integer != type->pure_integer)
            return false;
      }
   }
   for (unsigned j = 0; j < 4; j++) {
      unsigned c = 0;

      while (c < 4 && f[1]->swizzle[c] != j)
         c++;
      if (c == 4 || f[0]->swizzle[c] > PIPE_SWIZZLE_W)
         return false;
      map[f[1]->channel[j].shift / 8] = f[0]->channel[f[0]->swizzle[c]].shift / 8;
   }
   return true;
}

/* An unscaled colour conversion. util's per-texel entry points look the
 * formats up (with a call_once) on every call, which made a 1024x1024
 * RGBA8 -> BGRA8 blit cost ~55 ms on the A9. Resolve the conversion once:
 * a byte permutation per texel, or else chunks of a row gathered, converted
 * by one unpack and one pack call each (the same util functions), and
 * scattered.
 *
 * Linear and Morton offsets both increase with x and with y, so the
 * rectangle's last element is its largest offset: checking the first and
 * last element against the surface and its stable mapping, as g5p_element()
 * does per element, bounds every element in between. */
#define G5P_CONVERT_CHUNK 256

struct g5p_walk {
   uint8_t *base;
   size_t bpp, stride;
   unsigned bits, mask;
   bool twiddled;
};

static void
g5p_walk_init(const struct g5p_surface *s, struct g5p_walk *w)
{
   w->base = s->base + s->component_offset;
   w->bpp = s->bpp;
   w->stride = s->stride;
   w->twiddled = s->twiddled;
   w->bits = util_logbase2_ceil(MAX2(MIN2(s->twiddle_width, s->twiddle_height), 1));
   w->mask = (1u << w->bits) - 1;
}

/* g5p_offset() split into column and row terms. Inside the surface only the
 * major axis has bits above the mask, so the interleaved low bits and the
 * shifted high bits are disjoint and the two terms add. */
static inline size_t
g5p_walk_x(const struct g5p_walk *w, unsigned x)
{
   if (w->twiddled)
      return (size_t)((g5p_spread(x & w->mask) << 1) | ((x & ~w->mask) << w->bits)) * w->bpp;
   return (size_t)x * w->bpp;
}

static inline size_t
g5p_walk_y(const struct g5p_walk *w, unsigned y)
{
   if (w->twiddled)
      return (size_t)(g5p_spread(y & w->mask) | ((y & ~w->mask) << w->bits)) * w->bpp;
   return (size_t)y * w->stride * w->bpp;
}

static bool
g5p_rect_bounded(struct g5p_surface *s, const VkRect2D *r)
{
   return (uint64_t)r->offset.x + r->extent.width <= s->width &&
          (uint64_t)r->offset.y + r->extent.height <= s->height &&
          g5p_element(s, r->offset.x, r->offset.y) &&
          g5p_element(s, r->offset.x + r->extent.width - 1, r->offset.y + r->extent.height - 1);
}

static bool
g5p_convert(const struct pvr_rect_mapping *m, struct g5p_surface *src, struct g5p_surface *dst)
{
   const unsigned w = m->dst_rect.extent.width, h = m->dst_rect.extent.height;
   const unsigned ox = m->src_rect.offset.x, oy = m->src_rect.offset.y;
   const unsigned dx = m->dst_rect.offset.x, dy = m->dst_rect.offset.y;
   const struct util_format_unpack_description *unpack = util_format_unpack_description(src->format);
   uint8_t map[4] = { 0, 1, 2, 3 };
   const bool permute = g5p_byte_permutation(src->format, dst->format, map);
   size_t scol[G5P_CONVERT_CHUNK], dcol[G5P_CONVERT_CHUNK];
   uint8_t packed[G5P_CONVERT_CHUNK * 16];
   uint32_t texels[G5P_CONVERT_CHUNK * 4];
   struct g5p_walk s, d;

   assert(src->bpp <= 16 && dst->bpp <= 16);
   if (!g5p_rect_bounded(src, &m->src_rect) || !g5p_rect_bounded(dst, &m->dst_rect))
      return false;
   g5p_walk_init(src, &s);
   g5p_walk_init(dst, &d);
   /* little-endian bytes: destination byte b is source byte map[b] */
   const unsigned shift[4] = { 8 * map[0], 8 * map[1], 8 * map[2], 8 * map[3] };
   for (unsigned x0 = 0; x0 < w; x0 += G5P_CONVERT_CHUNK) {
      const unsigned n = MIN2(G5P_CONVERT_CHUNK, w - x0);

      for (unsigned i = 0; i < n; i++) {
         scol[i] = g5p_walk_x(&s, ox + (m->flip_x ? w - 1 - (x0 + i) : x0 + i));
         dcol[i] = g5p_walk_x(&d, dx + x0 + i);
      }
      for (unsigned y = 0; y < h; y++) {
         const uint8_t *srow = s.base + g5p_walk_y(&s, oy + (m->flip_y ? h - 1 - y : y));
         uint8_t *drow = d.base + g5p_walk_y(&d, dy + y);

         if (permute) {
            for (unsigned i = 0; i < n; i++) {
               uint32_t v, out;

               memcpy(&v, srow + scol[i], 4);
               out = ((v >> shift[0]) & 0xff) | ((v >> shift[1]) & 0xff) << 8 |
                     ((v >> shift[2]) & 0xff) << 16 | ((v >> shift[3]) & 0xff) << 24;
               memcpy(drow + dcol[i], &out, 4);
            }
            continue;
         }
         for (unsigned i = 0; i < n; i++)
            memcpy(packed + i * src->bpp, srow + scol[i], src->bpp);
         unpack->unpack_rgba(texels, packed, n);
         util_format_pack_rgba(dst->format, packed, texels, n);
         for (unsigned i = 0; i < n; i++)
            memcpy(drow + dcol[i], packed + i * dst->bpp, dst->bpp);
      }
   }
   return true;
}

static bool
g5p_copy(const struct pvr_rect_mapping *mapping, struct g5p_surface *src, struct g5p_surface *dst, bool linear)
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

   /* Matching linear rows need no per-texel address calculation or format
    * conversion. Plane components with interleaved storage retain the element
    * path. Check the whole rectangle before the first write, including pitch
    * padding and the stable mapping's last byte. Vulkan copies do not overlap. */
   if (same && !src->twiddled && !dst->twiddled && !m->flip_x &&
       sw == dw && sh == dh && src->cpp == src->bpp && dst->cpp == dst->bpp) {
      const unsigned sx = m->src_rect.offset.x, sy = m->src_rect.offset.y;
      const unsigned dx = m->dst_rect.offset.x, dy = m->dst_rect.offset.y;
      if (sx >= src->width || sy >= src->height || sw > src->width - sx || sh > src->height - sy ||
          dx >= dst->width || dy >= dst->height || dw > dst->width - dx || dh > dst->height - dy ||
          !g5p_element(src, sx + sw - 1, sy + sh - 1) ||
          !g5p_element(dst, dx + dw - 1, dy + dh - 1))
         return false;
      const size_t bytes = (size_t)dw * dst->bpp;
      for (unsigned y = 0; y < dh; y++) {
         const unsigned row = m->flip_y ? sh - 1 - y : y;
         memcpy(dst->base + g5p_offset(dst, dx, dy + y),
                src->base + g5p_offset(src, sx, sy + row), bytes);
      }
      return true;
   }

   /* A scaled linear blit filters. Integer and depth/stencil formats have no linear filter in
    * Vulkan, and an unscaled blit samples exactly at texel centres: both stay element copies. */
   if (linear && !integer && !util_format_is_depth_or_stencil(src->format) && (sw != dw || sh != dh)) {
      for (unsigned y = 0; y < dh; y++) {
         for (unsigned x = 0; x < dw; x++) {
            uint8_t *to = g5p_element(dst, m->dst_rect.offset.x + x, m->dst_rect.offset.y + y);
            float v[4];

            if (!to || !g5p_linear_sample(src, m, x, y, v, &last, block))
               return false;
            util_format_pack_rgba(dst->format, to, v, 1);
         }
      }
      return true;
   }

   if (!decode && !same && sw == dw && sh == dh && !src->blocks && !dst->blocks &&
       src->cpp == src->bpp && dst->cpp == dst->bpp && src->bpp <= 16 && dst->bpp <= 16 &&
       !util_format_is_depth_or_stencil(src->format) && !util_format_is_depth_or_stencil(dst->format) &&
       (!src->twiddled || (src->twiddle_width == src->width && src->twiddle_height == src->height)) &&
       (!dst->twiddled || (dst->twiddle_width == dst->width && dst->twiddle_height == dst->height))) {
      const struct util_format_unpack_description *u = util_format_unpack_description(src->format);
      const struct util_format_pack_description *p = util_format_pack_description(dst->format);

      if (u && u->unpack_rgba && p && (p->pack_rgba_float || p->pack_rgba_uint || p->pack_rgba_sint))
         return g5p_convert(m, src, dst);
   }

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
            memcpy(to, from, dst->cpp);
         } else if (util_format_is_depth_or_stencil(src->format) ||
                    util_format_is_depth_or_stencil(dst->format)) {
            const struct util_format_unpack_description *u = util_format_unpack_description(src->format);
            const struct util_format_pack_description *p = util_format_pack_description(dst->format);
            if (u->unpack_z_float && p->pack_z_float) {
               float z;
               util_format_unpack_z_float(src->format, &z, from, 1);
               util_format_pack_z_float(dst->format, to, &z, 1);
            } else if (u->unpack_s_8uint && p->pack_s_8uint) {
               uint8_t stencil;
               util_format_unpack_s_8uint(src->format, &stencil, from, 1);
               util_format_pack_s_8uint(dst->format, to, &stencil, 1);
            } else {
               mesa_loge("G5P transfer: unsupported depth/stencil conversion %u -> %u", src->format, dst->format);
               return false;
            }
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
g5p_copy_aspects(const struct pvr_rect_mapping *mapping, struct g5p_surface *src, struct g5p_surface *dst, bool linear)
{
   const struct pvr_image *image = dst->image ? dst->image : src->image;
   if (!image || !vk_format_is_depth_or_stencil(image->vk.format))
      return g5p_copy(mapping, src, dst, linear);
   VkImageAspectFlags aspects = src->image ? src->aspects : dst->aspects;
   if (src->image && dst->image) aspects &= dst->aspects;
   const struct pvr_g5p_ds_format *f = pvr_g5p_ds_format(image->vk.format);
   if (!f) return false;
   for (unsigned i = 0; i < 2; i++) {
      VkImageAspectFlagBits aspect = i ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
      if (!(aspects & aspect)) continue;
      struct g5p_surface s, d;
      if (!g5p_ds_plane(src, aspect, f, &s) || !g5p_ds_plane(dst, aspect, f, &d) ||
          !g5p_copy(mapping, &s, &d, false))
         return false;
   }
   return aspects != 0;
}

static bool
g5p_run(struct pvr_device *device, const struct pvr_transfer_cmd *cmd, bool log)
{
   struct g5p_surface dst;

   if (!g5p_surface_init(device, &cmd->dst, &dst))
      return false;
   if (log && (cmd->flags & PVR_TRANSFER_CMD_FLAGS_FILL))
      fprintf(stderr, "G5P perf transfer-cmd fill dst=%s %ux%u twiddled=%d scissor=%ux%u\n",
              util_format_short_name(dst.format), dst.width, dst.height, dst.twiddled,
              cmd->scissor.extent.width, cmd->scissor.extent.height);
   if (cmd->flags & PVR_TRANSFER_CMD_FLAGS_FILL) {
      if (dst.image && vk_format_is_depth_or_stencil(dst.image->vk.format)) {
         const struct pvr_g5p_ds_format *f = pvr_g5p_ds_format(dst.image->vk.format);
         if (!f) return false;
         for (unsigned i = 0; i < 2; i++) {
            VkImageAspectFlagBits aspect = i ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
            struct g5p_surface p;
            if ((dst.aspects & aspect) && (!g5p_ds_plane(&dst, aspect, f, &p) || !g5p_fill(cmd, &p, aspect)))
               return false;
         }
         return true;
      }
      return g5p_fill(cmd, &dst, 0);
   }
   for (unsigned i = 0; i < cmd->source_count; i++) {
      const struct pvr_transfer_cmd_source *source = &cmd->sources[i];
      struct g5p_surface src;

      if (!g5p_surface_init(device, &source->surface, &src))
         return false;
      if (log) {
         const VkRect2D *r = source->mapping_count ? &source->mappings[0].dst_rect : &cmd->scissor;
         fprintf(stderr, "G5P perf transfer-cmd copy src=%s %ux%u twiddled=%d dst=%s %ux%u twiddled=%d "
                 "mappings=%u first=%dx%d linear=%d\n",
                 util_format_short_name(src.format), src.width, src.height, src.twiddled,
                 util_format_short_name(dst.format), dst.width, dst.height, dst.twiddled,
                 source->mapping_count, (int)r->extent.width, (int)r->extent.height,
                 source->filter == PVR_FILTER_LINEAR);
      }
      if (!source->mapping_count) {
         /* the whole destination scissor, 1:1 */
         const struct pvr_rect_mapping m = { .src_rect = cmd->scissor, .dst_rect = cmd->scissor };

         if (!g5p_copy_aspects(&m, &src, &dst, false))
            return false;
      }
      for (unsigned k = 0; k < source->mapping_count; k++)
         if (!g5p_copy_aspects(&source->mappings[k], &src, &dst, source->filter == PVR_FILTER_LINEAR))
            return false;
   }
   return true;
}

VkResult
PVR_PER_ARCH(g5p_transfer_submit)(struct pvr_device *device, struct pvr_sub_cmd_transfer *sub_cmd,
                                  struct vk_sync *wait_sync, struct vk_sync *signal_sync)
{
   VkResult result;
   const char *perf = getenv("PVR_G5P_PERF_LOG");
   const bool log = perf && strtoul(perf, NULL, 0) >= 2;
   const uint64_t start = log ? os_time_get_nano() : 0;

   if (wait_sync) {
      result = vk_sync_wait(&device->vk, wait_sync, 0, VK_SYNC_WAIT_COMPLETE, UINT64_MAX);
      if (result != VK_SUCCESS)
         return result;
   }
   const uint64_t ready = log ? os_time_get_nano() : 0;
   unsigned commands = 0;
   list_for_each_entry (struct pvr_transfer_cmd, cmd, sub_cmd->transfer_cmds, link) {
      commands++;
      if (!g5p_run(device, cmd, log))
         return vk_errorf(device, VK_ERROR_DEVICE_LOST,
                          "G5P transfer surface exceeds its stable BO mapping");
   }
   if (log)
      fprintf(stderr, "G5P perf transfer pid=%ld start_ns=%" PRIu64 " wait_us=%" PRIu64
              " cpu_us=%" PRIu64 " commands=%u\n", (long)getpid(), start,
              (ready - start) / 1000, (os_time_get_nano() - ready) / 1000, commands);
   return vk_sync_signal(&device->vk, signal_sync, 0);
}
