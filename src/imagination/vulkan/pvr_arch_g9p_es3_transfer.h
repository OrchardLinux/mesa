/* SPDX-License-Identifier: MIT */
/*
 * -Dimagination-g9p-es3: the resolve of a 4x image on the CPU, for
 * vkCmdResolveImage and upstream's transfer resolve of a render. Included
 * by pvr_arch_g9p_transfer.c only, at its end and under PVR_G9P_ES3, so
 * that it uses that file's surfaces and twiddle.
 *
 * Our A9 driver's resolve (msaa.md, measured against Apple's GPU: 4096 of
 * 4096 pixels): a 4x surface is twiddled like a one-sample one (Morton, y
 * bit first, the order twiddle() has), with 16-byte elements, sample s at
 * +4s; the resolve is (sum + 2) / 4 per byte. 8-bit UNORM formats only
 * (the measured case); anything else is logged and left alone, as the
 * other unhandled transfers are. Untested on the A10: its multisample
 * surface layout is the A9's assumption.
 */

static void
es3_resolve(struct pvr_device *device, const struct pvr_transfer_cmd *cmd,
            const struct pvr_transfer_cmd_source *source, const struct surface *dst)
{
   const struct util_format_description *desc;
   const unsigned samples = source->surface.sample_count;
   struct surface src;

   if (!surface_init(device, &source->surface, &src))
      return;
   desc = util_format_description(src.format);
   if (samples != 4 || !src.twiddled || src.format != dst->format || src.bpp != 4 ||
       !util_format_is_unorm8(desc) || desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB ||
       cmd->dst.sample_count > 1) {
      mesa_loge("G9P transfer: resolve of %u samples of %s to %u samples not handled", samples,
                util_format_name(src.format), cmd->dst.sample_count);
      return;
   }
   src.bpp = 4 * 4;
   for (unsigned k = 0; k < MAX2(source->mapping_count, 1); k++) {
      const struct pvr_rect_mapping whole = { .src_rect = cmd->scissor, .dst_rect = cmd->scissor };
      const struct pvr_rect_mapping *m = source->mapping_count ? &source->mappings[k] : &whole;
      const unsigned w = MIN2(m->src_rect.extent.width, m->dst_rect.extent.width);
      const unsigned h = MIN2(m->src_rect.extent.height, m->dst_rect.extent.height);

      for (unsigned y = 0; y < h; y++) {
         const unsigned sy = m->src_rect.offset.y + y, dy = m->dst_rect.offset.y + y;

         if (sy >= src.height || dy >= dst->height)
            continue;
         for (unsigned x = 0; x < w; x++) {
            const unsigned sx = m->src_rect.offset.x + x, dx = m->dst_rect.offset.x + x;
            const uint8_t *e;
            uint8_t *o;

            if (sx >= src.width || dx >= dst->width)
               continue;
            e = src.base + column_offset(&src, sx) + row_offset(&src, sy);
            o = dst->base + column_offset(dst, dx) + row_offset(dst, dy);
            for (unsigned b = 0; b < 4; b++)
               o[b] = (e[b] + e[4 + b] + e[8 + b] + e[12 + b] + 2) / 4;
         }
      }
   }
}
