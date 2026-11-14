/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): the depth/stencil words (pvr_g5p_ds.h). Plain
 * arithmetic: csbgen's field packing and nothing of the driver, so a host
 * test can run it against the captured words.
 *
 * Every value here is a named register field. Their sources are our iOS
 * captures of 2026-11-01 (the depth-store, depth-load, D16, stencil and
 * depth-bias scenes), decoded against Rogue's cr.xml and ppp.xml, and the
 * footprints a device run measured. No word is copied from a capture.
 */

#include <math.h>
#include <string.h>

/* addresses are plain integers here: this file never includes pvr_csb.h */
#define __pvr_address_type uint64_t
#define __pvr_get_address(addr) (addr)
#define __pvr_make_address(addr) (addr)
#include "csbgen/rogue/cr.h"
#include "csbgen/rogue/ppp.h"

#include "pvr_g5p_ds.h"

static uint32_t
g5p_pot(uint32_t v)
{
   uint32_t p = 1;
   while (p < v)
      p <<= 1;
   return p;
}

static uint32_t
g5p_align(uint32_t v, uint32_t a)
{
   return (v + a - 1) & ~(a - 1);
}

static uint32_t
g5p_fui(float f)
{
   uint32_t u;
   memcpy(&u, &f, sizeof(u));
   return u;
}

/*
 * A 64-bit register's value from its packed dwords. csbgen's packers store
 * through uint32_t; reading those bytes back as a uint64_t lvalue would break
 * C's aliasing rules, so each 64-bit word is packed into a dword pair and
 * copied out.
 */
static uint64_t
g5p_dw64(const uint32_t dw[2])
{
   uint64_t v;
   memcpy(&v, dw, sizeof(v));
   return v;
}

static const struct pvr_g5p_ds_format g5p_ds_formats[] = {
   { VK_FORMAT_D16_UNORM, ROGUE_CR_ZLS_FORMAT_TYPE_16BITINT, 2, true, false, true },
   /* the depth plane reads as R32F (the captured shadow-sampling scene) */
   { VK_FORMAT_D32_SFLOAT, ROGUE_CR_ZLS_FORMAT_TYPE_F32Z, 4, true, false, true },
   { VK_FORMAT_S8_UINT, ROGUE_CR_ZLS_FORMAT_TYPE_F32Z, 0, false, true, true },
   { VK_FORMAT_D32_SFLOAT_S8_UINT, ROGUE_CR_ZLS_FORMAT_TYPE_F32Z, 4, true, true, true },
};

const struct pvr_g5p_ds_format *
pvr_g5p_ds_format(VkFormat format)
{
   for (unsigned i = 0; i < sizeof(g5p_ds_formats) / sizeof(g5p_ds_formats[0]); i++)
      if (g5p_ds_formats[i].format == format)
         return &g5p_ds_formats[i];
   return NULL;
}

bool
pvr_g5p_ds_layout(VkFormat format, uint32_t width, uint32_t height, uint32_t samples,
                  struct pvr_g5p_ds_plane *out)
{
   const struct pvr_g5p_ds_format *f = pvr_g5p_ds_format(format);

   memset(out, 0, sizeof(*out));
   if (!f || !width || !height || (samples != 1 && samples != 4))
      return false;
   out->side = g5p_pot(g5p_align(width > height ? width : height, 32));
   const uint64_t texels = (uint64_t)out->side * out->side * samples;
   out->depth_bytes = f->depth ? texels * f->depth_bytes : 0;
   if (f->stencil)
      out->stencil_offset = (out->depth_bytes + PVR_G5P_ZLS_PLANE_ALIGN - 1) & ~(uint64_t)(PVR_G5P_ZLS_PLANE_ALIGN - 1);
   const uint64_t end = f->stencil ? out->stencil_offset + texels : out->depth_bytes;
   out->size = (end + PVR_G5P_ZLS_PLANE_ALIGN - 1) & ~(uint64_t)(PVR_G5P_ZLS_PLANE_ALIGN - 1);
   return true;
}

void
pvr_g5p_ds_zls_none(struct pvr_g5p_zls_words *out)
{
   const struct ROGUE_CR_ISP_BGOBJVALS v = { .enablebgtag = true, .mask = true };

   memset(out, 0, sizeof(*out));
   out->bgobj_depth = g5p_fui(1.0f);
   ROGUE_CR_ISP_BGOBJVALS_pack(&out->bgobj_vals, &v);
}

/*
 * ZLSCTL: twiddled load and store (iPadOS stores depth in Morton order and
 * its loads read it back), the plane's format, a load and a store enable per
 * aspect, each with its force bit (captured: 0xa2004 a store pass, 0xaa006 a
 * load pass, 0xe2004 depth and stencil, 0x26002 a stencil reload). The
 * extents stay 0, as in every capture: ZLS_PIXELS sizes the surface. Bases:
 * the depth plane is ZLOAD and ZSTORE (ZLOAD only when the render uses the
 * depth aspect, as the captured stencil reload), the stencil plane | enable
 * is both stencil bases. ZLS_PIXELS = (w - 1) | (h - 1) << 15, both sides
 * aligned to 32 with a stencil plane, a side under 32 or four samples (the
 * iPadOS driver's rule).
 */
const char *
pvr_g5p_ds_zls(const struct pvr_g5p_ds_target *t, struct pvr_g5p_zls_words *out)
{
   const struct pvr_g5p_ds_format *f = pvr_g5p_ds_format(t->format);
   struct pvr_g5p_ds_plane plane;

   memset(out, 0, sizeof(*out));
   if (!f)
      return "a depth/stencil format the A9's ZLS does not store";
   if (!pvr_g5p_ds_layout(t->format, t->width, t->height, t->samples, &plane))
      return "a depth/stencil attachment of 0 pixels, or not 1 or 4 samples";
   const bool d = t->depth && f->depth, s = t->stencil && f->stencil;
   const bool zload = d && t->zload, zstore = d && t->zstore;
   const bool sload = s && t->sload, sstore = s && t->sstore;
   if (t->samples == 4 && (zload || zstore || sload || sstore))
      return "a load or store of a 4-sample depth/stencil attachment (no multisample ZLS capture: it stays on chip)";

   const struct ROGUE_CR_ISP_ZLSCTL ctl = {
      .loadtwiddled = true,
      .storetwiddled = true,
      .zloadformat = f->zls_type,
      .zstoreformat = f->zls_type,
      .zloaden = zload,
      .zstoreen = zstore,
      .sloaden = sload,
      .sstoreen = sstore,
      .forcezload = zload || sload,
      .forcezstore = zstore || sstore,
   };
   uint32_t dw[2];
   ROGUE_CR_ISP_ZLSCTL_pack(dw, &ctl);
   out->zls[0] = g5p_dw64(dw);

   if (f->depth) {
      const struct ROGUE_CR_ISP_ZLOAD_BASE base = { .addr = t->va };
      ROGUE_CR_ISP_ZLOAD_BASE_pack(dw, &base); /* ZSTORE_BASE has the same field */
      out->zls[2] = g5p_dw64(dw);
      if (d)
         out->zls[1] = out->zls[2];
   }
   if (s) {
      const struct ROGUE_CR_ISP_STENCIL_LOAD_BASE base = { .addr = t->stencil_va ? t->stencil_va : t->va + plane.stencil_offset, .enable = true };
      ROGUE_CR_ISP_STENCIL_LOAD_BASE_pack(dw, &base);
      out->zls[3] = g5p_dw64(dw);
      out->zls[4] = out->zls[3];
   }

   uint32_t w = t->width, h = t->height;
   if (f->stencil || (!t->texture_chain && (w < 32 || h < 32)) || t->samples == 4) {
      w = g5p_align(w, 32);
      h = g5p_align(h, 32);
   }
   const struct ROGUE_CR_ISP_ZLS_PIXELS px = { .x = w - 1, .y = h - 1 };
   uint32_t pixels;
   ROGUE_CR_ISP_ZLS_PIXELS_pack(&pixels, &px);
   out->zls[5] = pixels;

   for (unsigned i = 1; i <= 4; i++)
      if (out->zls[i])
         out->address_mask |= 1u << i;

   /* the depth clear in the plane's format: D16 as its 16-bit integer */
   float dc = t->depth_clear < 0.0f ? 0.0f : t->depth_clear > 1.0f ? 1.0f : t->depth_clear;
   out->bgobj_depth = f->zls_type == ROGUE_CR_ZLS_FORMAT_TYPE_16BITINT ? (uint32_t)lroundf(dc * 65535.0f)
                                                                        : g5p_fui(dc);
   const struct ROGUE_CR_ISP_BGOBJVALS v = { .enablebgtag = true, .mask = true, .stencil = s ? t->stencil_clear : 0 };
   ROGUE_CR_ISP_BGOBJVALS_pack(&out->bgobj_vals, &v);
   return NULL;
}

/*
 * The ISP words of a draw. Without a depth/stencil attachment: the captured
 * colour-only form (ISPCTL 0x03c40000, back words that single-sided ISPCTL
 * ignores). With one: Rogue's face setup (pvr_setup_isp_faces_and_control)
 * in the form the iPadOS driver emits it: the back words repeat the front
 * ones unless the faces differ, bpres and two_sided with a stencil test, the
 * stencil write mask kept as given (captured: ISPCTL 0x03c00000 depth only,
 * 0x03cc0000 with stencil). The user pass is 15 at one sample and 7 at four
 * (the captured 4x draws, ISPCTL 0x01c40000).
 */
void
pvr_g5p_ds_isp(const struct pvr_g5p_isp_state *s, uint32_t w[5])
{
   const enum ROGUE_TA_OBJTYPE obj = s->lines ? ROGUE_TA_OBJTYPE_LINE : ROGUE_TA_OBJTYPE_TRIANGLE;
   const uint32_t upass = s->no_colour ? 0 : s->samples == 4 ? 7 : 15;

   if (!s->ds_attachment) {
      const struct ROGUE_TA_STATE_ISPCTL ctl = { .upass = upass, .bpres = true, .scenable = s->scissor,
                                                 .dbenable = s->dbias, .tagwritedisable = s->no_colour };
      const struct ROGUE_TA_STATE_ISPA fa = { .objtype = obj, .passtype = s->passtype, .dwritedisable = true,
                                              .dfbztestenable = true, .dcmpmode = ROGUE_TA_CMPMODE_ALWAYS,
                                              .pointlinewidth = s->line_width };
      const struct ROGUE_TA_STATE_ISPB fb = { .scmpmode = ROGUE_TA_CMPMODE_ALWAYS };
      const struct ROGUE_TA_STATE_ISPA ba = { .objtype = obj, .passtype = s->passtype, .dfbztestenable = true,
                                              .pointlinewidth = s->line_width };
      ROGUE_TA_STATE_ISPCTL_pack(&w[0], &ctl);
      ROGUE_TA_STATE_ISPA_pack(&w[1], &fa);
      ROGUE_TA_STATE_ISPB_pack(&w[2], &fb);
      ROGUE_TA_STATE_ISPA_pack(&w[3], &ba);
      w[4] = 0;
      return;
   }

   const bool test = s->depth_test;
   struct ROGUE_TA_STATE_ISPA a = {
      .objtype = obj,
      .passtype = s->passtype,
      .dwritedisable = !(test && s->depth_write),
      .dfbztestenable = true,
      /* VkCompareOp and ROGUE_TA_CMPMODE share their numbering */
      .dcmpmode = test ? (enum ROGUE_TA_CMPMODE)s->depth_compare : ROGUE_TA_CMPMODE_ALWAYS,
      .pointlinewidth = s->line_width,
   };
   uint32_t fa, fb, ba, bb;
   if (s->stencil_test) {
      const struct pvr_g5p_stencil_face *face[2] = { &s->front, &s->back };
      uint32_t *wa[2] = { &fa, &ba }, *wb[2] = { &fb, &bb };
      for (unsigned i = 0; i < 2; i++) {
         a.sref = face[i]->reference;
         ROGUE_TA_STATE_ISPA_pack(wa[i], &a);
         /* VkStencilOp and ROGUE_TA_ISPB_STENCILOP share their numbering */
         const struct ROGUE_TA_STATE_ISPB b = {
            .scmpmode = (enum ROGUE_TA_CMPMODE)face[i]->compare,
            .sop1 = (enum ROGUE_TA_ISPB_STENCILOP)face[i]->fail,
            .sop2 = (enum ROGUE_TA_ISPB_STENCILOP)face[i]->depth_fail,
            .sop3 = (enum ROGUE_TA_ISPB_STENCILOP)face[i]->pass,
            .scmpmask = face[i]->compare_mask,
            .swmask = face[i]->write_mask,
         };
         ROGUE_TA_STATE_ISPB_pack(wb[i], &b);
      }
   } else {
      const struct ROGUE_TA_STATE_ISPB b = { .scmpmode = ROGUE_TA_CMPMODE_ALWAYS };
      ROGUE_TA_STATE_ISPA_pack(&fa, &a);
      ROGUE_TA_STATE_ISPB_pack(&fb, &b);
      ba = fa;
      bb = fb;
   }
   if (fa != ba || fb != bb) {
      if (s->cull_back) {
         ba = fa, bb = fb; /* one face drawn: the front's state */
      } else if (s->cull_front) {
         fa = ba, fb = bb;
      } else if (s->front_ccw) {
         uint32_t t = fa;
         fa = ba, ba = t;
         t = fb, fb = bb, bb = t;
      }
   }
   const struct ROGUE_TA_STATE_ISPCTL ctl = { .upass = upass, .bpres = s->stencil_test, .two_sided = s->stencil_test,
                                              .dbenable = s->dbias, .scenable = s->scissor,
                                              .tagwritedisable = s->no_colour };
   ROGUE_TA_STATE_ISPCTL_pack(&w[0], &ctl);
   w[1] = fa, w[2] = fb, w[3] = ba, w[4] = bb;
}

/*
 * The texture unit reads a ZLS plane in place. Both use Morton order with y
 * in bit 0 (the captured shadow-sampling scene, 64x64; the measured 100x75
 * footprint). The texture unit pads each side of the view to its own power
 * of two and twiddles a rectangle as Morton over the smaller power with the
 * larger side's remaining bits above. The ZLS works in whole 32-pixel tiles.
 * Candidate (g5p-1b-zls-sample, pending tools/g5p-1b-compute/vk-zls-order on
 * hardware): the ZLS orders a rectangle the same way over its 32-aligned
 * sides. The two orders then agree when both sides pad to the same power of
 * two (Morton over a square is a prefix of a larger one), and otherwise
 * by the bit-placement rule below (tools/g5p-1b-compute/zls-order/
 * plane-test.c checks it against both orders at every pixel of every size
 * up to 130x130). If the hardware stores a rectangle as Morton
 * over the larger side's square instead, only equal powers agree and this
 * returns to refusing every other extent.
 */
const char *
pvr_g5p_ds_texture_plane(VkFormat format, VkImageAspectFlags aspect, uint32_t width, uint32_t height,
                         uint64_t va, uint64_t *plane_va, VkFormat *texture_format)
{
   const struct pvr_g5p_ds_format *f = pvr_g5p_ds_format(format);
   struct pvr_g5p_ds_plane plane;

   if (!f || !f->sampled)
      return "a sampled depth/stencil plane without a texture key";
   /* the texture's padded sides tw x th, the ZLS's zw x zh (32-aligned first). Morton bit b of x sits at 2b + 1
    * and of y at 2b; past the smaller power the larger side's bits continue linearly. Wider: the extra x bits
    * agree only when both orders interleave the same number of bits (the height pads alike: zh == th). Taller:
    * the extra y bits agree when the width pads alike (zw == tw), or when there is exactly one extra y bit (y's
    * Morton slot 2b equals its linear slot m + b at b = m, y being the bit-0 coordinate). */
   const uint32_t tw = g5p_pot(width), th = g5p_pot(height);
   const uint32_t zw = g5p_pot(g5p_align(width, 32)), zh = g5p_pot(g5p_align(height, 32));
   if ((tw > th && zh != th) || (th > tw && zw != tw && th != 2 * tw))
      return "a sampled ZLS plane whose texture order differs from its store order (a side of 16 pixels or fewer "
             "the ZLS pads to a 32-pixel tile and the texture unit does not, next to a side of another power of two)";
   if (!pvr_g5p_ds_layout(format, width, height, 1, &plane))
      return "a sampled depth/stencil image of 0 pixels";
   switch (aspect) {
   case VK_IMAGE_ASPECT_DEPTH_BIT:
      if (!f->depth)
         return "the depth aspect of a format without depth";
      *plane_va = va;
      *texture_format = f->depth_bytes == 2 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
      return NULL;
   case VK_IMAGE_ASPECT_STENCIL_BIT:
      if (!f->stencil)
         return "the stencil aspect of a format without stencil";
      *plane_va = va + plane.stencil_offset;
      *texture_format = VK_FORMAT_R8_UINT;
      return NULL;
   default:
      return "a sampled depth/stencil view with both aspects";
   }
}

VkFormatFeatureFlags2
pvr_g5p_ds_format_features(VkFormat format)
{
   const struct pvr_g5p_ds_format *f = pvr_g5p_ds_format(format);

   if (!f)
      return 0;
   return VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT |
          VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
          (f->sampled ? VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT : 0) |
          (f->sampled && f->depth ? VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT : 0);
}
