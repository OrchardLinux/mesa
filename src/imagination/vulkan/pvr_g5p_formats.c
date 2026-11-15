/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): colour formats (pvr_g5p_formats.h).
 *
 * Upstream's pvr_get_image_format_features2() answers from Rogue's table,
 * which offers far more than the A9 path can do: integer and narrow render
 * targets, storage images, every Rogue texture format. G5P replaces the
 * colour part of that answer with its own table (pvr_g5p_format_table.h), so
 * that Zink and the compositors pick only what runs: what isn't in the table
 * gets no features, and the image-format query follows from the features.
 * Depth/stencil formats are lane (c)'s (pvr_g5p_ds.c).
 *
 * In the shape of G9P's own format gate (the G9P driver, g9p_image_format() in
 * pvr_formats.c): upstream's queries, with the A9's answer for its formats.
 */

#include <stdlib.h>
#include <string.h>

#include "util/log.h"
#include "util/u_debug.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"

#include "pvr_g5p_format_table.h"
#include "pvr_g5p_formats.h"

/* Formats that have not run on the device are not advertised */
bool
pvr_g5p_formats_probe(void)
{
   return false;
}

/* Qualified 16-bit packed texture formats are always enabled. */
static bool
tex16_row(VkFormat format)
{
   const struct pvr_g5p_format *f = pvr_g5p_format_find(format);
   return f && f->evidence == PVR_G5P_FMT_EV_AGX && f->cpp == 2;
}

static bool
row_probe(VkFormat format)
{
   return pvr_g5p_formats_probe() || tex16_row(format);
}

bool
pvr_g5p_format_tex(VkFormat format, uint32_t *key, bool *srgb)
{
   const struct pvr_g5p_format *f = pvr_g5p_format_find(format);

   if (!f || !(pvr_g5p_format_flags(f, row_probe(format)) & PVR_G5P_FMT_SAMPLED))
      return false;
   *key = f->key;
   *srgb = f->srgb;
   return true;
}

/* The physical one-dword store of already packed bytes (5..8 compact
 * targets): R32UI's PBE key, no format admission of its own. */
uint32_t
pvr_g5p_packed_word_pbe_key(void)
{
   return pvr_g5p_format_find(VK_FORMAT_R32_UINT)->key;
}

bool
pvr_g5p_format_pbe(VkFormat format, uint32_t *key, bool *srgb)
{
   return pvr_g5p_format_pbe_key(format, row_probe(format), key, srgb);
}

VkFormatFeatureFlags2
pvr_g5p_color_format_features(VkFormat format, VkImageTiling tiling, VkFormatFeatureFlags2 flags)
{
   const struct pvr_g5p_format *f = pvr_g5p_format_find(format);

   /* the A9's own answer; upstream's flags only say the format exists */
   (void)flags;
   return pvr_g5p_format_features(f, tiling == VK_IMAGE_TILING_LINEAR, row_probe(format));
}

VkResult
pvr_g5p_image_format_check(const VkPhysicalDeviceImageFormatInfo2 *info, VkImageTiling tiling)
{
   const char *why = NULL;

   if (vk_format_is_depth_or_stencil(info->format))
      return VK_SUCCESS; /* lane (c): pvr_g5p_ds.c */
   if (!pvr_g5p_format_find(info->format))
      why = "no A9 format key";
   else if (info->type == VK_IMAGE_TYPE_1D)
      why = "1D images (the A9 texture state takes 2D, arrays, 3D and cubes; 1D is not built)";
   else if (info->type == VK_IMAGE_TYPE_3D && vk_format_is_compressed(info->format))
      why = "3D block-compressed images (the A9's 3D order is built for texels only)";
   else if (info->type == VK_IMAGE_TYPE_3D && tiling != VK_IMAGE_TILING_OPTIMAL)
      why = "linear 3D images (Apple samples 3D twiddled only)";
   if (!why && (info->usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
      const struct pvr_g5p_format *f = pvr_g5p_format_find(info->format);
      if (!(f->flags & PVR_G5P_FMT_STORAGE))
         why = "no A9 storage pack path for this format";
      else if (info->type != VK_IMAGE_TYPE_2D || tiling != VK_IMAGE_TILING_OPTIMAL)
         why = "A9 storage images require optimal 2D images (including arrays/cube faces)";
   }
   if (!why)
      return VK_SUCCESS;
   mesa_logd("G5P: no %s image of %s, tiling %s, usage %#x, flags %#x: %s", vk_ImageType_to_str(info->type),
             vk_Format_to_str(info->format), vk_ImageTiling_to_str(tiling), info->usage, info->flags, why);
   return VK_ERROR_FORMAT_NOT_SUPPORTED;
}
