/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): colour formats (pvr_g5p_formats.h). Built only with
 * -Dimagination-g5p-experimental=true.
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
#include "vk_enum_to_str.h"
#include "vk_format.h"

#include "pvr_g5p_format_table.h"
#include "pvr_g5p_formats.h"

bool
pvr_g5p_formats_probe(void)
{
   static int probe = -1;

   if (probe < 0) {
      const char *e = getenv("PVR_G5P_FORMATS");
      probe = e && !strcmp(e, "probe");
      if (probe)
         mesa_logw("G5P: PVR_G5P_FORMATS=probe: formats not yet run on the device are advertised");
   }
   return probe;
}

bool
pvr_g5p_format_tex(VkFormat format, uint32_t *key, bool *srgb)
{
   const struct pvr_g5p_format *f = pvr_g5p_format_find(format);

   if (!f || !(pvr_g5p_format_flags(f, pvr_g5p_formats_probe()) & PVR_G5P_FMT_SAMPLED))
      return false;
   *key = f->key;
   *srgb = f->srgb;
   return true;
}

bool
pvr_g5p_format_pbe(VkFormat format, uint32_t *key, bool *srgb)
{
   return pvr_g5p_format_pbe_key(format, pvr_g5p_formats_probe(), key, srgb);
}

VkFormatFeatureFlags2
pvr_g5p_color_format_features(VkFormat format, VkImageTiling tiling, VkFormatFeatureFlags2 flags)
{
   const struct pvr_g5p_format *f = pvr_g5p_format_find(format);

   /* the A9's own answer; upstream's flags only say the format exists */
   (void)flags;
   return pvr_g5p_format_features(f, tiling == VK_IMAGE_TILING_LINEAR, pvr_g5p_formats_probe());
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
   if (!why)
      return VK_SUCCESS;
   mesa_logd("G5P: no %s image of %s, tiling %s, usage %#x, flags %#x: %s", vk_ImageType_to_str(info->type),
             vk_Format_to_str(info->format), vk_ImageTiling_to_str(tiling), info->usage, info->flags, why);
   return VK_ERROR_FORMAT_NOT_SUPPORTED;
}
