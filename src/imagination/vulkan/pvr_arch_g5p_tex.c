/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): the texture state of image views and the words of
 * samplers, written into upstream's pvr_* descriptors (pvr_g5p_tex.h).
 * Compiled only with -Dimagination-g5p-experimental=true.
 *
 * In the shape of G9P's image and sampler states (Henrik Klee,
 * pvr_arch_g9p_gfx.c "Image and sampler states"): upstream's objects, the
 * hardware words from a G5P function at the call site.
 */

#include <inttypes.h>
#include <string.h>

#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_log.h"
#include "vk_util.h"

#include "util/log.h"

#include "pvr_common.h"
#include "pvr_g5p_tex.h"
#include "pvr_g5p_tex_encode.h"
#include "pvr_image.h"

/*
 * Format keys of the rendered and textured rungs: format code and channel
 * selectors, our decode of the iOS descriptor tables (pass/pass-rgba,
 * fmt-rgb10a2, fmt-srgb8 captures; ds-isa-state.md §2.1). The code is a bit
 * layout: 0x28 8_8_8_8, 0x26 10_10_10_2. sRGB is q1 bit 45 on the same
 * key. The A8B8G8R8 packed formats are the R8G8B8A8 bytes on a
 * little-endian GPU. Lane (f)'s table (every format) replaces this one.
 */
#define TEX_FMT_BGRA8 0x28, 2, 1, 0, 3
#define TEX_FMT_RGBA8 0x28, 0, 1, 2, 3
#define TEX_FMT_RGB10A2 0x26, 0, 1, 2, 3
#define TEX_KEY(...) PVR_G5P_TEX_KEY(__VA_ARGS__)

static const struct pvr_g5p_tex_format g5p_tex_formats[] = {
   { VK_FORMAT_B8G8R8A8_UNORM, TEX_KEY(TEX_FMT_BGRA8), false },
   { VK_FORMAT_B8G8R8A8_SRGB, TEX_KEY(TEX_FMT_BGRA8), true },
   { VK_FORMAT_R8G8B8A8_UNORM, TEX_KEY(TEX_FMT_RGBA8), false },
   { VK_FORMAT_R8G8B8A8_SRGB, TEX_KEY(TEX_FMT_RGBA8), true },
   { VK_FORMAT_A8B8G8R8_UNORM_PACK32, TEX_KEY(TEX_FMT_RGBA8), false },
   { VK_FORMAT_A8B8G8R8_SRGB_PACK32, TEX_KEY(TEX_FMT_RGBA8), true },
   { VK_FORMAT_A2B10G10R10_UNORM_PACK32, TEX_KEY(TEX_FMT_RGB10A2), false },
};

static const struct pvr_g5p_tex_format *
g5p_tex_format(VkFormat format)
{
   for (unsigned i = 0; i < ARRAY_SIZE(g5p_tex_formats); i++) {
      if (g5p_tex_formats[i].format == format)
         return &g5p_tex_formats[i];
   }
   return NULL;
}

/* The three qwords of a view, or the reason it has none */
static const char *
g5p_view_state(const struct pvr_image_view *iview, uint64_t q[3])
{
   const struct pvr_image *image = pvr_image_view_get_image(iview);
   const struct pvr_image_plane *plane = pvr_single_plane_const(image);
   const struct pvr_g5p_tex_format *f = g5p_tex_format(iview->vk.format);

   if (vk_format_is_depth_or_stencil(iview->vk.format))
      return "depth/stencil view (sampled from its ZLS planes: not built here)";
   if (!f)
      return "format without a texture key";

   const uint32_t key = pvr_g5p_tex_swizzle_key(f->key, &iview->vk.swizzle);

   if (image->memlayout == PVR_MEMLAYOUT_LINEAR) {
      const struct pvr_mip_level *level = &plane->mip_levels[0];
      const uint64_t va = image->dev_addr.addr + plane->offset + level->offset +
                          iview->vk.base_array_layer * plane->layer_size;

      if (iview->vk.view_type != VK_IMAGE_VIEW_TYPE_2D || iview->vk.base_mip_level)
         return "array/3D/cube or mip view of a linear image (Apple samples those twiddled only)";
      if (va % 64 || level->pitch % 16)
         return "linear texture off a 64-byte base or 16-byte rows";
      pvr_g5p_tex_linear(q, key, f->srgb, va, iview->vk.extent.width, iview->vk.extent.height,
                         level->pitch);
      return NULL;
   }

   struct pvr_g5p_tex_twiddled_view v = {
      .key = key,
      .width = image->vk.extent.width,
      .height = image->vk.extent.height,
      .depth = 1,
      .base_level = iview->vk.base_mip_level,
      .level_count = iview->vk.level_count,
      .image_has_mips = image->vk.mip_levels > 1,
      .srgb = f->srgb,
      .va = image->dev_addr.addr + plane->offset + iview->vk.base_array_layer * plane->layer_size,
   };
   const bool layered = image->vk.array_layers > 1 || image->vk.image_type == VK_IMAGE_TYPE_3D;

   v.big_layer = layered && plane->layer_size >= PVR_G5P_TEX_BIG_LAYER;
   switch (iview->vk.view_type) {
   case VK_IMAGE_VIEW_TYPE_2D:
      v.layout = PVR_G5P_TEX_LAYOUT_2D;
      break;
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
      v.layout = PVR_G5P_TEX_LAYOUT_2D_ARRAY;
      v.depth = iview->vk.layer_count;
      break;
   case VK_IMAGE_VIEW_TYPE_3D:
      v.layout = PVR_G5P_TEX_LAYOUT_3D;
      v.depth = image->vk.extent.depth;
      break;
   case VK_IMAGE_VIEW_TYPE_CUBE:
      if (iview->vk.layer_count != 6)
         return "cube view without six faces";
      v.layout = PVR_G5P_TEX_LAYOUT_CUBE;
      break;
   default:
      return "1D or cube-array view (A9 has no cube arrays; 1D textures not built)";
   }
   if (v.va % 64)
      return "twiddled view of a layer that is not 64-byte aligned";
   pvr_g5p_tex_twiddled(q, &v);
   return NULL;
}

/*
 * The sampled state of a view: q0, q1 in words[], q2 in meta[0..1],
 * meta[2..3] zero, the 32 bytes the backend's image upload reads.
 */
const char *
PVR_PER_ARCH(g5p_pack_image_view)(const struct pvr_image_view *iview, struct pvr_image_descriptor *desc)
{
   uint64_t q[3];
   const char *why = g5p_view_state(iview, q);

   static_assert(sizeof(desc->words) + sizeof(desc->meta) == 32, "G5P image state is 32 bytes");
   memset(desc, 0, sizeof(*desc));
   if (why) {
      if (pvr_image_view_get_image(iview)->vk.usage & VK_IMAGE_USAGE_SAMPLED_BIT)
         mesa_logw("G5P: no texture state for a %s view of %s: %s",
                   vk_ImageViewType_to_str(iview->vk.view_type), vk_Format_to_str(iview->vk.format), why);
      return why;
   }
   desc->words[0] = q[0];
   desc->words[1] = q[1];
   desc->meta[0] = (uint32_t)q[2];
   desc->meta[1] = (uint32_t)(q[2] >> 32);
   return NULL;
}

/*
 * Both forms of a sampler's word: the general one (Vulkan NEAREST
 * mipmapping as Apple's mip-nearest) in words[0], for standalone samplers
 * and views with mips; the single-level one in meta[RSVD0..1], which
 * g5p_combine_sampler() takes for one-level views.
 */
const char *
PVR_PER_ARCH(g5p_pack_sampler)(const VkSamplerCreateInfo *info, struct pvr_sampler_descriptor *desc)
{
   const bool extension = vk_find_struct_const(info->pNext, SAMPLER_YCBCR_CONVERSION_INFO) ||
                          vk_find_struct_const(info->pNext, SAMPLER_REDUCTION_MODE_CREATE_INFO) ||
                          vk_find_struct_const(info->pNext, SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT);
   uint64_t general, single;
   const char *why;

   memset(desc, 0, sizeof(*desc));
   why = pvr_g5p_sampler_encode(info, extension, true, &general);
   if (why) {
      mesa_logw("G5P: sampler (filter %u/%u, mip %u, address %u/%u/%u) not encodable: %s", info->magFilter,
                info->minFilter, info->mipmapMode, info->addressModeU, info->addressModeV,
                info->addressModeW, why);
      return why;
   }
   pvr_g5p_sampler_encode(info, extension, false, &single);
   desc->words[0] = general;
   desc->meta[PCO_SAMPLER_META_RSVD0] = (uint32_t)single;
   desc->meta[PCO_SAMPLER_META_RSVD1] = (uint32_t)(single >> 32);
   return NULL;
}

/* A combined image/sampler descriptor: the single-level form for a view
 * with one level (the form the r4e desktop ran on), the general one else. */
void
PVR_PER_ARCH(g5p_combine_sampler)(struct pvr_combined_image_sampler_descriptor *desc,
                                  const struct pvr_image_view *iview)
{
   if (iview && iview->vk.level_count == 1) {
      desc->sampler.words[0] = desc->sampler.meta[PCO_SAMPLER_META_RSVD0] |
                               (uint64_t)desc->sampler.meta[PCO_SAMPLER_META_RSVD1] << 32;
   }
}
