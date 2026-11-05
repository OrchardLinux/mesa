/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): image memory and the sample counts the device
 * reports. Built only with -Dimagination-g5p-experimental=true.
 *
 * Colour images (lane b; the A9 ICD's image_layout()/twiddled_layout()):
 * Apple's image state takes a linear layout for plain 2D only, so the A9
 * twiddles exactly the images that need it:
 *  - mip chains, cube-compatible images, sampled arrays, compressed
 *    formats and 3D images are twiddled: Morton order with y in bit 0 over
 *    each level's own extent padded to powers of two, levels back to back
 *    from the image base, each the power-of-two level-0 extent shifted
 *    (upstream's twiddled levels, unchanged);
 *  - a layer of an array or cube is, when the image has levels, the whole 2D
 *    chain down to 1x1, rounded to 64 bytes, or to 16 KiB from 16 KiB up:
 *    the A9 derives the layer pitch from the level-0 size, the has-levels bit
 *    and the big-layer bit of the image state alone (device 2026-11-04,
 *    formats l-arr-npot: a 12x10 3-level array reads layer 1 at 1408 =
 *    the 5-level chain's 1364 rounded, not 1344; Apple's getMipmapSize);
 *  - a 3D image is a 2D array of its slices (the same device run: 8x8x8
 *    reads (4,0,0) at element 32 of slice 0, and level 1 at 64 elements, one
 *    slice's 2D chain; Apple's getLevelOffset has no depth term and
 *    agxsTwiddle3DAddressCommon steps slice z by the layer pitch): one
 *    slice's 2D levels, slice z of every level z layer pitches on. It keeps
 *    memlayout 3DTWIDDLED (upstream's view setup wants it); the transfers
 *    take each slice as a 2D-twiddled surface one layer pitch on
 *    (PVR_G5P_HOOK_slice_pitch, PVR_G5P_HOOK_transfer_slice);
 *  - every other optimal 2D colour image is linear, its rows a multiple of
 *    64 bytes (pvr_image_init()'s stride alignment on G5P). The G5P render
 *    path writes only linear colour targets, as the ICD's desktop did.
 *
 * Depth/stencil and 4-sample images (lane c): the ZLS stores each
 * depth/stencil plane in its twiddled order over a power-of-two square of
 * 32-pixel tiles, and the stencil plane after the depth plane
 * (pvr_g5p_ds_layout()), so a depth/stencil image takes exactly that; a
 * 4-sample colour image is the PBE's 2D-multisample surface, 16 bytes a pixel
 * over power-of-two sides (pvr_g5p_msaa_layout()). pvr_image_init()'s G5P hook
 * calls pvr_g5p_ds_image_layout() after upstream's own layout.
 */

#include "util/u_math.h"
#include "vk_format.h"
#include "vk_physical_device.h"

#include "pvr_g5p_ds.h"
#include "pvr_g5p_msaa.h"
#include "pvr_g5p_tex.h"
#include "pvr_image.h"
#include "pvr_physical_device.h"
#include "winsys/apple_g5p/pvr_g5p_winsys.h"

void
pvr_g5p_image_layout(struct pvr_image *image, unsigned stride_align)
{
   struct pvr_image_plane *plane = &image->planes[0];
   const VkDeviceSize alignment = image->alignment;

   const bool is3d = image->vk.image_type == VK_IMAGE_TYPE_3D && image->memlayout == PVR_MEMLAYOUT_3DTWIDDLED;

   if (image->plane_count != 1 || (image->vk.image_type != VK_IMAGE_TYPE_2D && !is3d) ||
       image->vk.samples != VK_SAMPLE_COUNT_1_BIT || vk_format_is_depth_or_stencil(image->vk.format) ||
       (!is3d && image->memlayout != PVR_MEMLAYOUT_TWIDDLED))
      return;

   const bool twiddled = is3d || image->vk.mip_levels > 1 ||
                         (image->vk.create_flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) ||
                         (image->vk.array_layers > 1 && (image->vk.usage & VK_IMAGE_USAGE_SAMPLED_BIT)) ||
                         vk_format_is_compressed(image->vk.format);

   if (!twiddled) {
      const unsigned cpp = vk_format_get_blocksize(image->vk.format);
      struct pvr_mip_level *level = &plane->mip_levels[0];

      image->memlayout = PVR_MEMLAYOUT_LINEAR;
      plane->physical_extent = image->vk.extent;
      plane->physical_extent.width = align(plane->physical_extent.width, stride_align);
      level->pitch = cpp * plane->physical_extent.width;
      level->height_pitch = plane->physical_extent.height;
      level->size = (VkDeviceSize)level->pitch * level->height_pitch;
      level->offset = 0;
      plane->layer_size = level->size;
      if (image->vk.array_layers > 1)
         plane->layer_size = align64(plane->layer_size, alignment);
   } else {
      const unsigned cpp = vk_format_get_blocksize(image->vk.format);
      const unsigned bw = vk_format_get_blockwidth(image->vk.format), bh = vk_format_get_blockheight(image->vk.format);
      const uint32_t pw = util_next_power_of_two(image->vk.extent.width);
      const uint32_t ph = util_next_power_of_two(image->vk.extent.height);
      /* the levels a layer holds: the whole chain when the image has levels */
      const uint32_t chain = image->vk.mip_levels > 1 ? util_logbase2(MAX2(pw, ph)) + 1 : 1;
      VkDeviceSize size = 0, own = 0;

      for (uint32_t l = 0; l < MAX2(chain, image->vk.mip_levels); l++) {
         const uint32_t ew = DIV_ROUND_UP(u_minify(pw, l), bw), eh = DIV_ROUND_UP(u_minify(ph, l), bh);
         const VkDeviceSize level_size = (VkDeviceSize)cpp * ew * eh;

         if (is3d && l < image->vk.mip_levels) {
            /* a 3D image's levels are one slice's: upstream's 3DTWIDDLED levels hold every slice */
            struct pvr_mip_level *level = &plane->mip_levels[l];

            level->pitch = cpp * ew;
            level->height_pitch = eh;
            level->size = level_size;
            level->offset = own;
         }
         if (l < chain)
            size += level_size;
         if (l < image->vk.mip_levels)
            own += level_size;
      }
      plane->layer_size = image->vk.array_layers > 1 || is3d
                             ? align64(size, size >= PVR_G5P_IMAGE_BIG_LAYER ? PVR_G5P_IMAGE_BIG_LAYER : 64)
                             : own;
   }
   plane->size = plane->layer_size * (is3d ? image->vk.extent.depth : image->vk.array_layers);
   plane->offset = 0;
   image->total_size = plane->size;
}

void
pvr_g5p_ds_image_layout(struct pvr_image *image)
{
   struct pvr_g5p_ds_plane zls;
   uint64_t layer = 0;

   if (image->vk.tiling != VK_IMAGE_TILING_OPTIMAL || image->plane_count != 1 || image->vk.mip_levels != 1)
      return;
   struct pvr_image_plane *plane = &image->planes[0];
   if (pvr_g5p_ds_layout(image->vk.format, image->vk.extent.width, image->vk.extent.height, image->vk.samples,
                         &zls)) {
      /* The ZLS planes are the image's whole memory: the ZLS loads and
       * stores them, the texture state samples them in place
       * (pvr_g5p_ds_texture_plane()), and the formats have no transfer
       * features (pvr_g5p_ds_format_features()), so nothing addresses
       * upstream's layout. That layout takes D32F_S8 as 8 bytes a texel
       * over each side's power of two: 32 MiB for a 2048x1536 panel, where
       * the depth and stencil planes take 16 + 4 MiB (Hyprland's per-monitor
       * GL DEPTH24_STENCIL8 texture, which Zink makes D32F_S8). */
      plane->layer_size = zls.size;
      plane->size = zls.size * image->vk.array_layers;
      image->total_size = align64(plane->offset + plane->size, image->alignment);
      return;
   }
   if (image->vk.samples == VK_SAMPLE_COUNT_4_BIT && vk_format_is_color(image->vk.format))
      layer = pvr_g5p_msaa_layout(image->vk.extent.width, image->vk.extent.height,
                                  vk_format_get_blocksize(image->vk.format));
   if (!layer)
      return;
   plane->layer_size = MAX2(plane->layer_size, layer);
   plane->size = MAX2(plane->size, plane->layer_size * image->vk.array_layers);
   image->total_size = MAX2(image->total_size, align64(plane->offset + plane->size, image->alignment));
}

/*
 * The device's sample counts: 1, and 4 when the kernel takes 4-sample passes,
 * for every framebuffer kind; sampled and storage images keep one sample
 * (sampling a 4-sample image is not built: no 2D-multisample texture state
 * was captured). Upstream's Rogue limits offer 4 everywhere.
 */
void
pvr_g5p_msaa_limits(struct pvr_physical_device *pdevice, struct vk_properties *p)
{
   const VkSampleCountFlags fb = pvr_g5p_msaa_sample_counts(pvr_g5p_winsys_msaa(pdevice->ws));

   p->framebufferColorSampleCounts = fb;
   p->framebufferDepthSampleCounts = fb;
   p->framebufferStencilSampleCounts = fb;
   p->framebufferNoAttachmentsSampleCounts = fb;
   p->framebufferIntegerColorSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->sampledImageColorSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->sampledImageIntegerSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->sampledImageDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->sampledImageStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT;
}

/*
 * An image format's sample counts: 4 as well only for what a 4x pass
 * renders (our ICD's rule): a 2D optimal colour attachment of 4-byte
 * samples, or one of the depth/stencil formats the ZLS keeps on chip, used
 * neither as a sampled, storage nor input attachment image, on a kernel with
 * 4-sample passes. Upstream's Rogue answer is every count up to 8.
 */
void
pvr_g5p_msaa_image_samples(struct pvr_physical_device *pdevice, const VkPhysicalDeviceImageFormatInfo2 *info,
                           VkImageTiling tiling, VkImageFormatProperties *props)
{
   const VkImageUsageFlags attach = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   const VkImageUsageFlags read = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                  VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
   const bool format_ok = pvr_g5p_ds_format(info->format) ||
                          (vk_format_is_color(info->format) && !vk_format_is_compressed(info->format) &&
                           !vk_format_is_int(info->format) && vk_format_get_blocksize(info->format) == 4);
   const bool four = pvr_g5p_winsys_msaa(pdevice->ws) && tiling == VK_IMAGE_TILING_OPTIMAL &&
                     info->type == VK_IMAGE_TYPE_2D && !(info->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
                     (info->usage & attach) && !(info->usage & read) && format_ok;

   props->sampleCounts &= four ? VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT;
}

/*
 * Whether the PBE resolves a colour attachment at the end of each tile:
 * 4 samples of a 4-byte, non-integer colour format, averaged (Apple's
 * msaa4-resolve, exact on the A9 with our ICD: (sum + 2) / 4 per byte).
 * Upstream decides this with gs_rta_support, which the A9 lacks, and would
 * store every 4x attachment for a transfer resolve instead.
 */
bool
pvr_g5p_msaa_pbe_resolvable(VkFormat format, uint32_t samples, VkResolveModeFlagBits mode)
{
   return samples == VK_SAMPLE_COUNT_4_BIT && mode == VK_RESOLVE_MODE_AVERAGE_BIT && vk_format_is_color(format) &&
          !vk_format_is_int(format) && !vk_format_is_compressed(format) && vk_format_get_blocksize(format) == 4;
}
