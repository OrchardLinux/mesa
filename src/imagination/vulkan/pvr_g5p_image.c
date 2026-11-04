/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): image memory and the sample counts the device
 * reports. The ZLS stores each depth/stencil plane in its twiddled order
 * over a power-of-two square of 32-pixel tiles, and the stencil plane after
 * the depth plane (pvr_g5p_ds_layout()), so a depth/stencil image takes at
 * least that; a 4-sample colour image is the PBE's 2D-multisample surface,
 * 16 bytes a pixel over power-of-two sides (pvr_g5p_msaa_layout()).
 * pvr_image_init()'s G5P hook calls pvr_g5p_ds_image_layout() after
 * upstream's own layout. Built only with -Dimagination-g5p-experimental=true.
 */

#include "util/u_math.h"
#include "vk_format.h"
#include "vk_physical_device.h"

#include "pvr_g5p_ds.h"
#include "pvr_g5p_msaa.h"
#include "pvr_image.h"
#include "pvr_physical_device.h"
#include "winsys/apple_g5p/pvr_g5p_winsys.h"

void
pvr_g5p_ds_image_layout(struct pvr_image *image)
{
   struct pvr_g5p_ds_plane zls;
   uint64_t layer = 0;

   if (image->vk.tiling != VK_IMAGE_TILING_OPTIMAL || image->plane_count != 1 || image->vk.mip_levels != 1)
      return;
   if (pvr_g5p_ds_layout(image->vk.format, image->vk.extent.width, image->vk.extent.height, image->vk.samples,
                         &zls))
      layer = zls.size;
   else if (image->vk.samples == VK_SAMPLE_COUNT_4_BIT && vk_format_is_color(image->vk.format))
      layer = pvr_g5p_msaa_layout(image->vk.extent.width, image->vk.extent.height,
                                  vk_format_get_blocksize(image->vk.format));
   if (!layer)
      return;
   struct pvr_image_plane *plane = &image->planes[0];
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
