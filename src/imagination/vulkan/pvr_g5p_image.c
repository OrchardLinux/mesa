/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): the memory of a depth/stencil image. The ZLS stores
 * each plane in its twiddled order over a power-of-two square of 32-pixel
 * tiles, and the stencil plane after the depth plane (pvr_g5p_ds_layout()),
 * so the image takes at least that; pvr_image_init()'s G5P hook calls this
 * after upstream's own layout. Built only with
 * -Dimagination-g5p-experimental=true.
 */

#include "util/u_math.h"

#include "pvr_g5p_ds.h"
#include "pvr_image.h"

void
pvr_g5p_ds_image_layout(struct pvr_image *image)
{
   struct pvr_g5p_ds_plane zls;

   if (image->vk.tiling != VK_IMAGE_TILING_OPTIMAL || image->plane_count != 1 || image->vk.mip_levels != 1 ||
       !pvr_g5p_ds_layout(image->vk.format, image->vk.extent.width, image->vk.extent.height, image->vk.samples,
                          &zls))
      return;
   struct pvr_image_plane *plane = &image->planes[0];
   plane->layer_size = MAX2(plane->layer_size, zls.size);
   plane->size = MAX2(plane->size, plane->layer_size * image->vk.array_layers);
   image->total_size = MAX2(image->total_size, align64(plane->offset + plane->size, image->alignment));
}
