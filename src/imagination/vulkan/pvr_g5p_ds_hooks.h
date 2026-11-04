/* SPDX-License-Identifier: MIT */
/*
 * The Apple A9 (G5P) depth/stencil hooks in the shared PowerVR driver files,
 * in the form of pvr_g5p_hooks.h: one line, PVR_G5P_HOOK(name, ...), in the
 * place of a blank line, which expands to nothing without
 * -Dimagination-g5p-experimental, so the option-off build keeps every line
 * and byte of the file without G5P.
 */

#ifndef PVR_G5P_DS_HOOKS_H
#define PVR_G5P_DS_HOOKS_H

#if defined(PVR_APPLE_G5P)
#   include "pvr_g5p_ds.h"
#endif

#ifndef PVR_G5P_HOOK
#   if defined(PVR_APPLE_G5P)
#      define PVR_G5P_HOOK(name, ...) PVR_G5P_HOOK_##name(__VA_ARGS__)
#   else
#      define PVR_G5P_HOOK(name, ...)
#   endif
#endif

/* pvr_image.c, pvr_image_init(): a depth/stencil image takes the ZLS
 * planes' size (pvr_g5p_ds_image_layout())
 */
#define PVR_G5P_HOOK_ds_image(device, image)                            \
   if (PVR_HAS_FEATURE(&(device)->pdevice->dev_info, apple_g5p)) \
      pvr_g5p_ds_image_layout(image);

/* pvr_formats.c, pvr_get_image_format_features2(): depth/stencil formats
 * are the four the A9's ZLS stores (pvr_g5p_ds_format(): D16, D32F, S8,
 * D32F_S8), as attachments only. D24S8, X8_D24 and D16_S8 get no features,
 * so Zink renders GL's DEPTH24_STENCIL8 with D32F_S8 as on our ICD. No
 * sampling until the texture state packs a ZLS plane
 * (pvr_g5p_ds_texture_plane()), and no transfers: the planes are in the
 * ZLS's Morton order over a square of 32-pixel tiles, not a layout the
 * CPU copies know.
 */
#define PVR_G5P_HOOK_ds_format_features(pdevice, vk_format, flags)                          \
   if (PVR_HAS_FEATURE(&(pdevice)->dev_info, apple_g5p) && vk_format_is_depth_or_stencil(vk_format)) \
      (flags) &= pvr_g5p_ds_format(vk_format) ? VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT : 0;

#endif /* PVR_G5P_DS_HOOKS_H */
