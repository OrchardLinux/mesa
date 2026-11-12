/* SPDX-License-Identifier: MIT */
/*
 * The Apple A9 (G5P) depth/stencil and multisample hooks in the shared
 * PowerVR driver files,
 * in the form of pvr_g5p_hooks.h: one line, PVR_G5P_HOOK(name, ...), in the
 * place of a blank line, which expands to nothing without
 * -Dimagination-g5p-experimental, so the option-off build keeps every line
 * and byte of the file without G5P.
 */

#ifndef PVR_G5P_DS_HOOKS_H
#define PVR_G5P_DS_HOOKS_H

#if defined(PVR_APPLE_G5P)
#   include "pvr_g5p_ds.h"
#   include "pvr_g5p_msaa.h"
#   include "pvr_g5p_draw_hooks.h"
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
 * D32F_S8), as attachments, and D32F and D32F_S8 sampled with depth
 * comparison as well (pvr_g5p_ds_format_features(): the texture state reads
 * a ZLS plane in place). D24S8, X8_D24 and D16_S8 get no features, so Zink
 * renders GL's DEPTH24_STENCIL8 with D32F_S8 as on our ICD. No linear
 * filtering and no transfers: the planes are in the ZLS's Morton order over
 * a square of 32-pixel tiles, not a layout the CPU copies know.
 */
#define PVR_G5P_HOOK_ds_format_features(pdevice, vk_format, flags)                          \
   if (PVR_HAS_FEATURE(&(pdevice)->dev_info, apple_g5p) && vk_format_is_depth_or_stencil(vk_format)) \
      (flags) &= pvr_g5p_ds_format_features(vk_format);

/* pvr_physical_device.c, pvr_physical_device_init(): the sample counts of
 * the A9 path (pvr_g5p_msaa_limits()) in place of upstream's 1 | 4 everywhere */
#define PVR_G5P_HOOK_msaa_limits(pdevice, props)                 \
   if (PVR_HAS_FEATURE(&(pdevice)->dev_info, apple_g5p)) \
      pvr_g5p_msaa_limits((pdevice), (props));

/* pvr_formats.c, pvr_get_image_format_properties(): an image format's
 * sample counts (pvr_g5p_msaa_image_samples()) */
#define PVR_G5P_HOOK_msaa_image_samples(pdevice, info, tiling, props) \
   if (PVR_HAS_FEATURE(&(pdevice)->dev_info, apple_g5p))      \
      pvr_g5p_msaa_image_samples((pdevice), (info), (tiling), (props));

/* pvr_arch_cmd_buffer.c, the dynamic render's colour attachments: a 4x
 * AVERAGE resolve of 4-byte colour samples is the PBE's at the end of tile
 * (pvr_g5p_msaa_pbe_resolvable()). Upstream answers with gs_rta_support,
 * which the A9 lacks, and stores the samples for a transfer resolve. */
#define PVR_G5P_HOOK_msaa_pbe_resolve(dev_info, attachment, format, samples, mode) \
   if (PVR_HAS_FEATURE((dev_info), apple_g5p))                                  \
      (attachment)->is_pbe_downscalable = pvr_g5p_msaa_pbe_resolvable((format), (samples), (mode));

#endif /* PVR_G5P_DS_HOOKS_H */
