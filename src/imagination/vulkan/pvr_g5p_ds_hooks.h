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

#endif /* PVR_G5P_DS_HOOKS_H */
