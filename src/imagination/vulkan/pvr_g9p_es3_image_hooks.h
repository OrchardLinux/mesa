/* SPDX-License-Identifier: MIT */
/* PR1-style hooks: each call consumes one original blank line before the next
 * line-sensitive statement. Definitions capture the call site locals. No #line. */
#ifndef PVR_G9P_ES3_IMAGE_HOOKS_H
#define PVR_G9P_ES3_IMAGE_HOOKS_H
#include "pvr_g9p_es3_hooks.h"
#ifdef PVR_G9P_ES3

/* pvr_image.c, after main line 66: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_image_66() \
   /* \
   * -Dimagination-g9p-es3: arrays and cube maps twiddled, the only way \
   * Apple's A9 driver samples them (pvr_arch_g9p_es3_gfx.h) \
   */ \
   else if (PVR_HAS_FEATURE(&device->pdevice->dev_info, apple_g9p) && \
   (image->vk.array_layers > 1 || \
   (image->vk.create_flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT))) \
   image->memlayout = PVR_MEMLAYOUT_TWIDDLED;

/* pvr_image.c, after main line 208: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_image_208() \
   /* \
   * -Dimagination-g9p-es3, G9P: an array's layer stride is the chain's \
   * size rounded to 64 bytes, to 16 KiB past 16 KiB (the A9's rule, \
   * hardware-proven there; the texture state's bit 61 says which) \
   */ \
   { \
   const struct pvr_device *es3_device = \
   container_of(image->vk.base.device, struct pvr_device, vk); \
   if (PVR_HAS_FEATURE(&es3_device->pdevice->dev_info, apple_g9p) && \
   image->vk.array_layers > 1 && image->memlayout == PVR_MEMLAYOUT_TWIDDLED) { \
   plane->layer_size = \
   align64(plane->layer_size, plane->layer_size >= 16384 ? 16384 : 64); \
   plane->size = plane->layer_size * image->vk.array_layers; \
   return; \
   } \
   }
#endif
#endif
