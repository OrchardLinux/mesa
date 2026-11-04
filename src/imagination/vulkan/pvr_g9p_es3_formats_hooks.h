/* SPDX-License-Identifier: MIT */
/* PR1-style hooks: each call consumes one original blank line before the next
 * line-sensitive statement. Definitions capture the call site locals. No #line. */
#ifndef PVR_G9P_ES3_FORMATS_HOOKS_H
#define PVR_G9P_ES3_FORMATS_HOOKS_H
#include "pvr_g9p_es3_hooks.h"
#ifdef PVR_G9P_ES3
#include "pvr_g9p_es3.h"

/* pvr_formats.c, after main line 366: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_formats_366() \
   const VkFormatFeatureFlags2 es3_ds_flags = flags;

/* pvr_formats.c, after main line 371: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_formats_371() \
   /* -Dimagination-g9p-es3: D16, S8 and D32F_S8 attachments too (pvr_arch_g9p_es3_gfx.h) */ \
   if (PVR_HAS_FEATURE(&pdevice->dev_info, apple_g9p)) \
   flags |= es3_ds_flags & pvr_g9p_es3_ds_features(vk_format);

/* pvr_formats.c, after main line 763: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_formats_763() \
   if (PVR_HAS_FEATURE(&pdevice->dev_info, apple_g9p)) \
   pImageFormatProperties->sampleCounts &= pvr_g9p_es3_sample_counts(info->usage);
#endif
#endif
