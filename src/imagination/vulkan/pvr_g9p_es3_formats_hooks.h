/* SPDX-License-Identifier: MIT */
/* PR1-style hooks: each call consumes one original blank line before the next
 * line-sensitive statement. Definitions capture the call site locals. No #line. */
#ifndef PVR_G9P_ES3_FORMATS_HOOKS_H
#define PVR_G9P_ES3_FORMATS_HOOKS_H
#include "pvr_g9p_es3_hooks.h"
#ifdef PVR_G9P_ES3
#include "pvr_g9p_es3.h"

/* pvr_formats.c, after main line 763: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_formats_763() \
   if (PVR_HAS_FEATURE(&pdevice->dev_info, apple_g9p)) \
   pImageFormatProperties->sampleCounts &= pvr_g9p_es3_sample_counts();
#endif
#endif
