/* SPDX-License-Identifier: MIT */
/* PR1-style hooks: each call consumes one original blank line before the next
 * line-sensitive statement. Definitions capture the call site locals. No #line. */
#ifndef PVR_G9P_ES3_PHYSICAL_DEVICE_HOOKS_H
#define PVR_G9P_ES3_PHYSICAL_DEVICE_HOOKS_H
#include "pvr_g9p_es3_hooks.h"
#ifdef PVR_G9P_ES3
#include "pvr_g9p_es3.h"

/* pvr_physical_device.c, after main line 1301: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_physical_device_1301() \
   /* G9P, -Dimagination-g9p-es3: what its paths implement (pvr_g9p_es3.c) */ \
   if (PVR_HAS_FEATURE(&pdevice->dev_info, apple_g9p)) \
   pvr_g9p_es3_limits(&supported_extensions, &supported_features, &supported_properties);
#endif
#endif
