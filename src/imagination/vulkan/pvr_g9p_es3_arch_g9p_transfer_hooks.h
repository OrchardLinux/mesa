/* SPDX-License-Identifier: MIT */
/* PR1-style hooks: each call consumes one original blank line before the next
 * line-sensitive statement. Definitions capture the call site locals. No #line. */
#ifndef PVR_G9P_ES3_ARCH_G9P_TRANSFER_HOOKS_H
#define PVR_G9P_ES3_ARCH_G9P_TRANSFER_HOOKS_H
#include "pvr_g9p_es3_hooks.h"
#ifdef PVR_G9P_ES3

/* pvr_arch_g9p_transfer.c, after main line 36: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_transfer_36() \
   struct surface; \
   struct pvr_transfer_cmd_source; \
   /* -Dimagination-g9p-es3, at the end of this file (pvr_arch_g9p_es3_transfer.h) */ \
   static void es3_resolve(struct pvr_device *device, const struct pvr_transfer_cmd *cmd, \
   const struct pvr_transfer_cmd_source *source, const struct surface *dst);

/* pvr_arch_g9p_transfer.c, after main line 235: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_g9p_transfer_235() \
   /* a multisample source: a resolve (vkCmdResolveImage, upstream's transfer resolve) */ \
   if (source->surface.sample_count > 1) { \
   es3_resolve(device, cmd, source, &dst); \
   continue; \
   }
#endif
#endif
