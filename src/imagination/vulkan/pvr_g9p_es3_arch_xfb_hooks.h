/* SPDX-License-Identifier: MIT */
/* PR1-style hooks: each call consumes one original blank line before the next
 * line-sensitive statement. Definitions capture the call site locals. No #line. */
#ifndef PVR_G9P_ES3_ARCH_XFB_HOOKS_H
#define PVR_G9P_ES3_ARCH_XFB_HOOKS_H
#include "pvr_g9p_es3_hooks.h"
#ifdef PVR_G9P_ES3
#include "pvr_g9p_es3.h"

/* pvr_arch_xfb.c, after main line 813: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_xfb_813() \
   /* G9P: the vertex shader stores the records (pvr_arch_g9p_es3_gfx.h) */ \
   if (PVR_HAS_FEATURE(&cmd_buffer->device->pdevice->dev_info, apple_g9p)) { \
   pvr_arch_g9p_es3_xfb_begin(cmd_buffer, firstCounterBuffer, counterBufferCount, \
   pCounterBuffers, pCounterBufferOffsets); \
   return; \
   }

/* pvr_arch_xfb.c, after main line 857: the option-on insertion. */
#define PVR_G9P_ES3_HOOK_arch_xfb_857() \
   /* \
   * G9P: the counters only; no render split (the G9P background would \
   * clear the second half's targets instead of loading them) \
   */ \
   if (PVR_HAS_FEATURE(&cmd_buffer->device->pdevice->dev_info, apple_g9p)) { \
   pvr_arch_g9p_es3_xfb_end(cmd_buffer, firstCounterBuffer, counterBufferCount, \
   pCounterBuffers, pCounterBufferOffsets); \
   return; \
   }
#endif
#endif
