/* SPDX-License-Identifier: MIT */
/* Typed G5P never serializes the generic framebuffer SPM address or programs.
 * Omit unreachable scratch by default; exact "1" restores it for comparison.
 * This does not add PM growth/partial-render support or fix the heavy-scene
 * stall. Audit: linux-a9 perf-slice2-ram3.md, address-path conclusion.
 */
#ifndef PVR_G5P_SPM_H
#define PVR_G5P_SPM_H
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
static inline bool pvr_g5p_spm_skip(void)
{
   const char *s = getenv("PVR_G5P_SPM_SCRATCH");
   return !s || !strcmp(s, "0");
}
#endif
