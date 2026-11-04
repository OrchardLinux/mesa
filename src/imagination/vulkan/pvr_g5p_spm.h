/* SPDX-License-Identifier: MIT */
/* G5P SPM allocation experiment. The typed pass has no generic SPM address,
 * but PM exhaustion/firmware-armed partial render is not yet qualified.
 * Retain the allocation by default. Only exact "0" opts into the candidate;
 * production promotion requires the PM-pressure witness and device pixels.
 */
#ifndef PVR_G5P_SPM_H
#define PVR_G5P_SPM_H
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
static inline bool pvr_g5p_spm_skip(void)
{
   const char *s = getenv("PVR_G5P_SPM_SCRATCH");
   return s && !strcmp(s, "0");
}
#endif
