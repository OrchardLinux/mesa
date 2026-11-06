/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 * Shared software fold contract; no captured bytes.
 */
#ifndef G5P_TEXEL_DEFS_H
#define G5P_TEXEL_DEFS_H
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#define PVR_G5P_TBO_WIDTH 4096u
#define PVR_G5P_TBO_SHIFT 12u
#define PVR_G5P_TBO_ALIGN 64u
#define PVR_G5P_TBO_LIMIT 65536u
#define PVR_G5P_TBO_MAGIC 0x54424f31u /* software metadata, never a GPU field */
#define PVR_G5P_TBO_NIR_FOLDED (1u << 31)
static inline bool pvr_g5p_tbo_probe(void)
{
   const char *s = getenv("PVR_G5P_TEXEL_BUFFER");
   return s && !strcmp(s, "probe");
}

#endif
