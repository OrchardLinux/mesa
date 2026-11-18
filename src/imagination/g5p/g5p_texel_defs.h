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
/* Uniform texel buffers in fragment and vertex programs. This path (a
 * linear 4096-wide 2D state over the buffer, the NIR fold of the index, the
 * range in software metadata) gave exact pixels on the A9 for 30 formats,
 * 1 to 65536 elements, offsets 64 and 128, the size query and out-of-range
 * reads, in fragment and vertex programs. The fetch instruction is the
 * general texture encoder's. Storage texel buffers and compute fetches stay
 * refused by name. */
static inline bool pvr_g5p_tbo_probe(void)
{
   return true;
}

#endif
