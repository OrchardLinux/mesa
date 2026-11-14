/* SPDX-License-Identifier: MIT
 * What of vertex texturing is on. No advertised limit depends on it.
 */
#ifndef G5P_VTEX_H
#define G5P_VTEX_H
#include <stdbool.h>
/* The vertex-texture recipe that passed on the device (focused R16 CTS: 283
 * exact passes, no wrong results, crash or hang): non-cube and cube sampling
 * with fetch and sample offsets. The forms below it have not run there and
 * stay refused by name. */
static inline bool g5p_vtex_probe(void) { return false; }
static inline bool g5p_vtex_qualified(void) { return true; }
static inline bool g5p_vtex_cube_probe(void) { return true; }
static inline bool g5p_vtex_fetch_offset_probe(void) { return true; }
static inline bool g5p_vtex_gradient_probe(void) { return false; }
static inline bool g5p_vtex_integer_probe(void) { return false; }
static inline bool g5p_vtex_xyz_offset_probe(void) { return false; }
static inline bool g5p_vtex_size_lod_probe(void) { return false; }
static inline bool g5p_vtex_shadow_probe(void) { return false; }
#endif
