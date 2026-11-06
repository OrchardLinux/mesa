/* SPDX-License-Identifier: MIT
 * R21 qualification switch. No advertised limit is raised by selecting it.
 */
#ifndef G5P_VTEX_H
#define G5P_VTEX_H
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
static inline bool g5p_vtex_probe(void)
{
   const char *v = getenv("PVR_G5P_VERTEX_TEXTURE");
   return v && !strcmp(v, "probe");
}
/* Staged promotion candidate: exact API rungs have passed, focused CTS is
 * still required. This mode never selects the separate bank0 experiment. */
static inline bool g5p_vtex_qualified(void)
{
   const char *v = getenv("PVR_G5P_VERTEX_TEXTURE");
   return v && !strcmp(v, "qualified");
}
#endif
