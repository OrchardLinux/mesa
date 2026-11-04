/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): the desktop speed work (docs/gpu/orchard-port-perf.md).
 * Slice 1: what a render job writes, for the winsys's resource list
 * (winsys/apple_g5p/g5p_ws_perf.h). Built only with
 * -Dimagination-g5p-experimental=true.
 */

#ifndef PVR_G5P_PERF_H
#define PVR_G5P_PERF_H

#include <stdint.h>
#include <string.h>

#include "pvr_macros.h"
#include "winsys/apple_g5p/g5p_ws_perf.h"

/* The tail the fragment stream carries after the pass: its size, or 0 when
 * it does not fit (the winsys then lists every own BO READ|WRITE) */
static inline uint32_t
pvr_g5p_perf_stream_tail(const struct g5p_perf_tail *t, uint8_t *buf, uint32_t room)
{
   if (t->magic != G5P_PERF_TAIL_MAGIC || room < sizeof(*t))
      return 0;
   memcpy(buf, t, sizeof(*t));
   return sizeof(*t);
}

#ifdef PVR_PER_ARCH

struct pvr_cmd_buffer;
struct pvr_renderpass_hwsetup_render;

/* job_init: the images the render writes (its colour targets and its
 * depth/stencil attachment) into the job's tail */
void PVR_PER_ARCH(g5p_perf_job)(const struct pvr_cmd_buffer *cmd_buffer,
                                const struct pvr_renderpass_hwsetup_render *hw_render,
                                struct g5p_perf_tail *tail);

#   define pvr_arch_g5p_perf_job PVR_PER_ARCH(g5p_perf_job)

#endif /* PVR_PER_ARCH */

#endif /* PVR_G5P_PERF_H */
