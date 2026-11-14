/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): the desktop speed work (docs/gpu/orchard-port-perf.md).
 * Slice 1: what a render job writes, for the winsys's resource list
 * (winsys/apple_g5p/g5p_ws_perf.h). Built only with
 * -Dimagination-g5p-experimental=true.
 */

#ifndef PVR_G5P_PERF_H
#define PVR_G5P_PERF_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "pvr_macros.h"
#include "winsys/apple_g5p/g5p_ws_perf.h"

/* What a pass's draws read outside the arena (pvr_arch_g5p_perf_draw()),
 * kept in the job until job init uploads it for the winsys */
struct pvr_g5p_perf_reads {
   uint32_t count;
   bool unknown; /* a read that isn't decoded: every own BO is listed */
   uint64_t va[G5P_PERF_MAX_READS];
   /* BO ranges, not individual suballocation addresses. Draws often read
    * hundreds of offsets in the same application memory allocation. */
   uint64_t size[G5P_PERF_MAX_READS];
};

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
 * depth/stencil attachment) and the draws' reads (an upload) into the job's
 * tail */
void PVR_PER_ARCH(g5p_perf_job)(struct pvr_cmd_buffer *cmd_buffer,
                                const struct pvr_renderpass_hwsetup_render *hw_render,
                                const struct pvr_g5p_perf_reads *reads, struct g5p_perf_tail *tail);
/* each G5P draw, once its graphics sub-command exists: what it can read
 * outside the arena, into the sub-command's job */
void PVR_PER_ARCH(g5p_perf_draw)(struct pvr_cmd_buffer *cmd_buffer, bool indexed);

#   define pvr_arch_g5p_perf_job PVR_PER_ARCH(g5p_perf_job)
#   define pvr_arch_g5p_perf_draw PVR_PER_ARCH(g5p_perf_draw)

#endif /* PVR_PER_ARCH */

#endif /* PVR_G5P_PERF_H */
