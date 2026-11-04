/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): a render job's resource list, named by what the job
 * touches. Built only with -Dimagination-g5p-experimental=true.
 *
 * Our kernel does the CPU cache maintenance of every resource a job lists
 * with DRM_G5P_RENDER_ACCESS_CACHE (a clean before the job; an invalidate
 * after it for a resource listed WRITE), over the listed range of the BO
 * (render_user_sync_range bit 1, a release8 default). It fences a resource
 * listed IMPLICIT in its dma_resv: a write fence when listed WRITE, a read
 * fence otherwise. So the list decides both the job's CPU cost and whom a
 * compositor or a client waits for.
 *
 * Two narrowings, both exact:
 *  - an arena chunk is listed from the lowest slice the arena has ever
 *    handed out to its end, not whole: slices are taken from the top of the
 *    arena (util_vma_heap prefers high addresses), so everything below that
 *    mark was never a slice and no job can reach it;
 *  - an own BO (a display buffer or an imported dma-buf) is listed WRITE only
 *    when the job writes it: one of its colour targets, its depth/stencil
 *    attachment, or a ZLS address of the pass lies in it. Every other own BO
 *    is listed READ. When the job's writes are not all known (a fragment
 *    stream without the tail below, or transform feedback in the command
 *    buffer), every own BO stays READ|WRITE, as before.
 *
 * The writes travel to the winsys in a tail after the TYPED_V3 pass in the
 * fragment fw_stream (struct g5p_perf_tail): the pass itself is unchanged.
 *
 * PVR_G5P_PERF_RES=0: the old list (every chunk whole, every own BO
 * READ|WRITE), for an A/B in one package.
 * PVR_G5P_PERF_LOG=1: a summary line every 120 jobs (bytes the jobs named
 * for maintenance, before and after the job, and the submit ioctl's time);
 * =2: one line per job. Both count the jobs whose pass has no draw (a stream of
 * under G5P_PERF_DRAWLESS_VDM bytes: the terminate and its PPP block).
 *
 * The rule is the A9 ICD's (Mesa g5p-perf60-hypr 139fe06c167: cache
 * maintenance only over the memory a job names; 09b21fbb143: in whole GPU
 * pages; the read-only rule of kernel a9/shmsync e9a53f9f59), on the
 * pvr_* winsys. docs/gpu/orchard-port-perf.md, slice 1.
 */

#ifndef G5P_WS_PERF_H
#define G5P_WS_PERF_H

#include <stdbool.h>
#include <stdint.h>

#include "apple_g5p_drm.h"

#define G5P_PERF_TAIL_MAGIC 0x31667270u /* "prf1" */
#define G5P_PERF_MAX_WRITES 11
/* every write of the job to memory outside the arena is in write_va */
#define G5P_PERF_WRITES_KNOWN (1u << 0)
/* no attachment is cleared (every load is LOAD or DONT_CARE): a draw-less job of such a pass does nothing (logged) */
#define G5P_PERF_NO_CLEAR (1u << 1)

/* After the pass in the fragment fw_stream: the base addresses of the images
 * the job writes (any address inside a BO marks that BO written) */
struct g5p_perf_tail {
   uint32_t magic;
   uint16_t flags;  /* G5P_PERF_* */
   uint16_t writes; /* entries of write_va */
   uint64_t write_va[G5P_PERF_MAX_WRITES];
};

/* What the winsys keeps (one per winsys) */
struct g5p_ws_perf {
   uint64_t arena_va, arena_end;
   uint64_t arena_lo; /* the lowest slice address handed out so far; arena_end before the first */
   bool on;           /* PVR_G5P_PERF_RES != 0 */
   unsigned log;      /* PVR_G5P_PERF_LOG */
   /* PVR_G5P_PERF_LOG: totals since the last summary line */
   uint64_t jobs, pre_bytes, post_bytes, own_listed, own_written, submit_ns, writes_known, drawless, drawless_noclear;
   bool last_known;    /* the last job's writes were known (the per-job line) */
   bool last_noclear;  /* the last job clears no attachment (G5P_PERF_NO_CLEAR) */
   uint32_t last_vdm;  /* the last job's stream bytes: a pass without a draw is its terminate block alone */
};

/* One job: what its tail and its pass say it writes */
struct g5p_ws_perf_job {
   bool writes_known;
   unsigned n;
   uint64_t va[G5P_PERF_MAX_WRITES + DRM_G5P_RENDER_ZLS_REGS];
};

void g5p_ws_perf_init(struct g5p_ws_perf *p, uint64_t arena_va, uint64_t arena_size);
/* a slice was taken at va (any thread) */
void g5p_ws_perf_slice(struct g5p_ws_perf *p, uint64_t va);
/* the job's writes, from the bytes after the pass in the fragment stream */
void g5p_ws_perf_job(struct g5p_ws_perf *p, const struct drm_g5p_render_pass_v3 *pass, const void *tail,
                     uint32_t tail_len, struct g5p_ws_perf_job *j);
/* the part of an arena chunk [va, va + size) a job lists: false when none */
bool g5p_ws_perf_chunk(const struct g5p_ws_perf *p, uint64_t va, uint64_t size, uint64_t *offset,
                       uint64_t *length);
/* the access kind of an own BO [va, va + size) in this job */
uint32_t g5p_ws_perf_own_kind(const struct g5p_ws_perf *p, const struct g5p_ws_perf_job *j, uint64_t va,
                              uint64_t size, bool shared);
uint64_t g5p_ws_perf_now(void);
/* after the submit ioctl that started at t0 (g5p_ws_perf_now()) */
void g5p_ws_perf_account(struct g5p_ws_perf *p, const struct drm_g5p_render_resource *res, unsigned n,
                         unsigned chunks, uint64_t t0);

#endif /* G5P_WS_PERF_H */
