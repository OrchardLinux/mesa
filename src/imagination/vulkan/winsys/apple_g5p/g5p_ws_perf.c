/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): a render job's resource list, named by what the job
 * touches (g5p_ws_perf.h). Built only with -Dimagination-g5p-experimental=true.
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "util/macros.h"
#include "util/u_atomic.h"

#include "g5p_ws_perf.h"

/* The kernel's GPU page (GET_INFO.gpu_page_shift on the A9: 16 KiB); a listed
 * range is widened to whole pages by the kernel, and slices are page-aligned */
#define G5P_PERF_PAGE 16384ull
#define G5P_PERF_SUMMARY_JOBS 120
/* a stream without a draw: STREAM_TERMINATE and the PPP state before it (12 bytes on g5p-1b); one draw is ~100 */
#define G5P_PERF_DRAWLESS_VDM 32

static unsigned
env_uint(const char *name, unsigned dflt)
{
   const char *s = getenv(name);
   return s && *s ? (unsigned)strtoul(s, NULL, 0) : dflt;
}

void
g5p_ws_perf_init(struct g5p_ws_perf *p, uint64_t arena_va, uint64_t arena_size, const void *arena_cpu)
{
   memset(p, 0, sizeof(*p));
   p->arena_va = arena_va;
   p->arena_cpu = arena_cpu;
   p->arena_end = arena_va + arena_size;
   p->arena_lo = p->arena_end;
   p->on = env_uint("PVR_G5P_PERF_RES", 1) != 0;
   p->log = env_uint("PVR_G5P_PERF_LOG", 0);
   p->reads_on = env_uint("PVR_G5P_PERF_READS", 1) != 0;
}

void
g5p_ws_perf_slice(struct g5p_ws_perf *p, uint64_t va)
{
   uint64_t lo = p_atomic_read(&p->arena_lo);
   while (va < lo) {
      const uint64_t seen = p_atomic_cmpxchg(&p->arena_lo, lo, va);
      if (seen == lo)
         break;
      lo = seen;
   }
}

void
g5p_ws_perf_job(struct g5p_ws_perf *p, const struct drm_g5p_render_pass_v3 *pass, const void *tail,
                uint32_t tail_len, struct g5p_ws_perf_job *j)
{
   struct g5p_perf_tail t;

   memset(j, 0, sizeof(*j));
   p->last_known = false;
   p->last_vdm = pass->base.vdm_size;
   p->last_noclear = false;
   p->last_reads = false;
   p->last_omitted = 0;
   if (!p->on || !tail || tail_len != sizeof(t))
      return;
   memcpy(&t, tail, sizeof(t));
   if (t.magic != G5P_PERF_TAIL_MAGIC || !(t.flags & G5P_PERF_WRITES_KNOWN) || t.writes > G5P_PERF_MAX_WRITES)
      return;
   p->last_noclear = t.flags & G5P_PERF_NO_CLEAR;
   for (unsigned i = 0; i < t.writes; i++)
      j->va[j->n++] = t.write_va[i];
   /* the ZLS words that are addresses (the kernel requires them in a listed READ|WRITE range): the
    * depth/stencil loads and stores */
   for (unsigned i = 0; i < DRM_G5P_RENDER_ZLS_REGS; i++)
      if (pass->ext.zls_address_mask & (1u << i))
         j->va[j->n++] = pass->ext.zls[i];
   j->writes_known = true;
   p->last_known = true;

   /* the reads: t.reads addresses at reads_va, an upload of the command buffer in the arena */
   if (!p->reads_on || !(t.flags & G5P_PERF_READS_KNOWN) || t.reads > G5P_PERF_MAX_READS)
      return;
   if (t.reads) {
      if (t.reads_va < p->arena_va || t.reads_va > p->arena_end - 8ull * t.reads || (t.reads_va & 7))
         return;
      memcpy(j->read_va, p->arena_cpu + (t.reads_va - p->arena_va), 8ull * t.reads);
      j->nr = t.reads;
   }
   j->reads_known = true;
   p->last_reads = true;
}

bool
g5p_ws_perf_chunk(const struct g5p_ws_perf *p, uint64_t va, uint64_t size, uint64_t *offset, uint64_t *length)
{
   uint64_t lo = va, hi = va + size;

   if (p->on) {
      /* [arena_lo, arena_end) holds every slice ever handed out; a chunk below it was never sliced */
      const uint64_t mark = p_atomic_read((uint64_t *)&p->arena_lo) & ~(G5P_PERF_PAGE - 1);
      lo = MAX2(lo, mark);
      hi = MIN2(hi, p->arena_end);
   }
   if (lo >= hi)
      return false;
   *offset = lo - va;
   *length = hi - lo;
   return true;
}

uint32_t
g5p_ws_perf_own_kind(struct g5p_ws_perf *p, const struct g5p_ws_perf_job *j, uint64_t va, uint64_t size, bool shared)
{
   uint32_t kind = DRM_G5P_RENDER_ACCESS_READ | DRM_G5P_RENDER_ACCESS_CACHE |
                   (shared ? DRM_G5P_RENDER_ACCESS_IMPLICIT : 0);
   bool written = !p->on || !j->writes_known;

   for (unsigned i = 0; i < j->n && !written; i++)
      written = j->va[i] >= va && j->va[i] - va < size;
   if (written)
      return kind | DRM_G5P_RENDER_ACCESS_WRITE;
   if (!j->reads_known)
      return kind;
   for (unsigned i = 0; i < j->nr; i++)
      if (j->read_va[i] >= va && j->read_va[i] - va < size)
         return kind;
   p->last_omitted++;
   return 0;
}

uint64_t
g5p_ws_perf_now(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

void
g5p_ws_perf_account(struct g5p_ws_perf *p, const struct drm_g5p_render_resource *res, unsigned n, unsigned chunks,
                    uint64_t t0)
{
   if (!p->log)
      return;
   const uint64_t ns = g5p_ws_perf_now() - t0;
   uint64_t pre = 0, post = 0;
   unsigned own_w = 0;

   for (unsigned i = 0; i < n; i++) {
      if (!(res[i].kind & DRM_G5P_RENDER_ACCESS_CACHE))
         continue;
      pre += res[i].size;
      if (res[i].kind & DRM_G5P_RENDER_ACCESS_WRITE) {
         post += res[i].size;
         own_w += i >= chunks;
      }
   }
   if (p->log >= 2)
      fprintf(stderr,
              "G5P perf job res=%u chunks=%u own=%u own_written=%u omitted=%u writes_known=%d reads_known=%d vdm=%u pre=%" PRIu64 " post=%" PRIu64
              " arena_lo=+%#" PRIx64 " submit_us=%" PRIu64 "\n",
              n, chunks, n - chunks, own_w, p->last_omitted, p->last_known, p->last_reads, p->last_vdm, pre, post, p->arena_lo - p->arena_va, ns / 1000);

   /* the summary: totals over the last G5P_PERF_SUMMARY_JOBS jobs (approximate under concurrent submits) */
   p_atomic_add(&p->pre_bytes, pre);
   p_atomic_add(&p->post_bytes, post);
   p_atomic_add(&p->own_listed, n - chunks);
   p_atomic_add(&p->own_written, own_w);
   p_atomic_add(&p->submit_ns, ns);
   if (p->last_known)
      p_atomic_inc(&p->writes_known);
   if (p->last_reads)
      p_atomic_inc(&p->reads_known);
   p_atomic_add(&p->own_omitted, p->last_omitted);
   if (p->last_vdm < G5P_PERF_DRAWLESS_VDM) {
      p_atomic_inc(&p->drawless);
      if (p->last_noclear)
         p_atomic_inc(&p->drawless_noclear);
   }
   const uint64_t jobs = p_atomic_inc_return(&p->jobs);
   if (jobs % G5P_PERF_SUMMARY_JOBS)
      return;
   const uint64_t pb = p_atomic_xchg(&p->pre_bytes, 0), qb = p_atomic_xchg(&p->post_bytes, 0);
   const uint64_t ol = p_atomic_xchg(&p->own_listed, 0), ow = p_atomic_xchg(&p->own_written, 0);
   const uint64_t sn = p_atomic_xchg(&p->submit_ns, 0), wk = p_atomic_xchg(&p->writes_known, 0),
                  dl = p_atomic_xchg(&p->drawless, 0), dn = p_atomic_xchg(&p->drawless_noclear, 0),
                  rk = p_atomic_xchg(&p->reads_known, 0), om = p_atomic_xchg(&p->own_omitted, 0);
   fprintf(stderr,
           "G5P perf summary jobs=%" PRIu64 " res=%s per_job: pre_kib=%" PRIu64 " post_kib=%" PRIu64
           " own=%.1f own_written=%.1f omitted=%.1f writes_known=%" PRIu64 "/%u reads_known=%" PRIu64 " drawless=%" PRIu64 " (no clear %" PRIu64 ") submit_us=%" PRIu64 " arena_used_kib=%" PRIu64 "\n",
           jobs, p->on ? "narrow" : "whole", pb / 1024 / G5P_PERF_SUMMARY_JOBS, qb / 1024 / G5P_PERF_SUMMARY_JOBS,
           (double)ol / G5P_PERF_SUMMARY_JOBS, (double)ow / G5P_PERF_SUMMARY_JOBS, (double)om / G5P_PERF_SUMMARY_JOBS, wk,
           G5P_PERF_SUMMARY_JOBS, rk, dl, dn,
           sn / 1000 / G5P_PERF_SUMMARY_JOBS, (p->arena_end - p->arena_lo) / 1024);
}
