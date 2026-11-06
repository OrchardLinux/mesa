/* SPDX-License-Identifier: MIT */
/*
 * Apple GPUs (G9P): the CPU side of transform feedback, without Vulkan. From
 * PR #6's A9 port (pvr_g5p_xfb_core.h).
 *
 * There is no stream-out the driver can program like Rogue's: the vertex
 * shader's XFB variant stores every captured record itself (the backend's
 * nir_lower_xfb_to_stores forms, 4-byte vertex stores), at
 *
 *    block.addr[b] + (instance * block.num_vertices + vertex index) * stride + offset
 *
 * with the hardware vertex index (the draw's first vertex + i). Everything
 * else is CPU work at execution time, in command order, with lavapipe's
 * semantics:
 *  - BEGIN reads the counter buffers (bytes written from the binding offset;
 *    0 without one), END writes them;
 *  - each captured draw gets its record addresses in its per-draw block, with
 *    the first vertex folded in, so the shader subtracts nothing;
 *  - a draw whose primitives don't all fit (the buffer with the least room
 *    decides, whole primitives only), or whose last primitive is incomplete,
 *    stores into staging, and the CPU copies the primitives that fit after
 *    the job, record ranges only, so stride padding stays untouched;
 *  - a byte-count draw (vkCmdDrawIndirectByteCountEXT) reads its counter and
 *    its vertex count is patched into the control stream;
 *  - stream queries count the primitives written and needed while capturing.
 * A vertex shaded twice rewrites the same bytes: address and data depend on
 * the vertex alone, so nothing here depends on GPU-side counters.
 *
 * Plain C over CPU-visible ranges ({va, map, size}).
 */

#ifndef PVR_G9P_XFB_CORE_H
#define PVR_G9P_XFB_CORE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define G9P_XFB_BUFFERS 4
#define G9P_XFB_MAX_RANGES 32
/* Query result flags: VkQueryResultFlagBits' values */
#define G9P_XFB_RESULT_64 0x1u
#define G9P_XFB_RESULT_AVAILABILITY 0x4u
#define G9P_XFB_RESULT_PARTIAL 0x8u

/* What a capturing vertex shader writes (from its nir_xfb_info) */
struct g9p_xfb_layout {
   uint8_t buffers;    /* buffers written */
   uint16_t stride[G9P_XFB_BUFFERS]; /* bytes per vertex record */
   /* common-store slots in the draw's block (8 bytes each): each buffer's
    * address, the vertices per instance, the offset mask */
   uint8_t slot[G9P_XFB_BUFFERS], num_vertices_slot, mask_slot;
   uint8_t nranges;
   struct {
      uint8_t buffer;
      uint16_t offset, size; /* bytes written per record */
   } ranges[G9P_XFB_MAX_RANGES];
};

/* A CPU-visible GPU range: a whole buffer */
struct g9p_xfb_mem {
   uint64_t va;
   uint8_t *map;
   uint64_t size;
};

struct g9p_xfb_ref {
   const struct g9p_xfb_mem *mem; /* NULL: none */
   uint64_t offset;
};

/* A stream query pool: per query, primitives written and needed (two
 * uint64_t at data + q * stride, stride >= 16) and its availability
 * (nonzero: available) */
struct g9p_xfb_query_pool {
   uint32_t count, stride;
   uint8_t *data;
   uint32_t *available;
};

static inline uint64_t *
g9p_xfb_query_data(const struct g9p_xfb_query_pool *p, uint32_t q)
{
   return (uint64_t *)(p->data + (uint64_t)q * p->stride);
}

enum g9p_xfb_kind {
   G9P_XFB_BEGIN,
   G9P_XFB_END,
   G9P_XFB_DRAW,
   G9P_XFB_QUERY_BEGIN,
   G9P_XFB_QUERY_END,
   G9P_XFB_QUERY_RESET,
   G9P_XFB_QUERY_COPY,
};

struct g9p_xfb_event {
   enum g9p_xfb_kind kind;
   union {
      struct {
         struct g9p_xfb_ref bind[G9P_XFB_BUFFERS]; /* the bindings at Begin */
         uint64_t size[G9P_XFB_BUFFERS];
         struct g9p_xfb_ref counter[G9P_XFB_BUFFERS]; /* resume offsets; none = 0 */
      } begin;
      struct g9p_xfb_ref end[G9P_XFB_BUFFERS]; /* counters written; none = not written */
      struct {
         uint8_t *block;   /* the per-draw block (CPU address), NULL: not captured */
         uint32_t *count;  /* the control stream's vertex count (byte-count draws) */
         uint32_t vertices, instances, first, vpp;
         bool byte_count;
         struct g9p_xfb_ref counter;
         uint32_t counter_bias, vertex_stride;
         struct g9p_xfb_layout layout;
      } draw;
      struct {
         struct g9p_xfb_query_pool *pool;
         uint32_t first, count;
         struct g9p_xfb_ref dst; /* QUERY_COPY */
         uint64_t stride;
         uint32_t flags;
      } query;
   };
};

/* A staged draw's buffer waits for the staging address: where it goes,
 * offset in staging, folded first vertex */
struct g9p_xfb_stage {
   uint8_t *at;
   uint64_t rel, fold;
};

/* Staging holds hw records per instance (the shader's instance * hw + i);
 * the stream takes the first `complete` of each instance, instance-major,
 * until `records` are copied. */
struct g9p_xfb_copy {
   struct g9p_xfb_ref dst; /* the binding's buffer at its written position */
   uint64_t src;           /* bytes from the staging start */
   uint64_t records;
   uint32_t hw, complete;
   unsigned b;
   struct g9p_xfb_layout layout;
};

struct g9p_xfb_exec {
   bool active;
   struct {
      struct g9p_xfb_ref ref; /* the binding: buffer and offset */
      uint64_t size, written; /* bytes from the binding offset */
   } buf[G9P_XFB_BUFFERS];
   struct g9p_xfb_query_pool *pool; /* the active stream query */
   uint32_t query;
   struct g9p_xfb_stage *stage;
   unsigned nstage, cap_stage;
   struct g9p_xfb_copy *copies;
   unsigned ncopies, cap_copies;
   uint64_t staging_bytes; /* what this job's staged draws need */
};

static inline void
g9p_xfb_exec_init(struct g9p_xfb_exec *x)
{
   memset(x, 0, sizeof(*x));
}

static inline void
g9p_xfb_exec_finish(struct g9p_xfb_exec *x)
{
   free(x->stage);
   free(x->copies);
   memset(x, 0, sizeof(*x));
}

static inline void
g9p_xfb_put32(uint8_t *p, uint32_t v)
{
   memcpy(p, &v, 4);
}

static inline void
g9p_xfb_put64(uint8_t *p, uint64_t v)
{
   memcpy(p, &v, 8);
}

static inline bool
g9p_xfb_grow(void **arr, unsigned *cap, unsigned n, size_t elem)
{
   if (n < *cap)
      return true;
   unsigned c = *cap ? 2 * *cap : 8;
   void *p = realloc(*arr, c * elem);
   if (!p)
      return false;
   *arr = p;
   *cap = c;
   return true;
}

/* A counter: 4 bytes inside its buffer */
static inline uint8_t *
g9p_xfb_at(struct g9p_xfb_ref r, uint64_t bytes)
{
   if (!r.mem || !r.mem->map || r.offset + bytes > r.mem->size)
      return NULL;
   return r.mem->map + r.offset;
}

/* Values in Vulkan's order: primitives written, primitives needed (then
 * availability). Returns whether every query was available. */
static inline bool
g9p_xfb_query_results(const struct g9p_xfb_query_pool *p, uint32_t first, uint32_t count, uint8_t *dst,
                      uint64_t stride, uint32_t flags)
{
   bool all = true;
   const bool wide = flags & G9P_XFB_RESULT_64;
   for (uint32_t i = 0; i < count; i++, dst += stride) {
      const uint32_t q = first + i;
      const bool avail = q < p->count && p->available[q];
      all &= avail;
      uint64_t v[3];
      v[0] = avail ? g9p_xfb_query_data(p, q)[0] : 0;
      v[1] = avail ? g9p_xfb_query_data(p, q)[1] : 0;
      v[2] = avail;
      const unsigned n = avail || (flags & G9P_XFB_RESULT_PARTIAL) ? 2 : 0;
      for (unsigned k = 0; k < n; k++) {
         if (wide) {
            memcpy(dst + 8 * k, &v[k], 8);
         } else {
            const uint32_t w = v[k] > UINT32_MAX ? UINT32_MAX : (uint32_t)v[k];
            memcpy(dst + 4 * k, &w, 4);
         }
      }
      if (flags & G9P_XFB_RESULT_AVAILABILITY) {
         if (wide) {
            memcpy(dst + 16, &v[2], 8);
         } else {
            const uint32_t w = avail;
            memcpy(dst + 8, &w, 4);
         }
      }
   }
   return all;
}

static inline void
g9p_xfb_reset_queries(struct g9p_xfb_query_pool *p, uint32_t first, uint32_t count)
{
   for (uint32_t q = first; q < first + count && q < p->count; q++) {
      g9p_xfb_query_data(p, q)[0] = g9p_xfb_query_data(p, q)[1] = 0;
      p->available[q] = 0;
   }
}

/* One captured or byte-count draw: vertex count, fit, record addresses,
 * counters and query. Staged buffers get their addresses once the staging
 * range exists (g9p_xfb_bind_staging). Returns NULL or a refusal. */
static inline const char *
g9p_xfb_resolve_draw(struct g9p_xfb_exec *x, const struct g9p_xfb_event *e)
{
   uint32_t vertices = e->draw.vertices;
   if (e->draw.byte_count) {
      const uint8_t *c = g9p_xfb_at(e->draw.counter, 4);
      uint32_t bytes;
      if (!c)
         return "transform feedback counter outside its buffer";
      memcpy(&bytes, c, 4);
      vertices = e->draw.vertex_stride && bytes > e->draw.counter_bias ?
                 (bytes - e->draw.counter_bias) / e->draw.vertex_stride : 0;
   }
   /* The VS runs for hw vertices per instance; a zero count becomes one
    * vertex, which forms no primitive in any accepted topology. */
   const uint32_t hw = vertices > 1 ? vertices : 1;
   if (e->draw.byte_count && e->draw.count)
      *e->draw.count = hw;
   if (!e->draw.block)
      return NULL; /* not captured */

   const struct g9p_xfb_layout *l = &e->draw.layout;
   const unsigned vpp = e->draw.vpp;
   const uint64_t prims = x->active && vpp ? (uint64_t)(vertices / vpp) * e->draw.instances : 0;
   uint64_t fit = prims;
   for (unsigned b = 0; b < G9P_XFB_BUFFERS; b++) {
      if (!(l->buffers & (1u << b)) || !x->buf[b].ref.mem)
         continue;
      const uint64_t room = x->buf[b].size > x->buf[b].written ? x->buf[b].size - x->buf[b].written : 0;
      const uint64_t per = (uint64_t)vpp * l->stride[b];
      if (per && room / per < fit)
         fit = room / per;
   }
   /* The shader stores hw * instances records: direct only when they are
    * exactly the primitives that fit. */
   const bool direct = x->active && fit == prims && (uint64_t)hw * e->draw.instances == prims * vpp;
   for (unsigned b = 0; b < G9P_XFB_BUFFERS; b++) {
      if (!(l->buffers & (1u << b)))
         continue;
      const uint64_t stride = l->stride[b];
      /* 32-bit record offsets: (instance * hw + first + i) * stride + offset */
      if (((uint64_t)e->draw.instances * hw + e->draw.first + 1) * stride >= (1ull << 32))
         return "transform feedback offsets beyond 4 GiB";
      const uint64_t fold = (uint64_t)e->draw.first * stride;
      const struct g9p_xfb_mem *mem = x->active ? x->buf[b].ref.mem : NULL;
      if (direct && mem) {
         g9p_xfb_put64(e->draw.block + 8 * l->slot[b],
                       mem->va + x->buf[b].ref.offset + x->buf[b].written - fold);
      } else {
         if (!g9p_xfb_grow((void **)&x->stage, &x->cap_stage, x->nstage, sizeof(*x->stage)))
            return "out of host memory";
         x->stage[x->nstage++] = (struct g9p_xfb_stage){ e->draw.block + 8 * l->slot[b], x->staging_bytes, fold };
         if (mem && fit) {
            if (!g9p_xfb_grow((void **)&x->copies, &x->cap_copies, x->ncopies, sizeof(*x->copies)))
               return "out of host memory";
            x->copies[x->ncopies++] = (struct g9p_xfb_copy){
               .dst = { mem, x->buf[b].ref.offset + x->buf[b].written },
               .src = x->staging_bytes,
               .records = fit * vpp,
               .hw = hw,
               .complete = vertices / vpp * vpp,
               .b = b,
               .layout = *l,
            };
         }
         x->staging_bytes += ((uint64_t)hw * e->draw.instances * stride + 63) & ~63ull;
      }
      if (mem)
         x->buf[b].written += fit * vpp * stride;
   }
   g9p_xfb_put32(e->draw.block + 8 * l->num_vertices_slot, hw);
   if (x->pool && x->query < x->pool->count) {
      g9p_xfb_query_data(x->pool, x->query)[0] += fit;
      g9p_xfb_query_data(x->pool, x->query)[1] += prims;
   }
   return NULL;
}

/* Every event but DRAW */
static inline const char *
g9p_xfb_execute(struct g9p_xfb_exec *x, const struct g9p_xfb_event *e)
{
   switch (e->kind) {
   case G9P_XFB_BEGIN:
      x->active = true;
      for (unsigned b = 0; b < G9P_XFB_BUFFERS; b++) {
         x->buf[b].ref = e->begin.bind[b];
         x->buf[b].size = e->begin.size[b];
         x->buf[b].written = 0;
         if (e->begin.counter[b].mem && x->buf[b].ref.mem) {
            const uint8_t *c = g9p_xfb_at(e->begin.counter[b], 4);
            uint32_t v;
            if (!c)
               return "transform feedback counter outside its buffer";
            memcpy(&v, c, 4);
            x->buf[b].written = v;
         }
      }
      return NULL;
   case G9P_XFB_END:
      /* the counters hold the bytes written from the binding offset */
      for (unsigned b = 0; b < G9P_XFB_BUFFERS; b++) {
         if (!e->end[b].mem)
            continue;
         uint8_t *c = g9p_xfb_at(e->end[b], 4);
         if (!c)
            return "transform feedback counter outside its buffer";
         g9p_xfb_put32(c, x->buf[b].written > UINT32_MAX ? UINT32_MAX : (uint32_t)x->buf[b].written);
      }
      x->active = false;
      return NULL;
   case G9P_XFB_DRAW:
      return g9p_xfb_resolve_draw(x, e);
   case G9P_XFB_QUERY_RESET:
      g9p_xfb_reset_queries(e->query.pool, e->query.first, e->query.count);
      return NULL;
   case G9P_XFB_QUERY_BEGIN:
      g9p_xfb_reset_queries(e->query.pool, e->query.first, 1);
      x->pool = e->query.pool;
      x->query = e->query.first;
      return NULL;
   case G9P_XFB_QUERY_END:
      if (e->query.first < e->query.pool->count)
         e->query.pool->available[e->query.first] = 1;
      x->pool = NULL;
      return NULL;
   case G9P_XFB_QUERY_COPY: {
      const unsigned values = 2 + !!(e->query.flags & G9P_XFB_RESULT_AVAILABILITY);
      const uint64_t need = e->query.count ?
         (e->query.count - 1) * e->query.stride + values * ((e->query.flags & G9P_XFB_RESULT_64) ? 8 : 4) : 0;
      uint8_t *dst = g9p_xfb_at(e->query.dst, need);
      if (!dst)
         return "query results outside the buffer";
      g9p_xfb_query_results(e->query.pool, e->query.first, e->query.count, dst, e->query.stride, e->query.flags);
      return NULL;
   }
   }
   return "unknown transform feedback event";
}

/* A job's events, in order: draws are resolved, the rest executed.
 * Afterwards x->staging_bytes says how much staging the staged draws need
 * (0: none). Returns NULL or a refusal. */
static inline const char *
g9p_xfb_before_job(struct g9p_xfb_exec *x, const struct g9p_xfb_event *events, unsigned n)
{
   x->nstage = x->ncopies = 0;
   x->staging_bytes = 0;
   for (unsigned i = 0; i < n; i++) {
      const char *why = g9p_xfb_execute(x, &events[i]);
      if (why)
         return why;
   }
   return NULL;
}

/* The staged draws' record addresses, once staging is at va */
static inline void
g9p_xfb_bind_staging(struct g9p_xfb_exec *x, uint64_t va)
{
   for (unsigned i = 0; i < x->nstage; i++) {
      const struct g9p_xfb_stage *s = &x->stage[i];
      g9p_xfb_put64(s->at, va + s->rel - s->fold);
   }
}

/* After the job: the primitives that fit, from staging into the streams,
 * record ranges only. Nothing if the job did not complete. */
static inline void
g9p_xfb_after_job(struct g9p_xfb_exec *x, const uint8_t *staging, bool ran)
{
   for (unsigned i = 0; ran && staging && i < x->ncopies; i++) {
      const struct g9p_xfb_copy *c = &x->copies[i];
      const uint32_t stride = c->layout.stride[c->b];
      for (uint64_t out = 0; out < c->records; out++) {
         const uint64_t from = (out / c->complete) * c->hw + out % c->complete;
         for (unsigned k = 0; k < c->layout.nranges; k++) {
            if (c->layout.ranges[k].buffer != c->b)
               continue;
            const uint64_t at = c->dst.offset + out * stride + c->layout.ranges[k].offset;
            if (at + c->layout.ranges[k].size > c->dst.mem->size)
               continue; /* fit was computed against the binding size: never outside */
            memcpy(c->dst.mem->map + at, staging + c->src + from * stride + c->layout.ranges[k].offset,
                   c->layout.ranges[k].size);
         }
      }
   }
   x->nstage = x->ncopies = 0;
   x->staging_bytes = 0;
}

#endif /* PVR_G9P_XFB_CORE_H */
