/* SPDX-License-Identifier: MIT */
/*
 * Apple GPUs (G9P): transform feedback. From PR #6's A9 port
 * (pvr_arch_g5p_xfb.c). The resolve is pvr_g9p_xfb_core.h; this file is its
 * Vulkan side: the command buffer's events and the queue's two CPU points.
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/os_time.h"
#include "util/u_dynarray.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_log.h"
#include "vk_sync.h"

#include "pvr_bo.h"
#include "pvr_buffer.h"
#include "pvr_cmd_buffer.h"
#include "pvr_device.h"
#include "pvr_g9p.h"
#include "pvr_g9p_xfb_core.h"
#include "pvr_query.h"
#include "pvr_queue.h"

/* A refusal, by name, on stderr too (vk_errorf() is silent in release builds) */
#define G9P_XFB_REFUSE(obj, ...)                                                                                   \
   (fprintf(stderr, "G9P refuses: " __VA_ARGS__), fprintf(stderr, "\n"),                                          \
    vk_errorf((obj), VK_ERROR_FEATURE_NOT_PRESENT, "G9P: " __VA_ARGS__))

/* A buffer range named by a command (a binding or a counter) */
struct g9p_xfb_range {
   struct pvr_buffer *buffer; /* NULL: none */
   VkDeviceSize offset, size;
};

/* One recorded event: the core's event, the buffers it names (their CPU
 * views are filled when it runs) and its graphics sub-command */
struct g9p_xfb_rec {
   const struct pvr_sub_cmd_gfx *sub;
   struct g9p_xfb_event e;
   struct g9p_xfb_range bind[G9P_XFB_BUFFERS], counter[G9P_XFB_BUFFERS];
   struct g9p_xfb_mem bind_mem[G9P_XFB_BUFFERS], counter_mem[G9P_XFB_BUFFERS];
   /* query events: the pool, and its CPU view when the event runs (a query
    * copy's destination is counter[0]) */
   struct {
      const struct pvr_query_pool *pool;
   } qpool;
   struct g9p_xfb_query_pool qview;
};

struct pvr_g9p_xfb_cmd {
   struct g9p_xfb_range bind[G9P_XFB_BUFFERS]; /* vkCmdBindTransformFeedbackBuffersEXT */
   bool capturing;
   struct util_dynarray recs; /* struct g9p_xfb_rec, in command order */
   unsigned next;             /* the first event the current submission has not run */
   /* execution (one submission at a time: no SIMULTANEOUS_USE) */
   struct g9p_xfb_exec exec;
   struct pvr_bo *staging;
   const struct pvr_sub_cmd_gfx *staging_sub;
};

static struct pvr_g9p_xfb_cmd *
g9p_xfb_cmd(struct pvr_cmd_buffer *cmd_buffer)
{
   struct pvr_g9p_xfb_cmd *st = cmd_buffer->state.g9p_xfb;
   if (st)
      return st;
   st = vk_zalloc(&cmd_buffer->vk.pool->alloc, sizeof(*st), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!st) {
      vk_command_buffer_set_error(&cmd_buffer->vk, vk_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY));
      return NULL;
   }
   util_dynarray_init(&st->recs, NULL);
   g9p_xfb_exec_init(&st->exec);
   cmd_buffer->state.g9p_xfb = st;
   return st;
}

static struct g9p_xfb_rec *
g9p_xfb_rec_new(struct pvr_cmd_buffer *cmd_buffer, struct pvr_g9p_xfb_cmd *st, enum g9p_xfb_kind kind)
{
   /* the graphics sub-command it belongs to; NULL outside one (events before a
    * render pass's first draw, query resets and copies): run in order before
    * the next render or at the end */
   const struct pvr_sub_cmd *sub = cmd_buffer->state.current_sub_cmd;
   struct g9p_xfb_rec *r = util_dynarray_grow(&st->recs, struct g9p_xfb_rec, 1);
   if (!r) {
      vk_command_buffer_set_error(&cmd_buffer->vk, vk_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY));
      return NULL;
   }
   memset(r, 0, sizeof(*r));
   r->sub = sub && sub->type == PVR_SUB_CMD_TYPE_GRAPHICS ? &sub->gfx : NULL;
   r->e.kind = kind;
   return r;
}

void
PVR_PER_ARCH(g9p_xfb_bind)(struct pvr_cmd_buffer *cmd_buffer, uint32_t first, uint32_t count,
                           const VkBuffer *buffers, const VkDeviceSize *offsets, const VkDeviceSize *sizes)
{
   struct pvr_g9p_xfb_cmd *st = g9p_xfb_cmd(cmd_buffer);
   if (!st)
      return;
   for (uint32_t i = 0; i < count && first + i < G9P_XFB_BUFFERS; i++) {
      VK_FROM_HANDLE(pvr_buffer, buffer, buffers[i]);
      const VkDeviceSize off = offsets[i];
      const VkDeviceSize size = !sizes || sizes[i] == VK_WHOLE_SIZE ? buffer->vk.size - off : sizes[i];
      st->bind[first + i] = (struct g9p_xfb_range){ buffer, off, size };
      if (buffer)
         buffer->g9p_gpu_written = true;
   }
}

static void
g9p_xfb_counters(struct g9p_xfb_range *out, uint32_t first, uint32_t count, const VkBuffer *counters,
                 const VkDeviceSize *offsets)
{
   for (uint32_t i = 0; counters && i < count && first + i < G9P_XFB_BUFFERS; i++) {
      if (counters[i] == VK_NULL_HANDLE)
         continue;
      VK_FROM_HANDLE(pvr_buffer, buffer, counters[i]);
      out[first + i] = (struct g9p_xfb_range){ buffer, offsets ? offsets[i] : 0, 4 };
   }
}

void
PVR_PER_ARCH(g9p_xfb_begin)(struct pvr_cmd_buffer *cmd_buffer, uint32_t first, uint32_t count,
                            const VkBuffer *counters, const VkDeviceSize *offsets)
{
   struct pvr_g9p_xfb_cmd *st = g9p_xfb_cmd(cmd_buffer);
   if (!st)
      return;
   if (cmd_buffer->usage_flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT) {
      vk_command_buffer_set_error(&cmd_buffer->vk,
                                  G9P_XFB_REFUSE(cmd_buffer, "transform feedback in a SIMULTANEOUS_USE command "
                                                 "buffer (its per-draw blocks are filled per submission)"));
      return;
   }
   struct g9p_xfb_rec *r = g9p_xfb_rec_new(cmd_buffer, st, G9P_XFB_BEGIN);
   if (!r)
      return;
   memcpy(r->bind, st->bind, sizeof(r->bind));
   g9p_xfb_counters(r->counter, first, count, counters, offsets);
   st->capturing = true;
}

void
PVR_PER_ARCH(g9p_xfb_end)(struct pvr_cmd_buffer *cmd_buffer, uint32_t first, uint32_t count,
                          const VkBuffer *counters, const VkDeviceSize *offsets)
{
   struct pvr_g9p_xfb_cmd *st = g9p_xfb_cmd(cmd_buffer);
   if (!st)
      return;
   struct g9p_xfb_rec *r = g9p_xfb_rec_new(cmd_buffer, st, G9P_XFB_END);
   if (!r)
      return;
   g9p_xfb_counters(r->counter, first, count, counters, offsets);
   st->capturing = false;
}

bool
PVR_PER_ARCH(g9p_xfb_capturing)(const struct pvr_cmd_buffer *cmd_buffer)
{
   return cmd_buffer->state.g9p_xfb && cmd_buffer->state.g9p_xfb->capturing;
}

static unsigned
g9p_xfb_vpp(VkPrimitiveTopology t)
{
   switch (t) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: return 1;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: return 2;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: return 3;
   default: return 0;
   }
}

VkResult
PVR_PER_ARCH(g9p_xfb_draw_check)(struct pvr_cmd_buffer *cmd_buffer, VkPrimitiveTopology topology)
{
   /* the stores follow the vertex invocations: they are the primitive
    * stream only for lists (GLES 3.0 captures lists only) */
   if (!g9p_xfb_vpp(topology))
      return G9P_XFB_REFUSE(cmd_buffer, "transform feedback with topology %d (point, line and triangle lists)",
                            topology);
   return VK_SUCCESS;
}

VkResult
PVR_PER_ARCH(g9p_xfb_draw)(struct pvr_cmd_buffer *cmd_buffer, const struct g9p_xfb_layout *layout, uint8_t *block,
                           uint32_t vertices, uint32_t instances, uint32_t first, VkPrimitiveTopology topology)
{
   struct pvr_g9p_xfb_cmd *st = g9p_xfb_cmd(cmd_buffer);
   if (!st)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   if (!block)
      return G9P_XFB_REFUSE(cmd_buffer, "a capturing vertex program without its transform feedback block");
   struct g9p_xfb_rec *r = g9p_xfb_rec_new(cmd_buffer, st, G9P_XFB_DRAW);
   if (!r)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   r->e.draw.block = block;
   r->e.draw.vertices = vertices;
   r->e.draw.instances = instances;
   r->e.draw.first = first;
   r->e.draw.vpp = g9p_xfb_vpp(topology);
   r->e.draw.layout = *layout;
   return VK_SUCCESS;
}

/* Stream queries (VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT): events too.
 * The counts are the resolve's (primitives written and needed by captured
 * draws), kept in the pool's own buffers as two uint64_t per query, and
 * available once the query's End has run: the counts don't depend on the
 * GPU. */
static void
g9p_xfb_query_event(struct pvr_cmd_buffer *cmd_buffer, enum g9p_xfb_kind kind, const struct pvr_query_pool *pool,
                    uint32_t first, uint32_t count)
{
   struct pvr_g9p_xfb_cmd *st = g9p_xfb_cmd(cmd_buffer);
   if (!st)
      return;
   if (pool->result_elem_stride < 2 * sizeof(uint64_t)) {
      vk_command_buffer_set_error(&cmd_buffer->vk,
                                  G9P_XFB_REFUSE(cmd_buffer, "a stream query pool of %u bytes per query",
                                                 pool->result_elem_stride));
      return;
   }
   struct g9p_xfb_rec *r = g9p_xfb_rec_new(cmd_buffer, st, kind);
   if (!r)
      return;
   r->qpool.pool = pool;
   r->e.query.first = first;
   r->e.query.count = count;
}

void
PVR_PER_ARCH(g9p_xfb_query_begin)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_query_pool *pool,
                                  uint32_t query, uint32_t index)
{
   if (index) {
      vk_command_buffer_set_error(&cmd_buffer->vk,
                                  G9P_XFB_REFUSE(cmd_buffer, "a stream query of stream %u (stream 0 only)", index));
      return;
   }
   g9p_xfb_query_event(cmd_buffer, G9P_XFB_QUERY_BEGIN, pool, query, 1);
}

void
PVR_PER_ARCH(g9p_xfb_query_end)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_query_pool *pool, uint32_t query)
{
   g9p_xfb_query_event(cmd_buffer, G9P_XFB_QUERY_END, pool, query, 1);
}

void
PVR_PER_ARCH(g9p_xfb_query_reset)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_query_pool *pool,
                                  uint32_t first, uint32_t count)
{
   g9p_xfb_query_event(cmd_buffer, G9P_XFB_QUERY_RESET, pool, first, count);
}

void
PVR_PER_ARCH(g9p_xfb_query_copy)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_query_pool *pool,
                                 uint32_t first, uint32_t count, VkBuffer dst, VkDeviceSize offset,
                                 VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(pvr_buffer, buffer, dst);
   g9p_xfb_query_event(cmd_buffer, G9P_XFB_QUERY_COPY, pool, first, count);
   struct pvr_g9p_xfb_cmd *st = cmd_buffer->state.g9p_xfb;
   const unsigned n = st ? util_dynarray_num_elements(&st->recs, struct g9p_xfb_rec) : 0;
   if (!n)
      return;
   struct g9p_xfb_rec *r = util_dynarray_element(&st->recs, struct g9p_xfb_rec, n - 1);
   if (r->e.kind != G9P_XFB_QUERY_COPY)
      return;
   r->counter[0] = (struct g9p_xfb_range){ buffer, offset, buffer->vk.size - offset };
   r->e.query.stride = stride;
   r->e.query.flags = flags;
}

/* A buffer's CPU view, as the CPU transfers map theirs */
static bool
g9p_xfb_mem(struct pvr_device *device, const struct g9p_xfb_range *range, struct g9p_xfb_mem *mem,
            struct g9p_xfb_ref *ref)
{
   memset(mem, 0, sizeof(*mem));
   *ref = (struct g9p_xfb_ref){ 0 };
   if (!range->buffer)
      return true;
   mem->va = range->buffer->dev_addr.addr;
   mem->map = PVR_PER_ARCH(g9p_cpu_addr)(device, mem->va);
   if (!mem->map)
      return false;
   mem->size = range->buffer->vk.size;
   *ref = (struct g9p_xfb_ref){ mem, range->offset };
   return true;
}

/* Run the events [st->next, end) in command order. A new submission starts
 * at the first event with a fresh state (no SIMULTANEOUS_USE: one at a time). */
static VkResult
g9p_xfb_run(struct pvr_device *device, struct pvr_g9p_xfb_cmd *st, unsigned end)
{
   struct g9p_xfb_rec *recs = st->recs.data;
   struct g9p_xfb_exec *x = &st->exec;
   if (st->next == 0) {
      g9p_xfb_exec_finish(x);
      g9p_xfb_exec_init(x);
   }
   x->nstage = x->ncopies = 0;
   x->staging_bytes = 0;
   for (; st->next < end; st->next++) {
      struct g9p_xfb_rec *r = &recs[st->next];
      bool ok = true;
      for (unsigned b = 0; b < G9P_XFB_BUFFERS; b++) {
         switch (r->e.kind) {
         case G9P_XFB_BEGIN:
            ok &= g9p_xfb_mem(device, &r->bind[b], &r->bind_mem[b], &r->e.begin.bind[b]);
            r->e.begin.size[b] = r->bind[b].size;
            ok &= g9p_xfb_mem(device, &r->counter[b], &r->counter_mem[b], &r->e.begin.counter[b]);
            break;
         case G9P_XFB_END:
            ok &= g9p_xfb_mem(device, &r->counter[b], &r->counter_mem[b], &r->e.end[b]);
            break;
         case G9P_XFB_QUERY_COPY:
            if (b == 0)
               ok &= g9p_xfb_mem(device, &r->counter[0], &r->counter_mem[0], &r->e.query.dst);
            break;
         default:
            break;
         }
      }
      if (r->qpool.pool) {
         /* the pool's own result and availability buffers, CPU-mapped (pvr_query.c) */
         r->qview = (struct g9p_xfb_query_pool){
            .count = r->qpool.pool->query_count,
            .stride = r->qpool.pool->result_elem_stride,
            .data = pvr_bo_suballoc_get_map_addr(r->qpool.pool->result_buffer),
            .available = pvr_bo_suballoc_get_map_addr(r->qpool.pool->availability_buffer),
         };
         r->e.query.pool = &r->qview;
      }
      if (!ok)
         return G9P_XFB_REFUSE(device, "a transform feedback buffer without a CPU mapping");
      const char *why = g9p_xfb_execute(x, &r->e);
      if (why)
         return G9P_XFB_REFUSE(device, "transform feedback: %s", why);
   }
   return VK_SUCCESS;
}

VkResult
PVR_PER_ARCH(g9p_xfb_before_job)(struct pvr_device *device, struct pvr_cmd_buffer *cmd_buffer,
                                 const struct pvr_sub_cmd_gfx *sub_cmd)
{
   struct pvr_g9p_xfb_cmd *st = cmd_buffer->state.g9p_xfb;
   if (!st)
      return VK_SUCCESS;
   /* every event up to this sub-command's last one (those recorded outside a
    * render pass before it included) */
   const struct g9p_xfb_rec *recs = st->recs.data;
   unsigned end = util_dynarray_num_elements(&st->recs, struct g9p_xfb_rec);
   while (end > st->next && recs[end - 1].sub != sub_cmd)
      end--;
   if (end == st->next)
      return VK_SUCCESS;
   VkResult result = g9p_xfb_run(device, st, end);
   if (result != VK_SUCCESS)
      return result;

   /* a draw that doesn't fit stores into staging; the CPU copies what fits after the job */
   struct g9p_xfb_exec *x = &st->exec;
   if (x->staging_bytes) {
      result = pvr_bo_alloc(device, device->heaps.general_heap, x->staging_bytes, 64, PVR_BO_ALLOC_FLAG_CPU_MAPPED,
                            &st->staging);
      if (result != VK_SUCCESS)
         return result;
      st->staging_sub = sub_cmd;
      g9p_xfb_bind_staging(x, st->staging->vma->dev_addr.addr);
   }
   return VK_SUCCESS;
}

VkResult
PVR_PER_ARCH(g9p_xfb_end_cmd_buffer)(struct pvr_device *device, struct pvr_cmd_buffer *cmd_buffer)
{
   struct pvr_g9p_xfb_cmd *st = cmd_buffer->state.g9p_xfb;
   if (!st)
      return VK_SUCCESS;
   /* the events after the last render (query copies and resets outside a
    * render pass); then the next submission starts over */
   VkResult result = g9p_xfb_run(device, st, util_dynarray_num_elements(&st->recs, struct g9p_xfb_rec));
   st->next = 0;
   return result;
}

VkResult
PVR_PER_ARCH(g9p_xfb_after_job)(struct pvr_device *device, struct pvr_queue *queue, struct pvr_cmd_buffer *cmd_buffer,
                                const struct pvr_sub_cmd_gfx *sub_cmd)
{
   struct pvr_g9p_xfb_cmd *st = cmd_buffer->state.g9p_xfb;
   if (!st || !st->staging || st->staging_sub != sub_cmd)
      return VK_SUCCESS;
   /* the job's stores are in staging once both stages retired */
   bool ran = true;
   for (unsigned t = PVR_JOB_TYPE_GEOM; t <= PVR_JOB_TYPE_FRAG; t++) {
      struct vk_sync *s = queue->last_job_signal_sync[t];
      if (s && vk_sync_wait(&device->vk, s, 0, VK_SYNC_WAIT_COMPLETE, UINT64_MAX) != VK_SUCCESS)
         ran = false;
   }
   g9p_xfb_after_job(&st->exec, st->staging->bo->map, ran);
   pvr_bo_free(device, st->staging);
   st->staging = NULL;
   st->staging_sub = NULL;
   return ran ? VK_SUCCESS : vk_error(device, VK_ERROR_DEVICE_LOST);
}

void
PVR_PER_ARCH(g9p_xfb_free)(struct pvr_cmd_buffer *cmd_buffer)
{
   struct pvr_g9p_xfb_cmd *st = cmd_buffer->state.g9p_xfb;
   if (!st)
      return;
   if (st->staging)
      pvr_bo_free(cmd_buffer->device, st->staging);
   g9p_xfb_exec_finish(&st->exec);
   util_dynarray_fini(&st->recs);
   vk_free(&cmd_buffer->vk.pool->alloc, st);
   cmd_buffer->state.g9p_xfb = NULL;
}
