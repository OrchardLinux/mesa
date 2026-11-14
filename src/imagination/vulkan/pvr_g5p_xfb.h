/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): transform feedback on the G5P paths.
 * Built only with -Dimagination-g5p-experimental=true (PVR_APPLE_G5P).
 *
 * The vertex program's capturing variant stores every record itself
 * (pvr_g5p_compile_vs_xfb()); the rest is CPU work when the job is
 * submitted, in command order (pvr_g5p_xfb_core.h): Begin reads the counter
 * buffers, each captured draw gets its record addresses in its per-draw
 * block, a draw that doesn't fit stores into staging and the CPU copies the
 * whole primitives that fit after the job, End writes the counters.
 *
 * The command buffer records events (Bind is state, Begin, End, captured
 * draws), each with the graphics sub-command it belongs to; the queue runs a
 * sub-command's events just before submitting it (pvr_g5p_xfb_before_job)
 * and, when a draw overflowed, waits for the job and copies after it
 * (pvr_g5p_xfb_after_job). A command buffer without transform feedback never
 * allocates any of it.
 *
 * Limits, refused by name: non-indexed point, line and triangle lists
 * (a strip, fan or index buffer reuses vertices: the stores follow the vertex
 * invocations, not the primitive stream), stream 0, no 16-bit captures, no
 * byte-count draws (transformFeedbackDraw is off), stream queries of
 * stream 0 only, no SIMULTANEOUS_USE (the per-draw blocks are filled per
 * submission).
 *
 * Stream queries are events too; their counts are the resolve's, in the
 * pool's own buffers, available once the query's End has run (they don't
 * depend on the GPU). Events outside a render pass (query resets and copies)
 * run in order before the next render, or after the command buffer's last
 * one (pvr_g5p_xfb_end_cmd_buffer).
 */

#ifndef PVR_G5P_XFB_H
#define PVR_G5P_XFB_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

#include "pvr_common.h"
#include "pvr_macros.h"

struct g5p_xfb_layout;
struct pvr_cmd_buffer;
struct pvr_device;
struct pvr_query_pool;
struct pvr_queue;
struct pvr_sub_cmd_gfx;

/* A G5P transform feedback stream query pool (pvr_g5p_xfb_query.c) */
bool pvr_g5p_xfb_query_pool(const struct pvr_device *device, const struct pvr_query_pool *pool);
/* vkGetQueryPoolResults() of a stream query pool (pvr_g5p_xfb_query.c):
 * both values, primitives written then needed (upstream's path writes one) */
VkResult pvr_g5p_xfb_query_results(struct pvr_device *device, const struct pvr_query_pool *pool, uint32_t first,
                                   uint32_t count, void *data, VkDeviceSize stride, VkQueryResultFlags flags);

#ifdef PVR_PER_ARCH

/* vkCmdBind/Begin/EndTransformFeedbackEXT on G5P */
void PVR_PER_ARCH(g5p_xfb_bind)(struct pvr_cmd_buffer *cmd_buffer, uint32_t first, uint32_t count,
                                const VkBuffer *buffers, const VkDeviceSize *offsets, const VkDeviceSize *sizes);
void PVR_PER_ARCH(g5p_xfb_begin)(struct pvr_cmd_buffer *cmd_buffer, uint32_t first, uint32_t count,
                                 const VkBuffer *counters, const VkDeviceSize *offsets);
void PVR_PER_ARCH(g5p_xfb_end)(struct pvr_cmd_buffer *cmd_buffer, uint32_t first, uint32_t count,
                               const VkBuffer *counters, const VkDeviceSize *offsets);

/* A draw: capturing (between Begin and End), the topology it may capture,
 * and its event (the variant's layout and its per-draw block's CPU mapping) */
bool PVR_PER_ARCH(g5p_xfb_capturing)(const struct pvr_cmd_buffer *cmd_buffer);
VkResult PVR_PER_ARCH(g5p_xfb_draw_check)(struct pvr_cmd_buffer *cmd_buffer, VkPrimitiveTopology topology);
VkResult PVR_PER_ARCH(g5p_xfb_draw)(struct pvr_cmd_buffer *cmd_buffer, const struct g5p_xfb_layout *layout,
                                    uint8_t *block, uint32_t vertices, uint32_t instances, uint32_t first,
                                    VkPrimitiveTopology topology);

/* The queue, around a graphics sub-command's submission */
VkResult PVR_PER_ARCH(g5p_xfb_before_job)(struct pvr_device *device, struct pvr_cmd_buffer *cmd_buffer,
                                          const struct pvr_sub_cmd_gfx *sub_cmd);
VkResult PVR_PER_ARCH(g5p_xfb_after_job)(struct pvr_device *device, struct pvr_queue *queue,
                                         struct pvr_cmd_buffer *cmd_buffer, const struct pvr_sub_cmd_gfx *sub_cmd);

/* The events after the command buffer's last render (pvr_process_cmd_buffer()) */
VkResult PVR_PER_ARCH(g5p_xfb_end_cmd_buffer)(struct pvr_device *device, struct pvr_cmd_buffer *cmd_buffer);

/* Stream queries (VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT): Begin, End,
 * vkCmdResetQueryPool, vkCmdCopyQueryPoolResults */
void PVR_PER_ARCH(g5p_xfb_query_begin)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_query_pool *pool,
                                       uint32_t query, uint32_t index);
void PVR_PER_ARCH(g5p_xfb_query_end)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_query_pool *pool,
                                     uint32_t query);
void PVR_PER_ARCH(g5p_xfb_query_reset)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_query_pool *pool,
                                       uint32_t first, uint32_t count);
void PVR_PER_ARCH(g5p_xfb_query_copy)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_query_pool *pool,
                                      uint32_t first, uint32_t count, VkBuffer dst, VkDeviceSize offset,
                                      VkDeviceSize stride, VkQueryResultFlags flags);

/* The command buffer's resources are freed (reset, destroy) */
void PVR_PER_ARCH(g5p_xfb_free)(struct pvr_cmd_buffer *cmd_buffer);

#   define pvr_arch_g5p_xfb_bind PVR_PER_ARCH(g5p_xfb_bind)
#   define pvr_arch_g5p_xfb_begin PVR_PER_ARCH(g5p_xfb_begin)
#   define pvr_arch_g5p_xfb_end PVR_PER_ARCH(g5p_xfb_end)
#   define pvr_arch_g5p_xfb_capturing PVR_PER_ARCH(g5p_xfb_capturing)
#   define pvr_arch_g5p_xfb_draw_check PVR_PER_ARCH(g5p_xfb_draw_check)
#   define pvr_arch_g5p_xfb_draw PVR_PER_ARCH(g5p_xfb_draw)
#   define pvr_arch_g5p_xfb_before_job PVR_PER_ARCH(g5p_xfb_before_job)
#   define pvr_arch_g5p_xfb_after_job PVR_PER_ARCH(g5p_xfb_after_job)
#   define pvr_arch_g5p_xfb_free PVR_PER_ARCH(g5p_xfb_free)
#   define pvr_arch_g5p_xfb_end_cmd_buffer PVR_PER_ARCH(g5p_xfb_end_cmd_buffer)
#   define pvr_arch_g5p_xfb_query_begin PVR_PER_ARCH(g5p_xfb_query_begin)
#   define pvr_arch_g5p_xfb_query_end PVR_PER_ARCH(g5p_xfb_query_end)
#   define pvr_arch_g5p_xfb_query_reset PVR_PER_ARCH(g5p_xfb_query_reset)
#   define pvr_arch_g5p_xfb_query_copy PVR_PER_ARCH(g5p_xfb_query_copy)

#endif /* PVR_PER_ARCH */

#endif /* PVR_G5P_XFB_H */
