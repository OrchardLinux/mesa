/* SPDX-License-Identifier: MIT */
/*
 * Apple Rogue-derived GPUs (G9P, has_apple_g9p): compute pipelines,
 * dispatches and the end of a compute pass (pvr_arch_g9p.c); graphics
 * pipelines, draws and render jobs (pvr_arch_g9p_gfx.c).
 */
#ifndef PVR_G9P_H
#define PVR_G9P_H

#include <stdint.h>

#include <vulkan/vulkan_core.h>

#include "pvr_macros.h"

struct pvr_cmd_buffer;
struct pvr_compute_pipeline;
struct pvr_device;
struct pvr_graphics_pipeline;
struct pvr_render_job;
struct pvr_renderpass_hwsetup_render;
struct pvr_sub_cmd_compute;
struct pvr_sub_cmd_gfx;
struct vk_graphics_pipeline_state;

/*
 * The parameter buffer of a render (its local free list, bytes): the Apple
 * firmware does not grow it (pvr_arch_render_target_dataset_create())
 */
#define G9P_FREE_LIST_SIZE (32U * 1024U * 1024U)

/* One draw: vertices (or indices) from @offset (vertexOffset when indexed) */
struct pvr_g9p_draw {
   bool indexed;
   uint64_t index_addr;
   unsigned index_size;
   uint32_t count, instances, offset;
};

#ifdef PVR_PER_ARCH

VkResult PVR_PER_ARCH(g9p_clear_attachments)(struct pvr_cmd_buffer *cmd_buffer,
                                             uint32_t attachment_count,
                                             const VkClearAttachment *attachments,
                                             uint32_t rect_count, const VkClearRect *rects);

VkResult PVR_PER_ARCH(g9p_device_init)(struct pvr_device *device);

/* The CPU address of a device address in a bound buffer or image, or NULL */
uint8_t *PVR_PER_ARCH(g9p_cpu_addr)(struct pvr_device *device, uint64_t addr);
struct pvr_sub_cmd_gfx;
void PVR_PER_ARCH(g9p_run_fills)(struct pvr_device *device, struct pvr_sub_cmd_gfx *sub_cmd);
void PVR_PER_ARCH(g9p_device_finish)(struct pvr_device *device);

/* Copy a render's unaligned vertex bindings; before its job is submitted */
void PVR_PER_ARCH(g9p_realign_vertex_bindings)(struct pvr_sub_cmd_gfx *sub_cmd);

VkResult PVR_PER_ARCH(g9p_compute_pipeline_compile)(
   struct pvr_device *device,
   const VkComputePipelineCreateInfo *create_info,
   struct pvr_compute_pipeline *pipeline);

VkResult PVR_PER_ARCH(g9p_cmd_dispatch)(struct pvr_cmd_buffer *cmd_buffer,
                                        struct pvr_sub_cmd_compute *sub_cmd,
                                        const uint32_t group_count[static 3]);

void PVR_PER_ARCH(g9p_end_pass)(struct pvr_cmd_buffer *cmd_buffer,
                                struct pvr_sub_cmd_compute *sub_cmd);

VkResult PVR_PER_ARCH(g9p_gfx_device_init)(struct pvr_device *device);
void PVR_PER_ARCH(g9p_gfx_device_finish)(struct pvr_device *device);

VkResult PVR_PER_ARCH(g9p_graphics_pipeline_compile)(
   struct pvr_device *device,
   const VkGraphicsPipelineCreateInfo *create_info,
   struct pvr_graphics_pipeline *pipeline,
   const struct vk_graphics_pipeline_state *vk_state);
void PVR_PER_ARCH(g9p_graphics_pipeline_destroy)(struct pvr_graphics_pipeline *pipeline);

VkResult PVR_PER_ARCH(g9p_cmd_draw)(struct pvr_cmd_buffer *cmd_buffer,
                                    const struct pvr_g9p_draw *draw);

VkResult PVR_PER_ARCH(g9p_job_init)(struct pvr_cmd_buffer *cmd_buffer,
                                    const struct pvr_renderpass_hwsetup_render *hw_render,
                                    struct pvr_render_job *job);

/* The job streams with the Apple register block; return their length */
uint32_t PVR_PER_ARCH(g9p_geom_stream)(const struct pvr_device *device,
                                       const struct pvr_render_job *job, uint8_t *buf,
                                       uint32_t size);
uint32_t PVR_PER_ARCH(g9p_frag_stream)(const struct pvr_device *device,
                                       const struct pvr_render_job *job, uint8_t *buf,
                                       uint32_t size);

/* Transform feedback (pvr_arch_g9p_xfb.c) */
struct g9p_xfb_layout;
struct pvr_query_pool;
struct pvr_queue;
struct pvr_sub_cmd_gfx;
void PVR_PER_ARCH(g9p_xfb_bind)(struct pvr_cmd_buffer *cmd_buffer, uint32_t first, uint32_t count,
                                const VkBuffer *buffers, const VkDeviceSize *offsets,
                                const VkDeviceSize *sizes);
void PVR_PER_ARCH(g9p_xfb_begin)(struct pvr_cmd_buffer *cmd_buffer, uint32_t first, uint32_t count,
                                 const VkBuffer *counters, const VkDeviceSize *offsets);
void PVR_PER_ARCH(g9p_xfb_end)(struct pvr_cmd_buffer *cmd_buffer, uint32_t first, uint32_t count,
                               const VkBuffer *counters, const VkDeviceSize *offsets);
bool PVR_PER_ARCH(g9p_xfb_capturing)(const struct pvr_cmd_buffer *cmd_buffer);
VkResult PVR_PER_ARCH(g9p_xfb_draw_check)(struct pvr_cmd_buffer *cmd_buffer,
                                          VkPrimitiveTopology topology);
VkResult PVR_PER_ARCH(g9p_xfb_draw)(struct pvr_cmd_buffer *cmd_buffer,
                                    const struct g9p_xfb_layout *layout, uint8_t *block,
                                    uint32_t vertices, uint32_t instances, uint32_t first,
                                    VkPrimitiveTopology topology);
void PVR_PER_ARCH(g9p_xfb_query_begin)(struct pvr_cmd_buffer *cmd_buffer,
                                       const struct pvr_query_pool *pool, uint32_t query,
                                       uint32_t index);
void PVR_PER_ARCH(g9p_xfb_query_end)(struct pvr_cmd_buffer *cmd_buffer,
                                     const struct pvr_query_pool *pool, uint32_t query);
void PVR_PER_ARCH(g9p_xfb_query_reset)(struct pvr_cmd_buffer *cmd_buffer,
                                       const struct pvr_query_pool *pool, uint32_t first,
                                       uint32_t count);
void PVR_PER_ARCH(g9p_xfb_query_copy)(struct pvr_cmd_buffer *cmd_buffer,
                                      const struct pvr_query_pool *pool, uint32_t first,
                                      uint32_t count, VkBuffer dst, VkDeviceSize offset,
                                      VkDeviceSize stride, VkQueryResultFlags flags);
VkResult PVR_PER_ARCH(g9p_xfb_before_job)(struct pvr_device *device,
                                          struct pvr_cmd_buffer *cmd_buffer,
                                          const struct pvr_sub_cmd_gfx *sub_cmd);
VkResult PVR_PER_ARCH(g9p_xfb_after_job)(struct pvr_device *device, struct pvr_queue *queue,
                                         struct pvr_cmd_buffer *cmd_buffer,
                                         const struct pvr_sub_cmd_gfx *sub_cmd);
VkResult PVR_PER_ARCH(g9p_xfb_end_cmd_buffer)(struct pvr_device *device,
                                              struct pvr_cmd_buffer *cmd_buffer);
void PVR_PER_ARCH(g9p_xfb_free)(struct pvr_cmd_buffer *cmd_buffer);

struct pvr_image_view;
void PVR_PER_ARCH(g9p_pack_image_view)(const struct pvr_image_view *iview, uint64_t words[2]);
uint64_t PVR_PER_ARCH(g9p_pack_sampler)(const VkSamplerCreateInfo *info);

struct pvr_sub_cmd_transfer;
struct vk_sync;
VkResult PVR_PER_ARCH(g9p_transfer_submit)(struct pvr_device *device,
                                           struct pvr_sub_cmd_transfer *sub_cmd,
                                           struct vk_sync *wait_sync,
                                           struct vk_sync *signal_sync);

#   define pvr_arch_g9p_transfer_submit PVR_PER_ARCH(g9p_transfer_submit)
#   define pvr_arch_g9p_pack_image_view PVR_PER_ARCH(g9p_pack_image_view)
#   define pvr_arch_g9p_pack_sampler PVR_PER_ARCH(g9p_pack_sampler)
#   define pvr_arch_g9p_gfx_device_init PVR_PER_ARCH(g9p_gfx_device_init)
#   define pvr_arch_g9p_gfx_device_finish PVR_PER_ARCH(g9p_gfx_device_finish)
#   define pvr_arch_g9p_graphics_pipeline_compile \
      PVR_PER_ARCH(g9p_graphics_pipeline_compile)
#   define pvr_arch_g9p_graphics_pipeline_destroy \
      PVR_PER_ARCH(g9p_graphics_pipeline_destroy)
#   define pvr_arch_g9p_cmd_draw PVR_PER_ARCH(g9p_cmd_draw)
#   define pvr_arch_g9p_job_init PVR_PER_ARCH(g9p_job_init)
#   define pvr_arch_g9p_geom_stream PVR_PER_ARCH(g9p_geom_stream)
#   define pvr_arch_g9p_frag_stream PVR_PER_ARCH(g9p_frag_stream)
#   define pvr_arch_g9p_device_init PVR_PER_ARCH(g9p_device_init)
#   define pvr_arch_g9p_device_finish PVR_PER_ARCH(g9p_device_finish)
#   define pvr_arch_g9p_compute_pipeline_compile \
      PVR_PER_ARCH(g9p_compute_pipeline_compile)
#   define pvr_arch_g9p_cmd_dispatch PVR_PER_ARCH(g9p_cmd_dispatch)
#   define pvr_arch_g9p_end_pass PVR_PER_ARCH(g9p_end_pass)

#endif /* PVR_PER_ARCH */

#endif /* PVR_G9P_H */
