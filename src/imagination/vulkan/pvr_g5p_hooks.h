/* SPDX-License-Identifier: MIT */
/*
 * The Apple A9 (G5P) hooks in the shared PowerVR driver files.
 *
 * Where a shared file has code after the hook that records its own line
 * number (vk_error() and friends pass __LINE__), the hook is one line,
 * PVR_G5P_HOOK(name, ...), and it takes the place of a blank line of that
 * file. A build without -Dimagination-g5p-experimental so keeps every line
 * number, and every byte of libvulkan_powervr_mesa.so, of the file without
 * G5P: the hook expands to nothing there. With the option it expands to the
 * PVR_G5P_HOOK_<name> below.
 *
 * Included by pvr_winsys.h, so every file that has a hook sees it.
 */

#ifndef PVR_G5P_HOOKS_H
#define PVR_G5P_HOOKS_H

#if defined(PVR_APPLE_G5P)
#   define PVR_G5P_HOOK(name, ...) PVR_G5P_HOOK_##name(__VA_ARGS__)
#else
#   define PVR_G5P_HOOK(name, ...)
#endif

/* Textures and samplers (pvr_g5p_tex.h). */
#if defined(PVR_APPLE_G5P)
#   include "pvr_g5p_tex.h"
#   include "pvr_g5p_xfb.h"
#endif

/* pvr_arch_image.c, pvr_CreateImageView(): the sampled state of a view is
 * the A9's three-qword image state (pvr_arch_g5p_pack_image_view()).
 */
#define PVR_G5P_HOOK_image_view(device, iview)                              \
   if (PVR_HAS_FEATURE(&(device)->pdevice->dev_info, apple_g5p))           \
      pvr_arch_g5p_pack_image_view(                                         \
         iview, &(iview)->image_state[PVR_TEXTURE_STATE_SAMPLE]);

/* pvr_arch_sampler.c, pvr_CreateSampler(): the A9's sampler words, both
 * forms (pvr_arch_g5p_pack_sampler()).
 */
#define PVR_G5P_HOOK_sampler(device, create_info, sampler)                 \
   if (PVR_HAS_FEATURE(&(device)->pdevice->dev_info, apple_g5p))           \
      pvr_arch_g5p_pack_sampler(create_info, &(sampler)->descriptor);

/* pvr_arch_descriptor_set.c, write_image_sampler(): a combined descriptor
 * takes the sampler form its view needs (pvr_arch_g5p_combine_sampler()).
 */
#define PVR_G5P_HOOK_image_sampler(set, image_info, desc)                     \
   if (PVR_HAS_FEATURE(&container_of((set)->base.device, struct pvr_device, \
                                     vk)->pdevice->dev_info,                \
                       apple_g5p))                                          \
      pvr_arch_g5p_combine_sampler(                                         \
         desc, (image_info)->imageView                                      \
                  ? pvr_image_view_from_handle((image_info)->imageView)     \
                  : NULL);

/* pvr_image.c, pvr_image_init(): linear rows are 64-byte aligned on the
 * A9, the ICD's G5P_LINEAR_STRIDE_ALIGN (the image state's stride field is
 * in 16-byte units; every linear row the A9 has rendered or sampled was a
 * multiple of 64 bytes): the alignment in pixels of this format.
 */
#define PVR_G5P_HOOK_stride_align(device, create_info, align)                \
   if (PVR_HAS_FEATURE(&(device)->pdevice->dev_info, apple_g5p)) {           \
      const unsigned cpp_ = vk_format_get_blocksize((create_info)->format); \
      align = cpp_ && 64 % cpp_ == 0 ? 64 / cpp_ : 64;                      \
   }

/* pvr_device.c, pvr_MapMemory2(): the CPU transfer path G5P shares with G9P
 * (pvr_arch_g9p_cpu_addr()) maps a bound BO and leaves it mapped. That is the
 * driver's mapping, not the application's, so a vkMapMemory of that memory
 * maps it again (the same CPU address, or a placed one) instead of being
 * refused as "already mapped".
 */
#define PVR_G5P_HOOK_map_again(device, mem, addr, offset, data)          \
   if (PVR_HAS_FEATURE(&(device)->pdevice->dev_info, apple_g5p) &&        \
       (mem)->bo->map) {                                                  \
      VkResult r_ = (device)->ws->ops->buffer_map((mem)->bo, addr);       \
      if (r_ == VK_SUCCESS)                                               \
         *(data) = (uint8_t *)(mem)->bo->map + (offset);                  \
      return r_;                                                          \
   }

/* pvr_arch_xfb.c, vkCmdBind/Begin/EndTransformFeedbackEXT(): G5P records
 * its own events (pvr_g5p_xfb.h); Rogue's stream-out programs and End's
 * render split (a colour LOAD) are not used.
 */
#define PVR_G5P_HOOK_xfb_bind(cmd_buffer, first, count, buffers, offsets, sizes) \
   if (PVR_HAS_FEATURE(&(cmd_buffer)->device->pdevice->dev_info, apple_g5p)) {   \
      pvr_arch_g5p_xfb_bind(cmd_buffer, first, count, buffers, offsets, sizes);   \
      return;                                                                     \
   }
#define PVR_G5P_HOOK_xfb_begin(cmd_buffer, first, count, counters, offsets)     \
   if (PVR_HAS_FEATURE(&(cmd_buffer)->device->pdevice->dev_info, apple_g5p)) { \
      pvr_arch_g5p_xfb_begin(cmd_buffer, first, count, counters, offsets);      \
      return;                                                                   \
   }
#define PVR_G5P_HOOK_xfb_end(cmd_buffer, first, count, counters, offsets)       \
   if (PVR_HAS_FEATURE(&(cmd_buffer)->device->pdevice->dev_info, apple_g5p)) { \
      pvr_arch_g5p_xfb_end(cmd_buffer, first, count, counters, offsets);        \
      return;                                                                   \
   }

/* pvr_arch_queue.c, pvr_process_graphics_cmd(): a sub-command's transform
 * feedback events run on the CPU just before it is submitted, and a draw
 * that overflowed is copied once the job retired (pvr_g5p_xfb.h). Nothing
 * for a command buffer without transform feedback.
 */
#define PVR_G5P_HOOK_xfb_before_job(device, cmd_buffer, sub_cmd)                       \
   {                                                                                   \
      const VkResult xfb_r_ = pvr_arch_g5p_xfb_before_job(device, cmd_buffer, sub_cmd); \
      if (xfb_r_ != VK_SUCCESS)                                                        \
         return xfb_r_;                                                                \
   }
#define PVR_G5P_HOOK_xfb_after_job(device, queue, cmd_buffer, sub_cmd)                        \
   {                                                                                          \
      const VkResult xfb_r_ = pvr_arch_g5p_xfb_after_job(device, queue, cmd_buffer, sub_cmd); \
      if (xfb_r_ != VK_SUCCESS)                                                               \
         return xfb_r_;                                                                       \
   }

/* pvr_arch_queue.c, pvr_process_cmd_buffer(): the transform feedback events
 * after the command buffer's last render (query resets and copies).
 */
#define PVR_G5P_HOOK_xfb_end_cmd_buffer(device, cmd_buffer)                              \
   {                                                                                     \
      const VkResult xfb_r_ = pvr_arch_g5p_xfb_end_cmd_buffer(device, cmd_buffer);         \
      if (xfb_r_ != VK_SUCCESS)                                                          \
         return xfb_r_;                                                                  \
   }

/* pvr_arch_cmd_query.c and pvr_query.c: a transform feedback stream query
 * pool's commands and results are G5P's (pvr_g5p_xfb.h); occlusion pools go
 * on as they are.
 */
#define PVR_G5P_XFB_QUERY_POOL(device, pool) pvr_g5p_xfb_query_pool(device, pool)
#define PVR_G5P_HOOK_xfb_query_begin(cmd_buffer, pool, query, index)     \
   if (PVR_G5P_XFB_QUERY_POOL((cmd_buffer)->device, pool)) {             \
      pvr_arch_g5p_xfb_query_begin(cmd_buffer, pool, query, index);      \
      return;                                                            \
   }
#define PVR_G5P_HOOK_xfb_query_end(cmd_buffer, pool, query)              \
   if (PVR_G5P_XFB_QUERY_POOL((cmd_buffer)->device, pool)) {             \
      pvr_arch_g5p_xfb_query_end(cmd_buffer, pool, query);               \
      return;                                                            \
   }
#define PVR_G5P_HOOK_xfb_query_reset(cmd_buffer, query_pool, first, count)                     \
   if (PVR_G5P_XFB_QUERY_POOL((cmd_buffer)->device, pvr_query_pool_from_handle(query_pool))) { \
      pvr_arch_g5p_xfb_query_reset(cmd_buffer, pvr_query_pool_from_handle(query_pool), first,  \
                                   count);                                                     \
      return;                                                                                  \
   }
#define PVR_G5P_HOOK_xfb_query_copy(cmd_buffer, query_pool, first, count, dst, offset, stride, flags) \
   if (PVR_G5P_XFB_QUERY_POOL((cmd_buffer)->device, pvr_query_pool_from_handle(query_pool))) {        \
      pvr_arch_g5p_xfb_query_copy(cmd_buffer, pvr_query_pool_from_handle(query_pool), first, count,   \
                                  dst, offset, stride, flags);                                        \
      return;                                                                                         \
   }
#define PVR_G5P_HOOK_xfb_query_results(device, pool, first, count, data, stride, flags) \
   if (PVR_G5P_XFB_QUERY_POOL(device, pool))                                            \
      return pvr_g5p_xfb_query_results(device, pool, first, count, data, stride, flags);

/* pvr_arch_cmd_buffer.c, pvr_cmd_buffer_free_resources() */
#define PVR_G5P_HOOK_xfb_free(cmd_buffer) pvr_arch_g5p_xfb_free(cmd_buffer);

/* rogue/pvr_blit.c, vkCmdClearAttachments(): Rogue's clear programs and
 * words do not run on the A9 (G9P's hung its geometry job); G5P's own
 * (pvr_arch_g5p_gfx.c).
 */
#define PVR_G5P_HOOK_clear_attachments(cmd_buffer, count, attachments, rect_count, rects)               \
   if (PVR_HAS_FEATURE(&(cmd_buffer)->device->pdevice->dev_info, apple_g5p)) {                        \
      VkResult r_ = pvr_arch_g5p_clear_attachments(cmd_buffer, count, attachments, rect_count, rects); \
      if (r_ != VK_SUCCESS)                                                                           \
         pvr_cmd_buffer_set_error_unwarned(cmd_buffer, r_);                                           \
      return;                                                                                         \
   }

#endif /* PVR_G5P_HOOKS_H */

/* The per-arch functions of the hooks, guarded on their own (as
 * pvr_g5p_ds.h): a file that includes this header before PVR_PER_ARCH is
 * defined still gets them from a later include. */
#if defined(PVR_APPLE_G5P) && defined(PVR_PER_ARCH) && !defined(PVR_G5P_HOOKS_ARCH_H)
#define PVR_G5P_HOOKS_ARCH_H

struct pvr_cmd_buffer;

VkResult PVR_PER_ARCH(g5p_clear_attachments)(struct pvr_cmd_buffer *cmd_buffer, uint32_t attachment_count,
                                             const VkClearAttachment *attachments, uint32_t rect_count,
                                             const VkClearRect *rects);
#   define pvr_arch_g5p_clear_attachments PVR_PER_ARCH(g5p_clear_attachments)

#endif /* PVR_G5P_HOOKS_ARCH_H */
