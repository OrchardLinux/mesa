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

/* Textures and samplers (pvr_g5p_tex.h), colour formats (pvr_g5p_formats.h). */
#if defined(PVR_APPLE_G5P)
#   include "pvr_g5p_formats.h"
#   include "pvr_g5p_layout.h"
#   include "pvr_g5p_tex.h"
#   include "pvr_g5p_xfb.h"
#   include "pvr_g5p_memory.h"
#   include "pvr_g5p_spm.h"
#   include "pvr_g5p_shm.h"
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
      const VkImageDrmFormatModifierExplicitCreateInfoEXT *shm_ =           \
         pvr_g5p_shm_explicit(create_info);                                \
      if (shm_ && shm_->drmFormatModifier == DRM_FORMAT_MOD_LINEAR &&                          \
          shm_->drmFormatModifierPlaneCount == 1) {                        \
         if (shm_->pPlaneLayouts[0].offset % 16 ||                          \
             shm_->pPlaneLayouts[0].rowPitch % 16)                         \
            return VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT; \
         if (cpp_ && 16 % cpp_ == 0) align = 16 / cpp_;                     \
      }                                                                    \
   }

/* Public Vulkan layout is relative to the bound memory. Internal transfer
 * layout stays relative to image->dev_addr, which already includes import_offset.
 */
#define PVR_G5P_HOOK_shm_layout_public(image, subresource, layout)          \
   if (PVR_HAS_FEATURE(&container_of((image)->vk.base.device, struct pvr_device, \
                                     vk)->pdevice->dev_info, apple_g5p)) { \
      pvr_get_image_subresource_layout(image, &(subresource)->imageSubresource, \
                                       &(layout)->subresourceLayout);     \
      (layout)->subresourceLayout.offset += (image)->import_offset;        \
      return;                                                            \
   }

/* A G5P 3D image is a 2D array of its slices (pvr_g5p_image_layout(); it
 * keeps memlayout 3DTWIDDLED for upstream's view setup).
 *
 * pvr_image.c, pvr_get_image_subresource_layout(): a level's next slice is
 * one layer pitch on, not one level slice on.
 */
#define PVR_G5P_IMAGE_IS_SLICED_3D(image)                                     \
   (PVR_HAS_FEATURE(&container_of((image)->vk.base.device, struct pvr_device, \
                                  vk)->pdevice->dev_info,                   \
                    apple_g5p) &&                                           \
    (image)->vk.image_type == VK_IMAGE_TYPE_3D &&                            \
    (image)->memlayout == PVR_MEMLAYOUT_3DTWIDDLED)
#define PVR_G5P_HOOK_slice_pitch(image, layout)                               \
   if (PVR_G5P_IMAGE_IS_SLICED_3D(image))                                    \
      (layout)->depthPitch = (layout)->arrayPitch;

/* rogue/pvr_blit.c, pvr_setup_transfer_surface(): the transfer surface of
 * such an image is the 2D-twiddled surface of its slice, so G9P's 2D copies
 * take it, and the blit and image-to-buffer slice loops step it by the
 * depthPitch above.
 */
#define PVR_G5P_HOOK_transfer_slice(image, surface, info, fdepth)             \
   if (PVR_G5P_IMAGE_IS_SLICED_3D(image)) {                                  \
      (surface)->dev_addr.addr += (info)->depthPitch * (uint32_t)(fdepth);   \
      (surface)->mem_layout = PVR_MEMLAYOUT_TWIDDLED;                        \
      (surface)->depth = 1;                                                  \
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

/* pvr_arch_framebuffer.c: experimental omission of unreachable generic SPM
 * allocations only. Keep zeroed per-render states for unconditional cleanup.
 * The existing G9P and generic branches remain unchanged.
 */
#define PVR_G5P_HOOK_spm_scratch(device, rstate)                         \
   if (PVR_HAS_FEATURE(&(device)->pdevice->dev_info, apple_g5p) &&        \
       pvr_g5p_spm_skip()) { (rstate)->scratch_buffer = NULL; } else
#define PVR_G5P_HOOK_spm_render(device)                                 \
   if (!PVR_HAS_FEATURE(&(device)->pdevice->dev_info, apple_g5p) ||       \
       !pvr_g5p_spm_skip())

/* Application memory only: internal programs and control streams keep GPU type. */
#define PVR_G5P_HOOK_memory_bo_type(device, info, type)                     \
   if (PVR_HAS_FEATURE(&(device)->pdevice->dev_info, apple_g5p) &&           \
       (type) == PVR_WINSYS_BO_TYPE_GPU &&                                  \
       pvr_g5p_memory_is_large((info)->allocationSize))                     \
      (type) = pvr_g5p_memory_is_dedicated_image(info)                      \
                  ? PVR_G5P_BO_TYPE_IMAGE_DATA : PVR_G5P_BO_TYPE_DATA;

#define PVR_G5P_HOOK_memory_dedicated(device, requirements)                 \
   if (PVR_HAS_FEATURE(&(device)->pdevice->dev_info, apple_g5p))             \
      pvr_g5p_memory_dedicated(requirements);

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

/* pvr_formats.c, pvr_get_image_format_features2(): a colour format's
 * features are the A9's (pvr_g5p_color_format_features(): its format table,
 * not Rogue's), before the G9P and depth/stencil gates.
 */
#define PVR_G5P_HOOK_color_format_features(pdevice, vk_format, tiling, flags) \
   if (PVR_HAS_FEATURE(&(pdevice)->dev_info, apple_g5p) &&                 \
       !vk_format_is_depth_or_stencil(vk_format))                          \
      (flags) = pvr_g5p_color_format_features((vk_format), (tiling), (flags));

/* pvr_formats.c, pvr_get_image_format_properties(): the image types the A9
 * has of a colour format (pvr_g5p_image_format_check()), after the features
 */
#define PVR_G5P_HOOK_image_format(pdevice, info, tiling, result)                   \
   if (PVR_HAS_FEATURE(&(pdevice)->dev_info, apple_g5p) &&                        \
       ((result) = pvr_g5p_image_format_check((info), (tiling))) != VK_SUCCESS) \
      goto err_unsupported_format;

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
