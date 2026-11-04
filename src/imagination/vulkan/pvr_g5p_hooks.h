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
 * PVR_G5P_HOOK_<name> below. A hook ending in `else` is the G5P branch of the
 * if/else chain that follows it, in front of the G9P branch, as G9P's own
 * branches sit in front of the Rogue ones.
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

/* pvr_instance.c, pvr_drm_device_is_compatible(): the A9 kernel's DRM driver
 * name is a PowerVR node too.
 */
#define PVR_G5P_HOOK_drm_name(is_pvr, name) \
   is_pvr |= !strcmp(name, PVR_G5P_DRIVER_NAME);

/* winsys/pvr_winsys.c, pvr_winsys_create(): the apple_g5p winsys, over the A9
 * kernel's typed UAPI (winsys/apple_g5p).
 */
#define PVR_G5P_HOOK_winsys(version, render_fd, display_fd, alloc, ws, result) \
   if (strcmp((version)->name, PVR_G5P_DRIVER_NAME) == 0)                       \
      result = pvr_g5p_winsys_create(render_fd, display_fd, alloc, ws);         \
   else

/* pvr_arch_cmd_buffer.c, pvr_sub_cmd_gfx_job_init(): a G5P render's
 * background, end of tile and pass words (pvr_arch_g5p_job_init()).
 */
#define PVR_G5P_HOOK_job_init(dev_info, cmd_buffer, sub_cmd, hw_render, job, \
                              result)                                        \
   if (PVR_HAS_FEATURE(dev_info, apple_g5p)) {                               \
      result = pvr_arch_g5p_job_init(cmd_buffer, sub_cmd, hw_render, job);    \
      if (result != VK_SUCCESS)                                              \
         return result;                                                      \
   } else

/* pvr_arch_cmd_buffer.c, pvr_sub_cmd_gfx_job_init(): the background object
 * is pvr_arch_g5p_job_init()'s too.
 */
#define PVR_G5P_HOOK_job_bgnd(dev_info)          \
   if (PVR_HAS_FEATURE(dev_info, apple_g5p)) { \
   } else

/* pvr_arch_cmd_buffer.c, pvr_perform_start_of_render_attachment_clear():
 * clears are background objects (pvr_arch_g5p_job_init()), as on G9P.
 */
#define PVR_G5P_HOOK_start_clear(cmd_buffer)                                  \
   if (PVR_HAS_FEATURE(&(cmd_buffer)->device->pdevice->dev_info, apple_g5p)) \
      return;

/* pvr_arch_cmd_buffer.c, CmdBeginRenderPass2(): no "soft" background load
 * op either; the clear is the background object.
 */
#define PVR_G5P_HOOK_load_op(cmd_buffer)                                        \
   if (PVR_HAS_FEATURE(&(cmd_buffer)->device->pdevice->dev_info, apple_g5p)) { \
   } else

/* pvr_arch_pipeline.c, pvr_graphics_pipeline_destroy(): a G5P pipeline's
 * programs (pvr_arch_g5p_graphics_pipeline_compile()).
 */
#define PVR_G5P_HOOK_pipeline_destroy(device, allocator, gfx_pipeline) \
   if (PVR_HAS_FEATURE(&(device)->pdevice->dev_info, apple_g5p)) {     \
      pvr_arch_g5p_graphics_pipeline_destroy(device, gfx_pipeline);    \
      pvr_pipeline_finish(device, &(gfx_pipeline)->base);              \
      vk_free2(&(device)->vk.alloc, allocator, gfx_pipeline);          \
      return;                                                          \
   }

/* pvr_arch_pipeline.c, pvr_graphics_pipeline_compile(): a G5P pipeline is
 * compiled by the G5P backend from NIR (pvr_arch_g5p_gfx.c).
 */
#define PVR_G5P_HOOK_pipeline_compile(device, info, pipeline, state) \
   if (PVR_HAS_FEATURE(&(device)->pdevice->dev_info, apple_g5p))     \
      return pvr_arch_g5p_graphics_pipeline_compile(device,          \
                                                    info,            \
                                                    pipeline,        \
                                                    state);

/* pvr_arch_cmd_buffer.c, CmdDraw(): the draw's VDM/PPP words and programs
 * (pvr_arch_g5p_cmd_draw()), from CmdDraw's own parameters (vertexCount,
 * instanceCount, firstVertex, firstInstance); then to its label `end`.
 */
#define PVR_G5P_HOOK_draw(cmd_buffer, end)                                     \
   if (PVR_HAS_FEATURE(&(cmd_buffer)->device->pdevice->dev_info, apple_g5p)) { \
      const struct pvr_g5p_draw g5p_draw_ = {                                  \
         .count = vertexCount,                                                 \
         .instances = instanceCount,                                           \
         .first = firstVertex,                                                 \
         .first_instance = firstInstance,                                      \
      };                                                                       \
      VkResult g5p_result_ = pvr_arch_g5p_cmd_draw(cmd_buffer, &g5p_draw_);    \
      if (g5p_result_ != VK_SUCCESS)                                           \
         vk_command_buffer_set_error(&(cmd_buffer)->vk, g5p_result_);          \
      goto end;                                                                \
   }

/* pvr_arch_cmd_buffer.c, CmdDrawIndexed(): refused by name for now
 * (pvr_arch_g5p_cmd_draw_indexed()); then to its label `end`.
 */
#define PVR_G5P_HOOK_draw_indexed(cmd_buffer, end)                             \
   if (PVR_HAS_FEATURE(&(cmd_buffer)->device->pdevice->dev_info, apple_g5p)) { \
      vk_command_buffer_set_error(&(cmd_buffer)->vk,                           \
                                  pvr_arch_g5p_cmd_draw_indexed(cmd_buffer));  \
      goto end;                                                                \
   }

#endif /* PVR_G5P_HOOKS_H */
