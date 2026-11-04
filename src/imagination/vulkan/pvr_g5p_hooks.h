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

/* Textures and samplers (pvr_g5p_tex.h). */
#if defined(PVR_APPLE_G5P)
#   include "pvr_g5p_tex.h"
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

#endif /* PVR_G5P_HOOKS_H */
