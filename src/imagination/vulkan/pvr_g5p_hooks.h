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

#endif /* PVR_G5P_HOOKS_H */
