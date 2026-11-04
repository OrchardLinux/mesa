/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): the CPU copies that know the A9's layouts
 * (pvr_arch_g5p_layout.c). Built only with -Dimagination-g5p-experimental=true;
 * reached from the shared files through pvr_g5p_hooks.h.
 */

#ifndef PVR_G5P_LAYOUT_H
#define PVR_G5P_LAYOUT_H

#include <vulkan/vulkan_core.h>

#include "pvr_macros.h"

struct pvr_device;
struct pvr_sub_cmd_transfer;
struct vk_sync;

#ifdef PVR_PER_ARCH

/* The transfer sub-commands of G5P (pvr_arch_queue.c): G9P's CPU copies,
 * unless one of them needs what only this file's copies do (a blit from a
 * block-compressed image, decoded per 4x4 block) */
VkResult PVR_PER_ARCH(g5p_transfer_submit)(struct pvr_device *device, struct pvr_sub_cmd_transfer *sub_cmd,
                                           struct vk_sync *wait_sync, struct vk_sync *signal_sync);
#   define pvr_arch_g5p_transfer_submit PVR_PER_ARCH(g5p_transfer_submit)

#endif /* PVR_PER_ARCH */

#endif /* PVR_G5P_LAYOUT_H */
