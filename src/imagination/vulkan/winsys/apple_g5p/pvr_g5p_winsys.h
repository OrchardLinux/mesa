/* SPDX-License-Identifier: MIT */
/*
 * The apple_g5p winsys: pvr_winsys_ops over our A9 kernel's typed UAPI
 * (g5p_drm_transport.h). Everything kernel-specific stays below this
 * interface, so the UAPI can change (a joint one with the A10 port's kernel)
 * without touching the rest of the driver.
 */

#ifndef PVR_G5P_WINSYS_H
#define PVR_G5P_WINSYS_H

#include <stdbool.h>
#include <vulkan/vulkan_core.h>

#define PVR_G5P_DRIVER_NAME "apple_g5p"

struct pvr_winsys;
struct pvr_winsys_vma;

VkResult pvr_g5p_winsys_create(int render_fd, int display_fd,
                               const VkAllocationCallbacks *alloc,
                               struct pvr_winsys **ws_out);

/* The kernel takes 4-sample passes (TYPED_V3 and DRM_G5P_CAP_RENDER_MSAA) */
bool pvr_g5p_winsys_msaa(struct pvr_winsys *ws);

/* Resolve a bound GPU address through the winsys's permanent BO mapping.
 * This never returns or changes the application's vkMapMemory mapping. */
bool pvr_g5p_winsys_vma_cpu_addr(struct pvr_winsys_vma *vma,
                                 uint64_t addr,
                                 uint64_t size,
                                 uint8_t **ptr_out,
                                 uint64_t *available_out);

#endif /* PVR_G5P_WINSYS_H */
