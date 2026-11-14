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

/* The device-local heap: ram_share (the RAM share Mesa's unified-memory
 * drivers report, os_get_gpu_heap_size()), bounded by what the kernel pins in
 * one VM (G5P_WS_VM_PIN_MAX). The arena may grow to the same bound. */
uint64_t pvr_g5p_winsys_heap_size(struct pvr_winsys *ws, uint64_t ram_share);

/* A VkBuffer or VkImage bound (bytes > 0) or released (bytes < 0) memory:
 * counted for the PVR_G5P_ARENA_LOG statistics only, which then tell the
 * application's images from its buffers (Zink's upload staging). */
void pvr_g5p_winsys_bound(struct pvr_winsys *ws, bool image, int64_t bytes);

/* The backing BO of an application GPU address. The range is used only to
 * deduplicate conservative read tracking, never to infer cache coherency. */
bool pvr_g5p_winsys_bo_range(struct pvr_winsys *ws, uint64_t addr,
                             uint64_t *base, uint64_t *size);

/* Resolve a bound GPU address through the winsys's permanent BO mapping.
 * This never returns or changes the application's vkMapMemory mapping. */
bool pvr_g5p_winsys_vma_cpu_addr(struct pvr_winsys_vma *vma,
                                 uint64_t addr,
                                 uint64_t size,
                                 uint8_t **ptr_out,
                                 uint64_t *available_out);

#endif /* PVR_G5P_WINSYS_H */
