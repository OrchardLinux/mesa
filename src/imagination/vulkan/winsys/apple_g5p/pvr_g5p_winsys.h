/* SPDX-License-Identifier: MIT */
/*
 * The apple_g5p winsys: pvr_winsys_ops over our A9 kernel's typed UAPI
 * (g5p_drm_transport.h). Everything kernel-specific stays below this
 * interface, so the UAPI can change (a joint one with Henrik's A10 kernel)
 * without touching the rest of the driver.
 */

#ifndef PVR_G5P_WINSYS_H
#define PVR_G5P_WINSYS_H

#include <vulkan/vulkan_core.h>

#define PVR_G5P_DRIVER_NAME "apple_g5p"

struct pvr_winsys;

VkResult pvr_g5p_winsys_create(int render_fd, int display_fd,
                               const VkAllocationCallbacks *alloc,
                               struct pvr_winsys **ws_out);

#endif /* PVR_G5P_WINSYS_H */
