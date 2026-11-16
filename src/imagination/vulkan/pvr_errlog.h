/* SPDX-License-Identifier: MIT */
#ifndef PVR_ERRLOG_H
#define PVR_ERRLOG_H
#include <vulkan/vulkan_core.h>
struct pvr_device;
VkResult pvr_errlog_install(struct pvr_device *device);
#endif
