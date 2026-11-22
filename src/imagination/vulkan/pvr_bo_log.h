/* SPDX-License-Identifier: MIT */
#ifndef PVR_BO_LOG_H
#define PVR_BO_LOG_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

struct pvr_bo;
struct pvr_buffer;
struct pvr_device;
struct pvr_device_memory;
struct pvr_image;
struct pvr_winsys_heap;

void pvr_bo_log_device_init(struct pvr_device *device);
void pvr_bo_log_device_finish(struct pvr_device *device);
void pvr_bo_log_summary(struct pvr_device *device, const char *why);
/* PVR_BO_LOG_PERIOD_S's summary when the period has passed (called per queue submission) */
void pvr_bo_log_tick(struct pvr_device *device);
/* a render with a depth/stencil attachment, and whether its ZLS loads or stores it */
void pvr_bo_log_zls(struct pvr_device *device, uint32_t w, uint32_t h, bool load, bool store);
void pvr_bo_log_driver_alloc(struct pvr_device *device,
                             const char *tag,
                             struct pvr_winsys_heap *heap,
                             uint64_t size,
                             uint64_t flags,
                             const struct pvr_bo *bo,
                             VkResult result);
void pvr_bo_log_memory_alloc(struct pvr_device *device,
                             const VkMemoryAllocateInfo *info,
                             const struct pvr_device_memory *mem,
                             VkResult result);
void pvr_bo_log_free(struct pvr_device *device, const void *key);
void pvr_bo_log_image(struct pvr_device *device,
                      const struct pvr_image *image,
                      const struct pvr_device_memory *mem,
                      uint64_t offset);
void pvr_bo_log_buffer(struct pvr_device *device,
                       const struct pvr_buffer *buffer,
                       const struct pvr_device_memory *mem,
                       uint64_t offset);
void pvr_bo_log_unbind(struct pvr_device *device, const void *object);

#endif /* PVR_BO_LOG_H */
