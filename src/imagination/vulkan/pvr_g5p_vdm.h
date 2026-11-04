/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): a render's VDM control stream as the kernel takes
 * it, one contiguous range ending in STREAM_TERMINATE, also when pvr_csb
 * recorded it in several linked blocks (pvr_arch_g5p_vdm.c). Built only with
 * -Dimagination-g5p-experimental=true.
 */

#ifndef PVR_G5P_VDM_H
#define PVR_G5P_VDM_H

#include <stdint.h>
#include <vulkan/vulkan_core.h>

#include "pvr_macros.h"

#ifdef PVR_PER_ARCH

struct pvr_cmd_buffer;
struct pvr_csb;

/* The stream's address and length through its terminate: the first block
 * itself when there is one, else one upload of the blocks' commands without
 * their chaining links; a stream past the kernel's limit is refused */
VkResult PVR_PER_ARCH(g5p_vdm_stream)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_csb *csb, uint64_t *va,
                                      uint32_t *bytes);

#   define pvr_arch_g5p_vdm_stream PVR_PER_ARCH(g5p_vdm_stream)

#endif /* PVR_PER_ARCH */

#endif /* PVR_G5P_VDM_H */
