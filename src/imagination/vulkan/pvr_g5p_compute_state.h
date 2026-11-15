/* SPDX-License-Identifier: MIT */
#ifndef PVR_G5P_COMPUTE_STATE_H
#define PVR_G5P_COMPUTE_STATE_H
#define PVR_G5P_COMPUTE_STATE_FIELDS unsigned a9_bank1_regs; bool a9_bodyless; bool a9_unit_z; unsigned a9_local_dwords; bool a9_has_barriers; bool a9_device_fence; unsigned a9_num_images; struct { unsigned set,binding; } a9_images[1]; \
   unsigned a9_num_uploads; struct { unsigned char kind, set, binding, element, combined, level, common, dwords; } a9_uploads[7];
#define PVR_G5P_NATIVE_STORAGE(d) PVR_HAS_FEATURE((d), apple_g5p)
#endif
