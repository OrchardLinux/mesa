/* SPDX-License-Identifier: MIT */
#ifndef PVR_G5P_COMPUTE_STATE_H
#define PVR_G5P_COMPUTE_STATE_H
#ifdef PVR_APPLE_G5P
#define PVR_G5P_COMPUTE_STATE_FIELDS unsigned a9_local_dwords; bool a9_has_barriers; bool a9_device_fence; unsigned a9_num_images; struct { unsigned set,binding; } a9_images[1];
#define PVR_G5P_NATIVE_STORAGE(d) PVR_HAS_FEATURE((d), apple_g5p)
#else
#define PVR_G5P_COMPUTE_STATE_FIELDS
#define PVR_G5P_NATIVE_STORAGE(d) false
#endif
#endif
