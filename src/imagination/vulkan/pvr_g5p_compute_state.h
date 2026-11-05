/* SPDX-License-Identifier: MIT */
#ifndef PVR_G5P_COMPUTE_STATE_H
#define PVR_G5P_COMPUTE_STATE_H
#ifdef PVR_APPLE_G5P
#define PVR_G5P_COMPUTE_STATE_FIELDS unsigned a9_local_dwords; bool a9_has_barriers; bool a9_device_fence;
#else
#define PVR_G5P_COMPUTE_STATE_FIELDS
#endif
#endif
