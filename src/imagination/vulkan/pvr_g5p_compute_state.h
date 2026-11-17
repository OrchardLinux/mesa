/* SPDX-License-Identifier: MIT */
#ifndef PVR_G5P_COMPUTE_STATE_H
#define PVR_G5P_COMPUTE_STATE_H
#define PVR_G5P_COMPUTE_STATE_FIELDS unsigned shared_spill_stride; int shared_spill_logged; char a9_pipeline_id[13]; bool a9_capture_stub; bool a9_compute_log; unsigned a9_log_local[3], a9_log_shared; unsigned a9_bank1_regs; bool a9_bodyless; bool a9_wgid_z; unsigned a9_local_dwords; bool a9_has_barriers; bool a9_device_fence; unsigned a9_num_images; struct { unsigned set,binding; } a9_images[1]; \
   unsigned a9_num_uploads; struct { unsigned char kind, set, binding, element, combined, level, common, dwords; uint32_t value; } a9_uploads[10];
#define PVR_G5P_NATIVE_STORAGE(d) PVR_HAS_FEATURE((d), apple_g5p)
#endif
