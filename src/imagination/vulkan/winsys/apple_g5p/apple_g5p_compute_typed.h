/* SPDX-License-Identifier: MIT */
#ifndef APPLE_G5P_COMPUTE_TYPED_H
#define APPLE_G5P_COMPUTE_TYPED_H
#include "apple_g5p_drm.h"
#ifndef DRM_G5P_CAP_TYPED_COMPUTE
/**
 * DRM_G5P_CAP_TYPED_COMPUTE - compute jobs on render contexts (SUBMIT_COMPUTE_TYPED)
 *
 * Reported together with TYPED_RENDER_V3 (a9_gpu_drm.render_user_compute, default on).  A
 * render context (DRM_G5P_CONTEXT_RENDER) runs compute jobs beside its TYPED_V3 render jobs,
 * in the same VM, in submission order, serialized with every other context's jobs.  The
 * kernel builds the firmware's compute work (W, microsequence, stamps, channel, slot); the
 * job carries the CDM control stream and the register images iOS's userspace hands its
 * kernel, which the kernel bounds but does not decode.  Compute needs an epoch a render job
 * brought up: on a device no render job has started, SUBMIT_COMPUTE_TYPED fails with EAGAIN.
 * Specification: docs/gpu/experiments/compute.md.
 */
#define DRM_G5P_CAP_TYPED_COMPUTE (1ULL << 30)

#define DRM_G5P_COMPUTE_MAX_STREAM_BYTES 0x10000
#define DRM_G5P_COMPUTE_STREAM_TERMINATE 0x80000000U

/* A PDS program reference: code and data offsets from GET_RENDER_INFO.pds_base (16-byte
 * aligned; code in a READ resource whose BO carries DRM_G5P_BO_EXEC, data in a READ resource)
 * and its opaque control word (Rogue's PDS1 position: sizes and temps).
 */
struct drm_g5p_compute_pds {
	__u32 code;
	__u32 data;
	__u32 control;
	__u32 pad;		/* zero */
};

/*
 * One compute job.  Every *_va is a GPU VA inside a listed resource with the stated access.
 *   cdm_va/cdm_size  the CDM control stream (W+0x70, R0+0x480): READ, 4-byte aligned,
 *                    cdm_size <= MAX_STREAM_BYTES, in a native (not PRIME-imported) BO
 *   cdm_end          bytes from cdm_va past the stream's terminate word (the dword at
 *                    cdm_va + cdm_end - 4 is STREAM_TERMINATE), <= cdm_size
 *   state_va[4]      CDM context-state words (W+0x1ec..0x204), READ|WRITE, 4-byte aligned
 *   state_tail_va    CDM context-state tail (W+0x60, R0+0x498; W+0x20c), READ|WRITE, 16 bytes,
 *                    16-byte aligned
 *   config[2]        CDM configuration words (W+0x10/0x14, R0+0x4c0/0x4c8), opaque
 *   store            context-store program (W+0x21c/0x224, R0+0x4a8/0x4b0)
 *   resume           context-resume program (W+0x22c/0x234)
 *   terminate        terminate program (W+0x68, R0+0x4b8); control must be zero
 */
struct drm_g5p_compute_pass {
	__u32 size;		/* sizeof(struct drm_g5p_compute_pass) */
	__u32 flags;		/* zero */
	__u64 cdm_va;
	__u32 cdm_size;
	__u32 cdm_end;
	__u64 state_va[4];
	__u64 state_tail_va;
	__u32 config[2];
	struct drm_g5p_compute_pds store;
	struct drm_g5p_compute_pds resume;
	struct drm_g5p_compute_pds terminate;
	__u64 reserved[4];	/* zero */
};

/* Command 0x15.  resources lists every BO the job references (1..RENDER_MAX_RESOURCES,
 * TYPED_V1 access-mask kinds, IMPLICIT and CACHE included); pass_size is exactly
 * sizeof(struct drm_g5p_compute_pass).  Syncs, job ids, results and the error contract are
 * SUBMIT_RENDER_TYPED's: every error other than EINTR/EFAULT on copy-out precedes fence
 * replacement and hardware publication.
 */
struct drm_g5p_submit_compute_typed {
	__u32 context;		/* a DRM_G5P_CONTEXT_RENDER context */
	__u32 flags;		/* zero */
	__u64 resources;	/* struct drm_g5p_render_resource[] */
	__u32 resource_count;
	__u32 sync_count;
	__u64 syncs;		/* struct drm_g5p_sync[]: exactly one SIGNAL, at most 16 inputs */
	__u64 pass;		/* struct drm_g5p_compute_pass */
	__u32 pass_size;
	__u32 pad;		/* zero */
	__u64 job_id;		/* output */
	__u64 reserved[2];	/* zero */
};

#define DRM_G5P_SUBMIT_COMPUTE_TYPED 0x15

#define DRM_IOCTL_G5P_SUBMIT_COMPUTE_TYPED \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_SUBMIT_COMPUTE_TYPED, \
		 struct drm_g5p_submit_compute_typed)


#endif
#endif
