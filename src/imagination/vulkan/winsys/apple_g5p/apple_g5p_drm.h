/* SPDX-License-Identifier: (GPL-2.0-only WITH Linux-syscall-note) OR MIT */
#ifndef _APPLE_G5P_DRM_H_
#define _APPLE_G5P_DRM_H_

#include "drm.h"

/* Experimental v1: a G5P winsys, not binary compatibility with pvr_drm.h. */
#define DRM_G5P_UAPI_VERSION 1
#define DRM_G5P_CAP_BO_CPU (1ULL << 0)
#define DRM_G5P_CAP_VM_PREPARE (1ULL << 1)
#define DRM_G5P_CAP_SYNCOBJ (1ULL << 2)
#define DRM_G5P_CAP_CL_DEPENDENCY (1ULL << 3)
#define DRM_G5P_CAP_COMPUTE (1ULL << 4)
#define DRM_G5P_CAP_RENDER (1ULL << 5)
#define DRM_G5P_CAP_PRIME (1ULL << 6)
#define DRM_G5P_CAP_CODE_BASE_VERIFIED (1ULL << 7)
#define DRM_G5P_CAP_VM_LIVE (1ULL << 8)
#define DRM_G5P_CAP_CONTEXT_LIVE (1ULL << 9)
#define DRM_G5P_CAP_JOB_STATUS (1ULL << 10)
#define DRM_G5P_CAP_CONTEXT_STATUS (1ULL << 11)
#define DRM_G5P_CAP_CONTEXT_ATTACH_PROBE (1ULL << 12)
#define DRM_G5P_CAP_FIXED_CDM_PROBE (1ULL << 14)
#define DRM_G5P_CAP_FAULT_INFO (1ULL << 13)

/* LIVE means the owned user catalogue has completed firmware attachment. */
#define DRM_G5P_CONTEXT_LIVE 3
#define DRM_G5P_CONTEXT_HOST_ONLY 0
#define DRM_G5P_CONTEXT_LOST 1
/* Empty-VM diagnostic attachment only; does not imply COMPUTE/VM_LIVE. */
#define DRM_G5P_CONTEXT_ATTACHED_PROBE 2
struct drm_g5p_get_context_status {
	__u32 context;
	__u32 flags;
	__u32 state;
	__u32 map_count;
	__u64 mapped_bytes;
	__u64 reserved;
};

/* First terminal firmware event, retained independently of last_event. */
#define DRM_G5P_FAULT_NONE 0
#define DRM_G5P_FAULT_RECOVERY 1
#define DRM_G5P_FAULT_PAGE_REQUEST 2
#define DRM_G5P_FAULT_CHANNEL 3
#define DRM_G5P_FAULT_UNKNOWN 4
#define DRM_G5P_FAULT_VALID_CHANNEL (1U << 0)
struct drm_g5p_get_fault_info {
	__u32 flags;
	__u32 kind;
	__u64 event_serial; /* One-based arrival order; zero if no fault. */
	__u32 valid;
	__u32 channel_class; /* Event8: 0 memory failure, 1 killed, others opaque. */
	__u32 channel_token; /* Firmware token, not a DRM context handle. */
	__u32 channel_stamp;
	__u32 raw[11]; /* Includes unknown fields; no decoded fault VA is claimed. */
	__u32 reserved;
};

struct drm_g5p_get_info {
	__u32 size;
	__u32 flags;
	__u32 version;
	__u32 gpu_page_shift;
	__u64 capabilities;
	__u64 va_start;
	__u64 va_end;
	__u64 reserved_start;
	__u64 reserved_end;
	__u64 code_base; /* Zero until CODE_BASE_VERIFIED. */
	__u64 max_bo_size;
	__u32 bvnc[4]; /* Static target, not a new MMIO measurement. */
	__u64 reserved[2];
};

/* GPU access roles; CPU initialization/mmap is permitted for every role. */
#define DRM_G5P_BO_READ (1U << 0)
#define DRM_G5P_BO_WRITE (1U << 1)
#define DRM_G5P_BO_EXEC (1U << 2)
struct drm_g5p_create_bo {
	__u64 size;
	__u32 flags;
	__u32 handle;
};

struct drm_g5p_bo_mmap_offset {
	__u32 handle;
	__u32 flags;
	__u64 offset;
};

#define DRM_G5P_BO_SYNC_TO_DEVICE 0
#define DRM_G5P_BO_SYNC_TO_CPU 1
struct drm_g5p_bo_sync {
	__u32 handle;
	__u32 op;
	__u64 offset;
	__u64 size;
};

struct drm_g5p_create_vm {
	__u32 flags;
	__u32 handle;
};

struct drm_g5p_destroy_vm {
	__u32 handle;
	__u32 flags;
};

#define DRM_G5P_VM_MAP 0
#define DRM_G5P_VM_UNMAP 1
struct drm_g5p_vm_bind {
	__u32 vm;
	__u32 bo;
	__u64 va;
	__u64 size;
	__u64 offset;
	__u32 op;
	__u32 flags;
};

#define DRM_G5P_CONTEXT_COMPUTE 0
struct drm_g5p_create_context {
	__u32 vm;
	__u32 type;
	__u32 flags;
	__u32 handle;
};

struct drm_g5p_destroy_context {
	__u32 handle;
	__u32 flags;
};

/* A point of zero uses a binary syncobj; nonzero uses a timeline point. */
#define DRM_G5P_SYNC_SIGNAL (1U << 0)
struct drm_g5p_sync {
	__u32 handle;
	__u32 flags;
	__u64 point;
};

#define DRM_G5P_JOB_CL_DEPENDENCY 0
#define DRM_G5P_JOB_COMPUTE 1
#define DRM_G5P_JOB_CONTEXT_ATTACH_PROBE 2
/* Privileged one-shot kernel fixture; all payload/BO fields must be zero.
 * The empty user context authorizes ownership only; its VM is not attached.
 * Does not advertise general COMPUTE or attach user-provided GPU addresses.
 */
#define DRM_G5P_JOB_FIXED_CDM_PROBE 3
struct drm_g5p_submit {
	__u32 context;
	__u32 type;
	__u32 flags;
	__u32 bo_count;
	__u64 bo_handles; /* Array of u32 GEM handles. */
	__u64 cdm;
	__u64 terminate;
	__u64 scratch[12]; /* Reserved; v1 dependency requires all zero. */
	__u64 syncs; /* Array of struct drm_g5p_sync; exactly one signal. */
	__u32 sync_count;
	__u32 pad;
	__u64 job_id; /* Output; zero on input. */
};

struct drm_g5p_get_status {
	__u32 flags;
	__u32 alive;
	__u32 failed;
	__s32 last_error;
	__u64 last_job_id;
	__u32 events;
	__u32 irq_count;
	__u32 stamp_notifications[4];
	__u32 last_event[11];
	__u32 reserved;
};

/* Geometry is all zero/NONE while CODE_BASE_VERIFIED is clear. */
#define DRM_G5P_BASE_NONE 0
#define DRM_G5P_BASE_DEVICE 1
#define DRM_G5P_BASE_VM 2
#define DRM_G5P_BASE_CONTEXT 3
struct drm_g5p_get_exec_info {
	__u32 size;
	__u32 flags;
	__u32 base_scope;
	__u32 pad;
	__u64 usc_base;
	__u64 pds_base;
	__u64 window_size;
	__u64 allocation_alignment;
	__u32 usc_offset_bits;
	__u32 pds_offset_bits;
	__u64 reserved[3];
};

#define DRM_G5P_JOB_PENDING 0
#define DRM_G5P_JOB_COMPLETE 1
struct drm_g5p_get_job_status {
	__u64 job_id;
	__u32 flags;
	__u32 state;
	__s32 error; /* -EINPROGRESS while pending; immutable after completion. */
	__u32 pad;
	__u64 reset_generation; /* Zero: no reset/recovery support. */
};

#define DRM_G5P_GET_INFO 0x00
#define DRM_G5P_CREATE_BO 0x01
#define DRM_G5P_BO_MMAP_OFFSET 0x02
#define DRM_G5P_BO_SYNC 0x03
#define DRM_G5P_CREATE_VM 0x04
#define DRM_G5P_DESTROY_VM 0x05
#define DRM_G5P_VM_BIND 0x06
#define DRM_G5P_CREATE_CONTEXT 0x07
#define DRM_G5P_DESTROY_CONTEXT 0x08
#define DRM_G5P_SUBMIT 0x09
#define DRM_G5P_GET_STATUS 0x0a
#define DRM_G5P_GET_EXEC_INFO 0x0b
#define DRM_G5P_GET_JOB_STATUS 0x0c
#define DRM_G5P_GET_CONTEXT_STATUS 0x0d
#define DRM_G5P_GET_FAULT_INFO 0x0e

#define DRM_IOCTL_G5P_GET_INFO \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_GET_INFO, struct drm_g5p_get_info)
#define DRM_IOCTL_G5P_CREATE_BO \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_CREATE_BO, struct drm_g5p_create_bo)
#define DRM_IOCTL_G5P_BO_MMAP_OFFSET \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_BO_MMAP_OFFSET, struct drm_g5p_bo_mmap_offset)
#define DRM_IOCTL_G5P_BO_SYNC \
	DRM_IOW(DRM_COMMAND_BASE + DRM_G5P_BO_SYNC, struct drm_g5p_bo_sync)
#define DRM_IOCTL_G5P_CREATE_VM \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_CREATE_VM, struct drm_g5p_create_vm)
#define DRM_IOCTL_G5P_DESTROY_VM \
	DRM_IOW(DRM_COMMAND_BASE + DRM_G5P_DESTROY_VM, struct drm_g5p_destroy_vm)
#define DRM_IOCTL_G5P_VM_BIND \
	DRM_IOW(DRM_COMMAND_BASE + DRM_G5P_VM_BIND, struct drm_g5p_vm_bind)
#define DRM_IOCTL_G5P_CREATE_CONTEXT \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_CREATE_CONTEXT, struct drm_g5p_create_context)
#define DRM_IOCTL_G5P_DESTROY_CONTEXT \
	DRM_IOW(DRM_COMMAND_BASE + DRM_G5P_DESTROY_CONTEXT, struct drm_g5p_destroy_context)
#define DRM_IOCTL_G5P_SUBMIT \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_SUBMIT, struct drm_g5p_submit)
#define DRM_IOCTL_G5P_GET_STATUS \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_GET_STATUS, struct drm_g5p_get_status)

#define DRM_IOCTL_G5P_GET_EXEC_INFO \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_GET_EXEC_INFO, struct drm_g5p_get_exec_info)
#define DRM_IOCTL_G5P_GET_JOB_STATUS \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_GET_JOB_STATUS, struct drm_g5p_get_job_status)

#define DRM_IOCTL_G5P_GET_CONTEXT_STATUS \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_GET_CONTEXT_STATUS, struct drm_g5p_get_context_status)

#define DRM_IOCTL_G5P_GET_FAULT_INFO \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_GET_FAULT_INFO, struct drm_g5p_get_fault_info)

#define DRM_G5P_CAP_BOUNDED_COMPUTE (1ULL << 15)
#define DRM_G5P_SUBMIT_COMPUTE 0x0f
#define DRM_G5P_COMPUTE_MAX_BOS 256
#define DRM_G5P_COMPUTE_MAX_SYNCS 17
#define DRM_G5P_COMPUTE_MAX_CDM_BYTES 0x4000

/* All flags/reserved/job_id zero on input. Pointer arrays copied once before
 * acceptance. bo_handles is u32[]; syncs is struct drm_g5p_sync[].
 * cdm_size includes the final 0x80000000 terminate dword, at cdm_va+size-4.
 * This initial dialect accepts only the characterized first-store stream;
 * a byte bound alone does not validate PDS/USC/data references or their roles.
 * No firmware, META, MMIO, physical address or raw work-object inputs.
 */
struct drm_g5p_submit_compute {
	__u32 context;
	__u32 flags;
	__u64 cdm_va;
	__u64 cdm_size;
	__u64 terminate_va;
	__u64 bo_handles;
	__u32 bo_count;
	__u32 sync_count;
	__u64 syncs;
	__u64 job_id;
	__u64 reserved[2];
};

#define DRM_IOCTL_G5P_SUBMIT_COMPUTE \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_SUBMIT_COMPUTE, struct drm_g5p_submit_compute)

/* The render VM layout and limits. GET_RENDER_INFO.flags is output only (input
 * zero): VARIANT_ACTIVE means a non-default job behaviour is latched. Render
 * programs are 32-bit offsets from usc_base (== pds_base), within code_window.
 */
#define DRM_G5P_RENDER_INFO_VARIANT_ACTIVE (1U << 0)
struct drm_g5p_get_render_info {
	__u32 size;			/* sizeof, exact */
	__u32 flags;			/* in: zero; out: RENDER_INFO_* */
	__u32 profiles;			/* bit N: profile N accepted */
	__u32 max_render_targets;
	__u64 private_va_start;		/* kernel window in render VMs */
	__u64 private_va_end;
	__u64 user_va_min;		/* lowest mappable VA in a render VM */
	__u64 usc_base;			/* common USC/PDS base, 32-bit offsets */
	__u64 pds_base;
	__u64 code_window;		/* bytes above usc_base usable for code */
	__u32 pm_blocks;		/* parameter buffer blocks per context */
	__u32 pm_block_bytes;
	__u32 max_queued_jobs;
	__u32 max_vdm_bytes;
	__u32 job_timeout_ms;
	__u32 retained_results;
	__u32 max_render_width;		/* 0: 64x64 only */
	__u32 max_render_height;
	__u32 max_pm_blocks;		/* PM pool growth bound per context */
	__u32 pad;
	__u64 hw_va_limit;		/* every VA the TA/3D read ends at or below it; 0: no statement */
	__u64 reserved;			/* in: zero */
};

#define DRM_G5P_GET_RENDER_INFO 0x13
#define DRM_IOCTL_G5P_GET_RENDER_INFO \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_GET_RENDER_INFO, struct drm_g5p_get_render_info)

/*
 * Typed render submission (TYPED_V3). The kernel owns every firmware record:
 * a pass names its streams and programs by address, the kernel validates each
 * against the job's resource list and the render layout above, and builds the
 * records itself. A render context (DRM_G5P_CONTEXT_RENDER) runs the passes; a
 * render target (CREATE_RENDER_TARGET) is the parameter-buffer geometry of one
 * width and height.
 */
#define DRM_G5P_CAP_TYPED_RENDER (1ULL << 20)
#define DRM_G5P_CAP_RENDER_USER_GEOMETRY (1ULL << 21)
#define DRM_G5P_CAP_TYPED_RENDER_V3 (1ULL << 26)
#define DRM_G5P_CAP_RENDER_PIXEL_OUTPUT_WIDTH (1ULL << 29)
#define DRM_G5P_CAP_RENDER_SKIP_EMPTY_TILES (1ULL << 32)

#define DRM_G5P_CONTEXT_RENDER 1

#define DRM_G5P_RENDER_PROFILE_TYPED_V1 2
#define DRM_G5P_RENDER_PROFILE_TYPED_V3 4

/* drm_g5p_render_resource.kind: how the job accesses the range */
#define DRM_G5P_RENDER_ACCESS_READ (1U << 0)
#define DRM_G5P_RENDER_ACCESS_WRITE (1U << 1)
#define DRM_G5P_RENDER_ACCESS_IMPLICIT (1U << 2)
#define DRM_G5P_RENDER_ACCESS_CACHE (1U << 3)

#define DRM_G5P_RENDER_MAX_RESOURCES 256
#define DRM_G5P_RENDER_MAX_SYNCS 17
#define DRM_G5P_RENDER_MAX_VDM_BYTES 0x10000
#define DRM_G5P_RENDER_MAX_PROGRAM_BYTES 0x1000
#define DRM_G5P_RENDER_ZLS_REGS 6
#define DRM_G5P_RENDER_MAX_TABLE_ENTRIES 65536

/* One range of one BO the job reads or writes: [va, va + size) inside the BO
 * at offset. */
struct drm_g5p_render_resource {
	__u32 handle;
	__u32 kind;			/* DRM_G5P_RENDER_ACCESS_* */
	__u64 offset;
	__u64 size;
	__u64 va;
};

/* The TYPED_V1 pass (360 bytes). Program words are offsets from pds_base. */
struct drm_g5p_render_pass {
	__u32 size;			/* TYPED_V3: sizeof(struct drm_g5p_render_pass_v3) */
	__u32 flags;
	__u64 vdm_va;			/* the VDM control stream */
	__u32 vdm_size;
	__u32 vdm_control;		/* TYPED_V3: zero */
	__u64 scratch_va;
	__u64 scratch_size;
	__u64 scratch_ptr[6];
	__u64 pool7_va;			/* V1 dialect only; the V2 extension names the tables */
	__u64 pool7_size;
	__u64 pool8_va;
	__u64 pool8_size;
	__u32 context_pds[2][6];	/* TA and 3D context store PDS words */
	__u32 resume_pds[2][4];		/* TA and 3D context resume PDS words */
	__u32 ta_pds[4];
	__u32 background[8];		/* PDS_BGRND0..3 */
	__u32 eot[4];			/* the end-of-tile (pixel event) PDS */
	__u32 eot_resume[4];
	__u32 restore[4];
	/* TYPED_V3 with DRM_G5P_CAP_RENDER_MSAA (kernel a9/gpu-msaa): 0 or 1
	 * sample needs a 1-sample render target, 4 a 4-sample one; zero keeps
	 * the one-sample pass */
	__u32 samples;			/* @296 */
	__u32 msaa_flags;		/* @300 DRM_G5P_RENDER_MSAA_* */
	__u64 sample_positions;		/* @304 x0 | y0 << 4 | x1 << 8 ..., 1/16 px */
	__u64 reserved[6];
};

/* RESOLVE: the end of tile's PBE averages the samples into a one-sample
 * target, and the kernel sets the 3D's resolve words with it (ISP_CTL bit
 * 17, W86c). It needs 4 samples. */
#define DRM_G5P_CAP_RENDER_MSAA (1ULL << 31)
#define DRM_G5P_RENDER_MSAA_RESOLVE (1U << 0)

/* The V2 extension (128 bytes). Non-zero selects the V2 dialect: the pools
 * become the named scissor and depth-bias tables, the pixel output width moves
 * here, and the ZLS words and background object values reach the 3D. */
struct drm_g5p_render_pass_ext {
	__u64 scissor_va;		/* ISP_SCISSOR_BASE, 8-byte entries */
	__u32 scissor_count;		/* 1..65536 */
	__u32 dbias_count;		/* 1..65536 */
	__u64 dbias_va;			/* ISP_DBIAS_BASE, {constant, slope, clamp} f32 */
	__u64 background_texture;	/* PDS_BGRND1_BASE, an offset from pds_base */
	__u32 bgobj_depth;		/* ISP_BGOBJDEPTH */
	__u32 bgobj_vals;		/* ISP_BGOBJVALS: stencil [7:0], mask [8], bgtag [9] */
	__u32 pixel_output_width;	/* 2, 4 or 8 */
	__u32 zls_address_mask;		/* bit i: zls[i] is an address in a listed RW range */
	__u64 zls[DRM_G5P_RENDER_ZLS_REGS]; /* ZLSCTL, ZLOAD, ZSTORE, STENCIL_LOAD, STENCIL_STORE, ZLS_PIXELS */
	__u64 reserved[4];
};

#define DRM_G5P_RENDER_PASS_SKIP_EMPTY_TILES (1U << 0)

/* The TYPED_V3 pass (536 bytes): a V1 pass, the V2 extension (all zero: the V1
 * dialect), the TA program and the end of the VDM stream. */
struct drm_g5p_render_pass_v3 {
	struct drm_g5p_render_pass base;
	struct drm_g5p_render_pass_ext ext;
	__u64 ta_program_va;		/* absolute VA */
	__u32 ta_program_size;		/* 16..0x1000, a multiple of 16 */
	__u32 vdm_end;			/* bytes from vdm_va past STREAM_TERMINATE */
	__u32 pixel_output_width;	/* V1 dialect: 0 (= 2), 4 or 8 */
	__u32 flags;			/* DRM_G5P_RENDER_PASS_* */
	__u64 reserved[3];
};

struct drm_g5p_submit_render_typed {
	__u32 context;			/* a DRM_G5P_CONTEXT_RENDER context */
	__u32 flags;
	__u32 profile;			/* DRM_G5P_RENDER_PROFILE_TYPED_V3 */
	__u32 render_target;
	__u64 resources;		/* struct drm_g5p_render_resource[] */
	__u32 resource_count;
	__u32 sync_count;
	__u64 syncs;			/* struct drm_g5p_sync[] */
	__u64 pass;			/* struct drm_g5p_render_pass_v3 */
	__u32 pass_size;
	__u32 pad;
	__u64 job_id;			/* out */
	__u64 reserved[2];
};

struct drm_g5p_rt_geometry {
	__u32 header_bytes;
	__u32 region_bytes;
	__u64 packed_state;
	__u32 state_word;
	__u32 padded_dims;
	__u64 scale[3];
	__u64 ta_state[2];
	__u64 ta_tiling[2];
	__u32 ta_tiling_word;
	__u32 tile_count;
	__u64 secondary_bytes;
	__u64 secondary_align;
	__u64 reserved[2];
};

/* Without USER_GEOMETRY the kernel computes the geometry and allocates the
 * parameter buffers itself; handle is out. */
#define DRM_G5P_RT_USER_GEOMETRY (1U << 0)
struct drm_g5p_create_render_target {
	__u32 context;
	__u32 flags;
	__u32 width;
	__u32 height;
	__u32 samples;
	__u32 layers;
	__u32 handle;			/* out */
	__u32 pad;
	__u64 main_va;
	__u64 main_bytes;
	__u64 secondary_va;
	__u64 secondary_bytes;
	struct drm_g5p_rt_geometry geometry;
	__u64 reserved[2];
};

struct drm_g5p_destroy_render_target {
	__u32 handle;
	__u32 flags;
};

#define DRM_G5P_CREATE_RENDER_TARGET 0x11
#define DRM_G5P_DESTROY_RENDER_TARGET 0x12
#define DRM_G5P_SUBMIT_RENDER_TYPED 0x14
#define DRM_IOCTL_G5P_CREATE_RENDER_TARGET \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_CREATE_RENDER_TARGET, struct drm_g5p_create_render_target)
#define DRM_IOCTL_G5P_DESTROY_RENDER_TARGET \
	DRM_IOW(DRM_COMMAND_BASE + DRM_G5P_DESTROY_RENDER_TARGET, struct drm_g5p_destroy_render_target)
#define DRM_IOCTL_G5P_SUBMIT_RENDER_TYPED \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_G5P_SUBMIT_RENDER_TYPED, struct drm_g5p_submit_render_typed)

#endif
