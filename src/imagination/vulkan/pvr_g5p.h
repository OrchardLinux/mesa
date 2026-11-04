/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P, has_apple_g5p): the G5P paths of the PowerVR driver.
 * Built only with -Dimagination-g5p-experimental=true (PVR_APPLE_G5P).
 *
 * G5P follows G9P's shape (pvr_g9p.h, Henrik Klee): it keeps the pvr_*
 * object model (devices, command buffers, render passes, descriptors,
 * images, the queue, WSI, pdsc and csbgen) and replaces the pieces that
 * face the hardware through hooks, each in a
 * PVR_HAS_FEATURE(dev_info, apple_g5p) branch of its own. The programs it
 * uploads come from pdsc (PDS) and from the G5P backend in
 * src/imagination/g5p (USC); none is a captured Apple binary.
 */

#ifndef PVR_G5P_H
#define PVR_G5P_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

#include "pvr_common.h"
#include "pvr_macros.h"

struct pvr_cmd_buffer;
struct pvr_device;
struct pvr_render_job;
struct pvr_renderpass_hwsetup_render;
struct pvr_sub_cmd_gfx;
struct pvr_suballoc_bo;

/* Colour-target counts with programs: 1, 2 and 4, the counts with an Apple witness (others are refused) */
#define PVR_G5P_MRT_VARIANTS 3

/*
 * The device-level programs every G5P render uses, uploaded once per
 * device. PDS programs are built with pdsc, USC programs by the G5P
 * backend's generators (src/imagination/g5p/g5p_mrt_programs.h,
 * g5p_usc.h). A kick is the shape the hardware's program registers take: 16
 * bytes of data (the USC program's DOUTU word) and, at +0x20, DOUTU + HALT.
 */
struct pvr_g5p_programs {
   /* PDS: a lone HALT with 16 zero data bytes (the context store and resume
    * tasks, as on G9P, and the TA context program, ta_pds; route (b) on
    * hardware) */
   struct pvr_pds_upload halt;
   /* USC: background clear and end of tile, per target-count variant */
   struct pvr_suballoc_bo *clear_usc[PVR_G5P_MRT_VARIANTS];
   struct pvr_suballoc_bo *eot_usc[PVR_G5P_MRT_VARIANTS];
   /* PDS kicks of the clear programs (PDS_BGRND0's shader) */
   struct pvr_pds_upload clear_kick[PVR_G5P_MRT_VARIANTS];
   /* PDS code: the clear colour's DOUTD (PDS_BGRND0's texunicode; its data,
    * {floats, control}, is per render) and the end of tile, DOUTD of the PBE
    * words then DOUTU of the end-of-tile program (data per render) */
   struct pvr_pds_upload clear_dma, eot;
   /* the TA's end-of-stream program (the render's TA program slot): a
    * branch to the next 64-byte block, padding, end of task; and its kick */
   struct pvr_suballoc_bo *ta_usc;
   struct pvr_pds_upload ta_kick;
   /* the TA scratch the kernel's records point into, and the V2 tables of
    * a render that has none of its own (the template's sizes) */
   struct pvr_suballoc_bo *scratch, *scissor_pool, *dbias_pool;
};

/* G5P's per-device state (pvr_device::g5p), allocated by g5p_device_init() */
struct pvr_g5p_device {
   struct pvr_g5p_programs programs;
};

/* What a G5P render job hands its fragment stream (pvr_render_job::g5p) */
struct pvr_g5p_job {
   uint64_t vdm_va;             /* the control stream, one block */
   uint32_t background[8];      /* PDS_BGRND0..3 (64-bit registers: 8-byte aligned) */
   uint32_t eot[4];             /* EVENT_PIXEL_PDS data, code, info */
   uint32_t vdm_bytes;          /* through STREAM_TERMINATE */
   uint32_t width, height;
   unsigned targets;            /* colour targets, 1, 2 or 4 */
   bool enabled;
};

#ifdef PVR_PER_ARCH

VkResult PVR_PER_ARCH(g5p_device_init)(struct pvr_device *device);
void PVR_PER_ARCH(g5p_device_finish)(struct pvr_device *device);
VkResult PVR_PER_ARCH(g5p_job_init)(struct pvr_cmd_buffer *cmd_buffer,
                                    struct pvr_sub_cmd_gfx *sub_cmd,
                                    const struct pvr_renderpass_hwsetup_render *hw_render,
                                    struct pvr_render_job *job);
uint32_t PVR_PER_ARCH(g5p_frag_stream)(const struct pvr_device *device,
                                       const struct pvr_render_job *job, uint8_t *buf,
                                       uint32_t size);

#   define pvr_arch_g5p_device_init PVR_PER_ARCH(g5p_device_init)
#   define pvr_arch_g5p_device_finish PVR_PER_ARCH(g5p_device_finish)
#   define pvr_arch_g5p_job_init PVR_PER_ARCH(g5p_job_init)
#   define pvr_arch_g5p_frag_stream PVR_PER_ARCH(g5p_frag_stream)

#endif /* PVR_PER_ARCH */

#endif /* PVR_G5P_H */
