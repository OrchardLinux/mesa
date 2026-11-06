/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): the vertex program's common-store uploads.
 * Built only with -Dimagination-g5p-experimental=true (PVR_APPLE_G5P).
 *
 * A vertex program that reads push constants, a uniform buffer or a buffer
 * address takes them from its common store, which a PDS loader fills before
 * the vertex data: one DOUTD per upload (source address, then the control
 * word), then HALT, run by a VDM PDS_STATE_UPDATE block between the varying
 * format and the vertex fetch data. That is the form our A9 driver used
 * (round 19, from the iOS texture capture: `20040001 <data> <code>`), and the
 * fragment program's loader (UNIFORMDATABASE) has the same program shape.
 *
 * The sources, per draw:
 *  - push constants: a copy of the bytes the program reads;
 *  - a uniform buffer: the bound buffer itself (its address, the dynamic
 *    offset and the upload's offset), read by the DOUTD at execution time;
 *  - a uniform or storage buffer's address (register-indexed loads, SSBO
 *    access): a copy of the 64-bit address;
 *  - the transform feedback block: zeroed words the caller fills before the
 *    job runs (pvr_g5p_xfb_core.h).
 */

#ifndef PVR_G5P_UPLOAD_H
#define PVR_G5P_UPLOAD_H

#include <stdint.h>
#include <vulkan/vulkan_core.h>

#include "pvr_common.h"
#include "pvr_macros.h"

struct pco_g5p_program;
struct pvr_cmd_buffer;
struct pvr_device;

#define PVR_G5P_MAX_STAGE_UPLOADS 16

/* One upload of a stage: what the backend's pco_g5p_upload names */
struct pvr_g5p_stage_upload {
   uint8_t kind; /* enum pco_g5p_upload_kind */
   uint8_t set, binding;
   uint16_t element; /* descriptor-array element */
   uint16_t offset; /* bytes */
   uint16_t dwords;
   uint16_t common; /* common-store dword */
};

/* A pipeline's vertex uploads and their loader */
struct pvr_g5p_stage_uploads {
   unsigned count, common_dwords;
   struct pvr_g5p_stage_upload u[PVR_G5P_MAX_STAGE_UPLOADS];
   struct pvr_pds_upload loader; /* the DOUTD loader's code */
};

/* The vertex PDS_STATE_UPDATE words of one draw (VDMCTRL PDS_STATE0..2),
 * and, when the program has a transform feedback block, its CPU mapping
 * and device address (else NULL / 0). */
struct pvr_g5p_vs_upload_draw {
   uint32_t pds_state[3];
   uint32_t *xfb_block;
   uint64_t xfb_block_va;
};

#ifdef PVR_PER_ARCH

/* Pipeline creation: NULL in *out for a vertex program without uploads, else
 * its upload list and loader (refused by name: kinds without a source on the
 * vertex stage). */
VkResult PVR_PER_ARCH(g5p_vs_uploads_create)(struct pvr_device *device, const struct pco_g5p_program *vs,
                                             struct pvr_g5p_stage_uploads **out);
void PVR_PER_ARCH(g5p_vs_uploads_destroy)(struct pvr_device *device, struct pvr_g5p_stage_uploads *up);

/* A draw: the sources of every upload, the loader's data, and the
 * PDS_STATE_UPDATE words that run it. Before the draw's words are written
 * (it uploads into the command buffer). */
VkResult PVR_PER_ARCH(g5p_upload_buffer)(struct pvr_cmd_buffer *cmd_buffer, unsigned set, unsigned binding,
                                         unsigned element, uint64_t *addr, uint64_t *size);
VkResult PVR_PER_ARCH(g5p_vs_uploads_draw)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g5p_stage_uploads *up,
                                           struct pvr_g5p_vs_upload_draw *out);

#   define pvr_arch_g5p_vs_uploads_create PVR_PER_ARCH(g5p_vs_uploads_create)
#   define pvr_arch_g5p_vs_uploads_destroy PVR_PER_ARCH(g5p_vs_uploads_destroy)
#   define pvr_arch_g5p_vs_uploads_draw PVR_PER_ARCH(g5p_vs_uploads_draw)
#   define pvr_arch_g5p_upload_buffer PVR_PER_ARCH(g5p_upload_buffer)

#endif /* PVR_PER_ARCH */

#endif /* PVR_G5P_UPLOAD_H */
