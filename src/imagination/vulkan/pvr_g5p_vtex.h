/* SPDX-License-Identifier: MIT */
#ifndef PVR_G5P_VTEX_H
#define PVR_G5P_VTEX_H
#include "pvr_g5p_upload.h"
#include "g5p_vtex.h"
struct pvr_g5p_vtex_uploads;
#ifdef PVR_PER_ARCH
VkResult PVR_PER_ARCH(g5p_vtex_create)(struct pvr_device *, const struct pco_g5p_program *,
                                      struct pvr_g5p_stage_uploads **);
void PVR_PER_ARCH(g5p_vtex_destroy)(struct pvr_device *, struct pvr_g5p_stage_uploads *);
VkResult PVR_PER_ARCH(g5p_vtex_draw)(struct pvr_cmd_buffer *, const struct pvr_g5p_stage_uploads *,
                                    struct pvr_g5p_vs_upload_draw *);
#define pvr_arch_g5p_vtex_create PVR_PER_ARCH(g5p_vtex_create)
#define pvr_arch_g5p_vtex_destroy PVR_PER_ARCH(g5p_vtex_destroy)
#define pvr_arch_g5p_vtex_draw PVR_PER_ARCH(g5p_vtex_draw)
#endif
#endif
