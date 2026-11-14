/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): blending. The A9 blends in the fragment program
 * (the G5P backend's epilogues, src/imagination/g5p/g5p_blend.c): the
 * pipeline's blend state becomes the backend's program key, and the
 * program's tile read sets the ISP pass type of its draws. Built only with
 * -Dimagination-g5p-experimental=true.
 */

#ifndef PVR_G5P_BLEND_H
#define PVR_G5P_BLEND_H

#include <stdbool.h>
#include <stdint.h>

#include "pvr_macros.h"

struct pco_g5p_fs_key;
struct vk_graphics_pipeline_state;

/* The ISP control fields a draw's program decides */
struct pvr_g5p_blend_isp {
   unsigned pass_type;   /* ROGUE_TA_PASSTYPE_* */
   unsigned upass;       /* ISPCTL upass: 0 for a program that writes nothing */
   bool tag_write_disable;
};

/* Clamp of the dynamic blend constants for the 8-bit UNORM target the
 * epilogue writes (the program reads them as four f32 common dwords) */
static inline void
pvr_g5p_blend_constants(const float in[4], float out[4])
{
   for (unsigned i = 0; i < 4; i++)
      out[i] = in[i] < 0.0f ? 0.0f : in[i] > 1.0f ? 1.0f : in[i];
}

#ifdef PVR_PER_ARCH

const char *PVR_PER_ARCH(g5p_blend_fs_key)(const struct vk_graphics_pipeline_state *state,
                                           unsigned targets, struct pco_g5p_fs_key *key);
void PVR_PER_ARCH(g5p_blend_isp)(unsigned blend_mode, bool shader_discard, uint8_t write_mask,
                                 bool rasterizer_discard, struct pvr_g5p_blend_isp *isp);

#   define pvr_arch_g5p_blend_fs_key PVR_PER_ARCH(g5p_blend_fs_key)
#   define pvr_arch_g5p_blend_isp PVR_PER_ARCH(g5p_blend_isp)

#endif /* PVR_PER_ARCH */

#endif /* PVR_G5P_BLEND_H */
