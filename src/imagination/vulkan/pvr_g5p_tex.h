/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): image-view texture state and sampler words for the
 * pvr_* descriptors. Built only with -Dimagination-g5p-experimental=true.
 *
 * The shape follows G9P's g9p_pack_image_view()/g9p_pack_sampler() (Henrik
 * Klee, pvr_arch_g9p_gfx.c): upstream's image, view and sampler objects,
 * with the hardware-facing words replaced by a G5P branch at their call
 * sites. The words themselves are pvr_g5p_tex_encode.h's (included by
 * pvr_arch_g5p_tex.c only: this header is reached from every shared file
 * through pvr_g5p_hooks.h).
 *
 * Descriptor layout the G5P backend reads (PCO_G5P_UPLOAD_IMAGE: 8 dwords
 * at +0; PCO_G5P_UPLOAD_SAMPLER: 2 dwords at +32):
 *   pvr_image_descriptor: words[0..1] = q0, q1; meta[0..1] = q2; meta[2..3]
 *     = 0. The PCO meta upstream writes there is replaced: the G5P backend
 *     doesn't read it.
 *   pvr_sampler_descriptor: words[0] = the general (mip) form; meta[RSVD0,
 *     RSVD1] = the single-level form; everything else 0.
 * Unencodable state leaves the words 0 and logs the reason once; the draw
 * path refuses a draw that uploads it (pvr_g5p_tex_state_valid(),
 * pvr_g5p_sampler_valid()).
 */

#ifndef PVR_G5P_TEX_H
#define PVR_G5P_TEX_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

#include "pvr_macros.h"

struct pvr_combined_image_sampler_descriptor;
struct pvr_image_descriptor;
struct pvr_image_view;
struct pvr_sampler_descriptor;

/* What the texture state needs of a format (pvr_arch_g5p_tex.c looks it
 * up). Lane (f)'s format table, pvr_arch_g5p_formats.c, is to provide it;
 * until then pvr_arch_g5p_tex.c has the formats of the rendered and
 * textured rungs only. */
struct pvr_g5p_tex_format {
   VkFormat format;
   uint32_t key; /* PVR_G5P_TEX_KEY(): format code and channel selectors */
   bool srgb;
};

#ifdef PVR_PER_ARCH

const char *PVR_PER_ARCH(g5p_pack_image_view)(const struct pvr_image_view *iview,
                                              struct pvr_image_descriptor *desc);
const char *PVR_PER_ARCH(g5p_pack_sampler)(const VkSamplerCreateInfo *info,
                                           struct pvr_sampler_descriptor *desc);
void PVR_PER_ARCH(g5p_combine_sampler)(struct pvr_combined_image_sampler_descriptor *desc,
                                       const struct pvr_image_view *iview);

#   define pvr_arch_g5p_pack_image_view PVR_PER_ARCH(g5p_pack_image_view)
#   define pvr_arch_g5p_pack_sampler PVR_PER_ARCH(g5p_pack_sampler)
#   define pvr_arch_g5p_combine_sampler PVR_PER_ARCH(g5p_combine_sampler)

#endif /* PVR_PER_ARCH */

#endif /* PVR_G5P_TEX_H */
