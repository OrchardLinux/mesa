/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): depth/stencil attachments. The ZLS words, the
 * depth/stencil planes and the draw's ISP words, as the A9 driver on iPadOS
 * builds them (our iOS captures of depth, stencil and depth-bias passes,
 * 2026-11-01). Built only with -Dimagination-g5p-experimental=true.
 *
 * The functions without PVR_PER_ARCH are plain arithmetic on the words
 * (csbgen field packing, no driver objects), so a host test can check them
 * against the captured values. The PVR_PER_ARCH ones read the driver's
 * objects (pvr_render_job::ds, the dynamic state) and call them.
 *
 * The ZLS words follow the placement of g9p_zlsctl() (pvr_arch_g9p_gfx.c,
 * the G9P driver): they are made when the fragment stream is written, after
 * pvr_sub_cmd_gfx_job_init() has filled pvr_render_job::ds.
 */

#ifndef PVR_G5P_DS_H
#define PVR_G5P_DS_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

/* A depth/stencil format the A9's ZLS stores: the ZLSCTL load/store format
 * (ROGUE_CR_ZLS_FORMAT_TYPE: F32Z 0, 16BITINT 2), bytes per depth sample,
 * the aspects, and whether its depth or stencil plane can be sampled.
 * Stencil is always a separate 8-bit plane with its own bases. D24S8 is not
 * a ZLS format on the A9 (Zink takes D32_SFLOAT_S8_UINT instead). */
struct pvr_g5p_ds_format {
   VkFormat format;
   uint8_t zls_type;
   uint8_t depth_bytes;
   bool depth, stencil, sampled;
};

const struct pvr_g5p_ds_format *pvr_g5p_ds_format(VkFormat format);

/* The memory of a depth/stencil image: each plane is a power-of-two square
 * of 32-pixel tiles in the ZLS's twiddled (Morton) order, which bounds the
 * footprint the device measured (100x75 D32F touches 61440 of 65536 bytes);
 * the stencil plane follows the depth plane at a 16 KiB boundary; four
 * samples take four times each plane (on chip only, see pvr_g5p_ds_zls()). */
#define PVR_G5P_ZLS_PLANE_ALIGN 0x4000u
struct pvr_g5p_ds_plane {
   uint32_t side;           /* pixels per side of the padded square */
   uint64_t depth_bytes;    /* 0 for S8 */
   uint64_t stencil_offset; /* 0 without stencil */
   uint64_t size;           /* the whole image, PVR_G5P_ZLS_PLANE_ALIGN-aligned */
};

bool pvr_g5p_ds_layout(VkFormat format, uint32_t width, uint32_t height, uint32_t samples,
                       struct pvr_g5p_ds_plane *out);

/* The depth/stencil attachment of one render, as the ZLS sees it */
struct pvr_g5p_ds_target {
   VkFormat format;
   uint64_t va;             /* the image's base (the depth plane) */
   uint32_t width, height;  /* the attachment's extent: ZLS_PIXELS */
   uint32_t samples;        /* 1 or 4 */
   bool depth, stencil;     /* the aspects the render uses */
   bool zload, zstore, sload, sstore;
   float depth_clear;       /* ISP_BGOBJDEPTH */
   uint8_t stencil_clear;   /* ISP_BGOBJVALS.stencil */
};

/* The ZLS words of the TYPED_V3 pass's V2 extension (drm_g5p_render_pass_ext):
 * zls[] = ZLSCTL, ZLOAD_BASE, ZSTORE_BASE, STENCIL_LOAD_BASE,
 * STENCIL_STORE_BASE, ZLS_PIXELS; address_mask bit i says zls[i] is an
 * address (inside the job's listed ranges); the background object's depth
 * and values. */
struct pvr_g5p_zls_words {
   uint64_t zls[6];
   uint32_t address_mask;
   uint32_t bgobj_depth;
   uint32_t bgobj_vals;
};

/* The words of a render with no depth/stencil attachment (depth 1.0f, tag
 * and mask enabled, as the captured colour-only pass) */
void pvr_g5p_ds_zls_none(struct pvr_g5p_zls_words *out);

/* The words of a render with one. Returns NULL, or why the render is refused
 * (the ZLS cannot do it, or no capture shows how). */
const char *pvr_g5p_ds_zls(const struct pvr_g5p_ds_target *t, struct pvr_g5p_zls_words *out);

/* The draw's ISP words. Stencil ops and compare ops are the Vulkan enums
 * (Rogue's ROGUE_TA_ISPB_STENCILOP and ROGUE_TA_CMPMODE share their
 * numbering). The state is the one vk_optimize_depth_stencil_state() leaves
 * for the attachment's aspects. */
struct pvr_g5p_stencil_face {
   VkStencilOp fail, pass, depth_fail;
   VkCompareOp compare;
   uint8_t compare_mask, write_mask, reference;
};

struct pvr_g5p_isp_state {
   bool lines;              /* a line primitive, else a triangle */
   uint32_t passtype;       /* ROGUE_TA_PASSTYPE of the fragment program */
   uint32_t line_width;     /* ISPA.pointlinewidth: 4.4 fixed minus one, 0..255 */
   bool ds_attachment;      /* the render has one (depth or stencil aspect) */
   bool depth_test, depth_write;
   VkCompareOp depth_compare;
   bool stencil_test;
   struct pvr_g5p_stencil_face front, back;
   bool cull_front, cull_back, front_ccw;
   bool scissor, dbias, no_colour;
   uint32_t samples;        /* 1 or 4: ISPCTL.upass */
};

/* w[0..4] = ISPCTL, front ISPA, front ISPB, back ISPA, back ISPB (the TA
 * state words after the header that presents all five) */
void pvr_g5p_ds_isp(const struct pvr_g5p_isp_state *s, uint32_t w[5]);

/* A sampled plane of a depth/stencil image, read in place by the texture
 * unit: the plane's VA and the colour format whose texture key reads it
 * (depth: R32_SFLOAT, captured; stencil: R8_UINT, inferred).
 * Returns NULL, or why the plane cannot be sampled. */
const char *pvr_g5p_ds_texture_plane(VkFormat format, VkImageAspectFlags aspect, uint32_t width,
                                     uint32_t height, uint64_t va, uint64_t *plane_va,
                                     VkFormat *texture_format);

/* The format features of a depth/stencil format on the A9 path: none for a
 * format the ZLS does not store; a depth/stencil attachment otherwise, and
 * sampled with depth comparison when its planes have a texture key
 * (pvr_g5p_ds_texture_plane()). No linear filtering (the A9 filters no
 * 32-bit float texture, and ES 3.0 asks none of DEPTH_COMPONENT32F), and no
 * transfers (the planes are in the ZLS's Morton order). */
VkFormatFeatureFlags2 pvr_g5p_ds_format_features(VkFormat format);

/* pvr_image_init(): a depth/stencil image of a ZLS format takes exactly the
 * ZLS planes' size (pvr_g5p_ds_layout()), not upstream's larger D32F_S8
 * layout; a 4-sample colour image at least the PBE's surface; other images
 * are left as they are (pvr_g5p_image.c) */
struct pvr_image;
void pvr_g5p_ds_image_layout(struct pvr_image *image);

#endif /* PVR_G5P_DS_H */

/* The per-arch functions, guarded on their own: a header included early
 * (through the hook headers) before PVR_PER_ARCH is defined still gives
 * them to a later per-arch file. */
#if defined(PVR_PER_ARCH) && !defined(PVR_G5P_DS_ARCH_H)
#define PVR_G5P_DS_ARCH_H

struct pvr_cmd_buffer;
struct pvr_render_job;
struct pvr_renderpass_hwsetup_render;
struct vk_dynamic_graphics_state;

/* job_init: refuses, by name, a depth/stencil attachment this lane does not
 * build. VK_SUCCESS without one. */
VkResult PVR_PER_ARCH(g5p_ds_admit)(struct pvr_cmd_buffer *cmd_buffer,
                                    const struct pvr_renderpass_hwsetup_render *hw_render);
/* frag_stream: the job's ZLS and background words from pvr_render_job::ds */
void PVR_PER_ARCH(g5p_ds_pass)(const struct pvr_render_job *job, struct pvr_g5p_zls_words *out);
/* the draw: w[0..4] as pvr_g5p_ds_isp(), from the dynamic state */
void PVR_PER_ARCH(g5p_ds_isp)(const struct vk_dynamic_graphics_state *dyn, VkImageAspectFlags aspects,
                              uint32_t samples, bool lines, uint32_t passtype, bool scissor, bool dbias,
                              bool no_colour, uint32_t w[5]);

/* the draw: the aspects of the current render's depth/stencil attachment
 * (0 without one) */
VkImageAspectFlags PVR_PER_ARCH(g5p_ds_aspects)(const struct pvr_cmd_buffer *cmd_buffer);
/* the draw: its depth-bias table entry ({constant, slope, clamp} as given,
 * the A9 driver's form), appended to the command buffer's table, which
 * upstream uploads per render (pvr_render_job::depth_bias_table_addr) */
VkResult PVR_PER_ARCH(g5p_ds_bias)(struct pvr_cmd_buffer *cmd_buffer, const struct vk_dynamic_graphics_state *dyn,
                                   uint32_t *index);

/* lane b's image-view packing: the texture state of a depth or stencil view,
 * its ZLS plane read in place (q0..q2 as pvr_g5p_tex_twiddled()), or why it
 * has none */
struct pvr_image_view;
const char *PVR_PER_ARCH(g5p_ds_view_state)(const struct pvr_image_view *iview, uint64_t q[3]);

#   define pvr_arch_g5p_ds_view_state PVR_PER_ARCH(g5p_ds_view_state)
#   define pvr_arch_g5p_ds_admit PVR_PER_ARCH(g5p_ds_admit)
#   define pvr_arch_g5p_ds_pass PVR_PER_ARCH(g5p_ds_pass)
#   define pvr_arch_g5p_ds_isp PVR_PER_ARCH(g5p_ds_isp)
#   define pvr_arch_g5p_ds_aspects PVR_PER_ARCH(g5p_ds_aspects)
#   define pvr_arch_g5p_ds_bias PVR_PER_ARCH(g5p_ds_bias)

#endif /* PVR_PER_ARCH */
