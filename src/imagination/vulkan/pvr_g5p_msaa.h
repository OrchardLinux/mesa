/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): 4x multisampling. What a 4-sample render changes
 * against a one-sample one, as the iPadOS driver's 4x passes do (our iOS
 * captures msaa4-store and msaa4-resolve, 2026-11-01). The kernel
 * (a9/gpu-msaa) writes the render target's AA words from the pass's
 * samples, flags and positions; userspace writes the PBE words, the
 * end-of-tile program's emit, the fragment DOUTU sample rate and the draw's
 * user pass. Built only with -Dimagination-g5p-experimental=true.
 *
 * The functions without PVR_PER_ARCH are plain arithmetic (no driver
 * objects), checked by a host test against the captured words.
 */

#ifndef PVR_G5P_MSAA_H
#define PVR_G5P_MSAA_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

/* The pass fields of kernel a9/gpu-msaa (drm_g5p_render_pass: samples @296,
 * msaa_flags @300, sample_positions @304); zero keeps the one-sample pass */
#define PVR_G5P_MSAA_FLAG_RESOLVE (1u << 0)
struct pvr_g5p_msaa_words {
   uint32_t samples;
   uint32_t flags;
   uint64_t positions;
};

/* Vulkan's standard 4x pattern (6,2) (14,6) (2,10) (10,14) in 1/16 pixel,
 * packed x0 | y0 << 4 | x1 << 8 | ... (the captured record's positions,
 * which the kernel writes to PPP_MULTISAMPLECTL and the 3D's positions) */
uint64_t pvr_g5p_msaa_positions(uint32_t samples);

/* The pass fields of a render with `samples` samples; resolve: the PBE
 * averages the samples into a one-sample target at the end of each tile */
struct pvr_g5p_msaa_words pvr_g5p_msaa_pass(uint32_t samples, bool resolve);

/* The framebuffer sample counts: 1, and 4 when the kernel takes 4-sample
 * render targets (DRM_G5P_CAP_RENDER_MSAA) */
VkSampleCountFlags pvr_g5p_msaa_sample_counts(bool kernel_msaa);

/* A 4-sample colour image: one 16-byte element of four samples per pixel in
 * Morton order (y in bit 0) over power-of-two sides, sample s at s * cpp
 * (the captured 64x64 store surface, all 16384 samples). Returns the size,
 * or 0 for a format without 4-byte samples. */
uint64_t pvr_g5p_msaa_layout(uint32_t width, uint32_t height, uint32_t cpp);
uint64_t pvr_g5p_msaa_sample_offset(uint32_t width, uint32_t height, uint32_t cpp, uint32_t x,
                                    uint32_t y, uint32_t s);

/* PBE words (the three qwords g5p_pbe_encode() makes for a one-sample
 * linear target, key = the format's PBE key):
 * - the multisample store: image layout 4 (2D multisample) with the
 *   twiddled bit, the pixel extent, no stride (captured q0
 *   0x00007e01fb05050c for 64x64 BGRA8);
 * - the resolve: the one-sample target's words with q2 bit 42 (captured
 *   dword 5 0x400). */
void pvr_g5p_msaa_pbe_store(uint64_t q[3], uint32_t key, uint64_t va, uint32_t width, uint32_t height,
                            bool srgb);
void pvr_g5p_msaa_pbe_resolve(uint64_t q[3]);

/* The end-of-tile program of a one-target 4x store: the one-target program
 * (g5p_mrt_eot_usc(out, 1)) with the emitpix's sample field 2 instead of 1
 * (captured 1c 05 -> 1c 09). The resolve keeps the one-sample program.
 * Returns its size (G5P_MRT_EOT_USC_MAX bytes at most). */
unsigned pvr_g5p_msaa_eot_usc(uint8_t *out);

/* A fragment-side DOUTU src0 (the background program's and the fragment
 * programs') at `samples`: sample rate src0[34:33] 2 at 4x (captured
 * 0x10... -> 0x14...) */
uint64_t pvr_g5p_msaa_doutu(uint64_t src0, uint32_t samples);

/* The draw's ISPCTL user pass: 15 at one sample, 7 at four */
uint32_t pvr_g5p_msaa_upass(uint32_t samples);

/* Resolve on the CPU (vkCmdResolveImage, and a STORE pass with a resolve
 * view): per byte (sum of the four samples + 2) / 4, as the GPU's resolve
 * (the captured 4096 resolved pixels, ties included). src is the 4-sample
 * layout above, dst a linear one-sample target with dst_stride bytes a row. */
void pvr_g5p_msaa_resolve_cpu(const uint8_t *src, uint8_t *dst, uint32_t dst_stride, uint32_t width,
                              uint32_t height, uint32_t cpp);

#ifdef PVR_PER_ARCH

struct pvr_cmd_buffer;
struct pvr_renderpass_hwsetup_render;

/* job_init: refuses, by name, a multisample render this lane does not
 * build (only 1 and 4 samples; at 4: one colour target, no LOAD, no
 * multisample depth/stencil load or store). VK_SUCCESS at one sample. */
VkResult PVR_PER_ARCH(g5p_msaa_admit)(struct pvr_cmd_buffer *cmd_buffer,
                                      const struct pvr_renderpass_hwsetup_render *hw_render);

#   define pvr_arch_g5p_msaa_admit PVR_PER_ARCH(g5p_msaa_admit)

#endif /* PVR_PER_ARCH */

#endif /* PVR_G5P_MSAA_H */
