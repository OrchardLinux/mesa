/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): the colour format table. Built only with
 * -Dimagination-g5p-experimental=true.
 *
 * One row per Vulkan colour format the A9 samples or renders: the image-state
 * key (format code and channel selectors, the same key in the texture state
 * and in the PBE state of a render target), sRGB (q1 bit 45 on both), the
 * texel or block size, and what is advertised. Plain values, no pvr_* types:
 * the driver (pvr_arch_g5p_formats.c) and the host check of the table
 * against our standalone ICD's (tools: g5p-fmt-table) include this one copy.
 *
 * Codes and selectors are our decode of the iOS descriptor tables of the
 * gles3/gles3b/textypes captures (gles3-tex.md §1.1, textypes.md §2): the
 * code is a bit layout | 0x80 signed | 0x100 integer | 0x200 float. Layouts:
 * 0x00 8, 0x0a 8_8, 0x28 8_8_8_8, 0x09 16, 0x23 16_16, 0x32 16_16_16_16,
 * 0x21 32, 0x31 32_32, 0x38 32_32_32_32, 0x26 10_10_10_2, 0x25 11_11_10,
 * 0x27 9_9_9_e5, 0x0b 5_6_5, 0x0d 1_5_5_5, 0x0c 4_4_4_4; block compressed
 * 0x58 ETC2 RGB8, 0x5a ETC2 RGB8A1, 0x59 ETC2 RGBA8 (EAC alpha), 0x5b EAC R11,
 * 0x5c EAC RG11. Selectors: 0-3 a component (component 0 is the lowest
 * field of a packed format), 4 one, 5 zero.
 *
 * What a row advertises is what has run on the A9 through this driver: every
 * row's rungs of the formats hardware test (src/imagination/g5p/tests/formats)
 * passed on the device, every pixel exact (2026-11-04: 118 rungs of the rows
 * advertised then, and 39 rungs of the rows decoded but not yet run, the
 * probes, promoted since). Before that:
 *  - sampling: the texture state is format-independent (lane b's
 *    pvr_arch_g5p_tex.c); each key below was sampled exactly on the device by
 *    our ICD (gles3tex T1, textypes §11), or derived by the code rule;
 *  - rendering: the fragment program writes four f16 in two output words for
 *    every target (the backend's only colour output), and the PBE packs them
 *    into the target's format. Apple's compiler uses that one 70-byte program
 *    for BGRA8, RGBA8, both sRGBs, RGB10A2, RGBA16F and RGBA8 snorm
 *    (gles3-tex.md §1.2); R8 and R16F, RG8 and RG16F, R32F, RG11B10F and
 *    B5G6R5 get other programs (narrower outputs, shader packing). The PBE
 *    packs R8, RG8, R16F, RG16F and RG11B10F from the two-word program too:
 *    exact on the device. Blending is the fragment program's, on the tile's
 *    f16 values before the PBE, for every rendered row;
 *  - probes (PVR_G5P_FORMATS=probe, for a device rung only): rows decoded or
 *    derived but not yet run, not advertised by default. None at present.
 */

#ifndef PVR_G5P_FORMAT_TABLE_H
#define PVR_G5P_FORMAT_TABLE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

/* The image-state key: layout 2D (bits 0-2) | twiddled (bit 3) | code << 5 |
 * selectors R, G, B, A at bits 15, 18, 21, 24 (pvr_g5p_tex_encode.h's
 * PVR_G5P_TEX_KEY(); the PBE and the linear texture state clear bit 3).
 */
#define PVR_G5P_FMT_KEY(code, r, g, b, a)                                                               \
   (0xau | (uint32_t)(code) << 5 | (uint32_t)(r) << 15 | (uint32_t)(g) << 18 | (uint32_t)(b) << 21 | \
    (uint32_t)(a) << 24)
#define PVR_G5P_FMT_KEY_CODE(key) (((key) >> 5) & 0x3ffu)

enum pvr_g5p_fmt_flags {
   PVR_G5P_FMT_SAMPLED = 1u << 0,      /* texture state (and copies) */
   PVR_G5P_FMT_FILTER = 1u << 1,       /* linear filtering */
   PVR_G5P_FMT_RENDER = 1u << 2,       /* colour attachment: the two-word program + this PBE key */
   PVR_G5P_FMT_BLEND = 1u << 3,        /* blending into it (the fragment program's) */
   PVR_G5P_FMT_COMPRESSED = 1u << 4,   /* 4x4 blocks, twiddled only */
   PVR_G5P_FMT_INTEGER = 1u << 5,      /* integer texel: texelFetch/unfiltered */
   /* not advertised unless PVR_G5P_FORMATS=probe (none at present) */
   PVR_G5P_FMT_PROBE_SAMPLED = 1u << 8, /* the sampled bits are a probe */
   PVR_G5P_FMT_PROBE_RENDER = 1u << 9,  /* the render (and blend) bits are a probe */
   PVR_G5P_FMT_PROBE_BLEND = 1u << 10,  /* the blend bit alone is a probe */
};

/* Where a row's key and its advertised use come from */
enum pvr_g5p_fmt_evidence {
   PVR_G5P_FMT_EV_G5P1B,   /* device-proven on g5p-1b (tex-blend 22/22, formats 118 + 39 rungs) */
   PVR_G5P_FMT_EV_ICD,     /* captured key, device-proven by our ICD (gles3tex, textypes) */
   PVR_G5P_FMT_EV_RULE,    /* derived from the decoded code rule, not captured */
   PVR_G5P_FMT_EV_CAPTURE, /* captured key, never run on the device */
};

struct pvr_g5p_format {
   VkFormat format;
   uint32_t key;
   uint8_t cpp; /* bytes per texel, or per 4x4 block */
   bool srgb;
   uint16_t flags;
   uint8_t evidence;
};

#define G5P_ONE 4
#define G5P_ZERO 5
#define G5P_SEL_RGBA 0, 1, 2, 3
#define G5P_SEL_BGRA 2, 1, 0, 3
#define G5P_SEL_RG01 0, 1, G5P_ZERO, G5P_ONE
#define G5P_SEL_R001 0, G5P_ZERO, G5P_ZERO, G5P_ONE
#define G5P_SEL_RGB1 0, 1, 2, G5P_ONE

#define G5P_S PVR_G5P_FMT_SAMPLED
#define G5P_SF (PVR_G5P_FMT_SAMPLED | PVR_G5P_FMT_FILTER)
#define G5P_SI (PVR_G5P_FMT_SAMPLED | PVR_G5P_FMT_INTEGER)
#define G5P_RT (PVR_G5P_FMT_RENDER | PVR_G5P_FMT_BLEND)
#define G5P_CT (PVR_G5P_FMT_SAMPLED | PVR_G5P_FMT_FILTER | PVR_G5P_FMT_COMPRESSED)
#define G5P_PR PVR_G5P_FMT_PROBE_RENDER
#define G5P_PS PVR_G5P_FMT_PROBE_SAMPLED
#define G5P_PB PVR_G5P_FMT_PROBE_BLEND

#define G5P_FMT(vk, cpp, srgb, flags, ev, code, ...)                                   \
   {                                                                                   \
      VK_FORMAT_##vk, PVR_G5P_FMT_KEY(code, __VA_ARGS__), cpp, srgb, (uint16_t)(flags), \
         PVR_G5P_FMT_EV_##ev                                                           \
   }

static const struct pvr_g5p_format pvr_g5p_format_table[] = {
   /* 8_8_8_8: pass/pass-rgba (BGRA8, RGBA8), fmt-srgb8 (q1 bit 45). The
    * A8B8G8R8 packed formats are the R8G8B8A8 bytes on a little-endian GPU. */
   G5P_FMT(B8G8R8A8_UNORM, 4, false, G5P_SF | G5P_RT, G5P1B, 0x28, G5P_SEL_BGRA),
   G5P_FMT(R8G8B8A8_UNORM, 4, false, G5P_SF | G5P_RT, G5P1B, 0x28, G5P_SEL_RGBA),
   G5P_FMT(A8B8G8R8_UNORM_PACK32, 4, false, G5P_SF | G5P_RT, G5P1B, 0x28, G5P_SEL_RGBA),
   G5P_FMT(B8G8R8A8_SRGB, 4, true, G5P_SF | G5P_RT, G5P1B, 0x28, G5P_SEL_BGRA),
   G5P_FMT(R8G8B8A8_SRGB, 4, true, G5P_SF | G5P_RT, G5P1B, 0x28, G5P_SEL_RGBA),
   G5P_FMT(A8B8G8R8_SRGB_PACK32, 4, true, G5P_SF | G5P_RT, G5P1B, 0x28, G5P_SEL_RGBA),
   /* 10_10_10_2 (fmt-rgb10a2). A2R10G10B10 swaps R and B by the selectors,
    * as BGRA8 does; sampled only (upstream's table has no render target of
    * it). */
   G5P_FMT(A2B10G10R10_UNORM_PACK32, 4, false, G5P_SF | G5P_RT, G5P1B, 0x26, G5P_SEL_RGBA),
   G5P_FMT(A2R10G10B10_UNORM_PACK32, 4, false, G5P_SF, G5P1B, 0x26, G5P_SEL_BGRA),
   G5P_FMT(A2B10G10R10_UINT_PACK32, 4, false, G5P_SI, G5P1B, 0x126, G5P_SEL_RGBA),
   /* R/RG 8 (fmt-r8, fmt-rg8). Apple renders them with narrower programs;
    * the two-word program with these PBE keys renders them exactly too. */
   G5P_FMT(R8_UNORM, 1, false, G5P_SF | G5P_RT, G5P1B, 0x00, G5P_SEL_R001),
   G5P_FMT(R8G8_UNORM, 2, false, G5P_SF | G5P_RT, G5P1B, 0x0a, G5P_SEL_RG01),
   /* snorm (g3b-sample-a t0-t2) */
   G5P_FMT(R8_SNORM, 1, false, G5P_SF, G5P1B, 0x80, G5P_SEL_R001),
   G5P_FMT(R8G8_SNORM, 2, false, G5P_SF, G5P1B, 0x8a, G5P_SEL_RG01),
   G5P_FMT(R8G8B8A8_SNORM, 4, false, G5P_SF, G5P1B, 0xa8, G5P_SEL_RGBA),
   G5P_FMT(A8B8G8R8_SNORM_PACK32, 4, false, G5P_SF, G5P1B, 0xa8, G5P_SEL_RGBA),
   /* 16-bit unorm/snorm (t3-t8; RGBA16 on the device) */
   G5P_FMT(R16_UNORM, 2, false, G5P_SF, G5P1B, 0x09, G5P_SEL_R001),
   G5P_FMT(R16G16_UNORM, 4, false, G5P_SF, G5P1B, 0x23, G5P_SEL_RG01),
   G5P_FMT(R16G16B16A16_UNORM, 8, false, G5P_SF, G5P1B, 0x32, G5P_SEL_RGBA),
   G5P_FMT(R16_SNORM, 2, false, G5P_SF, G5P1B, 0x89, G5P_SEL_R001),
   G5P_FMT(R16G16_SNORM, 4, false, G5P_SF, G5P1B, 0xa3, G5P_SEL_RG01),
   G5P_FMT(R16G16B16A16_SNORM, 8, false, G5P_SF, G5P1B, 0xb2, G5P_SEL_RGBA),
   /* half float (t9-t11, fmt-rg16f, fmt-rgba16f), rendered with the two-word
    * program (RGBA16F: an 8-byte PBE pixel) */
   G5P_FMT(R16_SFLOAT, 2, false, G5P_SF | G5P_RT, G5P1B, 0x209, G5P_SEL_R001),
   G5P_FMT(R16G16_SFLOAT, 4, false, G5P_SF | G5P_RT, G5P1B, 0x223, G5P_SEL_RG01),
   G5P_FMT(R16G16B16A16_SFLOAT, 8, false, G5P_SF | G5P_RT, G5P1B, 0x232, G5P_SEL_RGBA),
   /* 32-bit float (fmt-r32f, t12-t13): the A9 does not filter 32-bit float */
   G5P_FMT(R32_SFLOAT, 4, false, G5P_S, G5P1B, 0x221, G5P_SEL_R001),
   G5P_FMT(R32G32_SFLOAT, 8, false, G5P_S, G5P1B, 0x231, G5P_SEL_RG01),
   G5P_FMT(R32G32B32A32_SFLOAT, 16, false, G5P_S, G5P1B, 0x238, G5P_SEL_RGBA),
   /* packed float (t14-t15). Apple packs RG11B10F in a 66-byte program; the
    * PBE packs it from the two-word program's f16 too, exactly. */
   G5P_FMT(B10G11R11_UFLOAT_PACK32, 4, false, G5P_SF | G5P_RT, G5P1B, 0x225, G5P_SEL_RGB1),
   G5P_FMT(E5B9G9R9_UFLOAT_PACK32, 4, false, G5P_SF, G5P1B, 0x227, G5P_SEL_RGB1),
   /* 16-bit packed (g3b-sample-b t1-t3: Metal B5G6R5 [2,1,0,4], ABGR4
    * [3,2,1,0]). Metal lists a packed format from its lowest field, so those
    * are Vulkan's R5G6B5 and R4G4B4A4; B4G4R4A4 by the selectors. Only the
    * forms upstream's Rogue table has (it knows no R4G4B4A4, B5G6R5 or 5_5_5_1
    * with alpha low). */
   G5P_FMT(R5G6B5_UNORM_PACK16, 2, false, G5P_SF, G5P1B, 0x0b, 2, 1, 0, G5P_ONE),
   G5P_FMT(B4G4R4A4_UNORM_PACK16, 2, false, G5P_SF, G5P1B, 0x0c, 1, 2, 3, 0),
   /* integer (g3b-sample-b t5-t18, fmt-rgba8ui; textypes int-fetch and
    * int-sample on the device). RGBA32 UINT/SINT clear/draw/LOAD/masks
    * passed the integer-r4 core/compat device rungs. Narrow integer render
    * targets remain absent or probe-only; integer blending is not exposed. */
   G5P_FMT(R8_UINT, 1, false, G5P_SI, G5P1B, 0x100, G5P_SEL_R001),
   G5P_FMT(R8G8_UINT, 2, false, G5P_SI, G5P1B, 0x10a, G5P_SEL_RG01),
   G5P_FMT(R8G8B8A8_UINT, 4, false, G5P_SI | PVR_G5P_FMT_RENDER | PVR_G5P_FMT_PROBE_RENDER, G5P1B, 0x128, G5P_SEL_RGBA),
   G5P_FMT(A8B8G8R8_UINT_PACK32, 4, false, G5P_SI, G5P1B, 0x128, G5P_SEL_RGBA),
   G5P_FMT(R16_UINT, 2, false, G5P_SI, G5P1B, 0x109, G5P_SEL_R001),
   G5P_FMT(R16G16_UINT, 4, false, G5P_SI | PVR_G5P_FMT_RENDER | PVR_G5P_FMT_PROBE_RENDER, G5P1B, 0x123, G5P_SEL_RG01),
   G5P_FMT(R16G16B16A16_UINT, 8, false, G5P_SI | PVR_G5P_FMT_RENDER | PVR_G5P_FMT_PROBE_RENDER, G5P1B, 0x132, G5P_SEL_RGBA),
   G5P_FMT(R32_UINT, 4, false, G5P_SI | PVR_G5P_FMT_RENDER, G5P1B, 0x121, G5P_SEL_R001),
   G5P_FMT(R32G32_UINT, 8, false, G5P_SI | PVR_G5P_FMT_RENDER, G5P1B, 0x131, G5P_SEL_RG01),
   G5P_FMT(R32G32B32A32_UINT, 16, false, G5P_SI | PVR_G5P_FMT_RENDER, G5P1B, 0x138, G5P_SEL_RGBA),
   G5P_FMT(R8_SINT, 1, false, G5P_SI, G5P1B, 0x180, G5P_SEL_R001),
   G5P_FMT(R8G8_SINT, 2, false, G5P_SI, G5P1B, 0x18a, G5P_SEL_RG01),
   G5P_FMT(R8G8B8A8_SINT, 4, false, G5P_SI, G5P1B, 0x1a8, G5P_SEL_RGBA),
   G5P_FMT(A8B8G8R8_SINT_PACK32, 4, false, G5P_SI, G5P1B, 0x1a8, G5P_SEL_RGBA),
   G5P_FMT(R16_SINT, 2, false, G5P_SI, G5P1B, 0x189, G5P_SEL_R001),
   G5P_FMT(R16G16_SINT, 4, false, G5P_SI, G5P1B, 0x1a3, G5P_SEL_RG01),
   G5P_FMT(R16G16B16A16_SINT, 8, false, G5P_SI, G5P1B, 0x1b2, G5P_SEL_RGBA),
   G5P_FMT(R32_SINT, 4, false, G5P_SI | PVR_G5P_FMT_RENDER, G5P1B, 0x1a1, G5P_SEL_R001),
   G5P_FMT(R32G32_SINT, 8, false, G5P_SI | PVR_G5P_FMT_RENDER, G5P1B, 0x1b1, G5P_SEL_RG01),
   G5P_FMT(R32G32B32A32_SINT, 16, false, G5P_SI | PVR_G5P_FMT_RENDER, G5P1B, 0x1b8, G5P_SEL_RGBA),
   /* ETC2/EAC (g3b-compressed t0-t7; textypes etc2, etc2-8x8 and eac exact
    * on the device), twiddled 4x4 blocks; sRGB is q1 bit 45 as everywhere.
    * Uploads are upstream's block-unit copies through the CPU twiddle; blits
    * decode the source's blocks (pvr_arch_g5p_layout.c). */
   G5P_FMT(ETC2_R8G8B8_UNORM_BLOCK, 8, false, G5P_CT, G5P1B, 0x58, G5P_SEL_RGB1),
   G5P_FMT(ETC2_R8G8B8_SRGB_BLOCK, 8, true, G5P_CT, G5P1B, 0x58, G5P_SEL_RGB1),
   G5P_FMT(ETC2_R8G8B8A1_UNORM_BLOCK, 8, false, G5P_CT, G5P1B, 0x5a, G5P_SEL_RGBA),
   G5P_FMT(ETC2_R8G8B8A1_SRGB_BLOCK, 8, true, G5P_CT, G5P1B, 0x5a, G5P_SEL_RGBA),
   G5P_FMT(ETC2_R8G8B8A8_UNORM_BLOCK, 16, false, G5P_CT, G5P1B, 0x59, G5P_SEL_RGBA),
   G5P_FMT(ETC2_R8G8B8A8_SRGB_BLOCK, 16, true, G5P_CT, G5P1B, 0x59, G5P_SEL_RGBA),
   G5P_FMT(EAC_R11_UNORM_BLOCK, 8, false, G5P_CT, G5P1B, 0x5b, G5P_SEL_R001),
   G5P_FMT(EAC_R11_SNORM_BLOCK, 8, false, G5P_CT, G5P1B, 0xdb, G5P_SEL_R001),
   G5P_FMT(EAC_R11G11_UNORM_BLOCK, 16, false, G5P_CT, G5P1B, 0x5c, G5P_SEL_RG01),
   G5P_FMT(EAC_R11G11_SNORM_BLOCK, 16, false, G5P_CT, G5P1B, 0xdc, G5P_SEL_RG01),
};

#undef G5P_FMT
#undef G5P_S
#undef G5P_SF
#undef G5P_SI
#undef G5P_RT
#undef G5P_CT
#undef G5P_PR
#undef G5P_PS
#undef G5P_PB

/* The row of a format, or NULL: the A9 neither samples nor renders it */
static inline const struct pvr_g5p_format *
pvr_g5p_format_find(VkFormat format)
{
   for (size_t i = 0; i < sizeof(pvr_g5p_format_table) / sizeof(pvr_g5p_format_table[0]); i++) {
      if (pvr_g5p_format_table[i].format == format)
         return &pvr_g5p_format_table[i];
   }
   return NULL;
}

/* A row's flags with its probes resolved: probes count only when asked for */
static inline uint16_t
pvr_g5p_format_flags(const struct pvr_g5p_format *f, bool probe)
{
   uint16_t flags = f->flags;

   if (!probe && (flags & PVR_G5P_FMT_PROBE_SAMPLED))
      flags &= ~(PVR_G5P_FMT_SAMPLED | PVR_G5P_FMT_FILTER);
   if (!probe && (flags & PVR_G5P_FMT_PROBE_RENDER))
      flags &= ~(PVR_G5P_FMT_RENDER | PVR_G5P_FMT_BLEND);
   if (!probe && (flags & PVR_G5P_FMT_PROBE_BLEND))
      flags &= ~PVR_G5P_FMT_BLEND;
   return flags;
}

/*
 * The format features of a colour format on the A9, from its row alone:
 *  - SAMPLED_IMAGE, TRANSFER_SRC/DST for sampled rows (copies are the CPU's:
 *    the transfer path G5P shares with G9P), FILTER_LINEAR where the A9
 *    filters; block-compressed rows in optimal tiling only;
 *  - BLIT_SRC for sampled rows: the CPU blit converts through util_format,
 *    compressed sources decoded per block (pvr_arch_g5p_layout.c);
 *  - COLOR_ATTACHMENT (BLEND, BLIT_DST) for rendered rows, in either tiling
 *    (a linear 2D target is what the PBE writes);
 *  - nothing else: no storage images (no compute on G5P), no texel buffers.
 * linear: the features of VK_IMAGE_TILING_LINEAR.
 */
static inline VkFormatFeatureFlags2
pvr_g5p_format_features(const struct pvr_g5p_format *f, bool linear, bool probe)
{
   VkFormatFeatureFlags2 r = 0;

   if (!f)
      return 0;
   const uint16_t flags = pvr_g5p_format_flags(f, probe);

   if ((flags & PVR_G5P_FMT_SAMPLED) && !(linear && (flags & PVR_G5P_FMT_COMPRESSED))) {
      r |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
           VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_2_BLIT_SRC_BIT;
      if (flags & PVR_G5P_FMT_FILTER)
         r |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
   }
   if (flags & PVR_G5P_FMT_RENDER) {
      r |= VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_2_BLIT_DST_BIT |
           VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
      if (flags & PVR_G5P_FMT_BLEND)
         r |= VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT;
   }
   return r;
}

/* The PBE key of a colour target format, or false when the A9 doesn't render it */
static inline bool
pvr_g5p_format_pbe_key(VkFormat format, bool probe, uint32_t *key, bool *srgb)
{
   const struct pvr_g5p_format *f = pvr_g5p_format_find(format);

   if (!f || !(pvr_g5p_format_flags(f, probe) & PVR_G5P_FMT_RENDER))
      return false;
   *key = f->key;
   *srgb = f->srgb;
   return true;
}

#endif /* PVR_G5P_FORMAT_TABLE_H */
