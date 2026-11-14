/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): what the driver asks of a colour format
 * (pvr_g5p_formats.c, over the table in pvr_g5p_format_table.h). Built only
 * with -Dimagination-g5p-experimental=true; reached from the shared files
 * through pvr_g5p_hooks.h.
 *
 * Users:
 *  - pvr_formats.c (hooks): the format features and the image-format query
 *    of colour formats are the A9's, not upstream's Rogue table;
 *  - the texture state (pvr_arch_g5p_tex.c): a view's key and sRGB bit;
 *  - the render job (pvr_arch_g5p.c) and the pipeline check
 *    (pvr_arch_g5p_gfx.c): a colour target's PBE key and sRGB bit.
 *
 * PVR_G5P_FORMATS=probe (read once) also advertises the table's probe rows:
 * formats decoded but not yet run on the device, for the device rung that
 * runs them. Never set it for anything else.
 */

#ifndef PVR_G5P_FORMATS_H
#define PVR_G5P_FORMATS_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

/* Whether PVR_G5P_FORMATS=probe is set */
bool pvr_g5p_formats_probe(void);

/* The texture key (format code, channel selectors; PVR_G5P_TEX_KEY()'s form)
 * and sRGB bit of a sampled colour format, or false */
bool pvr_g5p_format_tex(VkFormat format, uint32_t *key, bool *srgb);

/* The PBE key and sRGB bit of a colour target format, or false: the A9
 * doesn't render it (the pipeline and the render job refuse it by name) */
bool pvr_g5p_format_pbe(VkFormat format, uint32_t *key, bool *srgb);

/* pvr_get_image_format_features2(): the features of a colour format on
 * the A9, in place of upstream's (flags) */
VkFormatFeatureFlags2 pvr_g5p_color_format_features(VkFormat format, VkImageTiling tiling,
                                                     VkFormatFeatureFlags2 flags);

/* pvr_get_image_format_properties(): VK_SUCCESS, or the reason the A9 has
 * no image of this colour format, type and flags (logged once) */
VkResult pvr_g5p_image_format_check(const VkPhysicalDeviceImageFormatInfo2 *info, VkImageTiling tiling);

#endif /* PVR_G5P_FORMATS_H */
