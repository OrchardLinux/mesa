/* SPDX-License-Identifier: MIT */
/*
 * Apple G9P (the A10), -Dimagination-g9p-es3: the OpenGL ES 3.0 enablers.
 *
 * Several colour targets, transform feedback, depth/stencil formats, array
 * and cube textures, 4x MSAA, and the physical-device numbers that describe
 * what the G9P paths do with them (pvr_g9p_es3.c). Ported from our Apple A9
 * (G5P) driver, where each piece runs on hardware; the A9 and A10 shader
 * ISAs match on every instruction form compared so far. Nothing here has run
 * on an A10: every A10-specific choice is marked in its comment, and the
 * whole set is built only with the option (PVR_G9P_ES3).
 *
 * The shape follows the G9P hooks in pvr_g9p.h; the functions are
 * called from his G9P paths, each behind PVR_G9P_ES3.
 */
#ifndef PVR_G9P_ES3_H
#define PVR_G9P_ES3_H

#ifdef PVR_G9P_ES3

#include <stdbool.h>
#include <stdint.h>

#include <vulkan/vulkan_core.h>

struct vk_device_extension_table;
struct vk_features;
struct vk_properties;

/*
 * The limits and features of the G9P paths as built (pvr_g9p_es3.c),
 * applied over upstream's tables in pvr_physical_device_create()
 */
void pvr_g9p_es3_limits(struct vk_device_extension_table *ext, struct vk_features *features,
                        struct vk_properties *props);

/* The sample counts a G9P attachment image may have */
VkSampleCountFlags pvr_g9p_es3_sample_counts(void);

#endif /* PVR_G9P_ES3 */

#endif /* PVR_G9P_ES3_H */
