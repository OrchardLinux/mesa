/*
 * SPDX-License-Identifier: MIT
 */

/*
 * Apple A9 GPU (S8000/S8003, iPad 5, iPhone 6s): PowerVR Rogue-derived,
 * with Apple's own shader ISA (G5P). BVNC 9.15.6.2 as the legacy
 * CORE_REVISION (0x00090f00) and CORE_ID (0x011b0602) registers decode;
 * Apple's kext checks branch 9 and version 15 and copies N = 6.
 *
 * Built only with -Dimagination-g5p-experimental=true. Nothing in the
 * PowerVR driver selects this entry yet: the A9 runs our own ICD and kernel
 * interface (step 1a of the joint plan; see g5p/README.md).
 *
 * Values marked "A9" are established on the hardware or from Apple's
 * drivers; the others are the A10 entry's placeholders (apple-a10.h, which
 * copies them from GX6650) and are not verified for this GPU.
 */

#ifndef APPLE_A9_H
#define APPLE_A9_H

#include <stdbool.h>

#include "pvr_device_info.h"

#define PVR_DEVICE_IDENT_9_V_6_2 \
   .device_id = 0x8000, .series_name = "Apple", \
   .public_name = "A9", .arch = PVR_DEVICE_ARCH_ROGUE

static const struct pvr_device_features pvr_device_features_9_V_6_2 = {
   .has_apple_g5p = true,
   .has_num_clusters = true,            /* A9 */
   .has_tile_size_x = true,             /* A9 */
   .has_tile_size_y = true,             /* A9 */

   /* A10 (GX6650) placeholders */
   .has_cluster_grouping = true,
   .has_common_store_size_in_dwords = true,
   .has_isp_max_tiles_in_flight = true,
   .has_isp_samples_per_pixel = true,
   .has_max_instances_per_pds_task = true,
   .has_max_multisample = true,
   .has_max_partitions = true,
   .has_max_usc_tasks = true,
   .has_num_raster_pipes = true,
   .has_slc_cache_line_size_bits = true,
   .has_slc_mcu_cache_controls = true,
   .has_unified_store_depth = true,
   .has_usc_min_output_registers_per_pix = true,
   .has_usc_slots = true,
   .has_uvs_banks = true,
   .has_uvs_pba_entries = true,
   .has_uvs_vtx_entries = true,
   .has_vdm_cam_size = true,
   .has_xt_top_infrastructure = true,

   .num_clusters = 6U,                  /* A9: CORE_ID N = 6 */
   .tile_size_x = 32U,                  /* A9: render jobs, 32x32 tiles */
   .tile_size_y = 32U,                  /* A9 */

   .common_store_size_in_dwords = 1280U * 4U * 4U,
   .isp_max_tiles_in_flight = 8U,
   .isp_samples_per_pixel = 2U,
   .max_instances_per_pds_task = 32U,
   .max_multisample = 8U,
   .max_partitions = 8U,
   .max_usc_tasks = 56U,
   .num_raster_pipes = 2U,
   .slc_cache_line_size_bits = 512U,
   .unified_store_depth = 256U,
   .usc_min_output_registers_per_pix = 2U,
   .usc_slots = 32U,
   .uvs_banks = 8U,
   .uvs_pba_entries = 320U,
   .uvs_vtx_entries = 288U,
   .vdm_cam_size = 256U,
};

static const struct pvr_device_enhancements pvr_device_enhancements_9_15_6_2 = { 0 };

static const struct pvr_device_quirks pvr_device_quirks_9_15_6_2 = { 0 };

static const struct pvr_device_info pvr_device_info_9_15_6_2 = {
   .ident = {
      PVR_DEVICE_IDENT_9_V_6_2,
      .b = 9,
      .v = 15,
      .n = 6,
      .c = 2,
   },
   .features = pvr_device_features_9_V_6_2,
   .enhancements = pvr_device_enhancements_9_15_6_2,
   .quirks = pvr_device_quirks_9_15_6_2,
};

#endif /* APPLE_A9_H */
