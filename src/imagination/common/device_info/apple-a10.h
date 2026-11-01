/*
 * SPDX-License-Identifier: MIT
 */

/*
 * Apple A10 GPU (T8010, iPad 6/7, iPhone 7): PowerVR Rogue-derived, with
 * Apple's own shader ISA (applegpu_g9p). BVNC 13.10.6.2 as the kernel
 * decodes the legacy CORE_ID (0x011b0602) and CORE_REVISION (0x000d0a00)
 * registers.
 *
 * Values marked "A10" are established on the hardware or from Apple's
 * drivers (see g9p/README.md for the A10 notes: PROVENANCE.md); the others are
 * placeholders copied from GX6650, the closest 6-cluster Rogue, and are not
 * verified for this GPU.
 */

#ifndef APPLE_A10_H
#define APPLE_A10_H

#include <stdbool.h>

#include "pvr_device_info.h"

#define PVR_DEVICE_IDENT_13_V_6_2 \
   .device_id = 0x8010, .series_name = "Apple", \
   .public_name = "A10", .arch = PVR_DEVICE_ARCH_ROGUE

static const struct pvr_device_features pvr_device_features_13_V_6_2 = {
   .has_apple_g9p = true,
   .has_compute = true,                 /* A10 */
   .has_isp_samples_per_pixel = true,   /* A10 */
   .has_num_clusters = true,            /* A10 */
   .has_tile_size_x = true,             /* A10 */
   .has_tile_size_y = true,             /* A10 */

   /* GX6650 placeholders */
   .has_cluster_grouping = true,
   .has_common_store_size_in_dwords = true,
   .has_isp_max_tiles_in_flight = true,
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

   .isp_samples_per_pixel = 2U,         /* A10: TE_AA 1/5/7 (iOS 10 kext) */
   .num_clusters = 6U,                  /* A10: CORE_ID, firmware units 0-5 */
   .tile_size_x = 32U,                  /* A10 */
   .tile_size_y = 32U,                  /* A10 */

   .common_store_size_in_dwords = 1280U * 4U * 4U,
   .isp_max_tiles_in_flight = 8U,
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

static const struct pvr_device_enhancements pvr_device_enhancements_13_10_6_2 = { 0 };

static const struct pvr_device_quirks pvr_device_quirks_13_10_6_2 = { 0 };

static const struct pvr_device_info pvr_device_info_13_10_6_2 = {
   .ident = {
      PVR_DEVICE_IDENT_13_V_6_2,
      .b = 13,
      .v = 10,
      .n = 6,
      .c = 2,
   },
   .features = pvr_device_features_13_V_6_2,
   .enhancements = pvr_device_enhancements_13_10_6_2,
   .quirks = pvr_device_quirks_13_10_6_2,
};

#endif /* APPLE_A10_H */
