/*
 * SPDX-License-Identifier: MIT
 */

/*
 * Apple A8 GPU (T7000, iPod touch 6, iPhone 6, iPad mini 4): a PowerVR
 * Rogue 6XT core (GX6450) run by Apple's own firmware (AGXG4P, firmware
 * g4p_b0). BVNC 3.10.4.5 as the kernel decodes the legacy CORE_REVISION
 * (0x00030a00, measured on an iPod touch 6 under Linux) and CORE_ID
 * (0x011b0405) registers, the same rule that gives 9.15.6.2 for the A9 and
 * 13.10.6.2 for the A10. Apple's kext also accepts minor 8; no entry is made
 * for it until a device reports it.
 *
 * The A8 takes the A9's paths of the driver (has_apple_g5p): the same
 * firmware family and job protocol, Rogue CDM control streams of seven words
 * (A10: eight), and a shader ISA of the same Apple family. Apple's LLVM
 * backend for these GPUs defines "g4", "g5" and "g9" instruction sets, the
 * A8 (g4p) being the base the A9 (g5p) adds to. has_apple_g4p marks where
 * the A8 differs from the A9. See g5p/README.md, "The A8".
 *
 * Values marked "A8" are established on the hardware or from Apple's
 * drivers; the others are the A9 entry's placeholders (apple-a9.h) and are
 * not verified for this GPU.
 */

#ifndef APPLE_A8_H
#define APPLE_A8_H

#include <stdbool.h>

#include "pvr_device_info.h"

#define PVR_DEVICE_IDENT_3_V_4_5 \
   .device_id = 0x7000, .series_name = "Apple", \
   .public_name = "A8", .arch = PVR_DEVICE_ARCH_ROGUE

static const struct pvr_device_features pvr_device_features_3_V_4_5 = {
   .has_apple_g5p = true,
   .has_apple_g4p = true,
   /* No has_apple_compute_dispatch: the residency parameters come from
    * AGXMetalA8 and are not established yet, so compute pipelines are
    * refused by name (pvr_arch_g5p_compute.c).
    */
   .has_num_clusters = true,            /* A8 */

   /* A9 placeholders (apple-a9.h) */
   .has_tile_size_x = true,
   .has_tile_size_y = true,
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

   .num_clusters = 4U,                  /* A8: CORE_ID N = 4; firmware chip info {2, 4, 0, 4} */

   .tile_size_x = 32U,                  /* Rogue 6XT (gx6250.h); A8 unverified */
   .tile_size_y = 32U,
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

static const struct pvr_device_enhancements pvr_device_enhancements_3_10_4_5 = { 0 };

static const struct pvr_device_quirks pvr_device_quirks_3_10_4_5 = { 0 };

static const struct pvr_device_info pvr_device_info_3_10_4_5 = {
   .ident = {
      PVR_DEVICE_IDENT_3_V_4_5,
      .b = 3,
      .v = 10,
      .n = 4,
      .c = 5,
   },
   .features = pvr_device_features_3_V_4_5,
   .enhancements = pvr_device_enhancements_3_10_4_5,
   .quirks = pvr_device_quirks_3_10_4_5,
};

#endif /* APPLE_A8_H */
