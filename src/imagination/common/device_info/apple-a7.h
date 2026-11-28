/*
 * SPDX-License-Identifier: MIT
 */

/*
 * Apple A7 GPU (S5L8960X, iPad Air, iPad mini 2/3, iPhone 5s): a PowerVR
 * Rogue Series6 core (G6430) run by Apple's own firmware (AGXG3). BVNC
 * 1.70.4.3 as the kernel decodes CORE_REVISION 0x00014600 and CORE_ID
 * 0x011b0403 (measured on a J71 iPad Air under Linux), 4 clusters.
 *
 * The kernel's Apple glue boots the A7's firmware as the A8's (T7000,
 * G4P) variant: the compute job stream, the CDM records, the PDS DOUT
 * layout and the memory layout (heaps, one code base for PDS and USC) are
 * the A8's and run on the A7. So this entry sets has_apple_g5p and
 * has_apple_g4p and takes the A8's job, stream, PDS and memory glue.
 *
 * The shader ISA is not the A8's. Apple's A7 backend is "agx0" (CPU
 * g3-b0), plain Rogue, the ISA upstream pco encodes, with A7 deltas
 * (a 2-byte single upper source, stores taking their data from the
 * registers after the address pair, no 4-byte st). has_apple_g3 routes
 * the A7 to its own compute path (g3/g3_compute.c, pvr_arch_g3_compute.c)
 * and keeps every Apple-agx1 USC program of the A8/A9 paths off the GPU.
 * Graphics is not built for the A7 yet and is refused by name.
 *
 * Values marked "A7" are measured on the J71 or read in Apple's A7
 * drivers; the rest are the A8 entry's placeholders (apple-a8.h).
 */

#ifndef APPLE_A7_H
#define APPLE_A7_H

#include <stdbool.h>

#include "pvr_device_info.h"

#define PVR_DEVICE_IDENT_1_V_4_3 \
   .device_id = 0x8960, .series_name = "Apple", \
   .public_name = "A7", .arch = PVR_DEVICE_ARCH_ROGUE

static const struct pvr_device_features pvr_device_features_1_V_4_3 = {
   .has_apple_g5p = true,
   .has_apple_g4p = true,
   .has_apple_g3 = true,
   .has_num_clusters = true,            /* A7 */

   /* A8 placeholders (apple-a8.h) */
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

   .num_clusters = 4U,                  /* A7: CORE_ID N = 4 */

   .tile_size_x = 32U,
   .tile_size_y = 32U,
   .common_store_size_in_dwords = 1280U * 4U * 4U,
   .isp_max_tiles_in_flight = 8U,
   .isp_samples_per_pixel = 2U,
   .max_instances_per_pds_task = 32U,   /* A7: 32 instances per task (lanes 0-31 measured) */
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

static const struct pvr_device_enhancements pvr_device_enhancements_1_70_4_3 = { 0 };

static const struct pvr_device_quirks pvr_device_quirks_1_70_4_3 = { 0 };

static const struct pvr_device_info pvr_device_info_1_70_4_3 = {
   .ident = {
      PVR_DEVICE_IDENT_1_V_4_3,
      .b = 1,
      .v = 70,
      .n = 4,
      .c = 3,
   },
   .features = pvr_device_features_1_V_4_3,
   .enhancements = pvr_device_enhancements_1_70_4_3,
   .quirks = pvr_device_quirks_1_70_4_3,
};

#endif /* APPLE_A7_H */
