/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 */
#pragma once
#include "compiler/nir/nir.h"
#include "poly/nir/poly_nir.h"

/* Private qualification interface, not an advertised Vulkan GS implementation.
 * Binding 0: poly's compact VS output records. Binding 1: expanded records.
 * Each occupied varying slot occupies 16 bytes, in increasing slot order.
 * The caller allocates exactly the reported sizes and dispatches exact grids
 * with local_size=1. No raw application GPU addresses enter this interface.
 */
struct pvr_g5p_swgs {
   nir_shader *vertex;
   nir_shader *geometry;
   struct poly_gs_info topology;
   uint64_t input_mask, output_mask;
   unsigned input_stride, output_stride;
   unsigned input_size, output_size;
   unsigned vertex_grid[3], geometry_grid[3];
   char refusal[192];
};

/* Does not consume or mutate either input; returned NIR belongs to mem_ctx.
 * Initial scope: resource-free VS, pure static stream-0 point-input GS,
 * non-indexed output, one invocation. Unsupported scope refuses by name.
 */
bool pvr_g5p_swgs_lower(const nir_shader *vs, const nir_shader *gs,
                        unsigned vertices, unsigned instances,
                        void *mem_ctx, struct pvr_g5p_swgs *out);
