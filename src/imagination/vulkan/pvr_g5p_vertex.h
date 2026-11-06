/* SPDX-License-Identifier: MIT */
/*
 * Vertex input of G5P (Apple A9) pipelines and draws: which vertex
 * attributes the vertex fetch PDS program DMAs into which unified-store
 * registers, the vertex program's NIR rewritten to read them there, and a
 * draw's index list. The draw path (pvr_arch_g5p_gfx.c) calls these.
 *
 * The A9 fetches raw dwords: one DDMAD (stride * index + base) per fetch
 * into the unified store, and the vertex program reads the registers. This
 * is our ICD's g5p_pipeline.c/g5p_record.c and Apple's g3b-vtx16 capture
 * (our decode: runs of attributes that touch adjacent dwords of one buffer
 * are one DDMAD, packed back to back in the store).
 */

#ifndef PVR_G5P_VERTEX_H
#define PVR_G5P_VERTEX_H

#include <stdbool.h>
#include <stdint.h>

#include "vulkan/vulkan_core.h"

#include "pco_g5p_general.h"
#include "pvr_g5p.h"

struct nir_shader;
struct vk_vertex_input_state;

/* The backend's unified store for vertex inputs: dwords r0..r31 */
#define PVR_G5P_VERTEX_INPUT_DWORDS 32

/*
 * A pipeline's vertex input layout. pvr_g5p_vertex_layout_init() takes the
 * pipeline's state; pvr_g5p_vertex_lower() (in the vertex stage's NIR, before
 * the backend) fills the rest. With every read location below 8 the layout
 * is our ICD's: location L in r4L..r4L+3, so the NIR is left as it is. Past
 * that, the attributes are packed back to back and their loads rewritten to
 * the packed registers (virtual locations).
 */
struct pvr_g5p_vertex_layout {
   /* in */
   const struct vk_vertex_input_state *vi;
   bool dynamic_strides;

   /* out: the locations the program reads, whether they are packed, the
    * components per (virtual) location for the backend (it reads (0,0,0,1)
    * beyond them), and the fetches */
   uint32_t read;
   bool packed;
   uint8_t components[PCO_G5P_MAX_ATTRIBS];
   unsigned fetch_count;
   struct pvr_g5p_fetch fetch_list[PVR_G5P_MAX_FETCH];

   /* why a layout was refused */
   char refusal[160];
};

void pvr_g5p_vertex_layout_init(struct pvr_g5p_vertex_layout *layout, const struct vk_vertex_input_state *vi,
                                bool dynamic_strides);

/* The vertex attribute formats the A9 fetches: dwords of one attribute, 0 if
 * refused */
unsigned pvr_g5p_vertex_format_dwords(VkFormat format);

/* A vertex attribute format G5P fetches: 32-bit float and integer
 * components read as they are (convert < 0), or 8- and 16-bit normalized,
 * scaled and f16 components that the vertex program converts from the
 * fetched dwords (convert: enum pco_g5p_vs_format). Component c of the
 * attribute is at byte bytes * swizzle[c] (B8G8R8A8: 2, 1, 0, 3). */
struct pvr_g5p_vertex_format {
   uint8_t components, bytes;
   int8_t convert;
   bool packed_1010102; /* one raw dword; components unpacked in NIR */
   uint8_t swizzle[4];
};

bool pvr_g5p_vertex_format_info(VkFormat format, struct pvr_g5p_vertex_format *f);

/* The vertex stage's NIR (after the I/O lowering, before the backend): its
 * layout, and its loads rewritten when packed. false with layout->refusal. */
bool pvr_g5p_vertex_lower(struct nir_shader *vs, struct pvr_g5p_vertex_layout *layout);

#ifdef PVR_PER_ARCH
struct pvr_cmd_buffer;
struct pvr_g5p_pipeline;

/* What a draw needs of its vertex input: refused by name otherwise (the
 * index type, the bound buffers' alignment) */
VkResult PVR_PER_ARCH(g5p_vertex_draw_check)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g5p_pipeline *g,
                                             const struct pvr_g5p_draw *draw);

/* The draw's VDM index list: [VDM_STATE0/1: the cut index, indexed draws
 * with primitive restart], INDEX_LIST0, [INDEX_LIST1: the index buffer],
 * count, instances - 1, first vertex (non-indexed) or vertex offset
 * (indexed), split count. *n words into w[8]. */
VkResult PVR_PER_ARCH(g5p_vertex_index_list)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g5p_draw *draw,
                                             uint32_t hw_topology, uint32_t split, uint32_t *w, unsigned *n);

#   define pvr_arch_g5p_vertex_draw_check PVR_PER_ARCH(g5p_vertex_draw_check)
#   define pvr_arch_g5p_vertex_index_list PVR_PER_ARCH(g5p_vertex_index_list)
#endif

#endif /* PVR_G5P_VERTEX_H */
