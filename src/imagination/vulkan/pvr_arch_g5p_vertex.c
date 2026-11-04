/* SPDX-License-Identifier: MIT */
/*
 * Vertex input of G5P (Apple A9) draws (pvr_g5p_vertex.h): what a draw's
 * vertex input needs, the VDM index list of plain, instanced and indexed
 * draws. Every word is derived as the A9 ICD's
 * recorder derives it (g5p_record.c g5p_job_draw(), section 11, "Index
 * list"; device-proven there with 8/16/32-bit indices, vertex offsets and
 * instances). Compiled only with -Dimagination-g5p-experimental=true.
 */

#include <inttypes.h>
#include <stdio.h>

#include "vk_command_buffer.h"
#include "vk_graphics_state.h"
#include "vk_log.h"
#include "vk_util.h"

#include "pvr_buffer.h"
#include "pvr_cmd_buffer.h"
#include "pvr_csb.h"
#include "pvr_device.h"
#include "pvr_g5p.h"
#include "pvr_pipeline.h"
#include "pvr_g5p_vertex.h"

/* A refusal, by name, on stderr too (vk_errorf() is silent in release builds) */
#define G5P_REFUSE(obj, ...)                                                                                       \
   (fprintf(stderr, "G5P refuses: " __VA_ARGS__), fprintf(stderr, "\n"),                                          \
    vk_errorf((obj), VK_ERROR_FEATURE_NOT_PRESENT, "G5P: " __VA_ARGS__))

VkResult
PVR_PER_ARCH(g5p_vertex_draw_check)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g5p_pipeline *g,
                                    const struct pvr_g5p_draw *draw)
{
   const struct pvr_cmd_buffer_state *cs = &cmd_buffer->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd_buffer->vk.dynamic_graphics_state;

   if (draw->indexed) {
      const VkIndexType t = cs->index_buffer_binding.type;
      if (!cs->index_buffer_binding.buffer)
         return G5P_REFUSE(cmd_buffer, "an indexed draw without an index buffer");
      if (t != VK_INDEX_TYPE_UINT16 && t != VK_INDEX_TYPE_UINT32 && t != VK_INDEX_TYPE_UINT8_KHR)
         return G5P_REFUSE(cmd_buffer, "index type %d", t);
   }
   /* the DDMAD reads dwords: every fetch's base and stride are dword-aligned */
   for (unsigned f = 0; f < g->fetch_count; f++) {
      const struct pvr_g5p_fetch *fe = &g->fetch_list[f];
      const struct pvr_vertex_binding *vb = &cs->vertex_bindings[fe->binding];
      if (!vb->buffer)
         return G5P_REFUSE(cmd_buffer, "no vertex buffer bound at binding %u", fe->binding);
      if ((vb->offset + fe->offset) % 4 || dyn->vi_binding_strides[fe->binding] % 4)
         return G5P_REFUSE(cmd_buffer, "vertex binding %u at offset %" PRIu64 ", stride %u (the fetch is dword-granular)",
                           fe->binding, (uint64_t)(vb->offset + fe->offset), dyn->vi_binding_strides[fe->binding]);
   }
   return VK_SUCCESS;
}

VkResult
PVR_PER_ARCH(g5p_vertex_index_list)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g5p_draw *draw,
                                    uint32_t hw_topology, uint32_t split, uint32_t *w, unsigned *n)
{
   const struct pvr_cmd_buffer_state *cs = &cmd_buffer->state;
   uint64_t index_va = 0;
   unsigned isize = 4;

   if (draw->indexed) {
      const struct pvr_buffer *ib = cs->index_buffer_binding.buffer;
      if (!ib)
         return G5P_REFUSE(cmd_buffer, "an indexed draw without an index buffer");
      isize = vk_index_type_to_bytes(cs->index_buffer_binding.type);
      index_va = ib->dev_addr.addr + cs->index_buffer_binding.offset + (uint64_t)draw->first * isize;
   }
   *n = 0;
   /* primitive restart (Apple's g3b-idx16-restart: a VDM state update of its
    * own before the index list, VDM_STATE0 with the cut index present and
    * enabled, then VDM_STATE1 the cut index, all ones of the index size).
    * The draw's own VDM_STATE0 (cut_index_enable 0) ends it at the next
    * draw. Its cam size is the draw's (g5p_cmd_draw()'s step 1). */
   if (draw->indexed && cmd_buffer->vk.dynamic_graphics_state.ia.primitive_restart_enable) {
      const unsigned vector = cs->gfx_pipeline->g5p->vs_outputs + 5;
      pvr_csb_pack (&w[(*n)++], VDMCTRL_VDM_STATE0, v0) {
         v0.cut_index_present = true;
         v0.cut_index_enable = true;
         v0.cam_size = vector <= 32 ? 95 : vector <= 48 ? 63 : vector <= 64 ? 31 : 15;
      }
      pvr_csb_pack (&w[(*n)++], VDMCTRL_VDM_STATE1, v1) {
         v1.cut_index = isize == 1 ? 0xff : isize == 2 ? 0xffff : 0xffffffff;
      }
   }
   pvr_csb_pack (&w[(*n)++], VDMCTRL_INDEX_LIST0, l0) {
      l0.index_count_present = true;
      l0.index_instance_count_present = true;
      l0.index_offset_present = true;
      l0.split_count_present = true;
      l0.index_size = isize == 1   ? ROGUE_VDMCTRL_INDEX_SIZE_B8
                      : isize == 2 ? ROGUE_VDMCTRL_INDEX_SIZE_B16
                                   : ROGUE_VDMCTRL_INDEX_SIZE_B32;
      l0.primitive_topology = hw_topology;
      if (draw->indexed) {
         l0.index_addr_present = true;
         l0.index_base_addrmsb = PVR_DEV_ADDR(index_va);
      }
   }
   if (draw->indexed) {
      pvr_csb_pack (&w[(*n)++], VDMCTRL_INDEX_LIST1, l1) {
         l1.index_base_addrlsb = PVR_DEV_ADDR(index_va);
      }
   }
   w[(*n)++] = draw->count;
   w[(*n)++] = draw->instances - 1;
   /* indexed: the vertex offset (added to every index, so the PDS's vertex
    * index is gl_VertexIndex); else the first vertex */
   w[(*n)++] = draw->indexed ? (uint32_t)draw->vertex_offset : draw->first;
   w[(*n)++] = split;
   return VK_SUCCESS;
}
