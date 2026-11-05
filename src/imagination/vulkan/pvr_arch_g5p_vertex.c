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

#include "pvr_bo.h"
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

/* Native A9 indirect INDEX_LIST: the VDM reads the Vulkan/Metal argument
 * layout directly. Two 16-bit conditional links skip instanceCount == 0,
 * avoiding the hardware's count-minus-one underflow. This is the protocol
 * emitted by the A9 draw-indirect encoder, expressed with csbgen fields.
 *
 * Put the whole self-addressing block in a separate allocation and enter it
 * with RETURN. Flattening a long main CSB must not relocate its branch
 * targets. No PDS/USC program reads or rewrites the argument buffer.
 */
VkResult
PVR_PER_ARCH(g5p_vertex_indirect)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g5p_draw *draw,
                                  uint32_t topology, uint32_t split, struct pvr_csb *csb)
{
   uint32_t words[14 + ROGUE_VDMCTRL_GUARD_SIZE_DEFAULT / 4] = { 0 };
   struct pvr_suballoc_bo *bo;
   VkResult result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, words, sizeof(words), &bo);
   if (result != VK_SUCCESS)
      return result;
   const uint64_t va = bo->dev_addr.addr;
   const unsigned end = draw->indexed ? 13 : 12;
   /* Low half nonzero: go to the draw. Otherwise high half zero: skip it. */
   for (unsigned c = 0; c < 2; c++) {
      const uint64_t compare = draw->indirect_addr + 4 + 2 * c;
      const uint64_t target = va + 4 * (c ? end : 8);
      pvr_csb_pack (&words[4 * c], VDMCTRL_STREAM_LINK0, link) {
         link.compare_present = true;
         link.compare_mode = c ? 2 : 5;
         link.link_addrmsb = PVR_DEV_ADDR(target);
      }
      pvr_csb_pack (&words[4 * c + 1], VDMCTRL_STREAM_LINK1, link) {
         link.link_addrlsb = PVR_DEV_ADDR(target);
      }
      /* STREAM_LINK comparison address is 40-bit, 16-bit aligned. */
      words[4 * c + 2] = compare >> 32;
      words[4 * c + 3] = (uint32_t)compare & ~1u;
   }
   uint64_t index_va = 0;
   unsigned isize = 4;
   if (draw->indexed) {
      const struct pvr_cmd_buffer_state *state = &cmd_buffer->state;
      isize = vk_index_type_to_bytes(state->index_buffer_binding.type);
      index_va = state->index_buffer_binding.buffer->dev_addr.addr + state->index_buffer_binding.offset;
   }
   unsigned n = 8;
   pvr_csb_pack (&words[n++], VDMCTRL_INDEX_LIST0, list) {
      list.indirect_addr_present = true;
      list.split_count_present = true;
      list.index_size = isize == 1 ? ROGUE_VDMCTRL_INDEX_SIZE_B8 :
                        isize == 2 ? ROGUE_VDMCTRL_INDEX_SIZE_B16 : ROGUE_VDMCTRL_INDEX_SIZE_B32;
      list.primitive_topology = topology;
      list.index_addr_present = draw->indexed;
      list.index_base_addrmsb = PVR_DEV_ADDR(index_va);
   }
   if (draw->indexed) {
      pvr_csb_pack (&words[n++], VDMCTRL_INDEX_LIST1, list) {
         list.index_base_addrlsb = PVR_DEV_ADDR(index_va);
      }
   }
   pvr_csb_pack (&words[n++], VDMCTRL_INDEX_LIST7, list) {
      list.indirect_base_addrmsb = PVR_DEV_ADDR(draw->indirect_addr);
   }
   pvr_csb_pack (&words[n++], VDMCTRL_INDEX_LIST8, list) {
      list.indirect_base_addrlsb = PVR_DEV_ADDR(draw->indirect_addr);
   }
   words[n++] = split;
   assert(n == end);
   pvr_csb_pack (&words[n++], VDMCTRL_STREAM_RETURN, ret);
   memcpy(pvr_bo_suballoc_get_map_addr(bo), words, sizeof(words));
   if (draw->indexed && cmd_buffer->vk.dynamic_graphics_state.ia.primitive_restart_enable) {
      const unsigned vector = cmd_buffer->state.gfx_pipeline->g5p->vs_outputs + 5;
      pvr_csb_emit (csb, VDMCTRL_VDM_STATE0, state) {
         state.cut_index_present = true;
         state.cut_index_enable = true;
         state.cam_size = vector <= 32 ? 95 : vector <= 48 ? 63 : vector <= 64 ? 31 : 15;
      }
      pvr_csb_emit (csb, VDMCTRL_VDM_STATE1, state) {
         state.cut_index = isize == 1 ? 0xff : isize == 2 ? 0xffff : 0xffffffff;
      }
   }
   pvr_csb_emit (csb, VDMCTRL_STREAM_LINK0, link) {
      link.with_return = true;
      link.link_addrmsb = PVR_DEV_ADDR(va);
   }
   pvr_csb_emit (csb, VDMCTRL_STREAM_LINK1, link) {
      link.link_addrlsb = PVR_DEV_ADDR(va);
   }
   if (getenv("PVR_G5P_INDIRECT_LOG")) {
      fprintf(stderr, "G5P-INDIRECT packet va=%#" PRIx64 " args=%#" PRIx64 " indexed=%u words=%u:",
              va, draw->indirect_addr, draw->indexed, n);
      for (unsigned i = 0; i < n; i++)
         fprintf(stderr, " %08x", words[i]);
      fprintf(stderr, "\n");
   }
   return csb->status;
}

VkResult
PVR_PER_ARCH(g5p_cmd_draw_indirect)(struct pvr_cmd_buffer *cmd_buffer, struct pvr_buffer *buffer,
                                    VkDeviceSize offset, uint32_t count, uint32_t stride, bool indexed)
{
   const char *probe = getenv("PVR_G5P_INDIRECT");
   if (!probe || strcmp(probe, "probe"))
      return G5P_REFUSE(cmd_buffer, "indirect graphics draw awaits native A9 qualification (PVR_G5P_INDIRECT=probe)");
   if (!count)
      return VK_SUCCESS;
   const unsigned size = indexed ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);
   if (!buffer || offset % 4 || offset > buffer->vk.size || size > buffer->vk.size - offset ||
       (count > 1 && (stride < size || stride % 4 ||
                     (uint64_t)(count - 1) * stride > buffer->vk.size - offset - size)))
      return G5P_REFUSE(cmd_buffer, "indirect graphics argument range/alignment");
   /* The native VDM supplies firstInstance in PDS persistent temporary 1.
    * The indirect vertex-fetch variant applies it on the GPU. XFB's CPU
    * resolve cannot account for unknown GPU counts and refuses.
    */
   for (unsigned i = 0; i < count; i++) {
      const struct pvr_g5p_draw draw = {
         .count = 1, .instances = 1, .indexed = indexed,
         .indirect_addr = buffer->dev_addr.addr + offset + (uint64_t)i * stride,
      };
      VkResult result = pvr_arch_g5p_cmd_draw(cmd_buffer, &draw);
      if (result != VK_SUCCESS)
         return result;
   }
   return VK_SUCCESS;
}
