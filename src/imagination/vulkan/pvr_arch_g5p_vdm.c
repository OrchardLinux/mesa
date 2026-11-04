/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): a render's VDM control stream of more than one
 * pvr_csb block (pvr_g5p_vdm.h). Compiled only with
 * -Dimagination-g5p-experimental=true.
 *
 * pvr_csb records a graphics control stream in blocks of
 * PVR_CMD_BUFFER_CSB_BO_SIZE (4 KiB), each in its own allocation, and chains
 * them with VDMCTRL_STREAM_LINK0/1 (pvr_arch_csb.c pvr_csb_buffer_extend():
 * the link is written where the block's last complete state update ended,
 * and that update moves to the next block). The A9's VDM follows such links
 * (Apple's own render streams carry STREAM_LINK `80000001 <addr>`, the same
 * encoding; docs/gpu/experiments/frida-agx-findings.md), but our kernel takes
 * a pass's stream as one range: [vdm_va, vdm_va + vdm_size) inside one
 * listed resource, at most DRM_G5P_RENDER_MAX_VDM_BYTES, with the stream's
 * STREAM_TERMINATE as its last dword at vdm_va + vdm_end - 4
 * (render-user-validate.h g5p_ru_pass3_check(), g5p_ru_pass3_content()).
 *
 * So a stream of several blocks is laid out again as one: each block's
 * commands up to its chaining link, then the last block's up to the
 * terminate, copied in order into one upload, followed by
 * ROGUE_VDMCTRL_GUARD_SIZE_DEFAULT zero bytes (the VDM reads ahead; no
 * terminate in the last 64 bytes of the last valid page). The commands are
 * unchanged: none of them addresses the stream itself (a state update that
 * crossed a block boundary was moved whole by pvr_csb, and a STREAM_LINK
 * with return to a secondary stream returns to the dword after it, wherever
 * that is). Only the chaining links, which jump to the next block, are left
 * out. A stream longer than the kernel's limit is refused by name.
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vk_log.h"

#include "pvr_bo.h"
#include "pvr_cmd_buffer.h"
#include "pvr_csb.h"
#include "pvr_g5p_vdm.h"
#include "winsys/apple_g5p/apple_g5p_drm.h"

#define G5P_VDM_REFUSE(cmd_buffer, ...)                                                                            \
   (fprintf(stderr, "G5P refuses: " __VA_ARGS__), fprintf(stderr, "\n"),                                          \
    vk_errorf((cmd_buffer), VK_ERROR_FEATURE_NOT_PRESENT, "G5P: " __VA_ARGS__))

/* The bytes of a block before its chaining link to `next`, or -1 when the
 * block holds no such link */
static int64_t
block_commands(const struct pvr_bo *block, pvr_dev_addr_t next)
{
   uint32_t link[2];

   pvr_csb_pack (&link[0], VDMCTRL_STREAM_LINK0, l) {
      l.link_addrmsb = next;
      l.with_return = false;
   }
   pvr_csb_pack (&link[1], VDMCTRL_STREAM_LINK1, l) {
      l.link_addrlsb = next;
   }
   const uint32_t *w = block->bo->map;
   const uint64_t dwords = block->bo->size / 4;
   for (uint64_t i = 0; i + 1 < dwords; i++)
      if (w[i] == link[0] && w[i + 1] == link[1])
         return (int64_t)i * 4;
   return -1;
}

VkResult
PVR_PER_ARCH(g5p_vdm_stream)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_csb *csb, uint64_t *va,
                             uint32_t *bytes)
{
   const unsigned blocks = list_length(&csb->pvr_bo_list);
   const uint32_t last = (uint32_t)((const uint8_t *)csb->next - (const uint8_t *)csb->start);

   if (blocks <= 1) {
      *va = pvr_csb_get_start_address(csb).addr;
      *bytes = last;
      return VK_SUCCESS;
   }

   /* the commands of every block but the last, up to its link to the next */
   uint64_t total = last;
   unsigned i = 0;
   list_for_each_entry (const struct pvr_bo, b, &csb->pvr_bo_list, link) {
      if (++i == blocks)
         break;
      const struct pvr_bo *n = list_entry(b->link.next, struct pvr_bo, link);
      const int64_t len = block_commands(b, n->vma->dev_addr);
      if (len < 0)
         return G5P_VDM_REFUSE(cmd_buffer, "control stream block %u of %u has no link to block %u", i, blocks,
                               i + 1);
      total += (uint64_t)len;
   }
   if (total > DRM_G5P_RENDER_MAX_VDM_BYTES)
      return G5P_VDM_REFUSE(cmd_buffer, "a control stream of %" PRIu64 " bytes in %u blocks (the kernel takes %u)",
                            total, blocks, DRM_G5P_RENDER_MAX_VDM_BYTES);

   const size_t size = total + ROGUE_VDMCTRL_GUARD_SIZE_DEFAULT;
   uint8_t *flat = calloc(1, size);
   if (!flat)
      return vk_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY);
   uint64_t at = 0;
   i = 0;
   list_for_each_entry (const struct pvr_bo, b, &csb->pvr_bo_list, link) {
      const uint64_t len = ++i == blocks ? last
                                         : (uint64_t)block_commands(b, list_entry(b->link.next, struct pvr_bo,
                                                                                  link)->vma->dev_addr);
      memcpy(flat + at, b->bo->map, len);
      at += len;
   }
   struct pvr_suballoc_bo *bo;
   VkResult result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, flat, size, &bo);
   free(flat);
   if (result != VK_SUCCESS)
      return result;
   if (getenv("PVR_G5P_WS_LOG"))
      fprintf(stderr, "g5p vdm: %u blocks, %" PRIu64 " bytes laid out as one stream at %#" PRIx64 "\n", blocks,
              total, bo->dev_addr.addr);
   *va = bo->dev_addr.addr;
   *bytes = (uint32_t)total;
   return VK_SUCCESS;
}
