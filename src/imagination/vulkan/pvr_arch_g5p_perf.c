/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): the desktop speed work (pvr_g5p_perf.h,
 * docs/gpu/orchard-port-perf.md). Compiled only with
 * -Dimagination-g5p-experimental=true.
 */

#include <string.h>

#include "vulkan/vulkan_core.h"
#include "vk_pipeline_layout.h"

#include "pvr_buffer.h"
#include "pvr_cmd_buffer.h"
#include "pvr_descriptor_set.h"
#include "pvr_device.h"
#include "pvr_g5p.h"
#include "pvr_g5p_perf.h"
#include "pvr_hw_pass.h"
#include "pvr_image.h"
#include "pvr_pipeline.h"

static void
note_write(struct g5p_perf_tail *t, const struct pvr_image_view *iview)
{
   if (!iview) {
      /* an attachment we can't place: the winsys must assume any own BO is written */
      t->flags &= ~G5P_PERF_WRITES_KNOWN;
      return;
   }
   const struct pvr_image *image = pvr_image_view_get_image(iview);
   const uint64_t va = image->dev_addr.addr;

   for (unsigned i = 0; i < t->writes; i++)
      if (t->write_va[i] == va)
         return;
   if (t->writes == G5P_PERF_MAX_WRITES) {
      t->flags &= ~G5P_PERF_WRITES_KNOWN;
      return;
   }
   t->write_va[t->writes++] = va;
}

/*
 * What the render writes outside the arena can only be its attachments: the
 * G5P backend compiles no fragment memory store (pco_g5p_general.c
 * store_refusal(): a store is a transform feedback store or refused), so the
 * one other writer is transform feedback, whose buffers this does not track:
 * a command buffer with transform feedback state leaves the writes (and the
 * reads) unknown. The base address of an attachment's image marks the BO it
 * lies in, so a view of any level or layer marks the right BO; a 4x render's
 * resolve target is its EOT surface's view (pvr_arch_g5p_job_init()).
 *
 * The reads the pass's draws collected (pvr_arch_g5p_perf_draw()) go into an
 * upload of the command buffer, in the arena, for the winsys to read at
 * submit; the tail carries its address.
 */
void
PVR_PER_ARCH(g5p_perf_job)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_renderpass_hwsetup_render *hw_render,
                           const struct pvr_g5p_perf_reads *reads, struct g5p_perf_tail *t)
{
   const struct pvr_render_pass_info *info = &cmd_buffer->state.render_pass_info;

   memset(t, 0, sizeof(*t));
   t->magic = G5P_PERF_TAIL_MAGIC;
   t->flags = G5P_PERF_WRITES_KNOWN;
   if (cmd_buffer->state.g5p_xfb)
      t->flags &= ~G5P_PERF_WRITES_KNOWN;

   for (unsigned s = 0; s < hw_render->eot_surface_count; s++) {
      const uint32_t idx = hw_render->eot_surfaces[s].attachment_idx;
      note_write(t, idx < info->attachment_count ? info->attachments[idx] : NULL);
   }
   /* a pass that clears nothing: a draw-less job of it changes no pixel (counted by the winsys's log; the next
    * slice skips it) */
   bool clears = hw_render->depth_init == VK_ATTACHMENT_LOAD_OP_CLEAR ||
                 hw_render->stencil_init == VK_ATTACHMENT_LOAD_OP_CLEAR;
   for (unsigned i = 0; i < hw_render->color_init_count; i++)
      clears |= hw_render->color_init[i].op == VK_ATTACHMENT_LOAD_OP_CLEAR;
   if (!clears)
      t->flags |= G5P_PERF_NO_CLEAR;
   if (hw_render->ds_attach_idx != VK_ATTACHMENT_UNUSED)
      note_write(t, hw_render->ds_attach_idx < info->attachment_count ? info->attachments[hw_render->ds_attach_idx]
                                                                      : NULL);

   /* the reads */
   if (!(t->flags & G5P_PERF_WRITES_KNOWN) || reads->unknown || reads->count > G5P_PERF_MAX_READS)
      return;
   if (reads->count) {
      struct pvr_suballoc_bo *bo;
      if (pvr_arch_cmd_buffer_upload_general(cmd_buffer, reads->va, 8 * reads->count, &bo) != VK_SUCCESS)
         return; /* reads unknown: every own BO is listed */
      t->reads_va = bo->dev_addr.addr;
      t->reads = reads->count;
   }
   t->flags |= G5P_PERF_READS_KNOWN;
}

/* ---- reads ---- */

struct reads_ctx {
   struct pvr_g5p_perf_reads *r;
   uint64_t arena_lo, arena_hi; /* the arena: every heap of the apple_g5p winsys names it */
};

static void
note_read(struct reads_ctx *c, uint64_t va)
{
   struct pvr_g5p_perf_reads *r = c->r;

   if (!va || (va >= c->arena_lo && va < c->arena_hi))
      return;
   for (unsigned i = 0; i < r->count; i++)
      if (r->va[i] == va)
         return;
   if (r->count == G5P_PERF_MAX_READS) {
      r->unknown = true;
      return;
   }
   r->va[r->count++] = va;
}

/* The address of a G5P image state (the A9's three-qword texture state the
 * image-view hook packs, pvr_g5p_tex_encode.h pvr_g5p_tex_linear() and
 * pvr_g5p_tex_twiddled()): q1 holds va >> 1 in its low 45 bits, the level
 * count and the has-mips bit in bits 0..4 (an image is at least 64-byte
 * aligned, so va >> 1 has them clear), and sRGB, depth, stride and layer
 * bits from bit 45 up. 0 for an empty descriptor (a valid state never has q0
 * = 0, pvr_g5p_tex_state_valid()). */
static uint64_t
image_state_va(const uint8_t *desc)
{
   uint64_t q[2];

   memcpy(q, desc, sizeof(q));
   if (!q[0])
      return 0;
   return (q[1] & ((1ull << 45) - 1) & ~0x1full) << 1;
}

static void
note_set(struct reads_ctx *c, const struct pvr_descriptor_set *set)
{
   const struct pvr_descriptor_set_layout *layout = set->layout;

   for (unsigned b = 0; b < layout->binding_count; b++) {
      const struct pvr_descriptor_set_layout_binding *lb = &layout->bindings[b];

      for (unsigned e = 0; e < lb->descriptor_count; e++) {
         const uint8_t *desc = (const uint8_t *)set->mapping + lb->offset + e * lb->stride;

         switch (lb->type) {
         case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
         case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
            struct pvr_buffer_descriptor bd;
            memcpy(&bd, desc, sizeof(bd));
            note_read(c, bd.addr);
            break;
         }
         case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
         case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            /* the base address marks the buffer's BO; the dynamic offset stays inside the buffer */
            note_read(c, set->dynamic_buffers[lb->dynamic_buffer_idx + e].addr);
            break;
         case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: /* the image state first */
         case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            note_read(c, image_state_va(desc));
            break;
         case VK_DESCRIPTOR_TYPE_SAMPLER:
         case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
            break;
         default:
            /* storage images, input attachments and texel buffers carry states the G5P view hook does not pack
             * (PVR_TEXTURE_STATE_STORAGE/ATTACHMENT, buffer views): their addresses aren't decoded here */
            c->r->unknown = true;
            return;
         }
      }
   }
}

/*
 * Everything a draw can read outside the arena: the descriptors of every set
 * its pipeline layout names (the backend loads uniforms, buffer addresses and
 * texture states from them, pvr_arch_g5p_gfx.c fs_uploads() and the vertex
 * uploads), the vertex buffers of its fetches, and the index buffer. Over-
 * collecting only lists a BO the job doesn't touch; a read that can't be
 * decoded marks the pass's reads unknown, which lists every own BO.
 */
void
PVR_PER_ARCH(g5p_perf_draw)(struct pvr_cmd_buffer *cmd_buffer, bool indexed)
{
   struct pvr_cmd_buffer_state *cs = &cmd_buffer->state;
   if (!cs->current_sub_cmd || cs->current_sub_cmd->type != PVR_SUB_CMD_TYPE_GRAPHICS)
      return;
   struct pvr_g5p_perf_reads *r = &cs->current_sub_cmd->gfx.job.g5p.reads;
   const struct pvr_winsys_heap *heap = cmd_buffer->device->heaps.general_heap;
   struct reads_ctx c = { r, heap->base_addr.addr, heap->base_addr.addr + heap->size };

   if (r->unknown)
      return;
   if (cs->g5p_xfb) {
      r->unknown = true;
      return;
   }
   const struct pvr_graphics_pipeline *pipeline = cs->gfx_pipeline;
   const struct pvr_g5p_pipeline *g = pipeline ? pipeline->g5p : NULL;
   if (!g) {
      r->unknown = true;
      return;
   }
   const struct vk_pipeline_layout *layout = pipeline->base.layout;
   for (unsigned s = 0; layout && s < layout->set_count && s < PVR_MAX_DESCRIPTOR_SETS && !r->unknown; s++) {
      const struct pvr_descriptor_set *set = cs->gfx_desc_state.sets[s];
      /* a set the layout names but nothing binds: the draw path refuses an upload from it */
      if (set && set->layout && set->mapping)
         note_set(&c, set);
   }
   for (unsigned f = 0; f < g->fetch_count; f++) {
      const struct pvr_vertex_binding *vb = &cs->vertex_bindings[g->fetch_list[f].binding];
      if (vb->buffer)
         note_read(&c, vb->buffer->dev_addr.addr);
   }
   if (indexed && cs->index_buffer_binding.buffer)
      note_read(&c, cs->index_buffer_binding.buffer->dev_addr.addr);
}
