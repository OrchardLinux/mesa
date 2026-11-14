/* SPDX-License-Identifier: MIT */
#ifndef PVR_G5P_META_H
#define PVR_G5P_META_H
#include "pvr_cmd_buffer.h"
#include "pvr_descriptor_set.h"
/* The application's graphics state that a meta draw replaces (hk_meta_save) */
struct g5p_meta_save {
   const struct pvr_graphics_pipeline *pipeline;
   struct pvr_vertex_binding vb0;
   struct pvr_descriptor_set *set0;
   bool has_push_set;
   struct pvr_push_descriptor_set push_set;
   struct pvr_push_constants push_consts[PVR_STAGE_ALLOCATION_COUNT];
   struct vk_dynamic_graphics_state dynamic;
};

static inline void
g5p_meta_begin(struct pvr_cmd_buffer *cmd_buffer, struct g5p_meta_save *save)
{
   struct pvr_cmd_buffer_state *state = &cmd_buffer->state;
   struct pvr_descriptor_state *desc = &state->gfx_desc_state;

   save->pipeline = state->gfx_pipeline;
   save->vb0 = state->vertex_bindings[0];
   save->set0 = desc->sets[0];
   save->has_push_set = desc->push_set != NULL;
   if (save->has_push_set)
      save->push_set = *desc->push_set;
   memcpy(save->push_consts, state->push_consts, sizeof(save->push_consts));
   save->dynamic = cmd_buffer->vk.dynamic_graphics_state;
}

static inline void
g5p_meta_end(struct pvr_cmd_buffer *cmd_buffer, struct g5p_meta_save *save)
{
   struct pvr_cmd_buffer_state *state = &cmd_buffer->state;
   struct pvr_descriptor_state *desc = &state->gfx_desc_state;
   struct vk_dynamic_graphics_state *dyn = &cmd_buffer->vk.dynamic_graphics_state;

   state->gfx_pipeline = save->pipeline;
   state->dirty.gfx_pipeline_binding = true;
   state->vertex_bindings[0] = save->vb0;
   state->dirty.vertex_bindings = true;
   desc->sets[0] = save->set0;
   desc->dirty_sets |= BITFIELD_BIT(0);
   if (save->has_push_set && desc->push_set) {
      *desc->push_set = save->push_set;
      desc->push_set_dirty = true;
   }
   state->dirty.gfx_desc_dirty = true;
   memcpy(state->push_consts, save->push_consts, sizeof(save->push_consts));
   for (unsigned s = 0; s < PVR_STAGE_ALLOCATION_COUNT; s++)
      state->push_consts[s].dirty = true;
   *dyn = save->dynamic;
   memcpy(dyn->dirty, dyn->set, sizeof(dyn->set));
}


#endif
