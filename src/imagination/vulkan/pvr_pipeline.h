/*
 * Copyright © 2022 Imagination Technologies Ltd.
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * based in part on radv driver which is:
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef PVR_PIPELINE_H
#define PVR_PIPELINE_H

#include "vk_object.h"
#include "vk_graphics_state.h"

#include "pvr_common.h"
#include "pvr_csb.h"
#include "pvr_g9p_xfb_core.h"
#include "pvr_pds.h"
#include "util/shader_stats.h"

struct pvr_suballoc_bo;

struct pvr_pipeline_stage_state {
   uint32_t pds_temps_count;
};

struct pvr_compute_shader_state {
   /* Pointer to a buffer object that contains the shader binary. */
   struct pvr_suballoc_bo *shader_bo;

   /* Buffer object for the coefficient update shader binary. */
   struct pvr_suballoc_bo *coeff_update_shader_bo;
   uint32_t coeff_update_shader_temps;
};

struct pvr_pds_attrib_program {
   struct pvr_pds_info info;
   /* The uploaded PDS program stored here only contains the code segment,
    * meaning the data size will be 0, unlike the data size stored in the
    * 'info' member above.
    */
   struct pvr_pds_upload program;
};

struct pvr_stage_allocation_descriptor_state {
   struct pvr_pds_upload pds_code;
   /* Since we upload the code segment separately from the data segment
    * pds_code->data_size might be 0 whilst
    * pds_info->data_size_in_dwords might be >0 in the case of this struct
    * referring to the code upload.
    */
   struct pvr_pds_info pds_info;

   /* Already setup compile time static consts. */
   struct pvr_suballoc_bo *static_consts;
};

struct pvr_vertex_shader_state {
   /* Pointer to a buffer object that contains the shader binary. */
   struct pvr_suballoc_bo *shader_bo;

   struct pvr_pds_attrib_program
      pds_attrib_programs[PVR_PDS_VERTEX_ATTRIB_PROGRAM_COUNT];

   struct pvr_pipeline_stage_state stage_state;
   /* FIXME: Move this into stage_state? */
   struct pvr_stage_allocation_descriptor_state descriptor_state;
};

struct pvr_fragment_shader_state {
   /* Pointer to a buffer object that contains the shader binary. */
   struct pvr_suballoc_bo *shader_bo;

   struct pvr_pipeline_stage_state stage_state;
   /* FIXME: Move this into stage_state? */
   struct pvr_stage_allocation_descriptor_state descriptor_state;
   enum ROGUE_TA_PASSTYPE pass_type;

   struct pvr_pds_coeff_loading_program pds_coeff_program;
   uint32_t *pds_coeff_program_buffer;

   struct pvr_pds_kickusc_program pds_fragment_program;
   uint32_t *pds_fragment_program_buffer;

   bool is_passthrough;
};

struct pvr_xfb_varying {
   uint32_t register_start;
   uint32_t register_count;
   uint32_t xfb_buffer;
   uint32_t xfb_offset;
   uint32_t xfb_stride;
};

struct pvr_xfb_shader_state {
   uint32_t varying_count;
   struct pvr_xfb_varying *varyings;
   uint32_t buffer_strides[PVR_XFB_MAX_BUFFERS];

   uint32_t buffer_group_count;
   uint32_t group_varying_count[PVR_XFB_MAX_BUFFERS];

   struct pvr_suballoc_bo *counter_buffer_offsets;

   struct pvr_suballoc_bo *indirect_byte_count;
};

struct pvr_pipeline {
   struct vk_object_base base;
   enum pvr_pipeline_type type;
   struct vk_pipeline_layout *layout;
   VkPipelineCreateFlags2KHR pipeline_flags;
};

/* Apple G9P (has_apple_g9p): the shader and what its dispatches load */
struct pvr_g9p_compute_state {
   struct pvr_suballoc_bo *shader_bo;
   unsigned num_buffers;
   struct {
      unsigned set, binding;
   } buffers[36]; /* G9P_MAX_BUFFERS */
   unsigned num_regs;
   bool uses_workgroup_id;
   unsigned local_size[3];
};

/* Apple G9P (has_apple_g9p): a graphics pipeline's programs (pvr_arch_g9p_gfx.c) */
struct pvr_g9p_graphics_state {
   /* a number for debug output (G9P_DEBUG prints it before its shaders) */
   unsigned id;
   /*
    * With cube samples: the same pipeline sampling them as 2D arrays per face,
    * for draws whose sampler asks for no filtering across faces
    */
   struct pvr_g9p_graphics_state *seamful;
   /* and with GL's magnification threshold in explicit levels */
   struct pvr_g9p_graphics_state *seamful_mag;
   bool is_seamful;
   struct pvr_suballoc_bo *vs_bo, *fs_bo;
   /* vertex kick (code and data), fragment kick bundle, DOUTI program */
   struct pvr_pds_upload vs_kick, fs_kick;
   /* The DOUTI program by provoking vertex; only [0] without flat varyings */
   struct pvr_pds_upload varyings[3];
   bool flat_varyings;
   /* the DOUTI data in 16-byte units */
   unsigned varying_units, varying_coeffs;
   unsigned num_varyings;
   /* vertex output dwords */
   unsigned vs_outputs;
   /* vertex shader registers (VDM_STATE5 sets their allocation) */
   unsigned vs_regs;
   /* the fragment shader may discard: punch-through pass type */
   bool uses_discard;
   bool uses_fbfetch;
   /* the vertex shader's common-store slots: vertex buffers, push constants, buffers */
   unsigned num_buffers;
   struct {
      unsigned set, binding;
   } buffers[36]; /* G9P_MAX_BUFFERS */
   /*
    * the fragment shader's buffers (uniform buffers, push constants), by
    * slot: their first dwords are DMA'd into the common store
    */
   unsigned fs_num_buffers;
   unsigned num_fs_runs;
   struct {
      uint16_t slot, first, count, dest;
   } fs_runs[64]; /* G9P_MAX_FS_RUNS */
   struct {
      unsigned set, binding, dwords;
   } fs_buffers[36]; /* G9P_MAX_BUFFERS */
   /* the fragment shader's common store in dwords */
   unsigned fs_common_dwords;
   /* the fragment shader's combined image samplers, by common-store slot */
   unsigned num_textures;
   struct {
      unsigned set, binding, index;
   } textures[16]; /* G9P_MAX_TEXTURES */
   /* the vertex shader's, and the size of its common store in dwords */
   unsigned vs_num_textures, vs_common_dwords;
   /* transform feedback: the vertex shader also stores what it captures */
   bool has_xfb;
   /* the fragment shader writes gl_FragDepth */
   bool writes_depth;
   /* and takes derivatives: no triangle merging, as for textures */
   bool uses_derivatives;
   /* samples with coordinates iterated straight to the sampler */
   bool straight_samples;
   bool writes_coverage;
   struct g9p_xfb_layout xfb_layout;
   bool vs_point_size;
   struct {
      unsigned set, binding, index;
   } vs_textures[16]; /* G9P_MAX_TEXTURES */
};

struct pvr_compute_pipeline {
   struct pvr_pipeline base;

   struct pvr_g9p_compute_state g9p;

   pco_data cs_data;

   struct pvr_compute_shader_state shader_state;
   struct pvr_stage_allocation_descriptor_state descriptor_state;

   struct pvr_pds_upload pds_cs_program;
   struct pvr_pds_info pds_cs_program_info;

   uint32_t *pds_cs_data_section;
   uint32_t base_workgroup_data_patching_offset;
   uint32_t num_workgroups_data_patching_offset;
   uint32_t num_workgroups_indirect_src_patching_offset;
   uint32_t num_workgroups_indirect_src_dma_patching_offset;

   /* Debug Info */
   struct pvr_stats *cs_stats;
   const char *cs_nir_str;
};

struct pvr_graphics_pipeline {
   struct pvr_pipeline base;

   struct pvr_g9p_graphics_state g9p;

   struct vk_dynamic_graphics_state dynamic_state;

   /* Derived and other state */
   size_t stage_indices[MESA_SHADER_STAGES];

   pco_data vs_data;
   pco_data fs_data;

   struct {
      struct pvr_vertex_shader_state vertex;
      struct pvr_fragment_shader_state fragment;
      struct pvr_xfb_shader_state xfb_shader_state;
   } shader_state;

   /* Debug Info */
   struct pvr_stats *vs_stats;
   struct pvr_stats *fs_stats;

   const char *vs_nir_str;
   const char *fs_nir_str;
};

struct pvr_private_compute_pipeline {
   /* Used by pvr_compute_update_kernel_private(). */
   uint32_t pds_code_offset;
   uint32_t pds_data_offset;
   uint32_t pds_data_size_dw;
   uint32_t pds_temps_used;
   uint32_t coeff_regs_count;
   uint32_t unified_store_regs_count;
   VkExtent3D workgroup_size;

   /* Used by pvr_compute_update_shared_private(). */
   uint32_t pds_shared_update_code_offset;
   uint32_t pds_shared_update_data_offset;
   uint32_t pds_shared_update_data_size_dw;

   /* Used by both pvr_compute_update_{kernel,shared}_private(). */
   uint32_t const_shared_regs_count;

   pvr_dev_addr_t const_buffer_addr;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(pvr_pipeline,
                               base,
                               VkPipeline,
                               VK_OBJECT_TYPE_PIPELINE)

static inline struct pvr_compute_pipeline *
to_pvr_compute_pipeline(struct pvr_pipeline *pipeline)
{
   assert(pipeline->type == PVR_PIPELINE_TYPE_COMPUTE);
   return container_of(pipeline, struct pvr_compute_pipeline, base);
}

static inline struct pvr_graphics_pipeline *
to_pvr_graphics_pipeline(struct pvr_pipeline *pipeline)
{
   assert(pipeline->type == PVR_PIPELINE_TYPE_GRAPHICS);
   return container_of(pipeline, struct pvr_graphics_pipeline, base);
}

static enum pvr_pipeline_stage_bits
pvr_stage_mask(VkPipelineStageFlags2 stage_mask)
{
   enum pvr_pipeline_stage_bits stages = 0;

   if (stage_mask & VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
      return PVR_PIPELINE_STAGE_ALL_BITS;

   if (stage_mask & (VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT))
      stages |= PVR_PIPELINE_STAGE_ALL_GRAPHICS_BITS;

   if (stage_mask & (VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                     VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
                     VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT)) {
      stages |= PVR_PIPELINE_STAGE_GEOM_BIT;
   }

   if (stage_mask & (VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                     VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)) {
      stages |= PVR_PIPELINE_STAGE_FRAG_BIT;
   }

   if (stage_mask & (VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)) {
      stages |= PVR_PIPELINE_STAGE_COMPUTE_BIT;
   }

   if (stage_mask & (VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                     VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT |
                     VK_PIPELINE_STAGE_2_COPY_BIT |
                     VK_PIPELINE_STAGE_2_BLIT_BIT |
                     VK_PIPELINE_STAGE_2_RESOLVE_BIT |
                     VK_PIPELINE_STAGE_2_CLEAR_BIT))
      stages |= PVR_PIPELINE_STAGE_TRANSFER_BIT;

   if (stage_mask & (VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT))
      stages |= PVR_PIPELINE_STAGE_QUERY_BIT;

   return stages;
}

static inline enum pvr_pipeline_stage_bits
pvr_stage_mask_src(VkPipelineStageFlags2 stage_mask)
{
   /* If the source is bottom of pipe, all stages will need to be waited for. */
   if (stage_mask & VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
      return PVR_PIPELINE_STAGE_ALL_BITS;

   return pvr_stage_mask(stage_mask);
}

static inline enum pvr_pipeline_stage_bits
pvr_stage_mask_dst(VkPipelineStageFlags2 stage_mask)
{
   /* If the destination is top of pipe, all stages should be blocked by prior
    * commands.
    */
   if (stage_mask & VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT)
      return PVR_PIPELINE_STAGE_ALL_BITS;

   return pvr_stage_mask(stage_mask);
}

#endif /* PVR_PIPELINE_H */
