/* SPDX-License-Identifier: MIT */
/*
 * Compute on Apple's Rogue-derived GPUs (G9P, the A10): pipelines compiled
 * by the G9P backend, dispatches and the end of a compute pass in Apple's
 * CDM record format (g9p_cdm.h). Used on devices with has_apple_g9p in place
 * of the pco/Rogue paths.
 */

#include <string.h>

#include "vk_pipeline.h"

#include "hwdef/pvr_hw_utils.h"

#include "util/ralloc.h"

#include "g9p_cdm.h"
#include "g9p_compile.h"

#include "pvr_bo.h"
#include "pvr_cmd_buffer.h"
#include "pvr_csb.h"
#include "pvr_descriptor_set.h"
#include "pvr_device.h"
#include "pvr_entrypoints.h"
#include "pvr_g9p.h"
#include "pvr_physical_device.h"
#include "pvr_pipeline.h"

/* Kernel task data: DOUTU source, local-ID and workgroup-ID control words */
#define G9P_KERNEL_DATA_DWORDS 8

static uint32_t
usc_offset(const struct pvr_device *device, const struct pvr_suballoc_bo *bo)
{
   return bo->dev_addr.addr - device->heaps.usc_heap->base_addr.addr;
}

VkResult
PVR_PER_ARCH(g9p_device_init)(struct pvr_device *device)
{
   const uint32_t cache_line_size =
      pvr_get_slc_cache_line_size(&device->pdevice->dev_info);
   uint16_t empty[64];
   uint32_t data[4] = { 0 }, code[4];
   VkResult result;

   for (unsigned i = 0; i < ARRAY_SIZE(empty); i++)
      empty[i] = 0xc000;
   g9p_usc_empty_kernel(empty);
   result = pvr_gpu_upload_usc(device, empty, sizeof(empty), 64,
                               &device->g9p.empty_kernel);
   if (result != VK_SUCCESS)
      return result;

   g9p_pds_kick_data(data, usc_offset(device, device->g9p.empty_kernel));
   g9p_pds_kick_program(code);
   result = pvr_gpu_upload_pds(device, data, ARRAY_SIZE(data),
                               ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT, code,
                               ARRAY_SIZE(code),
                               ROGUE_CDMCTRL_KERNEL2_CODE_ADDR_ALIGNMENT,
                               cache_line_size, &device->g9p.kick);
   if (result != VK_SUCCESS)
      goto err_free_empty;

   /* The fence blocks read zeroed data, as on iPadOS */
   memset(data, 0, sizeof(data));
   g9p_pds_fence_program(code);
   result = pvr_gpu_upload_pds(device, data, ARRAY_SIZE(data),
                               ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT, code,
                               ARRAY_SIZE(code),
                               ROGUE_CDMCTRL_KERNEL2_CODE_ADDR_ALIGNMENT,
                               cache_line_size, &device->g9p.fence);
   if (result != VK_SUCCESS)
      goto err_free_kick;

   return VK_SUCCESS;

err_free_kick:
   pvr_bo_suballoc_free(device->g9p.kick.pvr_bo);
err_free_empty:
   pvr_bo_suballoc_free(device->g9p.empty_kernel);
   return result;
}

void
PVR_PER_ARCH(g9p_device_finish)(struct pvr_device *device)
{
   pvr_bo_suballoc_free(device->g9p.fence.pvr_bo);
   pvr_bo_suballoc_free(device->g9p.kick.pvr_bo);
   pvr_bo_suballoc_free(device->g9p.empty_kernel);
}

VkResult
PVR_PER_ARCH(g9p_compute_pipeline_compile)(
   struct pvr_device *device,
   const VkComputePipelineCreateInfo *create_info,
   struct pvr_compute_pipeline *pipeline)
{
   struct pvr_g9p_compute_state *state = &pipeline->g9p;
   void *mem_ctx = ralloc_context(NULL);
   struct g9p_compute_shader shader;
   nir_shader *nir;
   VkResult result;

   result = vk_pipeline_shader_stage_to_nir(&device->vk,
                                            pipeline->base.pipeline_flags,
                                            &create_info->stage,
                                            g9p_spirv_options(),
                                            g9p_nir_options(), mem_ctx, &nir);
   if (result != VK_SUCCESS)
      goto out;

   if (!g9p_compile_compute(nir, &shader, mem_ctx)) {
      result = vk_errorf(device, VK_ERROR_UNKNOWN,
                         "G9P: shader not supported by the backend yet");
      goto out;
   }

   result = pvr_gpu_upload_usc(device, shader.code, shader.code_size, 64,
                               &state->shader_bo);
   if (result != VK_SUCCESS)
      goto out;

   state->num_buffers = shader.num_buffers;
   for (unsigned i = 0; i < shader.num_buffers; i++) {
      state->buffers[i].set = shader.buffers[i].set;
      state->buffers[i].binding = shader.buffers[i].binding;
   }
   state->uses_workgroup_id = shader.uses_workgroup_id;
   memcpy(state->local_size, shader.local_size, sizeof(state->local_size));

out:
   ralloc_free(mem_ctx);
   return result;
}

static void
emit_record(struct pvr_csb *csb, const struct g9p_cdm_record *rec)
{
   uint32_t *w = pvr_arch_csb_alloc_dwords(csb, 8);

   if (w)
      memcpy(w, rec->w, sizeof(rec->w));
}

/*
 * One dispatch: a DMA task loading the buffer addresses (from the bound
 * descriptor sets) and the thread count into the common store, then the
 * kernel, whose PDS program writes the workgroup ID into threadgroup words
 * 0-1 and kicks the USC.
 */
VkResult
PVR_PER_ARCH(g9p_cmd_dispatch)(struct pvr_cmd_buffer *cmd_buffer,
                               struct pvr_sub_cmd_compute *sub_cmd,
                               const uint32_t group_count[static 3])
{
   struct pvr_device *device = cmd_buffer->device;
   const struct pvr_compute_pipeline *pipeline = cmd_buffer->state.compute_pipeline;
   const struct pvr_g9p_compute_state *state = &pipeline->g9p;
   const struct pvr_descriptor_state *desc = &cmd_buffer->state.compute_desc_state;
   const unsigned n = state->num_buffers;
   const uint32_t threads =
      state->local_size[0] * state->local_size[1] * state->local_size[2];
   const uint32_t count = threads | threads << 16;
   struct pvr_suballoc_bo *count_bo;
   uint32_t bind_data[4 * (G9P_MAX_BUFFERS + 1) + 4] = { 0 };
   uint32_t bind_code[G9P_MAX_BUFFERS + 2];
   uint32_t kernel_data[G9P_KERNEL_DATA_DWORDS] = { 0 }, kernel_code[8];
   struct pvr_pds_upload bind, kernel;
   unsigned bind_dwords, ctrl0, nc = 0;
   VkResult result;

   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, &count, sizeof(count),
                                               &count_bo);
   if (result != VK_SUCCESS)
      return result;

   /*
    * Binding task data: the source addresses (64-bit each), then one DOUTD
    * control word per transfer: Rogue DOUTD src1, last (bit 31), common
    * store (bit 28), destination dword << 13, dwords.
    */
   ctrl0 = 2 * (n + 1);
   for (unsigned k = 0; k <= n; k++) {
      uint64_t src;
      uint32_t dest = 2 * k, dwords = 2;

      if (k < n) {
         const struct pvr_descriptor_set *set = desc->sets[state->buffers[k].set];
         const struct pvr_descriptor_set_layout_binding *binding;

         if (!set)
            return vk_error(cmd_buffer, VK_ERROR_UNKNOWN);
         binding = &set->layout->bindings[state->buffers[k].binding];
         /* pvr_buffer_descriptor: the address comes first */
         src = set->dev_addr.addr + binding->offset;
      } else {
         src = count_bo->dev_addr.addr;
         dwords = 1;
      }
      bind_data[2 * k] = src;
      bind_data[2 * k + 1] = src >> 32;
      bind_data[ctrl0 + k] =
         (k == n ? BITFIELD_BIT(31) : 0) | BITFIELD_BIT(28) | dest << 13 | dwords;
      bind_code[nc++] = g9p_pds_dout(G9P_DOUTD, k, G9P_PDS_DATA, ctrl0 + k,
                                     k == n ? G9P_PDS_END : 0);
   }
   bind_code[nc++] = g9p_pds_halt();
   bind_dwords = ALIGN_POT(ctrl0 + n + 1, 4);

   result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, bind_data, bind_dwords,
                                           ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT,
                                           bind_code, nc,
                                           ROGUE_CDMCTRL_KERNEL2_CODE_ADDR_ALIGNMENT,
                                           ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT,
                                           &bind);
   if (result != VK_SUCCESS)
      return result;

   /*
    * Kernel task: DOUTU source (USC offset, 0x10), the local-ID control
    * word (temps q0 -> r0) and the workgroup-ID one (temps q0, 64 bits ->
    * threadgroup words 0-1) in the coefficient task.
    */
   g9p_pds_kick_data(kernel_data, usc_offset(device, state->shader_bo));
   kernel_data[2] = 0x80000000;
   kernel_data[4] = 0x90000002;
   kernel_code[0] = g9p_pds_bra_not_if1(4);
   kernel_code[1] = g9p_pds_dout(G9P_DOUTW, 0, G9P_PDS_TEMPS, 4, G9P_PDS_END);
   kernel_code[2] = g9p_pds_halt();
   kernel_code[3] = g9p_pds_halt();
   kernel_code[4] = g9p_pds_dout(G9P_DOUTW, 0, G9P_PDS_TEMPS, 2, 0);
   kernel_code[5] = g9p_pds_dout(G9P_DOUTU, 0, G9P_PDS_DATA, 0,
                                 G9P_PDS_END | G9P_PDS_CC);
   kernel_code[6] = g9p_pds_halt();
   kernel_code[7] = g9p_pds_halt();

   result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, kernel_data,
                                           ARRAY_SIZE(kernel_data),
                                           ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT,
                                           kernel_code, ARRAY_SIZE(kernel_code),
                                           ROGUE_CDMCTRL_KERNEL2_CODE_ADDR_ALIGNMENT,
                                           ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT,
                                           &kernel);
   if (result != VK_SUCCESS)
      return result;

   {
      struct pvr_csb *csb = &sub_cmd->control_stream;
      const struct g9p_cdm_record bind_rec =
         g9p_cdm_dma_task(G9P_CDM_DATA_BLOCKS(bind_dwords / 4), bind.data_offset,
                          bind.code_offset);
      const struct g9p_cdm_record kernel_rec = g9p_cdm_kernel(
         G9P_CDM_KERNEL_SIZES | G9P_CDM_ANY_CLUSTER | G9P_CDM_FENCE |
            G9P_CDM_DATA_BLOCKS(G9P_KERNEL_DATA_DWORDS / 4),
         kernel.data_offset, kernel.code_offset, group_count[0], group_count[1],
         group_count[2], threads, 1);

      pvr_csb_set_relocation_mark(csb);
      emit_record(csb, &bind_rec);
      emit_record(csb, &kernel_rec);
      pvr_csb_clear_relocation_mark(csb);
   }

   return VK_SUCCESS;
}

/*
 * The end of a compute pass, before the terminate: iPadOS's four records
 * (g9p_cdm_end_pass() without its terminate, which the caller emits).
 */
void
PVR_PER_ARCH(g9p_end_pass)(struct pvr_cmd_buffer *cmd_buffer,
                           struct pvr_sub_cmd_compute *sub_cmd)
{
   const struct pvr_device *device = cmd_buffer->device;
   const struct g9p_pass_end_programs programs = {
      .kick_code = device->g9p.kick.code_offset,
      .kick_data = device->g9p.kick.data_offset,
      .fence_code = device->g9p.fence.code_offset,
      .fence_data = device->g9p.fence.data_offset,
   };
   struct pvr_csb *csb = &sub_cmd->control_stream;
   uint32_t words[33];
   uint32_t *w;

   g9p_cdm_end_pass(words, &programs);
   pvr_csb_set_relocation_mark(csb);
   w = pvr_arch_csb_alloc_dwords(csb, 32);
   if (w)
      memcpy(w, words, 32 * sizeof(uint32_t));
   pvr_csb_clear_relocation_mark(csb);
}
