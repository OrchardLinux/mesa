/* SPDX-License-Identifier: MIT */
/* A9 compute. Pipeline/descriptor setup adapted from G9P's
 * pvr_arch_g9p.c; A9 CDM records are encoded independently from Rogue fields.
 * No captured USC/PDS/context window or fixture allowlist is used. */
#include <string.h>
#include "vk_pipeline.h"
#include "vk_alloc.h"
#include "util/ralloc.h"
#include "g9p_cdm.h"
#include "g9p_compile.h"
#include "g5p_cdm.h"
#include "g5p_usc.h"
#include "pvr_bo.h"
#include "pvr_cmd_buffer.h"
#include "pvr_csb.h"
#include "pvr_descriptor_set.h"
#include "pvr_device.h"
#include "pvr_physical_device.h"
#include "pvr_pipeline.h"
#include "pvr_g5p.h"
#include "pvr_g5p_compute.h"
#include "pvr_job_context.h"
#include "pvr_winsys.h"
#define G9P_KERNEL_DATA_DWORDS 8
static uint32_t usc_offset(const struct pvr_device *d, const struct pvr_suballoc_bo *bo)
{ return bo->dev_addr.addr - d->heaps.usc_heap->base_addr.addr; }
VkResult
PVR_PER_ARCH(g5p_compute_pipeline_compile)(
   struct pvr_device *device,
   const VkComputePipelineCreateInfo *create_info,
   struct pvr_compute_pipeline *pipeline)
{
   struct pvr_g9p_compute_state *state = &pipeline->g9p;
   void *mem_ctx = ralloc_context(NULL);
   struct g9p_shader shader;
   nir_shader *nir;
   VkResult result;

   result = vk_pipeline_shader_stage_to_nir(&device->vk,
                                            pipeline->base.pipeline_flags,
                                            &create_info->stage,
                                            g9p_spirv_options(),
                                            g9p_nir_options(), mem_ctx, &nir);
   if (result != VK_SUCCESS)
      goto out;

   if (!g9p_a9_compile_compute(nir, &shader, mem_ctx)) {
      result = vk_errorf(device, VK_ERROR_FEATURE_NOT_PRESENT,
                         "G5P compute: shader not supported by the backend yet");
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
   state->num_regs = shader.num_regs;
   state->a9_local_dwords = shader.a9_local_dwords;
   state->a9_has_barriers = shader.a9_has_barriers;
   state->a9_device_fence = shader.a9_device_fence;
   state->uses_workgroup_id = shader.uses_workgroup_id;
   memcpy(state->local_size, shader.local_size, sizeof(state->local_size));

out:
   ralloc_free(mem_ctx);
   return result;
}

static void
emit_record(struct pvr_csb *csb, const struct g5p_cdm_record *rec)
{
   uint32_t *w = pvr_arch_csb_alloc_dwords(csb, 7);

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
PVR_PER_ARCH(g5p_cmd_dispatch)(struct pvr_cmd_buffer *cmd_buffer,
                               struct pvr_sub_cmd_compute *sub_cmd,
                               uint64_t indirect_va, const uint32_t base_group[static 3],
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
   struct pvr_suballoc_bo *uniforms_bo;
   uint32_t uniforms[2 * G9P_MAX_BUFFERS + 3] = { 0 };
   uint32_t bind_data[4 * (G9P_MAX_BUFFERS + 1) + 4] = { 0 };
   uint32_t bind_code[G9P_MAX_BUFFERS + 3];
   uint32_t kernel_data[12] = { 0 }, kernel_code[8];
   struct pvr_pds_upload bind, kernel;
   unsigned bind_dwords, ctrl0, nc = 0;
   VkResult result;

   if (base_group[0] || base_group[1] || base_group[2])
      return vk_errorf(cmd_buffer, VK_ERROR_FEATURE_NOT_PRESENT, "G5P: base dispatch not implemented yet");
   if (indirect_va && ((indirect_va & 3) || indirect_va >> 40))
      return vk_error(cmd_buffer, VK_ERROR_FEATURE_NOT_PRESENT);
   if (!indirect_va && (!group_count[0] || !group_count[1] || !group_count[2]))
      return VK_SUCCESS;
   if (!threads || threads > 512 || group_count[0] > 65536 || group_count[1] > 65536 || group_count[2] > 65536)
      return vk_error(cmd_buffer, VK_ERROR_FEATURE_NOT_PRESENT);

   /*
    * Values the DMA copies that are not in descriptor memory: the
    * push-constant buffer's address and the thread count.
    */
   for (unsigned k = 0; k < n; k++) {
      if (state->buffers[k].set == G9P_PUSH_CONSTANTS) {
         const struct pvr_push_constants *pc =
            &cmd_buffer->state.push_consts[PVR_STAGE_ALLOCATION_COMPUTE];

         uniforms[2 * k] = pc->dev_addr.addr;
         uniforms[2 * k + 1] = pc->dev_addr.addr >> 32;
      }
   }
   unsigned transfers=n+1+(state->a9_device_fence ? 1 : 0);
   if (state->a9_device_fence) {
      uint32_t zero=0;
      struct pvr_suballoc_bo *fence;
      result=pvr_arch_cmd_buffer_upload_general(cmd_buffer,&zero,sizeof(zero),&fence);
      if (result != VK_SUCCESS) return result;
      uniforms[2*n+1]=fence->dev_addr.addr;
      uniforms[2*n+2]=fence->dev_addr.addr>>32;
   }
   uniforms[2 * n] = count;
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, uniforms,
                                               sizeof(uint32_t) * (2 * n + 1 + (state->a9_device_fence ? 2 : 0)),
                                               &uniforms_bo);
   if (result != VK_SUCCESS)
      return result;

   /*
    * Binding task data: the source addresses (64-bit each), then one DOUTD
    * control word per transfer: Rogue DOUTD src1, last (bit 31), common
    * store (bit 28), destination dword << 13, dwords.
    */
   ctrl0 = 2 * transfers;
   for (unsigned k = 0; k < transfers; k++) {
      uint64_t src;
      uint32_t dest = 2 * k, dwords = 2;

      if (k < n && state->buffers[k].set != G9P_PUSH_CONSTANTS) {
         const struct pvr_descriptor_set *set = desc->sets[state->buffers[k].set];
         const struct pvr_descriptor_set_layout_binding *binding;

         if (!set)
            return vk_error(cmd_buffer, VK_ERROR_UNKNOWN);
         binding = &set->layout->bindings[state->buffers[k].binding];
         /* pvr_buffer_descriptor: the address comes first */
         src = set->dev_addr.addr + binding->offset;
      } else if (k > n) {
         src=uniforms_bo->dev_addr.addr + 4*(2*n+1);
         dest=56;
      } else {
         src = uniforms_bo->dev_addr.addr + 8 * k;
         if (k == n)
            dwords = 1;
      }
      bind_data[2 * k] = src;
      bind_data[2 * k + 1] = src >> 32;
      bind_data[ctrl0 + k] =
         (k+1 == transfers ? BITFIELD_BIT(31) : 0) | BITFIELD_BIT(28) | dest << 13 | dwords;
      bind_code[nc++] = g9p_pds_dout(G9P_DOUTD, k, G9P_PDS_DATA, ctrl0 + k,
                                     k+1 == transfers ? G9P_PDS_END : 0);
   }
   bind_code[nc++] = g9p_pds_halt();
   bind_dwords = ALIGN_POT(ctrl0 + transfers, 4);

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
   kernel_data[1] = 0x10 * DIV_ROUND_UP(MAX2(state->num_regs, 1), 8);
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

   unsigned kernel_dwords = G9P_KERNEL_DATA_DWORDS;
   if (state->a9_has_barriers) {
      /* IF1 zeros TG2/3 before the ID write. DATA q4 is the zero pair. */
      kernel_dwords = 12;
      kernel_data[6] = BITFIELD_BIT(28) | 2u << 13 | 2;
      kernel_code[0] = g9p_pds_bra_not_if1(5);
      kernel_code[1] = g9p_pds_dout(G9P_DOUTW, 4, G9P_PDS_DATA, 6, 0);
      kernel_code[2] = g9p_pds_dout(G9P_DOUTW, 0, G9P_PDS_TEMPS, 4, G9P_PDS_END);
      kernel_code[3] = kernel_code[4] = g9p_pds_halt();
      kernel_code[5] = g9p_pds_dout(G9P_DOUTW, 0, G9P_PDS_TEMPS, 2, 0);
      kernel_code[6] = g9p_pds_dout(G9P_DOUTU, 0, G9P_PDS_DATA, 0, G9P_PDS_END | G9P_PDS_CC);
   }
   result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, kernel_data,
                                           kernel_dwords,
                                           ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT,
                                           kernel_code, ARRAY_SIZE(kernel_code),
                                           ROGUE_CDMCTRL_KERNEL2_CODE_ADDR_ALIGNMENT,
                                           ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT,
                                           &kernel);
   if (result != VK_SUCCESS)
      return result;

   {
      struct pvr_csb *csb = &sub_cmd->control_stream;
      const struct g5p_cdm_record bind_rec =
         g5p_cdm_dma_task(G9P_CDM_DATA_BLOCKS(bind_dwords / 4), bind.data_offset,
                          bind.code_offset, state->a9_device_fence ? 4 : DIV_ROUND_UP(2 * n + 1, 16));
      const struct g5p_cdm_record kernel_rec = g5p_cdm_kernel(
         G9P_CDM_KERNEL_REGS(state->num_regs) | G9P_CDM_ANY_CLUSTER | G9P_CDM_FENCE |
            G9P_CDM_DATA_BLOCKS(kernel_dwords / 4),
         kernel.data_offset, kernel.code_offset, group_count[0], group_count[1],
         group_count[2], threads, DIV_ROUND_UP(state->a9_local_dwords, 16));

      pvr_csb_set_relocation_mark(csb);
      emit_record(csb, &bind_rec);
      if (indirect_va) {
         const struct g5p_cdm_indirect_record rec = g5p_cdm_indirect(
            kernel_rec.w[0], kernel.data_offset, kernel.code_offset,
            indirect_va, threads, 0);
         uint32_t *w = pvr_arch_csb_alloc_dwords(csb, 6);
         if (w) memcpy(w, rec.w, sizeof(rec.w));
      } else {
         emit_record(csb, &kernel_rec);
      }
      pvr_csb_clear_relocation_mark(csb);
   }

   return VK_SUCCESS;
}


/* A9's fenced task needs the single-word C4/01 control operation before
 * the ordinary branch-to-END sequence (compute.md §9.2). The instruction's
 * fence-ack role is inferred; its exact broader semantics remain unknown.
 * Encode that instruction and the existing branch/pad/end forms; no captured
 * program image is imported. CDM tasks end with the generic END word,
 * not the VDM-specific UVSW end-task instruction used by ta_kick. */
static VkResult
upload_cdm_task(struct pvr_cmd_buffer *cmd, bool fenced, struct pvr_pds_upload *out)
{
   uint8_t usc[128] = {0};
   const uint16_t control = (0xc4u << 8) | 1u;
   unsigned at = 0;
   if (fenced) {
      memcpy(usc, &control, sizeof(control));
      at = sizeof(control);
   }
   at += g5p_usc_branch(usc + at, 64 - at, false);
   while (at < 64)
      at += g5p_usc_special(usc + at, false);
   at += g5p_usc_special(usc + at, true);
   struct pvr_suballoc_bo *bo;
   VkResult r = pvr_gpu_upload_usc(cmd->device, usc, ALIGN_POT(at, 64), 64, &bo);
   if (r != VK_SUCCESS) return r;
   list_add(&bo->link, &cmd->bo_list);
   uint32_t data[4] = {usc_offset(cmd->device, bo), 0x10, 0, 0};
   /* Same unconditional DOUTU.END/HALT as the original ta_kick; only the
    * USC target changes, so probe2 does not also vary the IF1 launch. */
   uint32_t code[2] = {
      g9p_pds_dout(G9P_DOUTU, 0, G9P_PDS_DATA, 0, G9P_PDS_END),
      g9p_pds_halt(),
   };
   if (getenv("PVR_G5P_WS_LOG"))
      fprintf(stderr, "G5P-COMPUTE end_task control=%u usc=%#x bytes=%u branch_to=64 end=8000\n", fenced, data[0], at);
   return pvr_arch_cmd_buffer_upload_pds(cmd, data, 4, 16, code, 2, 32, 64, out);
}

VkResult
PVR_PER_ARCH(g5p_compute_end_pass)(struct pvr_cmd_buffer *cmd,
                                  struct pvr_sub_cmd_compute *sub)
{
   uint32_t data[4] = {0}, code[4];
   struct pvr_pds_upload fence, fenced, kick;
   VkResult task_result = upload_cdm_task(cmd, false, &kick);
   if (task_result == VK_SUCCESS)
      task_result = upload_cdm_task(cmd, true, &fenced);
   if (task_result != VK_SUCCESS) return task_result;
   g9p_pds_fence_program(code);
   VkResult r = pvr_arch_cmd_buffer_upload_pds(cmd, data, 4, 16, code, 4, 16, 16, &fence);
   if (r != VK_SUCCESS) return r;
   struct g5p_cdm_record rec;
   pvr_csb_set_relocation_mark(&sub->control_stream);
   rec = g5p_cdm_task(G9P_CDM_ANY_CLUSTER, kick.data_offset, G9P_CDM_K1_DMA, kick.code_offset);
   emit_record(&sub->control_stream, &rec);
   rec = g5p_cdm_task(G9P_CDM_FENCE_BLOCK | G9P_CDM_ANY_CLUSTER | G9P_CDM_FENCE,
                     fence.data_offset, G9P_CDM_K1_FENCE, fence.code_offset);
   emit_record(&sub->control_stream, &rec);
   rec = g5p_cdm_task(G9P_CDM_ANY_CLUSTER | G9P_CDM_FENCE, fenced.data_offset,
                     G9P_CDM_K1_FENCE, fenced.code_offset);
   emit_record(&sub->control_stream, &rec);
   rec = g5p_cdm_task(G9P_CDM_FENCE_BLOCK | G9P_CDM_ANY_CLUSTER | G9P_CDM_FENCE,
                     fence.data_offset, G9P_CDM_K1_FENCE, fence.code_offset);
   emit_record(&sub->control_stream, &rec);
   pvr_csb_clear_relocation_mark(&sub->control_stream);
   return pvr_csb_get_status(&sub->control_stream);
}

VkResult
PVR_PER_ARCH(g5p_compute_ctx_create)(struct pvr_device *device, struct pvr_compute_ctx **out)
{
   struct pvr_compute_ctx *ctx = vk_zalloc(&device->vk.alloc, sizeof(*ctx), 8,
                                           VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!ctx) return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   ctx->device = device;
   VkResult r = pvr_bo_alloc(device, device->heaps.general_heap, 0x8000, 0x4000,
      PVR_BO_ALLOC_FLAG_CPU_MAPPED | PVR_BO_ALLOC_FLAG_GPU_UNCACHED,
      &ctx->ctx_switch.compute_state_bo);
   if (r != VK_SUCCESS) { vk_free(&device->vk.alloc, ctx); return r; }
   memset(ctx->ctx_switch.compute_state_bo->bo->map, 0, 0x8000);
   struct pvr_winsys_compute_ctx_create_info info = {0};
   r = device->ws->ops->compute_ctx_create(device->ws, &info, &device->pdevice->dev_info,
                                          &ctx->ws_ctx);
   if (r != VK_SUCCESS) {
      pvr_bo_free(device, ctx->ctx_switch.compute_state_bo);
      vk_free(&device->vk.alloc, ctx);
      return r;
   }
   *out = ctx;
   return VK_SUCCESS;
}

void PVR_PER_ARCH(g5p_compute_ctx_destroy)(struct pvr_compute_ctx *ctx)
{
   struct pvr_device *d = ctx->device;
   d->ws->ops->compute_ctx_destroy(ctx->ws_ctx);
   pvr_bo_free(d, ctx->ctx_switch.compute_state_bo);
   vk_free(&d->vk.alloc, ctx);
}

VkResult
PVR_PER_ARCH(g5p_compute_job_submit)(struct pvr_compute_ctx *ctx,
 struct pvr_sub_cmd_compute *sub, struct vk_sync *wait, struct vk_sync *signal)
{
   struct pvr_device *d = ctx->device;
   const struct pvr_csb *cs = &sub->control_stream;
   /* cdm_end is relative to the first block and must name its terminate.
    * Chained CDM streams need the UAPI's terminal address, not a guessed
    * contiguous byte count. Refuse until that form has its own gate. */
   if (cs->pvr_bo != list_first_entry(&cs->pvr_bo_list, struct pvr_bo, link))
      return vk_errorf(d, VK_ERROR_FEATURE_NOT_PRESENT, "G5P: chained CDM stream not implemented yet");
   struct pvr_g5p_compute_stream stream = {
      .cdm_va = pvr_csb_get_start_address(cs).addr,
      .cdm_size = cs->pvr_bo->bo->size,
      .cdm_end = (char *)cs->next - (char *)cs->start,
      .state_va = ctx->ctx_switch.compute_state_bo->vma->dev_addr.addr,
      .tail_va = ctx->ctx_switch.compute_state_bo->vma->dev_addr.addr + 0x6020,
      .halt_code = d->g5p->programs.halt.code_offset,
      .halt_data = d->g5p->programs.halt.data_offset,
   };
   struct pvr_winsys_compute_submit_info info = { .wait = wait,
      .frame_num = d->global_queue_present_count, .job_num = d->global_cmd_buffer_submit_count,
      .fw_stream_len = sizeof(stream) };
   static_assert(sizeof(stream) <= sizeof(info.fw_stream), "G5P compute winsys stream fits");
   memcpy(info.fw_stream, &stream, sizeof(stream));
   return d->ws->ops->compute_submit(ctx->ws_ctx, &info, &d->pdevice->dev_info, signal);
}
