/* SPDX-License-Identifier: MIT */
/* Native A9 query kernels. Rogue's common query programs and CDM records are
 * not executable by the G5P. Build the same reset/availability/copy work as
 * ordinary internal NIR, then launch it through the measured A9 CDM shape. */
#include <stddef.h>
#include <string.h>

#include "compiler/nir/nir_builder.h"
#include "g5p_cdm.h"
#include "g5p_usc.h"
#include "g9p_cdm.h"
#include "g9p_compile.h"
#include "util/ralloc.h"
#include "vk_alloc.h"

#include "pvr_bo.h"
#include "pvr_buffer.h"
#include "pvr_cmd_buffer.h"
#include "pvr_csb.h"
#include "pvr_device.h"
#include "pvr_g5p_query.h"
#include "pvr_physical_device.h"
#include "pvr_query.h"

#define G5P_QUERY_LOCAL_SIZE 32
#define G5P_QUERY_KERNEL_DATA_DWORDS 8

enum g5p_query_program {
   G5P_QUERY_AVAILABILITY,
   G5P_QUERY_COPY,
   G5P_QUERY_RESET,
};

struct g5p_query_params {
   uint32_t count;
   uint32_t dest_stride;
   uint32_t flags;
};

static nir_def *query_buffer(nir_builder *b, unsigned binding)
{
   return nir_vulkan_resource_index(
      b,
      2,
      32,
      nir_imm_int(b, 0),
      .desc_set = 0,
      .binding = binding,
      .desc_type = nir_descriptor_type_storage_buffer,
      .resource_type = nir_resource_type_read_write_storage_buffer);
}

static nir_def *query_param(nir_builder *b, unsigned offset)
{
   return nir_load_push_constant(b,
                                 1,
                                 32,
                                 nir_imm_int(b, 0),
                                 .base = offset,
                                 .range = sizeof(uint32_t));
}

static nir_def *query_index(nir_builder *b)
{
   return nir_channel(b, nir_load_global_invocation_id(b, 32), 0);
}

static nir_def *query_offset(nir_builder *b, nir_def *index)
{
   return nir_ishl_imm(b, index, 2);
}

static nir_shader *build_availability_nir(void *mem_ctx)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  g9p_nir_options(),
                                                  "g5p-query-availability");
   nir_def *indices = query_buffer(&b, 0);
   nir_def *availability = query_buffer(&b, 1);
   nir_def *index = query_index(&b);
   nir_def *count = query_param(&b, offsetof(struct g5p_query_params, count));
   nir_if *in_range = nir_push_if(&b, nir_ult(&b, index, count));
   nir_def *offset = nir_load_ssbo(&b,
                                   1,
                                   32,
                                   indices,
                                   query_offset(&b, index),
                                   .align_mul = 4,
                                   .access = ACCESS_NON_WRITEABLE);

   nir_store_ssbo(&b,
                  nir_imm_int(&b, ~0u),
                  availability,
                  query_offset(&b, offset),
                  .align_mul = 4,
                  .write_mask = 1);
   nir_pop_if(&b, in_range);

   b.shader->info.workgroup_size[0] = G5P_QUERY_LOCAL_SIZE;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_validate_shader(b.shader, "G5P query availability");
   ralloc_steal(mem_ctx, b.shader);
   return b.shader;
}

static nir_shader *build_reset_nir(void *mem_ctx)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  g9p_nir_options(),
                                                  "g5p-query-reset");
   nir_def *result = query_buffer(&b, 0);
   nir_def *availability = query_buffer(&b, 1);
   nir_def *index = query_index(&b);
   nir_def *count = query_param(&b, offsetof(struct g5p_query_params, count));
   nir_if *in_range = nir_push_if(&b, nir_ult(&b, index, count));
   nir_def *offset = query_offset(&b, index);

   nir_store_ssbo(&b,
                  nir_imm_int(&b, 0),
                  result,
                  offset,
                  .align_mul = 4,
                  .write_mask = 1);
   nir_store_ssbo(&b,
                  nir_imm_int(&b, 0),
                  availability,
                  offset,
                  .align_mul = 4,
                  .write_mask = 1);
   nir_pop_if(&b, in_range);

   b.shader->info.workgroup_size[0] = G5P_QUERY_LOCAL_SIZE;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_validate_shader(b.shader, "G5P query reset");
   ralloc_steal(mem_ctx, b.shader);
   return b.shader;
}

static void store_query_value(nir_builder *b,
                              nir_def *dst,
                              nir_def *offset,
                              nir_def *value,
                              nir_def *is_64bit)
{
   nir_if *wide = nir_push_if(b, is_64bit);

   nir_store_ssbo(b, value, dst, offset, .align_mul = 4, .write_mask = 1);
   nir_store_ssbo(b,
                  nir_imm_int(b, 0),
                  dst,
                  nir_iadd_imm(b, offset, 4),
                  .align_mul = 4,
                  .write_mask = 1);
   nir_push_else(b, wide);
   nir_store_ssbo(b, value, dst, offset, .align_mul = 4, .write_mask = 1);
   nir_pop_if(b, wide);
}

static nir_shader *build_copy_nir(void *mem_ctx)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  g9p_nir_options(),
                                                  "g5p-query-copy");
   nir_def *dst = query_buffer(&b, 0);
   nir_def *availability = query_buffer(&b, 1);
   nir_def *result = query_buffer(&b, 2);
   nir_def *index = query_index(&b);
   nir_def *count = query_param(&b, offsetof(struct g5p_query_params, count));
   nir_def *dest_stride =
      query_param(&b, offsetof(struct g5p_query_params, dest_stride));
   nir_def *flags = query_param(&b, offsetof(struct g5p_query_params, flags));
   nir_if *in_range = nir_push_if(&b, nir_ult(&b, index, count));
   nir_def *src_offset = query_offset(&b, index);
   nir_def *dst_offset = nir_imul(&b, index, dest_stride);
   nir_def *available = nir_load_ssbo(&b,
                                      1,
                                      32,
                                      availability,
                                      src_offset,
                                      .align_mul = 4,
                                      .access = ACCESS_NON_WRITEABLE);
   nir_def *partial =
      nir_ine_imm(&b, nir_iand_imm(&b, flags, VK_QUERY_RESULT_PARTIAL_BIT), 0);
   nir_def *is_64bit =
      nir_ine_imm(&b, nir_iand_imm(&b, flags, VK_QUERY_RESULT_64_BIT), 0);
   nir_def *write_result = nir_ior(&b, nir_ine_imm(&b, available, 0), partial);
   nir_if *have_result = nir_push_if(&b, write_result);
   nir_def *value = nir_load_ssbo(&b,
                                  1,
                                  32,
                                  result,
                                  src_offset,
                                  .align_mul = 4,
                                  .access = ACCESS_NON_WRITEABLE);

   store_query_value(&b, dst, dst_offset, value, is_64bit);
   nir_pop_if(&b, have_result);

   nir_def *with_availability = nir_ine_imm(
      &b,
      nir_iand_imm(&b, flags, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
      0);
   nir_if *write_availability = nir_push_if(&b, with_availability);
   nir_def *availability_offset =
      nir_iadd(&b,
               dst_offset,
               nir_bcsel(&b, is_64bit, nir_imm_int(&b, 8), nir_imm_int(&b, 4)));

   store_query_value(&b, dst, availability_offset, available, is_64bit);
   nir_pop_if(&b, write_availability);
   nir_pop_if(&b, in_range);

   b.shader->info.workgroup_size[0] = G5P_QUERY_LOCAL_SIZE;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_validate_shader(b.shader, "G5P query copy");
   ralloc_steal(mem_ctx, b.shader);
   return b.shader;
}

static VkResult compile_query_program(struct pvr_device *device,
                                      enum g5p_query_program kind,
                                      struct pvr_compute_query_shader *program)
{
   void *mem_ctx = ralloc_context(NULL);
   struct g9p_shader shader;
   nir_shader *nir;
   VkResult result;

   memset(program, 0, sizeof(*program));
   switch (kind) {
   case G5P_QUERY_AVAILABILITY:
      nir = build_availability_nir(mem_ctx);
      break;
   case G5P_QUERY_COPY:
      nir = build_copy_nir(mem_ctx);
      break;
   case G5P_QUERY_RESET:
      nir = build_reset_nir(mem_ctx);
      break;
   default:
      UNREACHABLE("Invalid G5P query program");
   }

   if (!g9p_a9_compile_compute(nir, &shader, mem_ctx)) {
      result = vk_errorf(device,
                         VK_ERROR_FEATURE_NOT_PRESENT,
                         "G5P query program is not supported by the backend");
      goto out;
   }
   if (shader.num_buffers > ARRAY_SIZE(program->g5p_buffers)) {
      result = vk_errorf(device,
                         VK_ERROR_FEATURE_NOT_PRESENT,
                         "G5P query program uses too many buffers");
      goto out;
   }

   result = pvr_gpu_upload_usc(device,
                               shader.code,
                               shader.code_size,
                               64,
                               &program->usc_bo);
   if (result != VK_SUCCESS)
      goto out;

   program->g5p_num_regs = shader.num_regs;
   program->g5p_local_dwords = shader.a9_local_dwords;
   program->g5p_num_buffers = shader.num_buffers;
   for (unsigned i = 0; i < shader.num_buffers; i++) {
      program->g5p_buffers[i].set = shader.buffers[i].set;
      program->g5p_buffers[i].binding = shader.buffers[i].binding;
   }

out:
   ralloc_free(mem_ctx);
   return result;
}

VkResult PVR_PER_ARCH(g5p_query_programs_create)(struct pvr_device *device)
{
   VkResult result = compile_query_program(device,
                                           G5P_QUERY_AVAILABILITY,
                                           &device->availability_shader);
   if (result != VK_SUCCESS)
      return result;

   result = compile_query_program(device,
                                  G5P_QUERY_COPY,
                                  &device->copy_results_shader);
   if (result != VK_SUCCESS)
      goto err_availability;

   result = compile_query_program(device,
                                  G5P_QUERY_RESET,
                                  &device->reset_queries_shader);
   if (result != VK_SUCCESS)
      goto err_copy;

   return VK_SUCCESS;

err_copy:
   pvr_bo_suballoc_free(device->copy_results_shader.usc_bo);
err_availability:
   pvr_bo_suballoc_free(device->availability_shader.usc_bo);
   return result;
}

void PVR_PER_ARCH(g5p_query_programs_destroy)(struct pvr_device *device)
{
   pvr_bo_suballoc_free(device->availability_shader.usc_bo);
   pvr_bo_suballoc_free(device->copy_results_shader.usc_bo);
   pvr_bo_suballoc_free(device->reset_queries_shader.usc_bo);
}

static uint32_t usc_offset(const struct pvr_device *device,
                           const struct pvr_suballoc_bo *bo)
{
   return bo->dev_addr.addr - device->heaps.usc_heap->base_addr.addr;
}

static void emit_record(struct pvr_csb *csb,
                        const struct g5p_cdm_record *record)
{
   uint32_t *words = pvr_arch_csb_alloc_dwords(csb, ARRAY_SIZE(record->w));

   if (words)
      memcpy(words, record->w, sizeof(record->w));
}

static VkResult
emit_query_dispatch(struct pvr_cmd_buffer *cmd_buffer,
                    struct pvr_sub_cmd_compute *sub_cmd,
                    const struct pvr_compute_query_shader *program,
                    const uint64_t *buffers,
                    unsigned buffer_count,
                    const struct g5p_query_params *params)
{
   struct pvr_device *device = cmd_buffer->device;
   const unsigned num_buffers = program->g5p_num_buffers;
   const unsigned transfers = num_buffers + 1;
   uint32_t uniforms[2 * 4 + 1] = { 0 };
   uint32_t bind_data[4 * (4 + 1) + 4] = { 0 };
   uint32_t bind_code[4 + 5] = { 0 };
   uint32_t kernel_data[G5P_QUERY_KERNEL_DATA_DWORDS] = { 0 };
   uint32_t kernel_code[8] = { 0 };
   struct pvr_suballoc_bo *params_bo;
   struct pvr_suballoc_bo *uniforms_bo;
   struct pvr_pds_upload bind;
   struct pvr_pds_upload kernel;
   unsigned code_count = 0;
   VkResult result;

   if (!params->count)
      return VK_SUCCESS;

   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer,
                                               params,
                                               sizeof(*params),
                                               &params_bo);
   if (result != VK_SUCCESS)
      return result;

   for (unsigned i = 0; i < num_buffers; i++) {
      uint64_t address;

      if (program->g5p_buffers[i].set == G9P_PUSH_CONSTANTS) {
         address = params_bo->dev_addr.addr;
      } else {
         const unsigned binding = program->g5p_buffers[i].binding;

         if (program->g5p_buffers[i].set || binding >= buffer_count)
            return vk_errorf(cmd_buffer,
                             VK_ERROR_UNKNOWN,
                             "G5P query program has an invalid binding");
         address = buffers[binding];
      }
      uniforms[2 * i] = address;
      uniforms[2 * i + 1] = address >> 32;
   }
   uniforms[2 * num_buffers] = G5P_QUERY_LOCAL_SIZE | G5P_QUERY_LOCAL_SIZE
                                                         << 16;

   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer,
                                               uniforms,
                                               sizeof(uint32_t) *
                                                  (2 * num_buffers + 1),
                                               &uniforms_bo);
   if (result != VK_SUCCESS)
      return result;

   const unsigned control_base = 2 * transfers;
   for (unsigned i = 0; i < transfers; i++) {
      const uint64_t source = uniforms_bo->dev_addr.addr + 8 * i;
      const unsigned dwords = i == num_buffers ? 1 : 2;

      bind_data[2 * i] = source;
      bind_data[2 * i + 1] = source >> 32;
      bind_data[control_base + i] =
         (i + 1 == transfers ? BITFIELD_BIT(31) : 0) | BITFIELD_BIT(28) |
         (2 * i) << 13 | dwords;
      bind_code[code_count++] =
         g9p_pds_dout(G9P_DOUTD,
                      i,
                      G9P_PDS_DATA,
                      control_base + i,
                      i + 1 == transfers ? G9P_PDS_END : 0);
   }
   bind_code[code_count++] = g9p_pds_halt();
   const unsigned bind_dwords = ALIGN_POT(control_base + transfers, 4);

   result =
      pvr_arch_cmd_buffer_upload_pds(cmd_buffer,
                                     bind_data,
                                     bind_dwords,
                                     ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT,
                                     bind_code,
                                     code_count,
                                     ROGUE_CDMCTRL_KERNEL2_CODE_ADDR_ALIGNMENT,
                                     ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT,
                                     &bind);
   if (result != VK_SUCCESS)
      return result;

   g9p_pds_kick_data(kernel_data, usc_offset(device, program->usc_bo));
   kernel_data[1] = 0x10 * DIV_ROUND_UP(MAX2(program->g5p_num_regs, 1), 8);
   kernel_data[2] = 0x80000000;
   kernel_data[4] = 0x90000002;
   kernel_code[0] = g9p_pds_bra_not_if1(4);
   kernel_code[1] = g9p_pds_dout(G9P_DOUTW, 0, G9P_PDS_TEMPS, 4, G9P_PDS_END);
   kernel_code[2] = g9p_pds_halt();
   kernel_code[3] = g9p_pds_halt();
   kernel_code[4] = g9p_pds_dout(G9P_DOUTW, 0, G9P_PDS_TEMPS, 2, 0);
   kernel_code[5] =
      g9p_pds_dout(G9P_DOUTU, 0, G9P_PDS_DATA, 0, G9P_PDS_END | G9P_PDS_CC);
   kernel_code[6] = g9p_pds_halt();
   kernel_code[7] = g9p_pds_halt();

   result =
      pvr_arch_cmd_buffer_upload_pds(cmd_buffer,
                                     kernel_data,
                                     ARRAY_SIZE(kernel_data),
                                     ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT,
                                     kernel_code,
                                     ARRAY_SIZE(kernel_code),
                                     ROGUE_CDMCTRL_KERNEL2_CODE_ADDR_ALIGNMENT,
                                     ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT,
                                     &kernel);
   if (result != VK_SUCCESS)
      return result;

   const struct g5p_cdm_record bind_record =
      g5p_cdm_dma_task(G9P_CDM_DATA_BLOCKS(bind_dwords / 4),
                       bind.data_offset,
                       bind.code_offset,
                       DIV_ROUND_UP(2 * num_buffers + 1, 16));
   const uint32_t groups = DIV_ROUND_UP(params->count, G5P_QUERY_LOCAL_SIZE);
   const struct g5p_cdm_record kernel_record = g5p_cdm_kernel(
      G9P_CDM_KERNEL_REGS(program->g5p_num_regs) | G9P_CDM_ANY_CLUSTER |
         G9P_CDM_FENCE | G9P_CDM_DATA_BLOCKS(G5P_QUERY_KERNEL_DATA_DWORDS / 4),
      kernel.data_offset,
      kernel.code_offset,
      groups,
      1,
      1,
      g5p_cdm_workgroup_threads(G5P_QUERY_LOCAL_SIZE),
      DIV_ROUND_UP(program->g5p_local_dwords, 16));

   pvr_csb_set_relocation_mark(&sub_cmd->control_stream);
   emit_record(&sub_cmd->control_stream, &bind_record);
   emit_record(&sub_cmd->control_stream, &kernel_record);
   pvr_csb_clear_relocation_mark(&sub_cmd->control_stream);
   return VK_SUCCESS;
}

VkResult
PVR_PER_ARCH(g5p_add_query_program)(struct pvr_cmd_buffer *cmd_buffer,
                                    const struct pvr_query_info *query_info)
{
   struct pvr_device *device = cmd_buffer->device;
   const struct pvr_compute_query_shader *program;
   struct g5p_query_params params = { 0 };
   uint64_t buffers[3] = { 0 };
   unsigned buffer_count;
   VkResult result;

   result =
      pvr_arch_cmd_buffer_start_sub_cmd(cmd_buffer, PVR_SUB_CMD_TYPE_QUERY);
   if (result != VK_SUCCESS)
      return result;

   cmd_buffer->state.current_sub_cmd->is_dynamic_render =
      query_info->is_dynamic_render;
   cmd_buffer->state.current_sub_cmd->is_suspend = query_info->is_suspend;

   switch (query_info->type) {
   case PVR_QUERY_TYPE_AVAILABILITY_WRITE:
      program = &device->availability_shader;
      params.count = query_info->availability_write.num_query_indices;
      buffers[0] = query_info->availability_write.index_bo->dev_addr.addr;
      buffers[1] =
         query_info->availability_write.availability_bo->dev_addr.addr;
      buffer_count = 2;
      break;

   case PVR_QUERY_TYPE_COPY_QUERY_RESULTS: {
      VK_FROM_HANDLE(pvr_query_pool,
                     pool,
                     query_info->copy_query_results.query_pool);
      VK_FROM_HANDLE(pvr_buffer,
                     dst,
                     query_info->copy_query_results.dst_buffer);

      if (pool->query_type != VK_QUERY_TYPE_OCCLUSION ||
          pool->query_report_count != 1)
         return vk_errorf(cmd_buffer,
                          VK_ERROR_FEATURE_NOT_PRESENT,
                          "G5P native query copy only supports occlusion");

      const uint64_t result_offset =
         (uint64_t)query_info->copy_query_results.first_query *
         pool->result_elem_stride;
      const uint64_t availability_offset =
         (uint64_t)query_info->copy_query_results.first_query *
         sizeof(uint32_t);

      program = &device->copy_results_shader;
      params.count = query_info->copy_query_results.query_count;
      params.dest_stride = query_info->copy_query_results.stride;
      params.flags = query_info->copy_query_results.flags;
      buffers[0] =
         dst->dev_addr.addr + query_info->copy_query_results.dst_offset;
      buffers[1] =
         pool->availability_buffer->dev_addr.addr + availability_offset;
      buffers[2] = pool->result_buffer->dev_addr.addr + result_offset;
      buffer_count = 3;
      break;
   }

   case PVR_QUERY_TYPE_RESET_QUERY_POOL: {
      VK_FROM_HANDLE(pvr_query_pool,
                     pool,
                     query_info->reset_query_pool.query_pool);

      if (pool->query_type != VK_QUERY_TYPE_OCCLUSION ||
          pool->query_report_count != 1 ||
          pool->result_elem_stride != sizeof(uint32_t))
         return vk_errorf(cmd_buffer,
                          VK_ERROR_FEATURE_NOT_PRESENT,
                          "G5P native query reset only supports occlusion");

      const uint64_t result_offset =
         (uint64_t)query_info->reset_query_pool.first_query *
         pool->result_elem_stride;
      const uint64_t availability_offset =
         (uint64_t)query_info->reset_query_pool.first_query * sizeof(uint32_t);

      program = &device->reset_queries_shader;
      params.count = query_info->reset_query_pool.query_count;
      buffers[0] = pool->result_buffer->dev_addr.addr + result_offset;
      buffers[1] =
         pool->availability_buffer->dev_addr.addr + availability_offset;
      buffer_count = 2;
      break;
   }

   default:
      UNREACHABLE("Invalid G5P query program");
   }

   result = emit_query_dispatch(cmd_buffer,
                                &cmd_buffer->state.current_sub_cmd->compute,
                                program,
                                buffers,
                                buffer_count,
                                &params);
   if (result != VK_SUCCESS)
      return result;

   /* The generic query path emits a Rogue private-compute fence here. On G5P
    * that produces a foreign K0=0x7 record immediately before the native pass
    * tail. Native application compute does not emit that record: the common
    * G5P kick/fence/fenced-kick/fence tail waits for the kernel and publishes
    * completion. Keep query jobs byte-shaped like the proven native path. */
   return pvr_arch_cmd_buffer_end_sub_cmd(cmd_buffer);
}
