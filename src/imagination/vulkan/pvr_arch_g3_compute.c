/* SPDX-License-Identifier: MIT */
/*
 * The Apple A7's (G3) compute: pipelines compiled by g3/g3_compute.c and
 * dispatches in the A7's measured forms (compute/A7-ISA.md of the A7 bring-up):
 *
 *  - one CDM kernel record per workgroup: KERNEL0 = unified units << 12 |
 *    PDS temps 1 | data units << 2 | any cluster, no fence; KERNEL3-5 = 0
 *    (one workgroup); KERNEL8 = size_x - 1 << 18. Its PDS data carries the
 *    workgroup's buffer addresses and value addends (g3_pds_data()), so no
 *    workgroup ID, coefficient register or common store is needed;
 *  - the PDS program: branch past the (empty) IF1 part, DOUTWs of PDS data
 *    and of the lane (temps q0) into the unified store, DOUTU.END.CC, HALT;
 *  - the end of pass and the job stream are the A8's
 *    (g5p_compute_end_pass_g4p() with the A7's nop.end, g5p_compute_stream()).
 *
 * PVR_G3_DUMP=1 logs each pipeline's USC and PDS code and each record.
 */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "g3_compute.h"
#include "nir.h"
#include "pvr_bo.h"
#include "pvr_cmd_buffer.h"
#include "pvr_csb.h"
#include "pvr_descriptor_set.h"
#include "pvr_device.h"
#include "pvr_g3_compute.h"
#include "pvr_g5p.h"
#include "pvr_physical_device.h"
#include "pvr_pipeline.h"
#include "g9p_compile.h"
#include "util/log.h"
#include "util/ralloc.h"
#include "util/u_debug.h"
#include "vk_alloc.h"
#include "vk_pipeline.h"

static uint32_t
pipeline_usc_offset(const struct pvr_device *d, const struct pvr_suballoc_bo *bo)
{
   return bo->dev_addr.addr - PVR_G5P_USC_EXEC_BASE(d);
}

static bool
g3_dump(void)
{
   static int on = -1;
   if (on < 0)
      on = debug_get_bool_option("PVR_G3_DUMP", false);
   return on;
}

static void
g3_dump_bytes(const char *what, const uint8_t *b, unsigned n)
{
   char line[3 * G3_MAX_USC_BYTES + 1];
   unsigned at = 0;
   for (unsigned i = 0; i < n && at + 3 < sizeof(line); i++)
      at += snprintf(line + at, sizeof(line) - at, "%02x", b[i]);
   line[at] = 0;
   mesa_logi("pvr/g3: %s %s", what, line);
}

static void
g3_dump_words(const char *what, const uint32_t *w, unsigned n)
{
   char line[9 * (G3_MAX_PDS_DATA + 8) + 1];
   unsigned at = 0;
   for (unsigned i = 0; i < n && at + 10 < sizeof(line); i++)
      at += snprintf(line + at, sizeof(line) - at, "%s%08x", i ? " " : "", w[i]);
   line[at] = 0;
   mesa_logi("pvr/g3: %s %s", what, line);
}

VkResult
PVR_PER_ARCH(g3_compute_pipeline_compile)(struct pvr_device *device,
                                          const VkComputePipelineCreateInfo *create_info,
                                          struct pvr_compute_pipeline *pipeline)
{
   void *mem_ctx = ralloc_context(NULL);
   struct g3_program *p = NULL;
   nir_shader *nir;
   VkResult result;

   result = vk_pipeline_shader_stage_to_nir(&device->vk, pipeline->base.pipeline_flags, &create_info->stage,
                                            g9p_spirv_options(), g9p_nir_options(), mem_ctx, &nir);
   if (result != VK_SUCCESS)
      goto out;

   p = vk_zalloc(&device->vk.alloc, sizeof(*p), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!p) {
      result = vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto out;
   }
   if (!g3_compile_compute(nir, p)) {
      mesa_logw("pvr/g3: compute pipeline refused: %s", p->refusal);
      result = vk_errorf(device, VK_ERROR_FEATURE_NOT_PRESENT, "A7 compute: %s", p->refusal);
      goto out;
   }
   if (g3_dump()) {
      char d[512];
      uint32_t code[G3_MAX_PDS_CODE];
      g3_describe(p, d, sizeof(d));
      mesa_logi("pvr/g3: pipeline %s; registers %u", d, p->unified_regs);
      g3_dump_bytes("usc", p->code, p->code_size);
      g3_dump_words("pds", code, g3_pds_code(p, code));
   }
   result = pvr_gpu_upload_usc(device, p->code, p->code_size, 64, &pipeline->g9p.shader_bo);
   if (result != VK_SUCCESS)
      goto out;
   memcpy(pipeline->g9p.local_size, (uint32_t[3]){ p->local_size, 1, 1 }, sizeof(pipeline->g9p.local_size));
   pipeline->g3 = p;
   p = NULL;

out:
   if (p)
      vk_free(&device->vk.alloc, p);
   ralloc_free(mem_ctx);
   return pvr_refusal_to_api(result);
}

/* The address of store i's buffer, from the bound descriptor set */
static VkResult
g3_buffer_address(struct pvr_cmd_buffer *cmd, const struct g3_store *s, uint64_t *addr)
{
   const struct pvr_descriptor_state *desc = &cmd->state.compute_desc_state;
   const struct pvr_descriptor_set *set = s->set < PVR_MAX_DESCRIPTOR_SETS ? desc->sets[s->set] : NULL;

   if (!set || !set->layout || s->binding >= set->layout->binding_count)
      return vk_errorf(cmd, VK_ERROR_UNKNOWN, "A7 compute: set %u binding %u is not bound", s->set, s->binding);
   const struct pvr_descriptor_set_layout_binding *b = &set->layout->bindings[s->binding];
   if (b->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && b->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
      return vk_errorf(cmd, VK_ERROR_UNKNOWN, "A7 compute: set %u binding %u is not a storage buffer", s->set,
                       s->binding);
   if (pvr_descriptor_set_binding_is_dynamic(b)) {
      uint32_t range;
      if (!pvr_descriptor_set_dynamic_buffer(set, b, 0, addr, &range))
         return vk_errorf(cmd, VK_ERROR_UNKNOWN, "A7 compute: set %u binding %u holds no buffer", s->set,
                          s->binding);
      return VK_SUCCESS;
   }
   if (!set->mapping)
      return vk_errorf(cmd, VK_ERROR_UNKNOWN, "A7 compute: set %u has no descriptor memory", s->set);
   struct pvr_buffer_descriptor d;
   memcpy(&d, (const uint8_t *)set->mapping + b->offset, sizeof(d));
   if (!d.addr)
      return vk_errorf(cmd, VK_ERROR_UNKNOWN, "A7 compute: set %u binding %u holds no buffer", s->set, s->binding);
   *addr = d.addr + d.offset;
   return VK_SUCCESS;
}

VkResult
PVR_PER_ARCH(g3_cmd_dispatch)(struct pvr_cmd_buffer *cmd, struct pvr_sub_cmd_compute *sub, uint64_t indirect_va,
                              const uint32_t base_group[3], const uint32_t group_count[3])
{
   struct pvr_device *device = cmd->device;
   const struct g3_program *p = cmd->state.compute_pipeline->g3;
   uint64_t buffer[G3_MAX_STORES];
   uint32_t code[G3_MAX_PDS_CODE], data[G3_MAX_PDS_DATA];
   struct pvr_pds_upload first = { 0 };
   VkResult result;

   if (!p)
      return vk_errorf(cmd, VK_ERROR_UNKNOWN, "A7 compute: the pipeline has no A7 program");
   if (indirect_va)
      return vk_errorf(cmd, VK_ERROR_FEATURE_NOT_PRESENT, "A7 compute: an indirect dispatch");
   /* vkCmdDispatchBase along x: the workgroup IDs the CPU folds into each record start at the base */
   if (base_group[1] || base_group[2])
      return vk_errorf(cmd, VK_ERROR_FEATURE_NOT_PRESENT, "A7 compute: a base workgroup along y or z");
   if (!group_count[0] || !group_count[1] || !group_count[2])
      return VK_SUCCESS;
   if (group_count[1] != 1 || group_count[2] != 1)
      return vk_errorf(cmd, VK_ERROR_FEATURE_NOT_PRESENT, "A7 compute: %ux%ux%u workgroups (Nx1x1 only)",
                       group_count[0], group_count[1], group_count[2]);
   if (group_count[0] > G3_MAX_WORKGROUPS)
      return vk_errorf(cmd, VK_ERROR_FEATURE_NOT_PRESENT,
                       "A7 compute: %u workgroups (at most %u: one CDM record each, one control stream block)",
                       group_count[0], G3_MAX_WORKGROUPS);
   /* no store: no observable invocation (the A8 omits such a dispatch too) */
   if (!p->num_stores)
      return VK_SUCCESS;
   for (unsigned j = 0; j < p->num_stores; j++) {
      result = g3_buffer_address(cmd, &p->stores[j], &buffer[j]);
      if (result != VK_SUCCESS)
         return result;
   }

   const uint32_t usc = pipeline_usc_offset(device, cmd->state.compute_pipeline->g9p.shader_bo);
   const unsigned ncode = g3_pds_code(p, code);
   const uint32_t k8 = g3_cdm_kernel8(p);
   struct pvr_csb *csb = &sub->control_stream;

   for (uint32_t g = 0; g < group_count[0]; g++) {
      const unsigned ndata = g3_pds_data(p, usc, buffer, base_group[0] + g, data);
      struct pvr_pds_upload up;

      /* the code once, with workgroup 0's data; the other workgroups' data alone */
      result = pvr_arch_cmd_buffer_upload_pds(cmd, data, ndata, ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT,
                                              g ? NULL : code, g ? 0 : ncode,
                                              ROGUE_CDMCTRL_KERNEL2_CODE_ADDR_ALIGNMENT,
                                              ROGUE_CDMCTRL_KERNEL1_DATA_ADDR_ALIGNMENT, &up);
      if (result != VK_SUCCESS)
         return result;
      if (!g)
         first = up;
      const uint32_t rec[7] = {
         g3_cdm_kernel0(p, ndata), up.data_offset, first.code_offset, 0, 0, 0, k8,
      };
      if (g3_dump()) {
         char what[32];
         snprintf(what, sizeof(what), "wg %u data", base_group[0] + g);
         g3_dump_words(what, data, ndata);
         g3_dump_words("record", rec, 7);
      }
      pvr_csb_set_relocation_mark(csb);
      uint32_t *w = pvr_arch_csb_alloc_dwords(csb, 7);
      if (w)
         memcpy(w, rec, sizeof(rec));
      pvr_csb_clear_relocation_mark(csb);
      result = pvr_csb_get_status(csb);
      if (result != VK_SUCCESS)
         return result;
   }
   sub->g5p_kernel_since_end = true;
   return VK_SUCCESS;
}
