/* SPDX-License-Identifier: MIT */
/* A9 compute. Pipeline/descriptor setup adapted from G9P's
 * pvr_arch_g9p.c; A9 CDM records are encoded independently from Rogue fields.
 * No captured USC/PDS/context window or fixture allowlist is used. */
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "vk_pipeline.h"
#include "vk_alloc.h"
#include "util/ralloc.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"
#include "util/u_call_once.h"
#include "util/log.h"
#include "g9p_cdm.h"
#include "g9p_compile.h"
#include "g5p_cdm.h"
#include "g5p_usc.h"
#include "g5p_end_barrier.h"
#include "pvr_bo.h"
#include "pvr_compute_limits.h"
#include "pvr_cmd_buffer.h"
#include "pvr_csb.h"
#include "pds/pvr_rogue_pds_defs.h"
#include "pvr_descriptor_set.h"
#include "pvr_image.h"
#include "vk_enum_to_str.h"
#include "pvr_device.h"
#include "pvr_physical_device.h"
#include "pvr_pipeline.h"
#include "pvr_g5p.h"
#include "pvr_g5p_tex_encode.h"
#include "pvr_g5p_shader_dump.h"
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
   state->a9_compute_log = pvr_g5p_compute_log();
   pvr_g5p_shader_id(&create_info->stage, state->a9_pipeline_id);
   void *mem_ctx = ralloc_context(NULL);
   struct g9p_shader shader;
   nir_shader *nir;
   VkResult result;

   result = vk_pipeline_shader_stage_to_nir(&device->vk,
                                            pipeline->base.pipeline_flags,
                                            &create_info->stage,
                                            g9p_spirv_options(),
                                            g9p_nir_options(), mem_ctx, &nir);
   if (result != VK_SUCCESS) {
      pvr_g5p_shader_dump(&create_info->stage, create_info->layout, "comp", NULL, 0, "REFUSE",
                          "SPIR-V to NIR failed");
      goto out;
   }

   const bool compiled = g9p_a9_compile_compute(nir, &shader, mem_ctx);
   for (unsigned i = 0; i < 3; i++) state->a9_log_local[i] = nir->info.workgroup_size[i];
   state->a9_log_shared = nir->info.shared_size;
   if (!compiled) {
      /* PVR_DEBUG=shader_dump / PVR_DEBUG=capture_stub (pvr_g5p_shader_dump.h) */
      const bool stub = pvr_g5p_capture_stub();
      char reason[sizeof(shader.a9_refusal)];
      snprintf(reason, sizeof(reason), "%s", shader.a9_refusal[0] ? shader.a9_refusal : g9p_a9_last_refusal());
      pvr_g5p_shader_dump(&create_info->stage, create_info->layout, "comp", NULL, 0, stub ? "STUB" : "REFUSE",
                          reason);
      if (!stub) {
         mesa_logw("pvr/g5p: compute pipeline refused: %s", reason);
         result = vk_errorf(device, VK_ERROR_FEATURE_NOT_PRESENT,
                            "G5P compute: shader not supported by the backend yet");
         goto out;
      }
      state->a9_capture_stub = true;
      mesa_logw("G5P capture stub: compute shader refused (%s): empty kernel, dispatches omitted", reason);
      VkPipelineShaderStageCreateInfo stub_info;
      VkShaderModuleCreateInfo stub_module;
      pvr_g5p_stub_stage(VK_SHADER_STAGE_COMPUTE_BIT, &stub_info, &stub_module);
      result = vk_pipeline_shader_stage_to_nir(&device->vk, pipeline->base.pipeline_flags, &stub_info,
                                               g9p_spirv_options(), g9p_nir_options(), mem_ctx, &nir);
      if (result != VK_SUCCESS)
         goto out;
      const bool compiled = g9p_a9_compile_compute(nir, &shader, mem_ctx);
   for (unsigned i = 0; i < 3; i++) state->a9_log_local[i] = nir->info.workgroup_size[i];
   state->a9_log_shared = nir->info.shared_size;
   if (!compiled) {
         result = vk_errorf(device, VK_ERROR_FEATURE_NOT_PRESENT, "G5P compute: the capture stub was refused");
         goto out;
      }
   } else {
      pvr_g5p_shader_dump(&create_info->stage, create_info->layout, "comp", NULL, 0, "ADMIT", NULL);
   }
   {
      const unsigned threads = shader.local_size[0] * shader.local_size[1] * shader.local_size[2];
      unsigned max;
      if (!pvr_compute_wg_residency_ok(&device->pdevice->dev_info,
                                      shader.num_regs,
                                      shader.a9_bank1_regs,
                                      threads,
                                      shader.a9_has_barriers,
                                      &max)) {
         const unsigned t = 8 * DIV_ROUND_UP(MAX2(shader.num_regs, 1), 8);
         const unsigned f = 8 * DIV_ROUND_UP(shader.a9_bank1_regs ? shader.a9_bank1_regs : shader.num_regs, 8);
         mesa_logw("G5P refuses: a %u-lane workgroup with %u bank-0 and %u bank-1 registers per lane, above "
                   "the %u lanes Apple's A9 driver allows for that allocation", threads, t, f, max);
         result = vk_errorf(device, VK_ERROR_FEATURE_NOT_PRESENT,
                            "G5P compute: %u-lane workgroup above the %u-lane limit of its allocation", threads,
                            max);
         goto out;
      }
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
   state->a9_bank1_regs = shader.a9_bank1_regs;
   state->a9_bodyless = shader.a9_bodyless;
   state->a9_wgid_z = shader.a9_wgid_z;
   state->a9_local_dwords = shader.a9_local_dwords;
   state->a9_has_barriers = shader.a9_has_barriers;
   state->a9_device_fence = shader.a9_device_fence;
   state->a9_num_images=shader.num_textures;
   for(unsigned i=0;i<shader.num_textures;i++) {
      state->a9_images[i].set=shader.textures[i].set;
      state->a9_images[i].binding=shader.textures[i].binding;
   }
   /* sampled textures' descriptor words (g9p_a9_tex.h) */
   STATIC_ASSERT(ARRAY_SIZE(state->a9_uploads) == G9P_A9_MAX_UPLOADS);
   state->a9_num_uploads = shader.a9_num_uploads;
   for (unsigned i = 0; i < shader.a9_num_uploads; i++) {
      const struct g9p_a9_upload *u = &shader.a9_uploads[i];
      state->a9_uploads[i].kind = u->kind;
      state->a9_uploads[i].set = u->set;
      state->a9_uploads[i].binding = u->binding;
      state->a9_uploads[i].element = u->element;
      state->a9_uploads[i].combined = u->combined;
      state->a9_uploads[i].level = u->level;
      state->a9_uploads[i].common = u->common;
      state->a9_uploads[i].dwords = u->dwords;
      state->a9_uploads[i].value = u->value;
   }
   state->shared_spill_stride = shader.shared_spill_stride;
   state->uses_workgroup_id = shader.uses_workgroup_id;
   memcpy(state->local_size, shader.local_size, sizeof(state->local_size));

out:
   ralloc_free(mem_ctx);
   return pvr_refusal_to_api(result);
}

static void
emit_record(struct pvr_csb *csb, const struct g5p_cdm_record *rec)
{
   uint32_t *w = pvr_arch_csb_alloc_dwords(csb, 7);

   if (w)
      memcpy(w, rec->w, sizeof(rec->w));
}

/* Capture complete bound compute resources, including a stub's original
 * descriptors. No descriptor or pipeline pointers survive command recording. */
static VkResult
g5p_record_dispatch(struct pvr_cmd_buffer *cmd, struct pvr_sub_cmd_compute *sub,
                    uint64_t indirect_va, uint64_t indirect_offset, const uint32_t groups[3])
{
   const struct pvr_g9p_compute_state *s = &cmd->state.compute_pipeline->g9p;
   struct pvr_g5p_dispatch_log *l = calloc(1, sizeof(*l));
   if (!l) return vk_error(cmd, VK_ERROR_OUT_OF_HOST_MEMORY);
   size_t size;
   FILE *f = open_memstream(&l->line, &size);
   if (!f) { free(l); return vk_error(cmd, VK_ERROR_OUT_OF_HOST_MEMORY); }
   memcpy(l->pipeline, s->a9_pipeline_id, sizeof(l->pipeline));
   l->stub = s->a9_capture_stub;
   fprintf(f, "pipe=%s stub=%u local=%ux%ux%u shared=%u ", l->pipeline, l->stub,
           s->a9_log_local[0], s->a9_log_local[1], s->a9_log_local[2], s->a9_log_shared);
   if (indirect_va)
      fprintf(f, "groups=indirect buffer_va=%#" PRIx64 " offset=%" PRIu64 " args_va=%#" PRIx64,
              indirect_va - indirect_offset, indirect_offset, indirect_va);
   else
      fprintf(f, "groups=%ux%ux%u", groups[0], groups[1], groups[2]);
   fprintf(f, " res=[");
   const struct pvr_descriptor_state *desc = &cmd->state.compute_desc_state;
   const char *sep = "";
   for (unsigned si = 0; si < PVR_MAX_DESCRIPTOR_SETS; si++) {
      const struct pvr_descriptor_set *set = desc->sets[si];
      /* a set of dynamic buffers only has no descriptor memory (no mapping) */
      if (!set || !set->layout) continue;
      for (unsigned bi = 0; bi < set->layout->binding_count; bi++) {
         const struct pvr_descriptor_set_layout_binding *b = &set->layout->bindings[bi];
         if (!(b->stage_flags & VK_SHADER_STAGE_COMPUTE_BIT)) continue;
         if (!set->mapping && !pvr_descriptor_set_binding_is_dynamic(b)) continue;
         for (unsigned e = 0; e < b->descriptor_count; e++) {
            const uint8_t *d = set->mapping ? (uint8_t *)set->mapping + b->offset + e * b->stride : NULL;
            switch (b->type) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
               struct pvr_buffer_descriptor buf;
               if (b->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                   b->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
                  buf = set->dynamic_buffers[b->dynamic_buffer_idx + e];
               else
                  memcpy(&buf, d, sizeof(buf));
               fprintf(f, "%sbuffer %u/%u[%u] VA=%#" PRIx64 " range=%u", sep, si, bi, e, buf.addr + buf.offset, buf.size);
               sep = "; ";
               break;
            }
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
               const struct pvr_image_view *v;
               uintptr_t ptr;
               memcpy(&ptr, d + 24, sizeof(ptr));
               v = (const struct pvr_image_view *)ptr;
               if (!v) {
                  fprintf(f, "%simage %u/%u[%u] unbound", sep, si, bi, e);
               } else {
                  const struct pvr_image *im = pvr_image_view_get_image(v);
                  uint64_t q[3];
                  memcpy(q, d, sizeof(q));
                  const bool linear = !(q[0] & PVR_G5P_TEX_TWIDDLED);
                  uint64_t va = (q[1] & ((UINT64_C(1) << 39) - 1) & ~UINT64_C(31)) << 1;
                  if (linear && (q[1] >> 63)) va |= ((q[2] >> 58) & 3) << 4;
                  fprintf(f, "%s%s %u/%u[%u] VA=%#" PRIx64 " bytes=%" PRIu64
                          " %ux%ux%u format=%s view=%s", sep,
                          b->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? "storage" : "sampled",
                          si, bi, e, va, im->total_size,
                          u_minify(im->vk.extent.width, v->vk.base_mip_level),
                          u_minify(im->vk.extent.height, v->vk.base_mip_level), v->vk.layer_count,
                          vk_Format_to_str(v->vk.format), vk_ImageViewType_to_str(v->vk.view_type));
                  if (b->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                     fprintf(f, " dim=%#x", (q[0] & 7) == PVR_G5P_TEX_LAYOUT_2D_ARRAY ? 0xf7 : 0xf5);
               }
               sep = "; ";
               break;
            }
            default: break;
            }
         }
      }
   }
   fprintf(f, "]");
   const bool failed = ferror(f);
   if (fclose(f) || failed) { free(l->line); free(l); return vk_error(cmd, VK_ERROR_OUT_OF_HOST_MEMORY); }
   if (sub->g5p_log_last) sub->g5p_log_last->next = l;
   else sub->g5p_log_first = l;
   sub->g5p_log_last = l;
   sub->g5p_log_count++;
   return VK_SUCCESS;
}

/* This translation unit is A9-only. Read the override once, off the hot path. */
/* Bound per-dispatch shared-memory spill allocations to 64 MiB. */
#define PVR_G5P_SHARED_SPILL_MAX_BYTES (64u << 20)

/*
 * One dispatch: a DMA task loading the buffer addresses (from the bound
 * descriptor sets) and the thread count into the common store, then the
 * kernel, whose PDS program writes the workgroup ID into threadgroup words
 * 0-1 and kicks the USC.
 */
VkResult
PVR_PER_ARCH(g5p_cmd_dispatch)(struct pvr_cmd_buffer *cmd_buffer,
                               struct pvr_sub_cmd_compute *sub_cmd,
                               uint64_t indirect_va, uint64_t indirect_offset, const uint32_t base_group[static 3],
                               const uint32_t group_count[static 3])
{
   struct pvr_device *device = cmd_buffer->device;
   const struct pvr_compute_pipeline *pipeline = cmd_buffer->state.compute_pipeline;
   const struct pvr_g9p_compute_state *state = &pipeline->g9p;
   const struct pvr_descriptor_state *desc = &cmd_buffer->state.compute_desc_state;
   const unsigned n = state->num_buffers;
   const uint32_t threads =
      state->local_size[0] * state->local_size[1] * state->local_size[2];
   const uint32_t hw_threads = g5p_cdm_workgroup_threads(threads);
   const uint32_t count = threads | threads << 16;
   struct pvr_suballoc_bo *uniforms_bo;
   uint32_t uniforms[2 * G9P_MAX_BUFFERS + 4 + 8 * G9P_A9_MAX_UPLOADS] = { 0 };
   uint32_t bind_data[4 * (G9P_MAX_BUFFERS + 1) + 4 + 3 * G9P_A9_MAX_UPLOADS] = { 0 };
   uint32_t bind_code[G9P_MAX_BUFFERS + 5 + G9P_A9_MAX_UPLOADS];
   uint32_t kernel_data[12] = { 0 }, kernel_code[8];
   bool from_uniforms[G9P_MAX_BUFFERS] = { false };
   struct pvr_pds_upload bind, kernel;
   unsigned bind_dwords, ctrl0, nc = 0;
   VkResult result;

   if (state->a9_compute_log) {
      result = g5p_record_dispatch(cmd_buffer, sub_cmd, indirect_va, indirect_offset, group_count);
      if (result != VK_SUCCESS) return result;
   }

   if (base_group[0] || base_group[1] || base_group[2])
      return vk_errorf(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY, "G5P: base dispatch not implemented yet");
   if (indirect_va && ((indirect_va & 3) || indirect_va >> 40))
      return vk_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY);
   if (!indirect_va && (!group_count[0] || !group_count[1] || !group_count[2]))
      return VK_SUCCESS;
   if (!threads || threads > 512 || group_count[0] > 65536 || group_count[1] > 65536 || group_count[2] > 65536)
      return vk_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* A bodyless shader has no observable invocation.  Apple's standalone
    * empty program does not retire under our CDM launch contract, so do not
    * launch it: the sub-command's ordinary completion tail still establishes
    * command-buffer ordering and signals the submit fence. */
   if (state->a9_bodyless) {
      /* An indirect bodyless dispatch has no consumer of its dimensions,
       * so its argument read is unobservable and is omitted like the kernel
       * record. A standalone usc_common_shared PDS DMA task here (the
       * earlier form) left the CDM silent on A9 hardware: its shared common
       * store allocation has no following kernel to consume it. The ordered
       * pass tail alone retires (direct case). */
      return VK_SUCCESS;
   }

   struct pvr_bo *shared_spill = NULL;
   if (state->shared_spill_stride) {
      if (indirect_va) {
         if (p_atomic_cmpxchg(&state->shared_spill_logged, 0, 1) == 0)
            mesa_logw("G5P refuses: indirect dispatch of a spilling workgroup-memory pipeline");
         vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }

      uint64_t groups = (uint64_t)group_count[0] * group_count[1] * group_count[2];
      /* Divide before multiplying: even a valid grid can overflow bytes. */
      if (groups > PVR_G5P_SHARED_SPILL_MAX_BYTES / state->shared_spill_stride) {
         if (p_atomic_cmpxchg(&state->shared_spill_logged, 0, 1) == 0)
            mesa_logw("G5P refuses: a dispatch of %" PRIu64 " groups × %u bytes of spilled workgroup memory "
                      "(%.2f MB, above the cap)", groups, state->shared_spill_stride,
                      (double)groups * state->shared_spill_stride / (1024 * 1024));
         vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      /* bo_list is freed as pvr_suballoc_bo, despite its pvr_bo comment.
       * Give the dedicated generic BO a normal ownership wrapper. Reset
       * may cache it in suballoc_general, so use that allocator's mapping
       * flags too; a later ordinary upload must find a mapped BO. */
      struct pvr_suballoc_bo *owner = vk_zalloc(&device->vk.alloc, sizeof(*owner), 8,
                                                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!owner) {
         vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      result = pvr_bo_alloc(device, device->heaps.general_heap,
                           groups * state->shared_spill_stride, 64,
                           PVR_BO_ALLOC_FLAG_CPU_MAPPED, &shared_spill);
      if (result != VK_SUCCESS) {
         vk_free(&device->vk.alloc, owner);
         vk_command_buffer_set_error(&cmd_buffer->vk, result);
         return result;
      }
      owner->allocator = &device->suballoc_general;
      owner->bo = shared_spill;
      owner->dev_addr = shared_spill->vma->dev_addr;
      owner->size = groups * state->shared_spill_stride;
      list_add(&owner->link, &cmd_buffer->bo_list);
   }

   /*
    * Values the DMA copies that are not in descriptor memory: the
    * push-constant buffer's address and the thread count.
    */
   for (unsigned k = 0; k < n; k++) {
      /* gl_NumWorkGroups (the A9 compiler's G9P_NUM_WORKGROUPS slots, g9p_compile.c): x and y in binding 0's
       * slot, z in binding 1's (G9P's 496e44739dc ("pvr/g9p: gl_NumWorkGroups, workgroup ID z, storage buffer array elements in compute")). An indirect dispatch's counts are the GPU's. */
      if (state->buffers[k].set == G9P_NUM_WORKGROUPS) {
         if (indirect_va)
            return vk_errorf(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY,
                             "G5P: gl_NumWorkGroups in an indirect dispatch");
         uniforms[2 * k] = group_count[state->buffers[k].binding ? 2 : 0];
         uniforms[2 * k + 1] = state->buffers[k].binding ? 0 : group_count[1];
      }
      if (state->buffers[k].set == G9P_SHARED_SPILL) {
         uniforms[2 * k] = shared_spill->vma->dev_addr.addr;
         uniforms[2 * k + 1] = shared_spill->vma->dev_addr.addr >> 32;
      }
      if (state->buffers[k].set == G9P_PUSH_CONSTANTS) {
         const struct pvr_push_constants *pc =
            &cmd_buffer->state.push_consts[PVR_STAGE_ALLOCATION_COMPUTE];

         uniforms[2 * k] = pc->dev_addr.addr;
         uniforms[2 * k + 1] = pc->dev_addr.addr >> 32;
      }
      if (state->buffers[k].set < PVR_MAX_DESCRIPTOR_SETS) {
         /* A dynamic buffer has no descriptor in set memory (its binding
          * offset is ~0: a DMA from there reads set->dev_addr + 0xffffffff).
          * Its address with the bound dynamic offset, or its range for
          * arrayLength(), goes through this upload. */
         const unsigned element = (state->buffers[k].binding & ~G9P_BINDING_SIZE) >> G9P_BINDING_ELEMENT_SHIFT;
         const unsigned b = state->buffers[k].binding & BITFIELD_MASK(G9P_BINDING_ELEMENT_SHIFT);

         if (!pvr_descriptor_set_dynamic_slot(desc->sets[state->buffers[k].set], b, element,
                                              state->buffers[k].binding & G9P_BINDING_SIZE,
                                              &uniforms[2 * k], &from_uniforms[k]))
            return vk_errorf(cmd_buffer, VK_ERROR_UNKNOWN,
                             "G5P compute: set %u binding %u element %u holds no bound buffer",
                             state->buffers[k].set, b, element);
      }
   }
   unsigned images=state->a9_num_images;
   if(images && n>15)return vk_errorf(cmd_buffer,VK_ERROR_OUT_OF_HOST_MEMORY,"G5P image LOD common31 overlaps the16th buffer pointer");
   const unsigned base_transfers=n+1+images+(state->a9_device_fence ? 1 : 0)+(images ? 1 : 0);
   const unsigned tex0=2*n+1+(state->a9_device_fence ? 2 : 0)+(images ? 1 : 0);
   const unsigned nup=state->a9_num_uploads;
   unsigned transfers=base_transfers+nup;
   /* sampled textures' words, validated as the fragment path's (pvr_arch_g5p_gfx.c), from the
    * bound sets' CPU mappings into the uniforms upload, then DMAed to their common dwords */
   for(unsigned i=0;i<nup;i++) {
      const __typeof__(state->a9_uploads[0]) *u=&state->a9_uploads[i];
      uint32_t *out=&uniforms[tex0+8*i];
      if (u->kind == G9P_A9_UP_LOD) {
         out[0] = u->value;
         continue;
      }
      if(u->kind==G9P_A9_UP_FETCH_SAMPLER) {
         const uint64_t policy=pvr_g5p_fetch_sampler_word();
         memcpy(out,&policy,sizeof(policy));
         continue;
      }
      const struct pvr_descriptor_set *set=u->set<PVR_MAX_DESCRIPTOR_SETS ? desc->sets[u->set] : NULL;
      if(!set || !set->layout || !set->mapping || u->binding>=set->layout->binding_count)
         return vk_errorf(cmd_buffer,VK_ERROR_UNKNOWN,"G5P compute: texture set %u binding %u is not bound",u->set,u->binding);
      const struct pvr_descriptor_set_layout_binding *lb=&set->layout->bindings[u->binding];
      if(u->element>=MAX2(lb->descriptor_count,1))
         return vk_errorf(cmd_buffer,VK_ERROR_UNKNOWN,"G5P compute: set %u binding %u element %u past its descriptors",u->set,u->binding,u->element);
      const uint32_t *d=(const uint32_t *)((const uint8_t *)set->mapping+lb->offset+(size_t)u->element*lb->stride);
      switch(u->kind) {
      case G9P_A9_UP_IMAGE:
         if(lb->type==VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || !pvr_g5p_tex_state_valid(d)) {
            mesa_logw("G5P refuses: compute set %u binding %u: a view without A9 texture state",u->set,u->binding);
            return vk_errorf(cmd_buffer,VK_ERROR_OUT_OF_HOST_MEMORY,"G5P compute: a view without A9 texture state");
         }
         memcpy(out,d,4*8);
         break;
      case G9P_A9_UP_SAMPLER:
         /* a combined descriptor's sampler is at +32, a standalone sampler's at +0 */
         if(lb->type!=VK_DESCRIPTOR_TYPE_SAMPLER)
            d+=8;
         if(!pvr_g5p_sampler_valid(d)) {
            mesa_logw("G5P refuses: compute set %u binding %u: a sampler without A9 words",u->set,u->binding);
            return vk_errorf(cmd_buffer,VK_ERROR_OUT_OF_HOST_MEMORY,"G5P compute: a sampler without A9 words");
         }
         memcpy(out,d,4*2);
         break;
      case G9P_A9_UP_SIZE:
         if(!pvr_g5p_tex_size(d,u->level,out)) {
            mesa_logw("G5P refuses: compute set %u binding %u: an image size without A9 texture state",u->set,u->binding);
            return vk_errorf(cmd_buffer,VK_ERROR_OUT_OF_HOST_MEMORY,"G5P compute: an image size without A9 texture state");
         }
         break;
      default:
         return vk_error(cmd_buffer,VK_ERROR_UNKNOWN);
      }
   }
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
                                               sizeof(uint32_t) * (tex0 + 8 * nup),
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

      if (k >= base_transfers) {
         const unsigned i = k - base_transfers;
         src = uniforms_bo->dev_addr.addr + 4 * (tex0 + 8 * i);
         dest = state->a9_uploads[i].common;
         dwords = state->a9_uploads[i].dwords;
      } else if (k < n && !from_uniforms[k] && state->buffers[k].set != G9P_PUSH_CONSTANTS &&
                 state->buffers[k].set != G9P_NUM_WORKGROUPS &&
                 state->buffers[k].set != G9P_SHARED_SPILL) {
         const struct pvr_descriptor_set *set = desc->sets[state->buffers[k].set];
         /* an array's element (Zink keeps every storage buffer in one array) and the size flag (the
          * descriptor's size word at byte 8, arrayLength()): G9P's 496e44739dc ("pvr/g9p: gl_NumWorkGroups, workgroup ID z, storage buffer array elements in compute"), 1d51ff5cec6 */
         const unsigned element = (state->buffers[k].binding & ~G9P_BINDING_SIZE) >> G9P_BINDING_ELEMENT_SHIFT;
         const unsigned b = state->buffers[k].binding & BITFIELD_MASK(G9P_BINDING_ELEMENT_SHIFT);
         const struct pvr_descriptor_set_layout_binding *binding;

         if (!set || b >= set->layout->binding_count)
            return vk_error(cmd_buffer, VK_ERROR_UNKNOWN);
         binding = &set->layout->bindings[b];
         if (element >= MAX2(binding->descriptor_count, 1))
            return vk_error(cmd_buffer, VK_ERROR_UNKNOWN);
         /* pvr_buffer_descriptor: the address comes first, the size at byte 8 */
         src = set->dev_addr.addr + binding->offset + element * binding->stride +
               (state->buffers[k].binding & G9P_BINDING_SIZE ? 8 : 0);
      } else if(k>n && k<n+1+images) {
         unsigned i=k-n-1;
         const struct pvr_descriptor_set *set=desc->sets[state->a9_images[i].set];
         if(!set)return vk_error(cmd_buffer,VK_ERROR_UNKNOWN);
         const struct pvr_descriptor_set_layout_binding *binding=&set->layout->bindings[state->a9_images[i].binding];
         src=set->dev_addr.addr+binding->offset;
         dest=40+8*i;dwords=8;
      } else if(k>n && state->a9_device_fence && k==n+1+images) {
         src=uniforms_bo->dev_addr.addr + 4*(2*n+1);
         dest=56;
      } else if(k>n) {
         src=uniforms_bo->dev_addr.addr+4*(2*n+1+(state->a9_device_fence ? 2 : 0));
         dest=31;dwords=1;
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
    * threadgroup words 0-1, and z from q1's upper half -> word 3 when the
    * kernel reads it) in the coefficient task.
    */
   g9p_pds_kick_data(kernel_data, usc_offset(device, state->shader_bo));
   /* the DOUTU temps allocate bank 0 (homes up to r(num_regs - 1)); the record's register field below
    * allocates bank 1, the code's own registers (g9p_cdm.h G9P_CDM_KERNEL_REGS) */
   const unsigned bank1 = state->a9_bank1_regs ? state->a9_bank1_regs : state->num_regs;
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
   if (state->a9_wgid_z) {
      /* Workgroup ID z, only for a kernel that reads it: AGXMetalA9's EncodeComputePDSProgram (0x207d1ea4c, flags
       * bit 2) adds a DOUTW of temps q1's upper half (g9p_pds_dout(G9P_DOUTW, 1, G9P_PDS_TEMPS, ...); its source-1
       * word below: last, the common store, the destination word, the upper 32 bits) after the x/y one (all 64
       * bits), which then drops its last bit; G9P's 496e44739dc uses the same form on the A10.
       * Our word is G9P_A9_TG_WGID_Z (3): word 2 is the barrier counter, whose zero pair is written first. */
      const unsigned at = state->a9_has_barriers ? 2 : 1;
      kernel_data[4] = PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_DEST_COMMON_STORE |
                       PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_BSIZE_ALL64;
      kernel_data[5] = PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_LAST_EN |
                       PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_DEST_COMMON_STORE |
                       G9P_A9_TG_WGID_Z << PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_AO_SHIFT |
                       PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTW_SRC1_BSIZE_UPPER;
      kernel_code[at] = g9p_pds_dout(G9P_DOUTW, 0, G9P_PDS_TEMPS, 4, 0);
      kernel_code[at + 1] = g9p_pds_dout(G9P_DOUTW, 1, G9P_PDS_TEMPS, 5, G9P_PDS_END);
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
                          bind.code_offset, (state->a9_device_fence || images || nup) ? 4 : DIV_ROUND_UP(2 * n + 1, 16));
      const struct g5p_cdm_record kernel_rec = g5p_cdm_kernel(
         G9P_CDM_KERNEL_REGS(bank1) | G9P_CDM_ANY_CLUSTER |
            G9P_CDM_FENCE |
            G9P_CDM_DATA_BLOCKS(kernel_dwords / 4),
         kernel.data_offset, kernel.code_offset, group_count[0], group_count[1],
         group_count[2], hw_threads, DIV_ROUND_UP(state->a9_local_dwords, 16));

      pvr_csb_set_relocation_mark(csb);
      emit_record(csb, &bind_rec);
      sub_cmd->g5p_kernel_since_end = true;
      if (indirect_va) {
         const struct g5p_cdm_indirect_record rec = g5p_cdm_indirect(
            kernel_rec.w[0], kernel.data_offset, kernel.code_offset,
            indirect_va, hw_threads, 0);
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
   return pvr_arch_cmd_buffer_upload_pds(cmd, data, 4, 16, code, 2, 32, 64, out);
}

/* iOS's A9 compute end of pass is five records; the first is a DMA task on
 * ALL clusters (K0 usc_target ALL, the only such record in the pass) that
 * kicks the compute flush program (g5p_end_barrier.h). Every iOS compute job
 * runs it after every dispatch, including between the dispatches of one CDM
 * job (captures g3b-cs-atomic, g3b-cs-image, r19d; ios-witness-r1 w-wg-* and
 * w-cs-*):
 *   PDS  DOUTD: the 16-byte source -> common dwords 0..3;
 *        DOUTU: kick the flush program (last instruction); HALT
 *   USC  the compute flush program (g5p_end_barrier.h): r0:r1 = common
 *        dwords 0/1, device memory fence, wait, fence, zero-word emit,
 *        wait, branch/pad/END
 * Without this record, image stores whose workgroup rows are partial cache
 * lines (workgroup width < 32 for R32) lose the last workgroups' writes:
 * they never reach memory (release9d r3diag: 384 of 4096 words for 8x8,
 * 16x4, 8x1 and 16x1 stores). With it every shape is exact, natively and
 * through Zink (r3c, r3d, r3e).
 * It follows every pass that emitted a kernel record; a pass whose
 * dispatches were all bodyless (omitted) keeps the plain tail. */

/* The fence operand the flush program loads into r0:r1 (common dwords 0
 * and 1): the two words iOS's driver DMAs for this task. In compiler-built
 * kernels the fence operand is the ComputeMemFenceAddr address; what the
 * flush program's form does with it is not settled. */
#define G5P_END_BARRIER_FENCE_OPERAND_LO 0x40u
#define G5P_END_BARRIER_FENCE_OPERAND_HI 0x10u
/* The DOUTD copies the whole 16-byte source, in dwords (Rogue BSIZE). */
#define G5P_END_BARRIER_SOURCE_DWORDS 4u

static VkResult
upload_end_barrier(struct pvr_cmd_buffer *cmd, struct pvr_pds_upload *out)
{
   uint8_t usc[128] = {0};
   const unsigned at = g5p_end_barrier_helper(usc);
   if (!at)
      return vk_errorf(cmd, VK_ERROR_UNKNOWN, "G5P end-of-pass helper: %s", g5p_usc_error);
   struct pvr_suballoc_bo *bo, *source;
   VkResult r = pvr_gpu_upload_usc(cmd->device, usc, ALIGN_POT(at, 64), 64, &bo);
   if (r != VK_SUCCESS) return r;
   list_add(&bo->link, &cmd->bo_list);
   const uint32_t words[G5P_END_BARRIER_SOURCE_DWORDS] = {
      G5P_END_BARRIER_FENCE_OPERAND_LO, G5P_END_BARRIER_FENCE_OPERAND_HI, 0, 0,
   };
   r = pvr_arch_cmd_buffer_upload_general(cmd, words, sizeof(words), &source);
   if (r != VK_SUCCESS) return r;
   const uint64_t va = source->dev_addr.addr;
   assert(!(va >> PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC0_DOFFSET_SHIFT));
   /* q0: DOUTD source 0, the source address (SBASE), common store offset 0
    * (DOFFSET), cached (SLCMODE). */
   const uint64_t doutd_src0 = va << PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC0_SBASE_SHIFT;
   /* DOUTD source 1: the program's last DMA, into the common store, cached,
    * at address offset 0, BSIZE dwords. */
   const uint32_t doutd_src1 = PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_LAST_EN |
                               PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_DEST_COMMON_STORE |
                               PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_CMODE_CACHED |
                               0u << PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_AO_SHIFT |
                               G5P_END_BARRIER_SOURCE_DWORDS << PVR_ROGUE_PDSINST_DOUT_FIELDS_DOUTD_SRC1_BSIZE_SHIFT;
   uint32_t data[8] = {(uint32_t)doutd_src0, (uint32_t)(doutd_src0 >> 32), doutd_src1, 0};
   g9p_pds_kick_data(&data[4], usc_offset(cmd->device, bo)); /* q2: DOUTU source */
   uint32_t code[3] = {
      g9p_pds_dout(G9P_DOUTD, 0, G9P_PDS_DATA, 2, 0),
      g9p_pds_dout(G9P_DOUTU, 2, G9P_PDS_DATA, 0, G9P_PDS_END),
      g9p_pds_halt(),
   };
   return pvr_arch_cmd_buffer_upload_pds(cmd, data, 8, 16, code, 3, 32, 64, out);
}

VkResult
PVR_PER_ARCH(g5p_compute_end_pass)(struct pvr_cmd_buffer *cmd,
                                  struct pvr_sub_cmd_compute *sub)
{
   uint32_t data[4] = {0}, code[4];
   struct pvr_pds_upload fence, fenced, kick, barrier;
   /* after a dispatch only, as iOS: a pass whose dispatches were all
    * bodyless (omitted) keeps the plain tail */
   const bool end_barrier = sub->g5p_kernel_since_end;
   sub->g5p_kernel_since_end = false;
   VkResult task_result = upload_cdm_task(cmd, false, &kick);
   if (task_result == VK_SUCCESS)
      task_result = upload_cdm_task(cmd, true, &fenced);
   if (task_result == VK_SUCCESS && end_barrier)
      task_result = upload_end_barrier(cmd, &barrier);
   if (task_result != VK_SUCCESS) return task_result;
   g9p_pds_fence_program(code);
   VkResult r = pvr_arch_cmd_buffer_upload_pds(cmd, data, 4, 16, code, 4, 16, 16, &fence);
   if (r != VK_SUCCESS) return r;
   struct g5p_cdm_record rec;
   pvr_csb_set_relocation_mark(&sub->control_stream);
   if (end_barrier) {
      /* The flush task's record in Rogue's CDM kernel layout (csbgen
       * rogue/cdm.xml), with the values iOS's driver writes. Words 3-5 (the
       * workgroup counts minus one) stay 0: one workgroup. */
      rec = (struct g5p_cdm_record){ 0 };
      pvr_csb_pack (&rec.w[0], CDMCTRL_KERNEL0, k0) {
         k0.usc_common_size = 1; /* one 64-byte unit holds the 16-byte source */
         k0.pds_data_size = 1;   /* two 16-byte blocks, encoded minus one (G9P_CDM_DATA_BLOCKS) */
         k0.usc_target = ROGUE_CDMCTRL_USC_TARGET_ALL; /* every cluster runs it */
         k0.fence = false;
      }
      pvr_csb_pack (&rec.w[1], CDMCTRL_KERNEL1, k1) {
         k1.data_addr = PVR_DEV_ADDR(barrier.data_offset);
         k1.sd_type = ROGUE_CDMCTRL_SD_TYPE_USC;
         k1.usc_common_shared = true;
      }
      pvr_csb_pack (&rec.w[2], CDMCTRL_KERNEL2, k2) {
         k2.code_addr = PVR_DEV_ADDR(barrier.code_offset);
      }
      pvr_csb_pack (&rec.w[6], CDMCTRL_KERNEL8, k8) {
         k8.max_instances = 8; /* workgroup 1x1x1: the sizes minus one stay 0 */
      }
      emit_record(&sub->control_stream, &rec);
   }
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

/* A compute context: the context state buffer the job's stream names, and
 * the kernel's context. The static context state is optional on the A9 and
 * left empty: the context programs travel with each job (the HALT). */
VkResult
PVR_PER_ARCH(g5p_compute_ctx_create)(struct pvr_device *device, enum pvr_winsys_ctx_priority priority,
                                     struct pvr_compute_ctx **out)
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
   struct pvr_winsys_compute_ctx_create_info info = { .priority = priority };
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
   /* The job names its stream's terminate word, which is taken from the
    * first block here. A chained CDM stream is refused until that form has
    * run on the device. */
   if (cs->pvr_bo != list_first_entry(&cs->pvr_bo_list, struct pvr_bo, link))
      return vk_errorf(d, VK_ERROR_OUT_OF_HOST_MEMORY, "G5P: chained CDM stream not implemented yet");
   struct pvr_winsys_compute_submit_info info = { .wait = wait,
      .g5p_log = sub->g5p_log_first, .g5p_log_count = sub->g5p_log_count,
      .frame_num = d->global_queue_present_count, .job_num = d->global_cmd_buffer_submit_count };
   info.fw_stream_len =
      pvr_arch_g5p_compute_stream(d, pvr_csb_get_start_address(cs).addr,
                                  (uint32_t)((const char *)cs->next - (const char *)cs->start),
                                  ctx->ctx_switch.compute_state_bo->vma->dev_addr.addr,
                                  info.fw_stream, sizeof(info.fw_stream));
   return d->ws->ops->compute_submit(ctx->ws_ctx, &info, &d->pdevice->dev_info, signal);
}
