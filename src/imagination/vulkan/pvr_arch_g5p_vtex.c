/* SPDX-License-Identifier: MIT
 * R21 A9 vertex textures: a bounded common-store upload plan.
 *
 * AGXMetalA9 setVertexTexture/setVertexSamplerState populate the vertex
 * argument table; VertexProgramVariantPDSState calls encodeConstantPDS.
 * Use the same DOUTD -> common-store ABI as our proven vertex UBO loader.
 * CPU-produced descriptor words are coalesced, GPU UBO data is never read
 * back on the CPU. The PDS constant-register file remains bounded to 128
 * dwords (32 transfers at four constant dwords per DOUTD).
 */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include "vk_alloc.h"
#include "vk_log.h"
#include "util/ralloc.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#include "pco_g5p_general.h"
#include "pdsc.h"
#include "pdsc_internal.h"
#include "pdsc_isa_pack.h"
#include "pdsc_ops.h"
#include "pvr_bo.h"
#include "pvr_cmd_buffer.h"
#include "pvr_csb.h"
#include "pvr_descriptor_set.h"
#include "pvr_device.h"
#include "pvr_g5p_tex.h"
#include "pvr_g5p_tex_encode.h"
#include "pvr_g5p_vtex.h"
#include "pvr_g5p_texel_buffer.h"

#define VTEX_REFUSE(obj, ...) \
   (fprintf(stderr, "G5P refuses: vertex texture " __VA_ARGS__), fprintf(stderr, "\n"), \
    vk_errorf((obj), VK_ERROR_FEATURE_NOT_PRESENT, "G5P vertex texture: " __VA_ARGS__))
#include "pvr_g5p_vtex_plan.h"
static_assert(VTEX_COMMON >= PCO_G5P_COMMON_DWORDS, "vertex common-store plan bound");

static uint32_t vtex_control(unsigned common, unsigned dwords, bool last)
{
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-vtex-control");
   struct pdsc_doutd_src1 s = {
      .last_issue = last, .store_dest = PDSC_STORE_DEST_COMMON, .ao = common, .bsize = dwords,
   };
   uint32_t word = pdsc_doutd_src1_pack(p, &s);
   ralloc_free(p);
   return word;
}

VkResult
PVR_PER_ARCH(g5p_vtex_create)(struct pvr_device *device, const struct pco_g5p_program *vs,
                              struct pvr_g5p_stage_uploads **out)
{
   *out = NULL;
   struct pvr_g5p_vtex_uploads plan = {0};
   const char *why = pvr_g5p_vtex_plan(vs->uploads, vs->num_uploads, vs->common_dwords, &plan);
   if (why)
      return VTEX_REFUSE(device, "%s", why);
   for (unsigned i = 0; i < vs->num_uploads; i++)
      if (vs->uploads[i].set >= PVR_MAX_DESCRIPTOR_SETS)
         return VTEX_REFUSE(device, "set index outside device limit");
   struct pvr_g5p_stage_uploads *up = vk_zalloc(&device->vk.alloc, sizeof(*up), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!up) return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   up->vtex = vk_alloc(&device->vk.alloc, sizeof(plan), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!up->vtex) { vk_free(&device->vk.alloc, up); return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY); }
   *up->vtex = plan;
   up->count = plan.dmas;
   up->common_dwords = vs->common_dwords;
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-vtex-loader");
   for (unsigned i = 0; i < plan.dmas; i++)
      pdsc_dout(p, pdsc_ref_const64(2*i), pdsc_ref_const32(2*plan.dmas + 2*i), pdsc_ref_null(),
                .dstdout = PDSC_DSTDOUT_D, .end = i == plan.dmas - 1);
   pdsc_halt(p);
   uint32_t code[VTEX_DMAS + 1];
   unsigned n = util_dynarray_num_elements(&p->instrs, pdsc_instr);
   assert(n <= ARRAY_SIZE(code));
   for (unsigned i = 0; i < n; i++)
      code[i] = pdsc_instr_pack(p, i, util_dynarray_element(&p->instrs, pdsc_instr, i)->op);
   ralloc_free(p);
   VkResult result = pvr_gpu_upload_pds(device, NULL, 0, 0, code, n, 16, 64, &up->loader);
   if (result != VK_SUCCESS) {
      vk_free(&device->vk.alloc, up->vtex);
      vk_free(&device->vk.alloc, up);
      return result;
   }
   *out = up;
   return VK_SUCCESS;
}

void
PVR_PER_ARCH(g5p_vtex_destroy)(struct pvr_device *device, struct pvr_g5p_stage_uploads *up)
{
   pvr_bo_suballoc_free(up->loader.pvr_bo);
   vk_free(&device->vk.alloc, up->vtex);
   vk_free(&device->vk.alloc, up);
}

static VkResult
vtex_descriptor(struct pvr_cmd_buffer *cmd, const struct pco_g5p_upload *u, uint32_t *out)
{
   const struct pvr_descriptor_state *ds = &cmd->state.gfx_desc_state;
   const struct pvr_descriptor_set *set = ds->sets[u->set];
   if (!set || !set->layout || u->binding >= set->layout->binding_count)
      return VTEX_REFUSE(cmd, "set %u binding %u is not bound", u->set, u->binding);
   const struct pvr_descriptor_set_layout_binding *lb = &set->layout->bindings[u->binding];
   if (u->element >= lb->descriptor_count)
      return VTEX_REFUSE(cmd, "set %u binding %u descriptor index %u outside %u", u->set, u->binding,
                          u->element, lb->descriptor_count);
   const uint32_t *desc = (const uint32_t *)((const uint8_t *)set->mapping + lb->offset +
                                          (size_t)u->element * lb->stride);
   if (lb->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) {
      if (!pvr_g5p_tbo_upload(u->kind, u->offset, u->dwords, desc, out))
         return VTEX_REFUSE(cmd, "unqualified texel-buffer upload at set %u binding %u element %u",
                           u->set, u->binding, u->element);
      return VK_SUCCESS;
   }
   bool image = lb->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || lb->type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
   if (u->kind == PCO_G5P_UPLOAD_IMAGE) {
      if (!image || u->dwords != 8 || !pvr_g5p_tex_state_valid(desc))
         return VTEX_REFUSE(cmd, "set %u binding %u has no A9 image state", u->set, u->binding);
      memcpy(out, desc, 32);
   } else if (u->kind == PCO_G5P_UPLOAD_SAMPLER) {
      if (lb->type != VK_DESCRIPTOR_TYPE_SAMPLER && lb->type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
         return VTEX_REFUSE(cmd, "set %u binding %u is not a sampler", u->set, u->binding);
      if (lb->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) desc += 8;
      if (u->dwords != 2 || !pvr_g5p_sampler_valid(desc))
         return VTEX_REFUSE(cmd, "set %u binding %u has no A9 sampler state", u->set, u->binding);
      memcpy(out, desc, 8);
   } else {
      if (!image || u->dwords != 4 || !pvr_g5p_tex_size(desc, u->offset, out))
         return VTEX_REFUSE(cmd, "set %u binding %u has no A9 image size", u->set, u->binding);
   }
   return VK_SUCCESS;
}

VkResult
PVR_PER_ARCH(g5p_vtex_draw)(struct pvr_cmd_buffer *cmd, const struct pvr_g5p_stage_uploads *up,
                            struct pvr_g5p_vs_upload_draw *out)
{
   const struct pvr_g5p_vtex_uploads *p = up->vtex;
   const struct pvr_push_constants *push = &cmd->state.push_consts[PVR_STAGE_ALLOCATION_VERTEX_GEOMETRY];
   uint32_t words[VTEX_COMMON] = {0}, data[4 * VTEX_DMAS] = {0};
   uint64_t direct[PCO_G5P_MAX_UPLOADS] = {0};
   memset(out, 0, sizeof(*out));
   for (unsigned i = 0; i < p->count; i++) {
      const struct pco_g5p_upload *u = &p->u[i];
      uint32_t *dst = words + u->common;
      unsigned bytes = u->dwords * 4;
      uint64_t addr, size;
      VkResult result;
      switch (u->kind) {
      case PCO_G5P_UPLOAD_IMAGE: case PCO_G5P_UPLOAD_SAMPLER: case PCO_G5P_UPLOAD_IMAGE_SIZE:
         result = vtex_descriptor(cmd, u, dst);
         if (result != VK_SUCCESS) return result;
         break;
      case PCO_G5P_UPLOAD_FETCH_SAMPLER: {
         if (u->dwords != 2) return VTEX_REFUSE(cmd, "invalid samplerless read policy size");
         uint64_t policy = pvr_g5p_fetch_sampler_word();
         memcpy(dst, &policy, sizeof(policy));
         break;
      }
      case PCO_G5P_UPLOAD_PUSH:
         if (u->offset > PVR_MAX_PUSH_CONSTANTS_SIZE || bytes > PVR_MAX_PUSH_CONSTANTS_SIZE - u->offset)
            return VTEX_REFUSE(cmd, "push constant interval outside the device limit");
         memcpy(dst, push->data + u->offset, bytes);
         break;
      case PCO_G5P_UPLOAD_UBO: case PCO_G5P_UPLOAD_UBO_ADDR: case PCO_G5P_UPLOAD_SSBO_ADDR:
         result = pvr_arch_g5p_upload_buffer(cmd, u->set, u->binding, u->element, &addr, &size);
         if (result != VK_SUCCESS) return result;
         if (u->kind == PCO_G5P_UPLOAD_UBO) {
            if (u->offset > size || bytes > size - u->offset)
               return VTEX_REFUSE(cmd, "UBO interval exceeds the bound range");
            direct[i] = addr + u->offset;
         } else {
            if (u->dwords != 2 || (u->kind == PCO_G5P_UPLOAD_UBO_ADDR && u->offset >= size))
               return VTEX_REFUSE(cmd, "invalid buffer pointer upload");
            uint64_t va = addr + (u->kind == PCO_G5P_UPLOAD_UBO_ADDR ? u->offset : 0);
            memcpy(dst, &va, sizeof(va));
         }
         break;
      case PCO_G5P_UPLOAD_XFB: {
         if (u->dwords != PCO_G5P_XFB_DWORDS || out->xfb_block)
            return VTEX_REFUSE(cmd, "invalid transform feedback block");
         uint32_t zero[PCO_G5P_XFB_DWORDS] = {0};
         struct pvr_suballoc_bo *bo;
         result = pvr_arch_cmd_buffer_upload_general(cmd, zero, sizeof(zero), &bo);
         if (result != VK_SUCCESS) return result;
         out->xfb_block = pvr_bo_suballoc_get_map_addr(bo);
         out->xfb_block_va = direct[i] = bo->dev_addr.addr;
         break;
      }
      default:
         return VTEX_REFUSE(cmd, "unknown source kind %u", u->kind);
      }
   }
   struct pvr_suballoc_bo *bo;
   VkResult result = pvr_arch_cmd_buffer_upload_general(cmd, words, 4 * up->common_dwords, &bo);
   if (result != VK_SUCCESS) return result;
   for (unsigned i = 0; i < p->dmas; i++) {
      const struct vtex_dma *d = &p->dma[i];
      uint64_t src = d->upload < 0 ? bo->dev_addr.addr + 4 * d->common : direct[d->upload];
      data[2*i] = src;
      data[2*i+1] = src >> 32;
      data[2*p->dmas + 2*i] = vtex_control(d->common, d->dwords, i == p->dmas - 1);
   }
   struct pvr_pds_upload pds;
   result = pvr_arch_cmd_buffer_upload_pds(cmd, data, 4*p->dmas, 16, NULL, 0, 0, 16, &pds);
   if (result != VK_SUCCESS) return result;
   pvr_csb_pack(&out->pds_state[0], VDMCTRL_PDS_STATE0, s0) {
      s0.usc_common_size = MAX2(4, 2 * DIV_ROUND_UP(up->common_dwords, 16));
      s0.pds_data_size = p->dmas;
   }
   pvr_csb_pack(&out->pds_state[1], VDMCTRL_PDS_STATE1, s1) { s1.pds_data_addr = PVR_DEV_ADDR(pds.data_offset); }
   pvr_csb_pack(&out->pds_state[2], VDMCTRL_PDS_STATE2, s2) { s2.pds_code_addr = PVR_DEV_ADDR(up->loader.code_offset); }
   if (getenv("PVR_G5P_WS_LOG"))
      fprintf(stderr, "g5p vtex uploads: %u sources -> %u DMAs, common=%u, CPU staging=%u bytes\n",
              p->count, p->dmas, up->common_dwords, 4*up->common_dwords);
   return VK_SUCCESS;
}
