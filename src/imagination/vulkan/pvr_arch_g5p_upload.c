/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P): the vertex program's common-store uploads
 * (pvr_g5p_upload.h). Compiled only with -Dimagination-g5p-experimental=true.
 *
 * Ported from our A9 driver's round-19 vertex loader (loader_program() and
 * loader_data() of its recorder, PDS_STATE_UPDATE before the vertex data),
 * on pvr_*'s descriptor sets, push constants and command-buffer uploads.
 */

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "vk_alloc.h"
#include "vk_enum_to_str.h"
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
#include "pvr_g5p_upload.h"

/* A refusal, by name, on stderr too (vk_errorf() is silent in release builds) */
#define G5P_UP_REFUSE(obj, ...)                                                                                    \
   (fprintf(stderr, "G5P refuses: " __VA_ARGS__), fprintf(stderr, "\n"),                                          \
    vk_errorf((obj), VK_ERROR_FEATURE_NOT_PRESENT, "G5P: " __VA_ARGS__))

static bool
g5p_up_log(void)
{
   static int on = -1;
   if (on < 0)
      on = getenv("PVR_G5P_WS_LOG") != NULL;
   return on;
}

static void
g5p_up_dump(const char *what, uint64_t addr, const uint32_t *w, unsigned n)
{
   if (!g5p_up_log())
      return;
   fprintf(stderr, "g5p draw: %-16s @%#" PRIx64 ":", what, addr);
   for (unsigned i = 0; i < n; i++)
      fprintf(stderr, " %08x", w[i]);
   fprintf(stderr, "\n");
}

/* The loader: DOUTD c(2i):c(2i+1) -> common store with the control word in
 * c(2n+2i), the last one ends, then HALT (the fragment loader's shape) */
static unsigned
g5p_up_loader_program(uint32_t *dest, unsigned n)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-vs-loader");
   for (unsigned i = 0; i < n; i++)
      pdsc_dout(p, pdsc_ref_const64(2 * i), pdsc_ref_const32(2 * n + 2 * i), pdsc_ref_null(),
                .dstdout = PDSC_DSTDOUT_D, .end = i == n - 1);
   pdsc_halt(p);
   const unsigned count = util_dynarray_num_elements(&p->instrs, pdsc_instr);
   for (unsigned i = 0; i < count; i++)
      dest[i] = pdsc_instr_pack(p, i, util_dynarray_element(&p->instrs, pdsc_instr, i)->op);
   ralloc_free(p);
   return count;
}

/* DOUTD control: `dwords` from the source address into common-store dword `common` */
static uint32_t
g5p_up_doutd_control(unsigned common, unsigned dwords, bool last)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-doutd");
   struct pdsc_doutd_src1 s = { .last_issue = last, .store_dest = PDSC_STORE_DEST_COMMON, .ao = common,
                                .bsize = dwords };
   const uint32_t v = pdsc_doutd_src1_pack(p, &s);
   ralloc_free(p);
   return v;
}

static const char *
g5p_up_kind_str(unsigned kind)
{
   switch (kind) {
   case PCO_G5P_UPLOAD_SSBO_ADDR: return "a storage buffer address";
   case PCO_G5P_UPLOAD_IMAGE: return "an image state";
   case PCO_G5P_UPLOAD_SAMPLER: return "a sampler";
   case PCO_G5P_UPLOAD_PUSH: return "push constants";
   case PCO_G5P_UPLOAD_UBO: return "a uniform buffer";
   case PCO_G5P_UPLOAD_UBO_ADDR: return "a uniform buffer address";
   case PCO_G5P_UPLOAD_IMAGE_SIZE: return "an image size";
   case PCO_G5P_UPLOAD_XFB: return "the transform feedback block";
   case PCO_G5P_UPLOAD_BLEND_CONSTANTS: return "the blend constants";
   default: return "an unknown upload";
   }
}

VkResult
PVR_PER_ARCH(g5p_vs_uploads_create)(struct pvr_device *device, const struct pco_g5p_program *vs,
                                    struct pvr_g5p_stage_uploads **out)
{
   *out = NULL;
   if (!vs->num_uploads)
      return VK_SUCCESS;
   if (vs->num_uploads > PVR_G5P_MAX_STAGE_UPLOADS)
      return G5P_UP_REFUSE(device, "%u vertex-stage uploads (at most %u)", vs->num_uploads,
                           PVR_G5P_MAX_STAGE_UPLOADS);

   for (unsigned i = 0; i < vs->num_uploads; i++) {
      const struct pco_g5p_upload *u = &vs->uploads[i];
      switch (u->kind) {
      case PCO_G5P_UPLOAD_PUSH:
      case PCO_G5P_UPLOAD_UBO:
      case PCO_G5P_UPLOAD_UBO_ADDR:
      case PCO_G5P_UPLOAD_SSBO_ADDR:
      case PCO_G5P_UPLOAD_XFB:
         break;
      default:
         /* textures and samplers in a vertex program: lane b's descriptor
          * words exist, the vertex stage does not take them yet */
         return G5P_UP_REFUSE(device, "vertex-stage upload %u: %s", i, g5p_up_kind_str(u->kind));
      }
      if (!u->dwords || u->dwords > 0xfff || u->set >= PVR_MAX_DESCRIPTOR_SETS || u->binding > UINT8_MAX ||
          u->offset > UINT16_MAX || u->common > UINT16_MAX)
         return G5P_UP_REFUSE(device, "vertex-stage upload %u: set %u binding %u offset %u, %u dwords at common %u",
                              i, u->set, u->binding, u->offset, u->dwords, u->common);
   }

   struct pvr_g5p_stage_uploads *up = vk_zalloc(&device->vk.alloc, sizeof(*up), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!up)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   up->count = vs->num_uploads;
   up->common_dwords = vs->common_dwords;
   for (unsigned i = 0; i < up->count; i++) {
      const struct pco_g5p_upload *u = &vs->uploads[i];
      up->u[i] = (struct pvr_g5p_stage_upload){ .kind = u->kind, .set = u->set, .binding = u->binding,
                                                .offset = u->offset, .dwords = u->dwords, .common = u->common };
   }

   uint32_t code[4 * PVR_G5P_MAX_STAGE_UPLOADS + 4];
   const unsigned n = g5p_up_loader_program(code, up->count);
   VkResult result = pvr_gpu_upload_pds(device, NULL, 0, 0, code, n, 16, 64, &up->loader);
   if (result != VK_SUCCESS) {
      vk_free(&device->vk.alloc, up);
      return result;
   }
   *out = up;
   return VK_SUCCESS;
}

void
PVR_PER_ARCH(g5p_vs_uploads_destroy)(struct pvr_device *device, struct pvr_g5p_stage_uploads *up)
{
   if (!up)
      return;
   pvr_bo_suballoc_free(up->loader.pvr_bo);
   vk_free(&device->vk.alloc, up);
}

/* The buffer a set's binding names (element 0): its address, the dynamic
 * offset included, and its range from there */
static VkResult
g5p_up_buffer(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g5p_stage_upload *u, uint64_t *addr,
              uint64_t *size)
{
   const struct pvr_descriptor_state *ds = &cmd_buffer->state.gfx_desc_state;
   const struct pvr_descriptor_set *set = ds->sets[u->set];

   /* a set without a layout: a pushed set not instantiated (never dereferenced) */
   if (!set || !set->layout || u->binding >= set->layout->binding_count)
      return G5P_UP_REFUSE(cmd_buffer, "upload: set %u binding %u is not bound", u->set, u->binding);
   const struct pvr_descriptor_set_layout_binding *lb = &set->layout->bindings[u->binding];
   struct pvr_buffer_descriptor d;
   switch (lb->type) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      memcpy(&d, (const uint8_t *)set->mapping + lb->offset, sizeof(d));
      d.offset = 0;
      break;
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
      d = set->dynamic_buffers[lb->dynamic_buffer_idx];
      break;
   default:
      return G5P_UP_REFUSE(cmd_buffer, "upload: set %u binding %u is a %s, not a buffer", u->set,
                           u->binding, vk_DescriptorType_to_str(lb->type));
   }
   if (!d.addr)
      return G5P_UP_REFUSE(cmd_buffer, "upload: set %u binding %u holds no buffer", u->set, u->binding);
   /* a dynamic buffer's range starts at its dynamic offset (VkDescriptorBufferInfo.range from there) */
   *addr = d.addr + d.offset;
   *size = d.size;
   return VK_SUCCESS;
}

/* The same lookup for the fragment stage's uploads (pvr_arch_g5p_gfx.c fs_uploads()) */
VkResult
PVR_PER_ARCH(g5p_upload_buffer)(struct pvr_cmd_buffer *cmd_buffer, unsigned set, unsigned binding, uint64_t *addr,
                                uint64_t *size)
{
   const struct pvr_g5p_stage_upload u = { .set = set, .binding = binding };
   if (set >= PVR_MAX_DESCRIPTOR_SETS)
      return G5P_UP_REFUSE(cmd_buffer, "upload: set %u", set);
   return g5p_up_buffer(cmd_buffer, &u, addr, size);
}

VkResult
PVR_PER_ARCH(g5p_vs_uploads_draw)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g5p_stage_uploads *up,
                                  struct pvr_g5p_vs_upload_draw *out)
{
   const struct pvr_push_constants *push = &cmd_buffer->state.push_consts[PVR_STAGE_ALLOCATION_VERTEX_GEOMETRY];
   const unsigned n = up->count;
   uint64_t src[PVR_G5P_MAX_STAGE_UPLOADS];
   uint32_t data[4 * PVR_G5P_MAX_STAGE_UPLOADS];
   struct pvr_suballoc_bo *bo;
   VkResult result;

   memset(out, 0, sizeof(*out));
   for (unsigned i = 0; i < n; i++) {
      const struct pvr_g5p_stage_upload *u = &up->u[i];
      const unsigned bytes = 4 * u->dwords;
      uint64_t addr, size;

      switch (u->kind) {
      case PCO_G5P_UPLOAD_PUSH:
         /* the bytes the program reads, as they are now (a copy per draw) */
         if (u->offset + bytes > PVR_MAX_PUSH_CONSTANTS_SIZE)
            return G5P_UP_REFUSE(cmd_buffer, "vertex push constants [%u, %u) past %u bytes", u->offset,
                                 u->offset + bytes, PVR_MAX_PUSH_CONSTANTS_SIZE);
         result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, push->data + u->offset, bytes, &bo);
         if (result != VK_SUCCESS)
            return result;
         src[i] = bo->dev_addr.addr;
         g5p_up_dump("vs upload push", src[i], (const uint32_t *)(push->data + u->offset), u->dwords);
         break;
      case PCO_G5P_UPLOAD_UBO:
         /* the buffer itself: the DOUTD reads it when the job runs */
         result = g5p_up_buffer(cmd_buffer, u, &addr, &size);
         if (result != VK_SUCCESS)
            return result;
         if (u->offset + bytes > size)
            return G5P_UP_REFUSE(cmd_buffer, "vertex uniform buffer at set %u binding %u: [%u, %u) past its %" PRIu64
                                 " bytes", u->set, u->binding, u->offset, u->offset + bytes, size);
         src[i] = addr + u->offset;
         break;
      case PCO_G5P_UPLOAD_UBO_ADDR:
      case PCO_G5P_UPLOAD_SSBO_ADDR: {
         /* the 64-bit address a register-indexed or storage access adds to */
         result = g5p_up_buffer(cmd_buffer, u, &addr, &size);
         if (result != VK_SUCCESS)
            return result;
         if (u->dwords != 2 || (u->kind == PCO_G5P_UPLOAD_UBO_ADDR && u->offset >= size))
            return G5P_UP_REFUSE(cmd_buffer, "vertex %s: %u dwords at offset %u of %" PRIu64 " bytes",
                                 g5p_up_kind_str(u->kind), u->dwords, u->offset, size);
         const uint64_t va = addr + (u->kind == PCO_G5P_UPLOAD_UBO_ADDR ? u->offset : 0);
         result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, &va, sizeof(va), &bo);
         if (result != VK_SUCCESS)
            return result;
         src[i] = bo->dev_addr.addr;
         break;
      }
      case PCO_G5P_UPLOAD_XFB: {
         /* zeroed here; the transform feedback resolve fills it before the
          * job runs */
         uint32_t zero[PCO_G5P_XFB_DWORDS];
         memset(zero, 0, sizeof(zero));
         if (u->dwords != PCO_G5P_XFB_DWORDS || out->xfb_block)
            return G5P_UP_REFUSE(cmd_buffer, "a transform feedback block of %u dwords", u->dwords);
         result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, zero, sizeof(zero), &bo);
         if (result != VK_SUCCESS)
            return result;
         src[i] = bo->dev_addr.addr;
         out->xfb_block = pvr_bo_suballoc_get_map_addr(bo);
         out->xfb_block_va = bo->dev_addr.addr;
         break;
      }
      default:
         return G5P_UP_REFUSE(cmd_buffer, "vertex upload %u: %s", i, g5p_up_kind_str(u->kind));
      }
   }

   for (unsigned i = 0; i < n; i++) {
      data[2 * i] = (uint32_t)src[i];
      data[2 * i + 1] = (uint32_t)(src[i] >> 32);
      data[2 * n + 2 * i] = g5p_up_doutd_control(up->u[i].common, up->u[i].dwords, i == n - 1);
      data[2 * n + 2 * i + 1] = 0;
   }
   struct pvr_pds_upload pds;
   result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, 4 * n, 16, NULL, 0, 0, 16, &pds);
   if (result != VK_SUCCESS)
      return result;
   g5p_up_dump("vs upload data", pds.data_offset, data, 4 * n);

   /* PDS_STATE_UPDATE for the VDM: one 16-byte data unit per DOUTD; the
    * common store sized as AGXMetalA9's VertexProgramVariantPDSState does,
    * 2 * ceil(dwords / 16) units, at least 4 (the captured 18-dword loader) */
   pvr_csb_pack (&out->pds_state[0], VDMCTRL_PDS_STATE0, s0) {
      s0.usc_common_size = MAX2(4, 2 * DIV_ROUND_UP(up->common_dwords, 16));
      s0.pds_data_size = n;
   }
   pvr_csb_pack (&out->pds_state[1], VDMCTRL_PDS_STATE1, s1) {
      s1.pds_data_addr = PVR_DEV_ADDR(pds.data_offset);
   }
   pvr_csb_pack (&out->pds_state[2], VDMCTRL_PDS_STATE2, s2) {
      s2.pds_code_addr = PVR_DEV_ADDR(up->loader.code_offset);
   }
   g5p_up_dump("vs loader code", up->loader.code_offset,
               (const uint32_t *)pvr_bo_suballoc_get_map_addr(up->loader.pvr_bo), up->loader.code_size);
   return VK_SUCCESS;
}
