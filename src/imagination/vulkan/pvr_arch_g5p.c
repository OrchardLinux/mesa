/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P, has_apple_g5p): the device-level state of the G5P
 * paths (pvr_g5p.h), its device-level programs, and its render jobs: the
 * per-render data of the background clear and the end of tile
 * (g5p_job_init) and the TYPED_V3 pass the A9 kernel takes
 * (g5p_frag_stream). Compiled only with -Dimagination-g5p-experimental=true.
 *
 * Route (b) (docs: hardware question 1, 2026-11-03): the A9 renders and
 * retires jobs whose context store/resume programs are a bare PDS HALT,
 * with no prologue and no epilogue, as G9P runs the A10. So a G5P render is
 * the stream pvr_* records, the device programs below, and the pass.
 */

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "vk_alloc.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_log.h"

#include "util/ralloc.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"

#include "g5p_mrt_programs.h"
#include "g5p_color_source.h"
#include "g5p_pbe.h"
#include "util/half_float.h"
#include "pco_g5p_general.h"
#include "g5p_usc.h"
#include "pdsc.h"
#include "pdsc_internal.h"
#include "pdsc_isa_pack.h"
#include "pdsc_ops.h"
#include "pvr_bo.h"
#include "pvr_cmd_buffer.h"
#include "pvr_csb.h"
#include "pvr_device.h"
#include "pvr_g5p.h"
#include "pvr_g5p_dgl.h"
#include "pvr_g5p_ds.h"
#include "pvr_g5p_formats.h"
#include "pvr_g5p_msaa.h"
#include "pvr_g5p_tex_encode.h"
#include "pvr_g5p_tex.h"
#include "pvr_g5p_mrt.h"
#include "pvr_g5p_perf.h"
#include "pvr_g5p_vdm.h"
#include "pvr_hw_pass.h"
#include "pvr_image.h"
#include "pvr_job_render.h"
#include "pvr_query.h"
#include "pvr_rogue_pds_encode.h"
#include "winsys/apple_g5p/apple_g5p_drm.h"

static const unsigned g5p_mrt_variant_targets[PVR_G5P_MRT_VARIANTS] = { 1, 2, 4 };

/* The DOUTU temps of every G5P program (G5P_USC_TEMPS of the A9 ICD) */
#define G5P_TEMPS 8
/* A kick: 16 bytes of data, the code at +0x20 (the program registers name the data) */
#define G5P_KICK_CODE 0x20

/* The template-derived register words of a TYPED_V3 pass (the kernel
 * contract's layout, a9_contracts.h; the context words are the layout G9P
 * uses on the A10). Named here, values as the A9 ICD's passes carry them on
 * the device. */
#define G5P_CTX_STORE_TA_SIZES0 0x24001002u
#define G5P_CTX_STORE_TA_SIZES1 0x00030005u
#define G5P_CTX_STORE_3D_SIZES0 0x21001002u
#define G5P_CTX_STORE_3D_SIZES1 0x00030021u
#define G5P_CTX_STORE_SIZES4 0x00004820u
#define G5P_CTX_RESUME_SIZES 0x20000001u
#define G5P_TA_CTX_SIZES 0x24000081u
#define G5P_RESTORE0 0x01800000u
/* The TA scratch the kernel's records point into, and its pointer offsets */
#define G5P_SCRATCH_BYTES 0x8000u
#define G5P_NULL_RT_SIDE 16u  /* the null colour target: 16x16 R8, 16-byte rows (g5p_job_init()) */
#define G5P_NULL_RT_PITCH 16u
/* the same buffer as a 16x16 four-byte target, 64-byte rows: the padding 4th target of a 3-target pass */
#define G5P_NULL_RT_PITCH4 64u
static const uint32_t g5p_scratch_ptr_off[6] = { 0x20, 0x24, 0x28, 0x2c, 0x6240, 0x6040 };
/* The default scissor and depth-bias tables of a render without tables of its
 * own. Its draws index entry 0 only (scindex/dbindex 0, scissor and bias
 * disabled): a draw that clips or biases makes the render's own table
 * (upstream's per-render tables, pvr_arch_g5p_gfx.c). One page each, not
 * the 65536-entry V1 pools (512 + 768 KiB of every device's arena). */
#define G5P_SCISSOR_POOL_BYTES 0x4000u
#define G5P_DBIAS_POOL_BYTES 0x3000u

/* ---- pdsc programs ---- */

static unsigned
g5p_pds_pack(pdsc_program *p, uint32_t *dest)
{
   const unsigned n = util_dynarray_num_elements(&p->instrs, pdsc_instr);
   for (unsigned i = 0; i < n; i++)
      dest[i] = pdsc_instr_pack(p, i, util_dynarray_element(&p->instrs, pdsc_instr, i)->op);
   return n;
}

/* DOUTU from data dwords 0-1, then HALT: the code of a kick */
static unsigned
g5p_pds_kick_code(uint32_t *code)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-kick");
   pdsc_dout(p, pdsc_ref_const64(0), pdsc_ref_null(), pdsc_ref_null(), .dstdout = PDSC_DSTDOUT_U,
             .end = true);
   pdsc_halt(p);
   const unsigned n = g5p_pds_pack(p, code);
   ralloc_free(p);
   return n;
}

/* DOUTD from data dwords 0-1 with the control in dword 2, then (eot) DOUTU
 * from dwords 4-5; HALT (pdsc's const indices count dwords) */
static unsigned
g5p_pds_dma_code(uint32_t *code, bool eot)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, eot ? "g5p-eot" : "g5p-clear-dma");
   pdsc_dout(p, pdsc_ref_const64(0), pdsc_ref_const32(2), pdsc_ref_null(), .dstdout = PDSC_DSTDOUT_D,
             .end = !eot);
   if (eot) /* pdsc's const64 index counts dwords: dwords 4-5 */
      pdsc_dout(p, pdsc_ref_const64(4), pdsc_ref_null(), pdsc_ref_null(), .dstdout = PDSC_DSTDOUT_U,
                .end = true);
   pdsc_halt(p);
   const unsigned n = g5p_pds_pack(p, code);
   ralloc_free(p);
   return n;
}

/* n DOUTDs, source i from data dwords 2i..2i+1 and its control in dword
 * 2n + 2i, the last one ending; HALT (the A9 ICD's uniform loader) */
static unsigned
g5p_pds_loader_code(uint32_t *code, unsigned n)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-loader");
   for (unsigned i = 0; i < n; i++)
      pdsc_dout(p, pdsc_ref_const64(2 * i), pdsc_ref_const32(2 * n + 2 * i), pdsc_ref_null(),
                .dstdout = PDSC_DSTDOUT_D, .end = i == n - 1);
   pdsc_halt(p);
   const unsigned c = g5p_pds_pack(p, code);
   ralloc_free(p);
   return c;
}

static uint64_t
g5p_doutu_src0(uint32_t exe_off, unsigned temps)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-doutu");
   struct pdsc_doutu_src0 s = { .exe_off = exe_off, .temps = temps };
   const uint64_t v = pdsc_doutu_src0_pack(p, &s);
   ralloc_free(p);
   return v;
}

static uint32_t
g5p_doutd_control_at(unsigned common, unsigned dwords, bool last)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-doutd");
   struct pdsc_doutd_src1 s = { .last_issue = last, .store_dest = PDSC_STORE_DEST_COMMON, .ao = common,
                                .bsize = dwords };
   const uint32_t v = pdsc_doutd_src1_pack(p, &s);
   ralloc_free(p);
   return v;
}

static uint32_t
g5p_doutd_control(unsigned dwords)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-doutd");
   struct pdsc_doutd_src1 s = { .last_issue = true, .store_dest = PDSC_STORE_DEST_COMMON, .ao = 0,
                                .bsize = dwords };
   const uint32_t v = pdsc_doutd_src1_pack(p, &s);
   ralloc_free(p);
   return v;
}

/* USC offsets are from the USC heap base (the A9 kernel's usc_base == pds_base) */
static uint32_t
g5p_usc_off(const struct pvr_device *device, const struct pvr_suballoc_bo *bo)
{
   return (uint32_t)(bo->dev_addr.addr - device->heaps.usc_heap->base_addr.addr);
}

/* A kick of a USC program, its code at data + 0x20 as the registers expect;
 * a fragment-side program at `samples` (4: sample rate, pvr_g5p_msaa_doutu()) */
static VkResult
g5p_upload_kick_temps(struct pvr_device *device, const struct pvr_suballoc_bo *usc, unsigned temps,
                      uint32_t samples, struct pvr_pds_upload *out)
{
   const uint64_t doutu = pvr_g5p_msaa_doutu(g5p_doutu_src0(g5p_usc_off(device, usc), temps), samples);
   const uint32_t data[4] = { (uint32_t)doutu, (uint32_t)(doutu >> 32), 0, 0 };
   uint32_t code[8];
   const unsigned n = g5p_pds_kick_code(code);
   VkResult result = pvr_gpu_upload_pds(device, data, ARRAY_SIZE(data), 16, code, n, G5P_KICK_CODE,
                                        64, out);
   if (result == VK_SUCCESS && out->code_offset != out->data_offset + G5P_KICK_CODE)
      return vk_errorf(device, VK_ERROR_INITIALIZATION_FAILED,
                       "G5P: a kick's code is at data + %#x, not + %#x", out->code_offset - out->data_offset,
                       G5P_KICK_CODE);
   return result;
}

static VkResult
g5p_upload_kick(struct pvr_device *device, const struct pvr_suballoc_bo *usc, uint32_t samples,
                struct pvr_pds_upload *out)
{
   return g5p_upload_kick_temps(device, usc, G5P_TEMPS, samples, out);
}

static void
g5p_free_programs(struct pvr_device *device, struct pvr_g5p_programs *p)
{
   for (unsigned i = 0; i < PVR_G5P_MRT_VARIANTS; i++) {
      pvr_bo_suballoc_free(p->eot_usc[i]);
      pvr_bo_suballoc_free(p->clear_usc[i]);
      pvr_bo_suballoc_free(p->clear_kick[i].pvr_bo);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(p->raw_load); i++) {
      if (p->raw_load[i]) {
         g5p_free_programs(device, p->raw_load[i]);
         vk_free(&device->vk.alloc, p->raw_load[i]);
      }
   }
   for (unsigned i = 0; i < ARRAY_SIZE(p->raw_clear_usc); i++) {
      pvr_bo_suballoc_free(p->raw_clear_usc[i]);
      pvr_bo_suballoc_free(p->raw_clear_kick[i].pvr_bo);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(p->raw_eot_usc); i++)
      pvr_bo_suballoc_free(p->raw_eot_usc[i]);
   pvr_bo_suballoc_free(p->narrow_eot_usc);
   pvr_bo_suballoc_free(p->halt.pvr_bo);
   pvr_bo_suballoc_free(p->clear_dma.pvr_bo);
   pvr_bo_suballoc_free(p->eot.pvr_bo);
   pvr_bo_suballoc_free(p->ta_kick.pvr_bo);
   pvr_bo_suballoc_free(p->ta_usc);
   pvr_bo_suballoc_free(p->scratch);
   pvr_bo_suballoc_free(p->null_rt);
   pvr_bo_suballoc_free(p->scissor_pool);
   pvr_bo_suballoc_free(p->dbias_pool);
   pvr_bo_suballoc_free(p->eot_usc_ms);
   pvr_bo_suballoc_free(p->clear_kick_ms.pvr_bo);
   pvr_bo_suballoc_free(p->load_usc);
   pvr_bo_suballoc_free(p->load_kick.pvr_bo);
   pvr_bo_suballoc_free(p->load_dma.pvr_bo);
}

/* The load background's programs (a loaded colour attachment): the program
 * the backend compiles from NIR, its kick, its uploads' DMA code. A program
 * the backend refuses is not an error here: a LOAD render is refused by
 * name then. */
static VkResult
g5p_upload_load(struct pvr_device *device, struct pvr_g5p_programs *p, unsigned raw)
{
   struct pco_g5p_program prog;
   VkResult result = VK_SUCCESS;

   if (!(raw == 10 || raw == 11 ? pvr_g5p_compile_load_scalar8(&prog, raw == 11) : (raw == 8 || raw == 9) ? pvr_g5p_compile_load_packed_sint(&prog, raw == 8 ? 8 : 16) : raw == 7 ? pvr_g5p_compile_load_rgb10a2ui(&prog) : raw == 6 ? pvr_g5p_compile_load_rgba8ui(&prog) : raw == 5 ? pvr_g5p_compile_load_rgba16ui(&prog) : raw ? pvr_g5p_compile_load_raw(&prog, raw == 2) : pvr_g5p_compile_load(&prog))) {
      snprintf(p->load_refusal, sizeof(p->load_refusal), "%s", prog.refusal);
      free(prog.code);
      return VK_SUCCESS;
   }
   if (prog.num_uploads < 1 || prog.num_uploads > ARRAY_SIZE(p->load_upload)) {
      snprintf(p->load_refusal, sizeof(p->load_refusal), "a load program with %u uploads", prog.num_uploads);
      free(prog.code);
      return VK_SUCCESS;
   }
   for (unsigned i = 0; i < prog.num_uploads; i++) {
      const struct pco_g5p_upload *u = &prog.uploads[i];
      if ((u->kind != PCO_G5P_UPLOAD_IMAGE && u->kind != PCO_G5P_UPLOAD_SAMPLER &&
           u->kind != PCO_G5P_UPLOAD_IMAGE_SIZE) || u->set || u->binding) {
         snprintf(p->load_refusal, sizeof(p->load_refusal), "a load program upload of kind %u at %u.%u",
                  u->kind, u->set, u->binding);
         free(prog.code);
         return VK_SUCCESS;
      }
      p->load_upload[i].kind = u->kind;
      p->load_upload[i].dwords = u->dwords;
      p->load_upload[i].common = u->common;
   }
   p->load_uploads = prog.num_uploads;
   p->load_temps = prog.temps;
   p->load_common_dwords = prog.common_dwords;

   result = pvr_gpu_upload_usc(device, prog.code, prog.size, 64, &p->load_usc);
   if (result == VK_SUCCESS && getenv("PVR_G5P_WS_LOG")) {
      fprintf(stderr, "g5p load program: %zu bytes, temps %u, common %u dwords, %u uploads:", prog.size,
              prog.temps, prog.common_dwords, prog.num_uploads);
      for (unsigned i = 0; i < prog.size; i++)
         fprintf(stderr, "%s%02x", i % 4 ? "" : " ", prog.code[i]);
      fprintf(stderr, "\n");
   }
   free(prog.code);
   if (result != VK_SUCCESS)
      return result;
   result = g5p_upload_kick_temps(device, p->load_usc, p->load_temps, 1, &p->load_kick);
   if (result != VK_SUCCESS)
      return result;
   uint32_t code[2 * ARRAY_SIZE(p->load_upload) + 2];
   const unsigned n = g5p_pds_loader_code(code, p->load_uploads);
   return pvr_gpu_upload_pds(device, NULL, 0, 0, code, n, 16, 64, &p->load_dma);
}

static VkResult
g5p_upload_zeroed(struct pvr_device *device, uint32_t size, struct pvr_suballoc_bo **out)
{
   void *zeros = calloc(1, size);
   if (!zeros)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   VkResult result = pvr_gpu_upload(device, device->heaps.general_heap, zeros, size, 64, out);
   free(zeros);
   return result;
}

/* The programs of pvr_g5p_programs, uploaded into the device's PDS and USC
 * heaps (64-byte aligned USC code, as the end-of-tile program's branch to the
 * next 64-byte boundary assumes). */
/* The raw clear program's variant: total output words 2, 4 or 8 */
static unsigned
g5p_raw_variant(unsigned words)
{
   return words == 1 ? 3 : words == 8 ? 2 : words == 4;
}

/* Called only for an integer pass; qualification uses FORMATS=probe.
 * words: the pass's total output words; targets: 1, or 2 raw32 targets. */
static VkResult
g5p_raw_programs_unlocked(struct pvr_device *device, unsigned raw, unsigned words, unsigned targets, unsigned source_bits)
{
   struct pvr_g5p_programs *p = &device->g5p->programs;
   VkResult result;
   const unsigned variant = g5p_raw_variant(words);
   if (!p->raw_clear_usc[variant]) {
      uint8_t code[6 * G5P_USC_MAX];
      size_t size = pvr_g5p_dgl_clear_words(code, sizeof(code), words);
      if (!size)
         return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
      result = pvr_gpu_upload_usc(device, code, size, 64, &p->raw_clear_usc[variant]);
      if (result != VK_SUCCESS)
         return result;
   }
   if (!p->raw_clear_kick[variant].pvr_bo) {
      result = g5p_upload_kick(device, p->raw_clear_usc[variant], 1, &p->raw_clear_kick[variant]);
      if (result != VK_SUCCESS)
         return result;
   }
   /* witness-r16: R16UI/I EOT bit4 is clear; R32UI sets it. Cache
    * by source layout so a preceding raw32 pass cannot poison this choice. */
   const bool narrow=source_bits<32;
   struct pvr_suballoc_bo **eot = narrow ? &p->narrow_eot_usc : &p->raw_eot_usc[targets - 1];
   if (!*eot) {
      uint8_t code[G5P_MRT_EOT_USC_MAX];
      struct g5p_color_source_abi source;
      if (!g5p_color_source_make(source_bits,1,G5P_COLOR_SOURCE_UINT,&source))
         return vk_error(device,VK_ERROR_INITIALIZATION_FAILED);
      const unsigned size = narrow ? g5p_color_source_eot(code,&source)
                                    : g5p_mrt_eot_usc_mode(code,targets,true);
      result = pvr_gpu_upload_usc(device, code, size, 64, eot);
      if (result != VK_SUCCESS)
         return result;
   }
   if ((raw <= 2 || (raw >= 5 && raw <= 11)) && !p->raw_load[raw - 1]) {
      p->raw_load[raw - 1] = vk_zalloc(&device->vk.alloc, sizeof(*p), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
      if (!p->raw_load[raw - 1])
         return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
      result = g5p_upload_load(device, p->raw_load[raw - 1], raw);
      if (result != VK_SUCCESS)
         return result;
   }
   return VK_SUCCESS;
}

static VkResult
g5p_raw_programs(struct pvr_device *device, unsigned raw, unsigned words, unsigned targets, unsigned source_bits)
{
   simple_mtx_lock(&device->g5p->clear_lock);
   VkResult result = g5p_raw_programs_unlocked(device, raw, words, targets, source_bits);
   simple_mtx_unlock(&device->g5p->clear_lock);
   return result;
}

static VkResult
g5p_upload_programs(struct pvr_device *device, struct pvr_g5p_programs *p)
{
   const uint32_t halt = pvr_pds_inst_encode_halt(0);
   const uint32_t zero_data[4] = { 0 };
   VkResult result = pvr_gpu_upload_pds(device, zero_data, ARRAY_SIZE(zero_data), 16, &halt, 1, 16, 64,
                                        &p->halt);
   if (result != VK_SUCCESS)
      return result;
   /* a draw's stream-out program is this HALT too: its 16 data bytes, the code right after */
   if (p->halt.code_offset != p->halt.data_offset + 16)
      return vk_errorf(device, VK_ERROR_INITIALIZATION_FAILED, "G5P: the HALT's code is at data + %#x, not + 0x10",
                       p->halt.code_offset - p->halt.data_offset);

   for (unsigned i = 0; i < PVR_G5P_MRT_VARIANTS; i++) {
      const unsigned n = g5p_mrt_variant_targets[i];
      uint8_t code[MAX2(G5P_MRT_CLEAR_USC_MAX, G5P_MRT_EOT_USC_MAX)];
      unsigned size = g5p_mrt_clear_usc(code, n);

      result = pvr_gpu_upload_usc(device, code, size, 64, &p->clear_usc[i]);
      if (result != VK_SUCCESS)
         return result;
      size = g5p_mrt_eot_usc(code, n);
      result = pvr_gpu_upload_usc(device, code, size, 64, &p->eot_usc[i]);
      if (result != VK_SUCCESS)
         return result;
      result = g5p_upload_kick(device, p->clear_usc[i], 1, &p->clear_kick[i]);
      if (result != VK_SUCCESS)
         return result;
   }
   /* 4x (one target): the multisample store's end of tile, and the clear's
    * kick at sample rate */
   {
      uint8_t code[G5P_MRT_EOT_USC_MAX];
      const unsigned size = pvr_g5p_msaa_eot_usc(code);
      result = pvr_gpu_upload_usc(device, code, size, 64, &p->eot_usc_ms);
      if (result != VK_SUCCESS)
         return result;
      result = g5p_upload_kick(device, p->clear_usc[0], 4, &p->clear_kick_ms);
      if (result != VK_SUCCESS)
         return result;
   }

   uint32_t pds[8];
   unsigned n = g5p_pds_dma_code(pds, false);
   result = pvr_gpu_upload_pds(device, NULL, 0, 0, pds, n, 16, 64, &p->clear_dma);
   if (result != VK_SUCCESS)
      return result;
   n = g5p_pds_dma_code(pds, true);
   result = pvr_gpu_upload_pds(device, NULL, 0, 0, pds, n, 16, 64, &p->eot);
   if (result != VK_SUCCESS)
      return result;

   /* the TA's end-of-stream program: branch to the next 64-byte block,
    * 0xc000 padding (an end right after the branch hangs, G9P's notes), end */
   {
      uint8_t usc[128] = { 0 };
      unsigned at = g5p_usc_branch(usc, 0x40, false);
      while (at < 0x40)
         at += g5p_usc_special(usc + at, false);
      at += g5p_usc_task_end(usc + at);
      result = pvr_gpu_upload_usc(device, usc, ALIGN_POT(at, 64), 64, &p->ta_usc);
      if (result != VK_SUCCESS)
         return result;
      result = g5p_upload_kick(device, p->ta_usc, 1, &p->ta_kick);
      if (result != VK_SUCCESS)
         return result;
   }

   result = g5p_upload_zeroed(device, G5P_SCRATCH_BYTES, &p->scratch);
   if (result == VK_SUCCESS)
      result = g5p_upload_zeroed(device, G5P_NULL_RT_PITCH4 * G5P_NULL_RT_SIDE, &p->null_rt);
   if (result == VK_SUCCESS)
      result = g5p_upload_zeroed(device, G5P_SCISSOR_POOL_BYTES, &p->scissor_pool);
   if (result == VK_SUCCESS)
      result = g5p_upload_zeroed(device, G5P_DBIAS_POOL_BYTES, &p->dbias_pool);
   if (result == VK_SUCCESS)
      result = g5p_upload_load(device, p, 0);
   return result;
}

VkResult
PVR_PER_ARCH(g5p_device_init)(struct pvr_device *device)
{
   device->g5p = vk_zalloc(&device->vk.alloc, sizeof(*device->g5p), 8,
                           VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device->g5p)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* the bound memory the CPU transfers address (pvr_arch_g9p_transfer.c) */
   simple_mtx_init(&device->g9p_vmas.lock, mtx_plain);
   util_dynarray_init(&device->g9p_vmas.vmas, NULL);

   /* vkCmdClearAttachments' clear quad: its pipelines are made at first use (pvr_arch_g5p_gfx.c) */
   simple_mtx_init(&device->g5p->clear_lock, mtx_plain);

   VkResult result = g5p_upload_programs(device, &device->g5p->programs);
   if (result == VK_SUCCESS)
      result = PVR_PER_ARCH(g5p_meta_init)(device);
   if (result != VK_SUCCESS)
      PVR_PER_ARCH(g5p_device_finish)(device);
   return result;
}

void
PVR_PER_ARCH(g5p_device_finish)(struct pvr_device *device)
{
   if (!device->g5p)
      return;

   PVR_PER_ARCH(g5p_meta_finish)(device);

   /* the clear quad's pipelines and their layout */
   const VkDevice vk_dev = pvr_device_to_handle(device);
   for (unsigned i = 0; i < device->g5p->clear_count; i++)
      device->vk.dispatch_table.DestroyPipeline(vk_dev, device->g5p->clear[i].pipeline, NULL);
   if (device->g5p->clear_layout)
      device->vk.dispatch_table.DestroyPipelineLayout(vk_dev, device->g5p->clear_layout, NULL);
   simple_mtx_destroy(&device->g5p->clear_lock);

   g5p_free_programs(device, &device->g5p->programs);
   util_dynarray_fini(&device->g9p_vmas.vmas);
   simple_mtx_destroy(&device->g9p_vmas.lock);
   vk_free(&device->vk.alloc, device->g5p);
   device->g5p = NULL;
}

/* ---- render jobs ---- */

/* PBE state of a linear colour target (the A9 ICD's g5p_pbe_encode). The
 * key and the sRGB bit (q1 bit 45, fmt-srgb8's EOT PBE) are the A9 format
 * table's (pvr_g5p_formats.h): the formats the two-word fragment program
 * renders. */
#define G5P_PBE_SRGB_BIT 45

static bool
g5p_pbe_key(VkFormat format, uint32_t *key, bool *srgb)
{
   return pvr_g5p_format_pbe(format, key, srgb);
}

static void
g5p_pbe_encode(uint64_t q[3], uint32_t key, bool srgb, uint64_t va, uint32_t width, uint32_t height,
               uint32_t row_stride)
{
   g5p_pbe_2d(q,key,srgb,va,width,height,row_stride,false);
}

/* A refusal, by name, on stderr too (vk_errorf() is silent in release builds) */
#define G5P_REFUSE(cmd_buffer, ...)                                                                                \
   (fprintf(stderr, "G5P refuses: " __VA_ARGS__), fprintf(stderr, "\n"),                                          \
    vk_errorf((cmd_buffer), VK_ERROR_FEATURE_NOT_PRESENT, "G5P: " __VA_ARGS__))

/*
 * The background of a loaded colour attachment (loadOp LOAD; plan step 7,
 * rung B): the load program (a texelFetch of the attachment at the
 * fragment's pixel, compiled by the backend) with its uploads, the
 * attachment's own texture state and a nearest, clamped single-level
 * sampler, DMA'd into the common store. BGRND0 = {the program's kick, the
 * DMA code}, BGRND2 = the DMA data (the sources, then the controls), BGRND3
 * = the data and common-store sizes, as the A9 ICD's load background has
 * them. Apple's LOAD program is not used.
 */
static VkResult
g5p_load_background(struct pvr_cmd_buffer *cmd_buffer, const uint64_t state[4], uint32_t width, uint32_t height,
                    uint64_t bgnd[4], unsigned raw)
{
   const struct pvr_g5p_programs *p = &cmd_buffer->device->g5p->programs;
   if (raw)
      p = p->raw_load[raw - 1];
   const VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   uint64_t sampler[2] = { 0 };
   const char *why = pvr_g5p_sampler_encode(&sci, false, false, &sampler[0]);
   if (why)
      return G5P_REFUSE(cmd_buffer, "the load sampler: %s", why);

   /* textureSize()'s block (PCO_G5P_UPLOAD_IMAGE_SIZE): width, height, layers, levels */
   const uint32_t size[4] = { width, height, 1, 1 };
   struct pvr_suballoc_bo *state_bo, *sampler_bo, *size_bo;
   VkResult result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, state, 4 * sizeof(state[0]), &state_bo);
   if (result != VK_SUCCESS)
      return result;
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, sampler, sizeof(sampler), &sampler_bo);
   if (result != VK_SUCCESS)
      return result;
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, size, sizeof(size), &size_bo);
   if (result != VK_SUCCESS)
      return result;

   const unsigned n = p->load_uploads;
   uint32_t data[4 * ARRAY_SIZE(p->load_upload)] = { 0 };
   for (unsigned i = 0; i < n; i++) {
      const unsigned kind = p->load_upload[i].kind;
      const uint64_t src = kind == PCO_G5P_UPLOAD_IMAGE   ? state_bo->dev_addr.addr
                           : kind == PCO_G5P_UPLOAD_SAMPLER ? sampler_bo->dev_addr.addr
                                                            : size_bo->dev_addr.addr;
      if (p->load_upload[i].dwords > (kind == PCO_G5P_UPLOAD_IMAGE ? 8u : kind == PCO_G5P_UPLOAD_SAMPLER ? 2u : 4u))
         return G5P_REFUSE(cmd_buffer, "a load program upload of %u dwords", p->load_upload[i].dwords);
      data[2 * i] = (uint32_t)src;
      data[2 * i + 1] = (uint32_t)(src >> 32);
      data[2 * n + 2 * i] = g5p_doutd_control_at(p->load_upload[i].common, p->load_upload[i].dwords, i == n - 1);
   }
   struct pvr_pds_upload load_data;
   result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, 4 * n, 16, NULL, 0, 0, 16, &load_data);
   if (result != VK_SUCCESS)
      return result;

   pvr_csb_pack (&bgnd[0], CR_PDS_BGRND0_BASE, b0) {
      b0.shader_addr = PVR_DEV_ADDR(p->load_kick.data_offset);
      b0.texunicode_addr = PVR_DEV_ADDR(p->load_dma.code_offset);
   }
   pvr_csb_pack (&bgnd[2], CR_PDS_BGRND2_BASE, b2) {
      b2.uniformdata_addr = PVR_DEV_ADDR(load_data.data_offset);
   }
   pvr_csb_pack (&bgnd[3], CR_PDS_BGRND3_SIZEINFO, b3) {
      b3.pds_uniformsize = DIV_ROUND_UP(4 * n, ROGUE_CR_PDS_BGRND3_SIZEINFO_PDS_UNIFORMSIZE_UNIT_SIZE);
      b3.usc_sharedsize = DIV_ROUND_UP(p->load_common_dwords, ROGUE_CR_PDS_BGRND3_SIZEINFO_USC_SHAREDSIZE_UNIT_SIZE);
   }
   if (getenv("PVR_G5P_WS_LOG")) {
      const uint32_t *w = (const uint32_t *)state, *sw = (const uint32_t *)sampler;
      fprintf(stderr, "g5p load: state %08x %08x %08x %08x %08x %08x %08x %08x sampler %08x %08x data", w[0],
              w[1], w[2], w[3], w[4], w[5], w[6], w[7], sw[0], sw[1]);
      for (unsigned i = 0; i < 4 * n; i++)
         fprintf(stderr, " %08x", data[i]);
      fprintf(stderr, "\n");
   }
   return VK_SUCCESS;
}

VkResult
PVR_PER_ARCH(g5p_job_init)(struct pvr_cmd_buffer *cmd_buffer, struct pvr_sub_cmd_gfx *sub_cmd,
                           const struct pvr_renderpass_hwsetup_render *hw_render,
                           struct pvr_render_job *job)
{
   struct pvr_device *device = cmd_buffer->device;
   const struct pvr_g5p_programs *p = &device->g5p->programs;
   const struct pvr_render_pass_info *info = &cmd_buffer->state.render_pass_info;
   struct pvr_g5p_job *g = &job->g5p;
   /* A render without colour attachments (a depth-only or stencil-only pass): its end of tile stores the
    * one-target form into the device's null target, a 16x16 R8 surface. The PBE cuts its writes to the
    * surface it is given (the edge tiles' pixels past an attachment are not written), so the store touches
    * those 256 bytes and nothing else; the render's extent is the framebuffer's. Its draws write no colour
    * (their pipelines take write mask 0, pvr_arch_g5p_gfx.c): the no-colour ISP form, depth/stencil only.
    * Apple's own depth-only pass is not captured (bench-gaps.md section 6.5); this takes only the forms the
    * one-target pass already runs. */
   const bool null_target = hw_render->eot_surface_count == 0;
   const unsigned n = null_target ? 1 : hw_render->eot_surface_count;
   unsigned v;

   /* everything but the reads the pass's draws collected (pvr_g5p_perf.h) */
   memset(g, 0, offsetof(struct pvr_g5p_job, reads));
   /* three targets: Apple's 4-target pass with a fourth that stores to the null target (admitted by
    * pvr_arch_g5p_mrt_pass_check(): one 8-bit RGBA format; the program writes it zeros, pvr_g5p_nir.c) */
   const unsigned nt = n == 3 ? 4 : n;
   switch (nt) {
   case 1: v = 0; break;
   case 2: v = 1; break;
   case 4: v = 2; break;
   default: return G5P_REFUSE(cmd_buffer, "%u colour targets (1, 2, 3 or 4 are built)", n);
   }
   /* several targets: Apple's pass shape (pvr_g5p_mrt.h) */
   VkResult mrt = pvr_arch_g5p_mrt_pass_check(cmd_buffer, hw_render);
   if (mrt != VK_SUCCESS)
      return mrt;
   /* a depth/stencil attachment: what the ZLS builds (pvr_arch_g5p_ds.c) */
   VkResult admit = pvr_arch_g5p_ds_admit(cmd_buffer, hw_render);
   if (admit != VK_SUCCESS)
      return admit;
   /* 4 samples: what the captured 4x pass is (pvr_arch_g5p_msaa.c) */
   admit = pvr_arch_g5p_msaa_admit(cmd_buffer, hw_render);
   if (admit != VK_SUCCESS)
      return admit;
   g->samples = MAX2(hw_render->sample_count, 1);
   if (null_target && g->samples != 1)
      return G5P_REFUSE(cmd_buffer, "a %u-sample render without colour attachments", g->samples);
   /* the render's depth-bias table (upstream's, uploaded before this hook):
    * the kernel takes its entries from ISP_DBIAS_BASE */
   if (sub_cmd->depth_bias_bo)
      g->dbias_count = sub_cmd->depth_bias_bo->size / sizeof(struct pvr_depth_bias_state);
   /* the draws' ISP scissor entries (pvr_arch_g5p_gfx.c): upstream's per-render table, ISP_SCISSOR_BASE */
   if (sub_cmd->scissor_bo)
      g->scissor_count = sub_cmd->scissor_bo->size / sizeof(struct pvr_scissor_words);

   /* the control stream: one range through STREAM_TERMINATE, also from several linked blocks (pvr_g5p_vdm.h) */
   VkResult vdm = pvr_arch_g5p_vdm_stream(cmd_buffer, &sub_cmd->control_stream, &g->vdm_va, &g->vdm_bytes);
   if (vdm != VK_SUCCESS)
      return vdm;

   /* targets: PBE words and clear colours */
   uint64_t pbe[4][3];
   unsigned raw = 0;
   const VkFormat first_format = null_target ? VK_FORMAT_UNDEFINED
      : info->attachments[hw_render->eot_surfaces[0].attachment_idx]->vk.format;
   /* 32-bit integer targets (R, RG, RGBA; unsigned 1, signed 2): the four-word raw program, the PBE storing the
    * target's channels (pvr_g5p_raw32_kind()); RGBA16UI clears: five (below) */
   raw = pvr_g5p_raw32_kind(first_format);
   if (pvr_g5p_packed_uint_bits(first_format))
      raw = pvr_g5p_packed_is_signed(first_format) ? (pvr_g5p_packed_uint_bits(first_format) == 8 ? 8 : 9) :
            pvr_g5p_packed_uint_bits(first_format) == 10 ? 7 :
            pvr_g5p_packed_uint_bits(first_format) == 8 ? 6 : 5;
   /* output words per target, and of the pass: two raw32 targets of one format (admitted by
    * pvr_arch_g5p_mrt_pass_check(), behind the format probe) take four each, eight in all */
   const bool byte_source = first_format == VK_FORMAT_R8_UINT || first_format == VK_FORMAT_R8_SINT;
   if (byte_source)
      raw = first_format == VK_FORMAT_R8_SINT ? 11 : 10;
   const bool narrow = raw >= 5 &&
      (byte_source || first_format == VK_FORMAT_R16_UINT || first_format == VK_FORMAT_R16_SINT ||
       first_format == VK_FORMAT_R8G8_UINT || first_format == VK_FORMAT_R8G8_SINT);
   struct g5p_color_source_abi source = {0};
   if (raw >= 5 && !g5p_color_source_make(byte_source ? 8 : narrow ? 16 : 32,
         (first_format == VK_FORMAT_R16G16B16A16_UINT || first_format == VK_FORMAT_R16G16B16A16_SINT) ? 2 : 1,
         G5P_COLOR_SOURCE_UINT,&source))
      return G5P_REFUSE(cmd_buffer,"unqualified physical colour source ABI");
   const unsigned raw_words = raw <= 2 ? 4 : source.output_words;
   const unsigned raw_total = raw_words * n;
   if (raw) {
      if (!(n == 1 || (n == 2 && raw <= 2)) || g->samples != 1)
         return G5P_REFUSE(cmd_buffer, "integer pass requires one single-sample target, or two 32-bit integer "
                           "targets of one format");
      VkResult result = g5p_raw_programs(device, raw, raw_total, n, raw >= 5 ? source.channel_bits : 32);
      if (result != VK_SUCCESS)
         return result;
      g->raw_words = raw_total;
      g->raw_pixel_width = raw >= 5 ? source.pixel_output_width : raw_total;
   }
   float clear[4][4] = { { 0 } };
   /* a loaded target: its texture state (q0, q1, q2, 0), the load program's image upload */
   uint64_t load_state[4] = { 0 };
   if (null_target) {
      uint32_t key;
      bool srgb;
      if (!g5p_pbe_key(VK_FORMAT_R8_UNORM, &key, &srgb))
         return G5P_REFUSE(cmd_buffer, "the null colour target's R8 key");
      g5p_pbe_encode(pbe[0], key, srgb, p->null_rt->dev_addr.addr, G5P_NULL_RT_SIDE, G5P_NULL_RT_SIDE,
                     G5P_NULL_RT_PITCH);
      g->width = info->rstate->width;
      g->height = info->rstate->height;
   }
   for (unsigned t = 0; t < n && !null_target; t++) {
      const struct pvr_image_view *iview = info->attachments[hw_render->eot_surfaces[t].attachment_idx];
      const struct pvr_image *image = pvr_image_view_get_image(iview);
      const struct pvr_image_plane *plane = pvr_single_plane_const(image);
      const struct pvr_mip_level *level = &plane->mip_levels[iview->vk.base_mip_level];
      const uint64_t addr = image->dev_addr.addr + plane->offset + level->offset +
                            iview->vk.base_array_layer * plane->layer_size;
      uint32_t key;
      bool srgb;
      /* 4x (admitted above: one target): the samples stored as they are, or the PBE resolve into the
       * one-sample view this surface names (pvr_g5p_msaa.c) */
      const bool ms_store = g->samples == 4 && !hw_render->eot_surfaces[t].need_resolve;

      const bool twiddled=image->memlayout!=PVR_MEMLAYOUT_LINEAR && !ms_store;
      if(twiddled) {
         const char *probe=getenv("PVR_G5P_TWIDDLED_PBE");
         if(!probe || strcmp(probe,"probe"))
            return G5P_REFUSE(cmd_buffer,"a non-linear colour target (twiddled PBE is step 7)");
         if(image->memlayout!=PVR_MEMLAYOUT_TWIDDLED || image->vk.image_type!=VK_IMAGE_TYPE_2D ||
            iview->vk.layer_count!=1 || (iview->vk.view_type!=VK_IMAGE_VIEW_TYPE_2D &&
                                       iview->vk.view_type!=VK_IMAGE_VIEW_TYPE_2D_ARRAY))
            return G5P_REFUSE(cmd_buffer,"twiddled PBE needs a single 2D layer (3D brick slices are not qualified)");
      }
      if (!g5p_pbe_key(iview->vk.format, &key, &srgb))
         return G5P_REFUSE(cmd_buffer, "colour target format %s", vk_Format_to_str(iview->vk.format));
      if ((level->pitch % 16 && !ms_store && !twiddled) || addr % 64)
         return G5P_REFUSE(cmd_buffer, "a colour target at %#" PRIx64 " with pitch %u (64-byte base, 16-byte rows)",
                           addr, level->pitch);
      /* SINT32 and UINT32 share the same physical channel layout. Store
       * signed bits with the unsigned PBE representation and raw EOT mode;
       * do not perform a numeric conversion. LOAD/sampling below retains
       * the original signed image key. Blending is rejected for integers.
       */
      uint32_t store_key = raw ? key & ~(0x80u << 5) : key;
      if (raw >= 5 && raw <= 11) {
         /* Equal-size raw physical view; logical sampling stays typed.
          * Native witness-r16 supplies the narrow16-bit source layout. */
         bool physical_srgb;
         VkFormat physical = byte_source ? VK_FORMAT_R8_UINT : narrow ? VK_FORMAT_R16_UINT :
            (first_format == VK_FORMAT_R16G16B16A16_UINT || first_format == VK_FORMAT_R16G16B16A16_SINT)
               ? VK_FORMAT_R32G32_UINT : VK_FORMAT_R32_UINT;
         if (!g5p_pbe_key(physical, &store_key, &physical_srgb))
            return G5P_REFUSE(cmd_buffer, "RGBA16UI clear needs the RG32UI physical store");
         /* Native write selectors are RRRR/RGRR, not sampled R001/RG01.
          * Keep the logical sampling key untouched for integer LOAD. */
         store_key = source.write_key;
      }
      if (ms_store) {
         pvr_g5p_msaa_pbe_store(pbe[t], key, addr, iview->vk.extent.width, iview->vk.extent.height,
                                vk_format_is_srgb(iview->vk.format));
      } else {
         g5p_pbe_2d(pbe[t],store_key,srgb,addr,iview->vk.extent.width,iview->vk.extent.height,level->pitch,twiddled);
         if (g->samples == 4) {
            pvr_g5p_msaa_pbe_resolve(pbe[t]);
            g->msaa_resolve = true;
         }
      }
      if (raw && getenv("PVR_G5P_DGL_LOG"))
         fprintf(stderr, "DGL-PBE target=%u/%u words=%u total=%u source_half=%u source=uint-raw key=%08x store_key=%08x "
                 "state=%016" PRIx64 ",%016" PRIx64 ",%016" PRIx64 "\n",
                 t, n, raw_words, raw_total, 2 * raw_words * t, key, store_key, pbe[t][0], pbe[t][1], pbe[t][2]);
      if (!t) {
         if(twiddled) {
            struct pvr_image_descriptor desc;
            /* A one-layer array attachment is still a 2D store/restore
             * surface. Bake its base-layer offset into a 2D LOAD view;
             * the background uses a two-coordinate texelFetch. */
            struct pvr_image_view load_view=*iview;
            load_view.vk.view_type=VK_IMAGE_VIEW_TYPE_2D;
            const char *why=pvr_arch_g5p_pack_image_view(&load_view,&desc);
            if(why)return G5P_REFUSE(cmd_buffer,"twiddled LOAD view: %s",why);
            load_state[0]=desc.words[0];load_state[1]=desc.words[1];
            load_state[2]=(uint64_t)desc.meta[0]|(uint64_t)desc.meta[1]<<32;
         } else
            pvr_g5p_tex_linear(load_state, key, srgb, addr, iview->vk.extent.width, iview->vk.extent.height,level->pitch);
      }
      if (!t) {
         g->width = iview->vk.extent.width;
         g->height = iview->vk.extent.height;
      }
   }
   if (nt == 4 && n == 3) {
      uint32_t key;
      bool srgb;
      if (!g5p_pbe_key(first_format, &key, &srgb))
         return G5P_REFUSE(cmd_buffer, "the padding target's %s key", vk_Format_to_str(first_format));
      g5p_pbe_encode(pbe[3], key, srgb, p->null_rt->dev_addr.addr, G5P_NULL_RT_SIDE, G5P_NULL_RT_SIDE,
                     G5P_NULL_RT_PITCH4);
   }
   /* A render area smaller than the attachment: every tile runs the
    * background and is stored, so the tiles outside the area must load what
    * they hold (the A9 ICD's rule: the LOAD background whatever the load op;
    * the draws are clipped to the area, pvr_arch_g5p_gfx.c). A CLEAR there
    * would need a clear of the area alone. Several targets in a partial area
    * are refused before (pvr_arch_g5p_mrt_pass_check()). */
   const VkRect2D *area = &info->render_area;
   /* the null target holds nothing to keep: no colour LOAD whatever the area */
   const bool partial = !null_target && (area->offset.x || area->offset.y || area->extent.width != g->width ||
                                         area->extent.height != g->height);
   /* A size-limited continuation stores then loads the previous pass,
    * including its original CLEAR. Never clear the continuation again. */
   bool load = partial || (!null_target && sub_cmd->barrier_load);
   if (sub_cmd->barrier_load && !null_target && n != 1)
      return G5P_REFUSE(cmd_buffer, "VDM continuation needs one loadable colour target");
   for (unsigned i = 0; i < hw_render->color_init_count; i++) {
      const unsigned t = i; /* color_init follows the EOT surfaces' order */
      if (partial && !sub_cmd->barrier_load && hw_render->color_init[i].op == VK_ATTACHMENT_LOAD_OP_CLEAR)
         return G5P_REFUSE(cmd_buffer, "a CLEAR of colour attachment %u in a partial render area (%d,%d %ux%u of "
                           "%ux%u): no clear of the area alone", t, area->offset.x, area->offset.y,
                           area->extent.width, area->extent.height, g->width, g->height);
      if (hw_render->color_init[i].op == VK_ATTACHMENT_LOAD_OP_LOAD) {
         if (n != 1 || t)
            return G5P_REFUSE(cmd_buffer, "LOAD of colour attachment %u of %u (one target is built)", t, n);
         load = true;
      }
      if (t < n && hw_render->color_init[i].op == VK_ATTACHMENT_LOAD_OP_CLEAR &&
          hw_render->color_init[i].index < info->clear_value_count)
         memcpy(clear[t], info->clear_values[hw_render->color_init[i].index].color.float32, sizeof(clear[t]));
   }
   if (load && raw > 2 && !(raw >= 5 && raw <= 11))
      return G5P_REFUSE(cmd_buffer, "integer boundary qualification supports CLEAR only");
   const struct pvr_g5p_programs *load_program = raw && (raw <= 2 || (raw >= 5 && raw <= 11)) ? p->raw_load[raw - 1] : p;
   if (load && !load_program->load_usc)
      return G5P_REFUSE(cmd_buffer, "LOAD of a colour attachment%s: the load program: %s",
                        partial ? " (a partial render area)" : "", load_program->load_refusal);
   if (load && g->samples == 4)
      return G5P_REFUSE(cmd_buffer, "LOAD of a 4x colour attachment%s (the load program runs per pixel)",
                        partial ? " (a partial render area)" : "");
   g->targets = nt;

   /* per render: the clear floats and the PBE block, then their DOUTD data */
   if (raw) {
      uint32_t value[4]; memcpy(value, clear[0], sizeof(value));
      if (raw == 10 || raw == 11) {
         value[0] = g5p_color_source_scalar8(value[0], raw == 11);
         value[1] = value[2] = value[3] = 0;
      } else if (raw == 8 || raw == 9) {
         unsigned bits = raw == 8 ? 8 : 16;
         int lo = -(1 << (bits-1)), hi = (1 << (bits-1))-1;
         for (unsigned c = 0; c < 4; c++)
            value[c] = (uint32_t)CLAMP((int32_t)value[c],lo,hi) & ((1u << bits)-1);
      }
      if (raw == 6 || raw == 7 || raw == 8) {
         uint32_t packed = 0;
         unsigned bits = raw == 7 ? 10 : 8;
         for (unsigned c = 0; c < 4; c++) {
            unsigned max = raw == 7 ? (c == 3 ? 3 : 1023) : 255;
            packed |= MIN2(value[c],max) << (bits*c);
         }
         value[0] = packed; value[1] = value[2] = value[3] = 0;
      } else if (raw != 10 && raw != 11) {
         pvr_g5p_dgl_clear_pack(value, value, raw == 5 || raw == 9);
      }
      memcpy(clear[0], value, sizeof(value));
   }
   struct pvr_suballoc_bo *floats, *block;
   uint32_t pbe_words[G5P_MRT_PBE_STRIDE * G5P_MRT_MAX];
   /* each target's source: output half 4t (8-bit RGBA, two dwords), 8t (raw32, four words) */
   const unsigned pbe_dwords = g5p_mrt_pbe_block_halves(pbe_words, nt, (const uint64_t(*)[3])pbe,
                                                        raw ? 2 * raw_words : 4);
   VkResult result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, clear, 16 * nt, &floats);
   if (result != VK_SUCCESS)
      return result;
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, pbe_words, 4 * pbe_dwords, &block);
   if (result != VK_SUCCESS)
      return result;

   struct pvr_pds_upload clear_data, eot_data;
   {
      const uint32_t data[4] = { (uint32_t)floats->dev_addr.addr, (uint32_t)(floats->dev_addr.addr >> 32),
                                 raw ? g5p_doutd_control_at(0, raw_total, true) : g5p_doutd_control(G5P_MRT_CLEAR_DWORDS * nt), 0 };
      result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, ARRAY_SIZE(data), 16, NULL, 0, 0, 16,
                                              &clear_data);
      if (result != VK_SUCCESS)
         return result;
   }
   {
      /* a 4x store's end of tile emits per sample (pvr_g5p_msaa_eot_usc()); a resolve keeps the 1x one */
      const struct pvr_suballoc_bo *eot_usc = raw ? (narrow ? p->narrow_eot_usc : p->raw_eot_usc[n - 1]) : g->samples == 4 && !g->msaa_resolve ? p->eot_usc_ms : p->eot_usc[v];
      const uint64_t doutu = g5p_doutu_src0(g5p_usc_off(device, eot_usc), G5P_TEMPS);
      const uint32_t data[8] = { (uint32_t)block->dev_addr.addr, (uint32_t)(block->dev_addr.addr >> 32),
                                 g5p_doutd_control(pbe_dwords), 0, (uint32_t)doutu, (uint32_t)(doutu >> 32),
                                 0, 0 };
      result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, ARRAY_SIZE(data), 16, NULL, 0, 0, 16,
                                              &eot_data);
      if (result != VK_SUCCESS)
         return result;
   }

   /* PDS_BGRND0..3: the clear kick, the clear DMA, its data; sizes */
   uint64_t bgnd[4] = { 0 };
   if (load) {
      result = g5p_load_background(cmd_buffer, load_state, g->width, g->height, bgnd, raw);
      if (result != VK_SUCCESS)
         return result;
   } else {
      pvr_csb_pack (&bgnd[0], CR_PDS_BGRND0_BASE, b0) {
         b0.shader_addr = PVR_DEV_ADDR(raw ? p->raw_clear_kick[g5p_raw_variant(raw_total)].data_offset : g->samples == 4 ? p->clear_kick_ms.data_offset
                                                     : p->clear_kick[v].data_offset);
         b0.texunicode_addr = PVR_DEV_ADDR(p->clear_dma.code_offset);
      }
      pvr_csb_pack (&bgnd[2], CR_PDS_BGRND2_BASE, b2) {
         b2.uniformdata_addr = PVR_DEV_ADDR(clear_data.data_offset);
      }
      pvr_csb_pack (&bgnd[3], CR_PDS_BGRND3_SIZEINFO, b3) {
         b3.pds_uniformsize = DIV_ROUND_UP(4, ROGUE_CR_PDS_BGRND3_SIZEINFO_PDS_UNIFORMSIZE_UNIT_SIZE);
         b3.usc_sharedsize = DIV_ROUND_UP(raw ? raw_total : G5P_MRT_CLEAR_DWORDS * nt,
                                          ROGUE_CR_PDS_BGRND3_SIZEINFO_USC_SHAREDSIZE_UNIT_SIZE);
      }
   }
   memcpy(g->background, bgnd, sizeof(g->background));
   g->eot[0] = eot_data.data_offset;
   g->eot[1] = p->eot.code_offset;
   g->eot[2] = g5p_mrt_eot_info(nt);
   /* what the render writes, for the winsys's resource list (pvr_g5p_perf.h) */
   pvr_arch_g5p_perf_job(cmd_buffer, hw_render, &g->reads, &g->perf);
   g->perf.width = g->width;
   g->perf.height = g->height;
   g->perf.surfaces = hw_render->eot_surface_count;
   g->perf.color_load = partial || sub_cmd->barrier_load || load ? VK_ATTACHMENT_LOAD_OP_LOAD :
      hw_render->color_init_count ? hw_render->color_init[0].op : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
   g->perf.color_store = !null_target;
   g->perf.zload = job->ds.load.d;
   g->perf.zstore = job->ds.store.d;
   g->perf.sload = job->ds.load.s;
   g->perf.sstore = job->ds.store.s;
   if (!null_target) {
      const struct pvr_image_view *iv = info->attachments[hw_render->eot_surfaces[0].attachment_idx];
      const struct pvr_image *im = pvr_image_view_get_image(iv);
      const struct pvr_image_plane *pl = pvr_single_plane_const(im);
      g->perf.eot_va = im->dev_addr.addr + pl->offset +
         pl->mip_levels[iv->vk.base_mip_level].offset + iv->vk.base_array_layer * pl->layer_size;
   }
   g->enabled = true;
   return VK_SUCCESS;
}

/* The TYPED_V3 pass of a G5P render job, in the fragment fw_stream (the
 * apple_g5p winsys submits it as it is). Returns its size, or 0. */
uint32_t
PVR_PER_ARCH(g5p_frag_stream)(const struct pvr_device *device, const struct pvr_render_job *job,
                              uint8_t *buf, uint32_t size)
{
   const struct pvr_g5p_programs *p = &device->g5p->programs;
   const struct pvr_g5p_job *g = &job->g5p;
   struct drm_g5p_render_pass_v3 pass;

   if (size < sizeof(pass) || !g->enabled)
      return 0;
   memset(&pass, 0, sizeof(pass));
   struct drm_g5p_render_pass *b = &pass.base;
   b->size = sizeof(pass);
   b->vdm_va = g->vdm_va;
   b->vdm_size = g->vdm_bytes;
   b->scratch_va = p->scratch->dev_addr.addr;
   b->scratch_size = G5P_SCRATCH_BYTES;
   for (unsigned i = 0; i < ARRAY_SIZE(g5p_scratch_ptr_off); i++)
      b->scratch_ptr[i] = b->scratch_va + g5p_scratch_ptr_off[i];

   /* context store and resume: the HALT (route (b)) */
   const uint32_t halt = p->halt.code_offset, data = p->halt.data_offset;
   const uint32_t store[2][6] = {
      { G5P_CTX_STORE_TA_SIZES0, G5P_CTX_STORE_TA_SIZES1, halt, 0, G5P_CTX_STORE_SIZES4, data },
      { G5P_CTX_STORE_3D_SIZES0, G5P_CTX_STORE_3D_SIZES1, halt, 0, G5P_CTX_STORE_SIZES4, data },
   };
   const uint32_t resume[2][4] = {
      { G5P_CTX_RESUME_SIZES, data | 5, halt, 0 },
      { G5P_CTX_RESUME_SIZES, data | 1, halt, 0 },
   };
   memcpy(b->context_pds, store, sizeof(store));
   memcpy(b->resume_pds, resume, sizeof(resume));
   /* the TA context program: the HALT too, with the TA resume's data word */
   b->ta_pds[0] = G5P_TA_CTX_SIZES;
   b->ta_pds[1] = data | 5;
   b->ta_pds[2] = halt;

   memcpy(b->background, g->background, sizeof(b->background));
   memcpy(b->eot, g->eot, sizeof(b->eot));
   memcpy(b->eot_resume, g->eot, sizeof(b->eot_resume));
   b->restore[0] = G5P_RESTORE0;
   b->restore[1] = g5p_mrt_restore1(g->targets);
   /* 4x: the kernel's AA words take the sample count, positions and the PBE resolve (pvr_g5p_msaa.c); all
    * zero at one sample, the one-sample pass as before */
   STATIC_ASSERT(PVR_G5P_MSAA_FLAG_RESOLVE == DRM_G5P_RENDER_MSAA_RESOLVE);
   const struct pvr_g5p_msaa_words ms = pvr_g5p_msaa_pass(g->samples, g->msaa_resolve);
   b->samples = ms.samples;
   b->msaa_flags = ms.flags;
   b->sample_positions = ms.positions;

   /* the V2 extension: the device's tables, the background object, the output width */
   struct drm_g5p_render_pass_ext *e = &pass.ext;
   const struct pvr_sub_cmd_gfx *sub_cmd =
      container_of(job, const struct pvr_sub_cmd_gfx, job);
   e->scissor_va = p->scissor_pool->dev_addr.addr;
   e->scissor_count = MIN2(G5P_SCISSOR_POOL_BYTES / 8, DRM_G5P_RENDER_MAX_TABLE_ENTRIES);
   e->dbias_va = p->dbias_pool->dev_addr.addr;
   e->dbias_count = MIN2(G5P_DBIAS_POOL_BYTES / 12, DRM_G5P_RENDER_MAX_TABLE_ENTRIES);
   /* the depth/stencil attachment's ZLS words and background object
    * (pvr_arch_g5p_ds.c; without one, depth 1.0f and tag | mask) */
   struct pvr_g5p_zls_words zls;
   pvr_arch_g5p_ds_pass(job, &zls);
   memcpy(e->zls, zls.zls, sizeof(e->zls));
   e->zls_address_mask = zls.address_mask;
   e->bgobj_depth = zls.bgobj_depth;
   e->bgobj_vals = zls.bgobj_vals;
   /* the draws' scissor entries, when a draw is clipped: upstream's per-render table */
   if (g->scissor_count && job->scissor_table_addr.addr) {
      e->scissor_va = job->scissor_table_addr.addr;
      e->scissor_count = MIN2(g->scissor_count, DRM_G5P_RENDER_MAX_TABLE_ENTRIES);
   }
   /* the draws' depth-bias entries: upstream's per-render table */
   if (g->dbias_count && job->depth_bias_table_addr.addr) {
      e->dbias_va = job->depth_bias_table_addr.addr;
      e->dbias_count = MIN2(g->dbias_count, DRM_G5P_RENDER_MAX_TABLE_ENTRIES);
   }
   e->pixel_output_width = g->raw_words ? g->raw_pixel_width : g5p_mrt_pixel_width(g->targets);
   if (sub_cmd->query_pool) {
      e->occlusion_query_va =
         sub_cmd->query_pool->result_buffer->dev_addr.addr;
      e->occlusion_query_size = sub_cmd->query_pool->result_buffer->size;
   }

   pass.ta_program_va = device->heaps.pds_heap->base_addr.addr + p->ta_kick.data_offset;
   pass.ta_program_size = 0x40;
   pass.vdm_end = g->vdm_bytes;
   memcpy(buf, &pass, sizeof(pass));
   /* then what the job writes (pvr_g5p_perf.h) */
   return sizeof(pass) + pvr_g5p_perf_stream_tail(&g->perf, buf + sizeof(pass), size - sizeof(pass));
}
