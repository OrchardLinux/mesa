/* SPDX-License-Identifier: MIT */
/*
 * Graphics on Apple's G9P (the A10): pipelines compiled by the G9P backend,
 * draws as Rogue VDM/PPP words with G9P PDS programs, the end-of-tile and
 * background programs of a render, and the Apple register block of the
 * render job streams. Each piece follows a layout tested on hardware without
 * Mesa (the A10 notes, see g9p/README.md: tools/pvr-tri.c, RENDER.md "A
 * triangle from scratch", "Varyings and textures with our encodings").
 *
 * What this does not do yet: depth/stencil, culling state from the
 * pipeline, MSAA, multiple colour attachments, more than one texture per
 * fragment shader, non-32-bit vertex formats, indirect draws.
 */

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "vk_blend.h"
#include "vk_format.h"
#include "vk_pipeline.h"

#include "hwdef/pvr_hw_utils.h"
#include "hwdef/rogue_hw_utils.h"

#include "util/u_atomic.h"
#include "util/format/u_format.h"
#include "util/ralloc.h"

#include "g9p_cdm.h"
#include "g9p_compile.h"
#include "g9p_formats.h"
#include "g9p_isa.h"

#include "pvr_bo.h"
#include "pvr_buffer.h"
#include "pvr_cmd_buffer.h"
#include "pvr_csb.h"
#include "pvr_descriptor_set.h"
#include "pvr_device.h"
#include "pvr_g9p.h"
#include "pvr_hw_pass.h"
#include "pvr_image.h"
#include "pvr_job_render.h"
#include "pvr_physical_device.h"
#include "pvr_pipeline.h"

static_assert(ARRAY_SIZE(((struct pvr_g9p_graphics_state *)0)->buffers) == G9P_MAX_BUFFERS &&
                 ARRAY_SIZE(((struct pvr_g9p_graphics_state *)0)->fs_buffers) == G9P_MAX_BUFFERS,
              "pvr_pipeline.h sizes the shaders' buffer slots by G9P_MAX_BUFFERS");

/* A pipeline the G9P backend cannot build yet: said on stderr too */
#define g9p_error(device, ...)                                                \
   (fprintf(stderr, "g9p: " __VA_ARGS__), fprintf(stderr, "\n"),              \
    vk_errorf(device, VK_ERROR_UNKNOWN, __VA_ARGS__))

/* DOUT type 4: iterator issue (DOUTI) */
#define G9P_DOUTI 4u

/* G9P_DUMP=1: print what is built (programs, stream words, job streams) */
static bool
dumping(void)
{
   static int on = -1;

   if (on < 0)
      on = getenv("G9P_DUMP") != NULL;
   return on;
}

static void
dump32(const char *what, uint64_t addr, const uint32_t *w, unsigned n)
{
   if (!dumping())
      return;
   fprintf(stderr, "g9p: %s @%#" PRIx64 ":", what, addr);
   for (unsigned i = 0; i < n; i++)
      fprintf(stderr, " %08x", w[i]);
   fprintf(stderr, "\n");
}

static void
dump16(const char *what, uint64_t addr, const uint16_t *w, unsigned n)
{
   if (!dumping())
      return;
   fprintf(stderr, "g9p: %s @%#" PRIx64 ":", what, addr);
   for (unsigned i = 0; i < n; i++)
      fprintf(stderr, " %04x", w[i]);
   fprintf(stderr, "\n");
}

static void
dump_pds(const char *what, const struct pvr_pds_upload *up)
{
   const uint32_t *map;

   if (!dumping() || !up->pvr_bo)
      return;
   map = pvr_bo_suballoc_get_map_addr(up->pvr_bo);
   fprintf(stderr, "g9p: %s: data @%#x (%u dw), code @%#x (%u dw):", what, up->data_offset,
           up->data_size, up->code_offset, up->code_size);
   for (unsigned i = 0; i < up->data_size + up->code_size + 8 && i < 64; i++)
      fprintf(stderr, " %08x", map[i]);
   fprintf(stderr, "\n");
}

static uint32_t
usc_offset(const struct pvr_device *device, const struct pvr_suballoc_bo *bo)
{
   return bo->dev_addr.addr - device->heaps.usc_heap->base_addr.addr;
}

/* Kick data word 1: the USC temp count, units of 4 registers (ISA.md) */
static uint32_t
usc_temps(unsigned regs)
{
   return DIV_ROUND_UP(MAX2(regs, 8), 4) << 3;
}

/* DOUTD control word: last, common store, destination dword, dwords */
static uint32_t
doutd_ctrl(unsigned dest, unsigned dwords, bool last)
{
   return (last ? BITFIELD_BIT(31) : 0) | BITFIELD_BIT(28) | dest << 13 | dwords;
}

static VkResult
upload_usc(struct pvr_device *device, struct g9p_code *c,
           struct pvr_suballoc_bo **bo_out)
{
   return pvr_gpu_upload_usc(device, c->w, 2 * c->n, 64, bo_out);
}

/*
 * A kick bundle: data {USC offset, temps} at the start, DOUTU + HALT at
 * +0x20 (what PPP's shader base and PDS_BGRND0 point at)
 */
static VkResult
upload_kick(struct pvr_device *device, const struct pvr_suballoc_bo *usc,
            unsigned regs, uint32_t flags, struct pvr_pds_upload *out)
{
   const uint32_t data[2] = { usc_offset(device, usc), usc_temps(regs) };
   const uint32_t code[2] = {
      g9p_pds_dout(G9P_DOUTU, 0, G9P_PDS_DATA, 0, G9P_PDS_END | flags),
      g9p_pds_halt(),
   };

   return pvr_gpu_upload_pds(device, data, ARRAY_SIZE(data), 16, code,
                             ARRAY_SIZE(code), 32, 64, out);
}

/* --- Device programs ------------------------------------------------------ */

VkResult
PVR_PER_ARCH(g9p_gfx_device_init)(struct pvr_device *device)
{
   struct pvr_g9p_gfx_programs *p = &device->g9p.gfx;
   struct g9p_code *c = calloc(1, sizeof(*c));
   const uint32_t halt = g9p_pds_halt();
   const uint32_t dma[2] = {
      g9p_pds_dout(G9P_DOUTD, 0, G9P_PDS_DATA, 2, G9P_PDS_END),
      g9p_pds_halt(),
   };
   const uint32_t eot[3] = {
      g9p_pds_dout(G9P_DOUTD, 0, G9P_PDS_DATA, 2, 0), /* image state */
      g9p_pds_dout(G9P_DOUTU, 2, G9P_PDS_DATA, 0, G9P_PDS_END),
      g9p_pds_halt(),
   };
   const uint32_t stream_out_data[4] = { 0 };
   uint8_t zeros[0x400] = { 0 };
   VkResult result;

   if (!c)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* Clear: half h of the output = f16 of the f32 uniform h */
   c->fragment = true;
   for (unsigned h = 0; h < 4; h++)
      G9P_EMIT(c, 0x018a, 0x0100 | h, 0x0101, 0x01a0, 0x0101 | 4 * h, 0x4140, 0x0104,
               0x00b0);
   g9p_end_fragment(c, true);
   result = upload_usc(device, c, &p->clear_usc);
   if (result != VK_SUCCESS)
      goto out;

   /* End of tile: r0 = 0, the PBE emit with the image state in the common store */
   c->n = 0;
   g9p_mov_imm(c, 0, 0);
   G9P_EMIT(c, 0x051c, 0x5100, 0x0800);
   g9p_end(c);
   result = upload_usc(device, c, &p->eot_usc);
   if (result != VK_SUCCESS)
      goto out;

   /* Register 0x470's program: a jump to `015e a000` (as captured) */
   c->n = 0;
   {
      unsigned jump = 0, end = 64;

      G9P_EMIT(c, 0x0100, 0x0100 | (end - jump), 0x0000);
      while (2 * c->n < end)
         G9P_EMIT(c, 0xc000);
      G9P_EMIT(c, 0x015e, 0xa000);
      while (c->n % 32)
         G9P_EMIT(c, 0x0000);
   }
   result = upload_usc(device, c, &p->vdm_end_usc);
   if (result != VK_SUCCESS)
      goto out;

   /*
    * Load op (iPadOS's `load` capture): the pixel position (211c 6800, x
    * and y as u16 halves in 1/16 pixels, the sample centre), u2f each,
    * times 1/(16 width) and 1/(16 height) (common-store dwords 12-13) into
    * r0-r1, the texture sample with the state at dword 0 and the sampler at
    * 8 into the colour halves, output
    */
   c->n = 0;
   c->fragment = true;
   G9P_EMIT(c, 0x211c, 0x6800);
   g9p_wait(c);
   g9p_convert(c, 2, 0, G9P_U2F);
   g9p_convert(c, 3, 1, G9P_U2F);
   G9P_EMIT(c, 0x0199, g9p_dst(c, 0), 0x0181 | 12 << 2 | 2 << 10, 0x4100, G9P_FMUL);
   G9P_EMIT(c, 0x0199, g9p_dst(c, 1), 0x0181 | 13 << 2 | 3 << 10, 0x4100, G9P_FMUL);
   G9P_EMIT(c, 0xf518, 0x0100, 0x8108, 0x0700, 0x0200);
   g9p_wait(c);
   g9p_end_fragment(c, true);
   result = upload_usc(device, c, &p->load_usc);
   if (result != VK_SUCCESS)
      goto out;

   /*
    * Clear rectangles' vertex shader (pvr-tri's): (x, y) at common-store
    * dwords 0-1's address + 8 x the vertex index, z 0, w 1
    */
   c->n = 0;
   c->fragment = false;
   g9p_addr(c, 2, 0, 0, 3, false);
   g9p_mem_at(c, false, G9P_SIZE_2X32, 4, 2);
   g9p_wait(c);
   g9p_mov_imm(c, 6, 0);
   g9p_mov_imm(c, 7, fui(1.0f));
   for (unsigned i = 0; i < 3; i++)
      g9p_vs_output(c, 4 + i, i);
   g9p_end_vertex(c, 7, 3);
   result = upload_usc(device, c, &p->clear_vs_usc);
   if (result != VK_SUCCESS)
      goto out;
   {
      const uint32_t data[4] = { usc_offset(device, p->clear_vs_usc), usc_temps(8),
                                 0x80000000u, 0 };
      const uint32_t code[3] = {
         g9p_pds_dout(G9P_DOUTW, 0, G9P_PDS_TEMPS, 2, 0),
         g9p_pds_dout(G9P_DOUTU, 0, G9P_PDS_DATA, 0, G9P_PDS_END | G9P_PDS_CC),
         g9p_pds_halt(),
      };

      result = pvr_gpu_upload_pds(device, data, ARRAY_SIZE(data), 16, code, ARRAY_SIZE(code),
                                  16, 64, &p->clear_vs_kick);
      if (result != VK_SUCCESS)
         goto out;
   }

   result = upload_kick(device, p->clear_usc, 8, 0, &p->clear_kick);
   if (result == VK_SUCCESS)
      result = upload_kick(device, p->load_usc, 8, 0, &p->load_kick);
   if (result == VK_SUCCESS)
      result = upload_kick(device, p->vdm_end_usc, 8, 0, &p->vdm_end);
   if (result == VK_SUCCESS)
      result = pvr_gpu_upload_pds(device, NULL, 0, 0, &halt, 1, 16, 64, &p->halt);
   if (result == VK_SUCCESS)
      result = pvr_gpu_upload_pds(device, NULL, 0, 0, dma, ARRAY_SIZE(dma), 16, 64,
                                  &p->dma);
   if (result == VK_SUCCESS)
      result = pvr_gpu_upload_pds(device, NULL, 0, 0, eot, ARRAY_SIZE(eot), 16, 64,
                                  &p->eot);
   /* Stream-out program: 16 bytes of data, HALT right after */
   if (result == VK_SUCCESS)
      result = pvr_gpu_upload_pds(device, stream_out_data, ARRAY_SIZE(stream_out_data), 16,
                                  &halt, 1, 16, 64, &p->stream_out);
   /* Two VDM context state buffers */
   if (result == VK_SUCCESS)
      result = pvr_gpu_upload(device, device->heaps.general_heap, zeros, sizeof(zeros),
                              64, &p->ctx_state);
   if (result == VK_SUCCESS)
      result = pvr_gpu_upload(device, device->heaps.general_heap, zeros, 64, 64, &p->zeros);

out:
   free(c);
   return result;
}

void
PVR_PER_ARCH(g9p_gfx_device_finish)(struct pvr_device *device)
{
   struct pvr_g9p_gfx_programs *p = &device->g9p.gfx;

   pvr_bo_suballoc_free(p->zeros);
   pvr_bo_suballoc_free(p->ctx_state);
   if (p->stream_out.pvr_bo)
      pvr_bo_suballoc_free(p->stream_out.pvr_bo);
   if (p->eot.pvr_bo)
      pvr_bo_suballoc_free(p->eot.pvr_bo);
   if (p->dma.pvr_bo)
      pvr_bo_suballoc_free(p->dma.pvr_bo);
   if (p->halt.pvr_bo)
      pvr_bo_suballoc_free(p->halt.pvr_bo);
   if (p->vdm_end.pvr_bo)
      pvr_bo_suballoc_free(p->vdm_end.pvr_bo);
   if (p->clear_kick.pvr_bo)
      pvr_bo_suballoc_free(p->clear_kick.pvr_bo);
   pvr_bo_suballoc_free(p->vdm_end_usc);
   pvr_bo_suballoc_free(p->eot_usc);
   pvr_bo_suballoc_free(p->clear_usc);
   pvr_bo_suballoc_free(p->clear_vs_usc);
   pvr_bo_suballoc_free(p->load_usc);
   pvr_bo_suballoc_free(p->load_kick.pvr_bo);
   pvr_bo_suballoc_free(p->clear_vs_kick.pvr_bo);
}

/* --- Pipelines ------------------------------------------------------------ */

/*
 * Vertex attribute formats the shader loads: word-aligned, and either
 * 32-bit channels or channels the vertex shader unpacks from its words
 * (g9p_compile_vertex())
 */
static bool
attrib_format_ok(VkFormat format, unsigned offset)
{
   const struct util_format_description *desc =
      util_format_description(vk_format_to_pipe_format(format));
   bool all32 = true;

   if (!desc || desc->layout != UTIL_FORMAT_LAYOUT_PLAIN || offset % 4)
      return false;
   for (unsigned i = 0; i < desc->nr_channels; i++) {
      /* fixed point, and half floats (16-bit ALU work), not yet */
      if (desc->channel[i].type == UTIL_FORMAT_TYPE_FIXED ||
          (desc->channel[i].type == UTIL_FORMAT_TYPE_FLOAT && desc->channel[i].size != 32))
         return false;
      all32 &= desc->channel[i].size == 32;
   }
   return all32 || desc->block.bits <= 32 || desc->is_array;
}

static VkResult
stage_nir(struct pvr_device *device, struct pvr_graphics_pipeline *pipeline,
          const VkPipelineShaderStageCreateInfo *stage, void *mem_ctx, nir_shader **nir)
{
   return vk_pipeline_shader_stage_to_nir(&device->vk, pipeline->base.pipeline_flags, stage,
                                          g9p_spirv_options(), g9p_nir_options(), mem_ctx,
                                          nir);
}

/*
 * Blending of the first colour attachment, done in the fragment shader as
 * Apple's compiler does (it reads the tile's colour, `blend` capture and
 * offline probes); false when the attachment is written as is
 */
static bool
blend_options(const struct vk_graphics_pipeline_state *vk_state, nir_lower_blend_options *opts)
{
   const struct vk_color_blend_state *cb = vk_state->cb;
   const struct vk_color_blend_attachment_state *rt;

   if (!cb || !cb->attachment_count || !vk_state->rp ||
       !vk_state->rp->color_attachment_count)
      return false;
   rt = &cb->attachments[0];
   if (!rt->blend_enable && !cb->logic_op_enable && (rt->write_mask & 0xf) == 0xf)
      return false;

   memset(opts, 0, sizeof(*opts));
   opts->scalar_blend_const = true;
   opts->logicop_enable = cb->logic_op_enable;
   opts->logicop_func = vk_logic_op_to_pipe(cb->logic_op);
   opts->rt[0].format = vk_format_to_pipe_format(vk_state->rp->color_attachment_formats[0]);
   opts->rt[0].colormask = rt->write_mask;
   if (rt->blend_enable && !cb->logic_op_enable) {
      opts->rt[0].rgb.func = vk_blend_op_to_pipe(rt->color_blend_op);
      opts->rt[0].rgb.src_factor = vk_blend_factor_to_pipe(rt->src_color_blend_factor);
      opts->rt[0].rgb.dst_factor = vk_blend_factor_to_pipe(rt->dst_color_blend_factor);
      opts->rt[0].alpha.func = vk_blend_op_to_pipe(rt->alpha_blend_op);
      opts->rt[0].alpha.src_factor = vk_blend_factor_to_pipe(rt->src_alpha_blend_factor);
      opts->rt[0].alpha.dst_factor = vk_blend_factor_to_pipe(rt->dst_alpha_blend_factor);
   } else {
      opts->rt[0].rgb.func = opts->rt[0].alpha.func = PIPE_BLEND_ADD;
      opts->rt[0].rgb.src_factor = opts->rt[0].alpha.src_factor = PIPE_BLENDFACTOR_ONE;
      opts->rt[0].rgb.dst_factor = opts->rt[0].alpha.dst_factor = PIPE_BLENDFACTOR_ZERO;
   }
   return true;
}

static VkResult
compile_stage(struct pvr_device *device, nir_shader *nir,
              const VkPipelineShaderStageCreateInfo *stage,
              const struct g9p_vertex_attrib *attribs, int frag_coord_slot,
              const nir_lower_blend_options *blend, struct g9p_shader *out, void *mem_ctx)
{
   bool ok;

   if (stage->stage == VK_SHADER_STAGE_VERTEX_BIT)
      ok = g9p_compile_vertex(nir, attribs, frag_coord_slot, out, mem_ctx);
   else
      ok = g9p_compile_fragment(nir, frag_coord_slot, blend, out, mem_ctx);
   if (!ok)
      return g9p_error(device,
                       "G9P: %s shader not supported by the backend yet",
                       stage->stage == VK_SHADER_STAGE_VERTEX_BIT ? "vertex" : "fragment");
   return VK_SUCCESS;
}

/*
 * The DOUTI program (RENDER.md "Varyings and textures with our encodings"):
 * the W iterate, then per varying n (a vec4 at component 4n, location 2 +
 * 8n in halves, destination 2n in units of two components), the data padded
 * to 16 bytes and the code after it.
 */
static VkResult
upload_varyings(struct pvr_device *device, struct pvr_g9p_graphics_state *state,
                const struct g9p_shader *fs)
{
   const unsigned n = state->num_varyings;
   uint32_t data[2 * (G9P_MAX_VARYINGS + 1)] = { 0x03000000, 0 };
   uint32_t code[G9P_MAX_VARYINGS + 2];
   unsigned nc = 0;

   if (!n)
      return VK_SUCCESS;

   for (unsigned v = 0; v < n; v++) {
      const bool persp = v >= fs->num_varyings || fs->varying_perspective[v];
      const unsigned loc = 2 + 8 * v;

      data[2 + 2 * v] = 0x03000000 | (3u << 2 | persp) << 16 | loc << 8 | loc;
      data[3 + 2 * v] = (persp ? BITFIELD_BIT(22) : 0) | (2 * v) << 23 |
                        (v + 1 == n ? BITFIELD_BIT(31) : 0);
   }
   for (unsigned v = 0; v <= n; v++)
      code[nc++] = g9p_pds_dout(G9P_DOUTI, v, G9P_PDS_DATA, 0, v == n ? G9P_PDS_END : 0);
   code[nc++] = g9p_pds_halt();

   state->varying_units = DIV_ROUND_UP(8 * (n + 1), 16);
   return pvr_gpu_upload_pds(device, data, 2 * (n + 1), 16, code, nc, 16, 64,
                             &state->varyings);
}

VkResult
PVR_PER_ARCH(g9p_graphics_pipeline_compile)(struct pvr_device *device,
                                            const VkGraphicsPipelineCreateInfo *create_info,
                                            struct pvr_graphics_pipeline *pipeline,
                                            const struct vk_graphics_pipeline_state *vk_state)
{
   struct pvr_g9p_graphics_state *state = &pipeline->g9p;
   const VkPipelineShaderStageCreateInfo *vs_info = NULL, *fs_info = NULL;
   struct g9p_vertex_attrib attribs[G9P_MAX_ATTRIBS] = { 0 };
   void *mem_ctx = ralloc_context(NULL);
   struct g9p_shader vs, fs = { 0 };
   static unsigned next_id;
   VkResult result;


   memset(state, 0, sizeof(*state));
   state->id = p_atomic_inc_return(&next_id);
   if (getenv("G9P_DEBUG"))
      fprintf(stderr, "g9p: pipeline %u\n", state->id);
   for (unsigned i = 0; i < create_info->stageCount; i++) {
      if (create_info->pStages[i].stage == VK_SHADER_STAGE_VERTEX_BIT)
         vs_info = &create_info->pStages[i];
      else if (create_info->pStages[i].stage == VK_SHADER_STAGE_FRAGMENT_BIT)
         fs_info = &create_info->pStages[i];
   }
   if (!vs_info || !fs_info) {
      result = g9p_error(device,
                         "G9P: pipelines need a vertex and a fragment shader so far");
      goto out;
   }

   if (vk_state->vi) {
      u_foreach_bit (loc, vk_state->vi->attributes_valid) {
         const struct vk_vertex_attribute_state *a = &vk_state->vi->attributes[loc];

         if (loc >= G9P_MAX_ATTRIBS || !attrib_format_ok(a->format, a->offset)) {
            result = g9p_error(device, "G9P: vertex attribute %u: format %s not supported yet",
                               loc, vk_Format_to_str(a->format));
            goto out;
         }
         const struct vk_vertex_binding_state *b = &vk_state->vi->bindings[a->binding];

         if (b->divisor != 1) {
            result = g9p_error(device, "G9P: vertex binding %u: divisor %u not supported yet",
                               a->binding, b->divisor);
            goto out;
         }
         attribs[loc] = (struct g9p_vertex_attrib){
            .present = true,
            .per_instance = b->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE,
            .binding = a->binding,
            .offset = a->offset,
            .format = vk_format_to_pipe_format(a->format),
         };
      }
   }

   {
      /* gl_FragCoord's varying follows the vertex shader's own */
      nir_shader *vs_nir, *fs_nir;
      nir_lower_blend_options blend;
      unsigned slot;

      result = stage_nir(device, pipeline, vs_info, mem_ctx, &vs_nir);
      if (result == VK_SUCCESS)
         result = stage_nir(device, pipeline, fs_info, mem_ctx, &fs_nir);
      if (result != VK_SUCCESS)
         goto out;
      slot = g9p_frag_coord_slot(vs_nir);
      result = compile_stage(device, fs_nir, fs_info, NULL, slot,
                             blend_options(vk_state, &blend) ? &blend : NULL, &fs, mem_ctx);
      if (result == VK_SUCCESS)
         result = compile_stage(device, vs_nir, vs_info, attribs,
                                fs.uses_frag_coord ? (int)slot : -1, NULL, &vs, mem_ctx);
   }
   if (result != VK_SUCCESS)
      goto out;
   if (fs.num_varyings > vs.num_varyings) {
      result = g9p_error(device,
                         "G9P: the fragment shader reads varyings the vertex shader "
                         "does not write");
      goto out;
   }
   state->fs_num_buffers = fs.num_buffers;
   for (unsigned i = 0; i < fs.num_buffers; i++) {
      state->fs_buffers[i].set = fs.buffers[i].set;
      state->fs_buffers[i].binding = fs.buffers[i].binding;
      state->fs_buffers[i].dwords = fs.buffers[i].dwords;
      if (fs.buffers[i].dwords)
         state->fs_common_dwords = MAX2(state->fs_common_dwords,
                                        G9P_FS_BUF_BASE + G9P_FS_BUF_DWORDS * i +
                                           fs.buffers[i].dwords);
   }
   state->num_textures = fs.num_textures;
   for (unsigned i = 0; i < fs.num_textures; i++) {
      state->textures[i].set = fs.textures[i].set;
      state->textures[i].binding = fs.textures[i].binding;
   }

   result = pvr_gpu_upload_usc(device, vs.code, vs.code_size, 64, &state->vs_bo);
   if (result != VK_SUCCESS)
      goto out;
   result = pvr_gpu_upload_usc(device, fs.code, fs.code_size, 64, &state->fs_bo);
   if (result != VK_SUCCESS)
      goto out;

   state->num_varyings = vs.num_varyings;
   state->vs_outputs = vs.num_outputs;
   state->uses_discard = fs.uses_discard;
   state->uses_fbfetch = fs.uses_fbfetch;
   state->vs_regs = vs.num_regs;
   state->num_buffers = vs.num_buffers;
   for (unsigned i = 0; i < vs.num_buffers; i++) {
      state->buffers[i].set = vs.buffers[i].set;
      state->buffers[i].binding = vs.buffers[i].binding;
   }

   /*
    * Vertex kick: temps q0 holds the vertex index (low half) and the
    * instance index (high half); one DOUTW each into r0 and r1 (Rogue
    * control: AO destination at bit 13, BSIZE 0 lower / 1 upper; the
    * `instanced` capture), then DOUTU (CC as iPadOS's)
    */
   {
      const uint32_t data[4] = { usc_offset(device, state->vs_bo), usc_temps(vs.num_regs),
                                 0, 0x80000000u | 1u << 13 | 1u };
      const uint32_t code[4] = {
         g9p_pds_dout(G9P_DOUTW, 0, G9P_PDS_TEMPS, 2, 0),
         g9p_pds_dout(G9P_DOUTW, 0, G9P_PDS_TEMPS, 3, 0),
         g9p_pds_dout(G9P_DOUTU, 0, G9P_PDS_DATA, 0, G9P_PDS_END | G9P_PDS_CC),
         g9p_pds_halt(),
      };

      result = pvr_gpu_upload_pds(device, data, ARRAY_SIZE(data), 16, code,
                                  ARRAY_SIZE(code), 16, 64, &state->vs_kick);
      if (result != VK_SUCCESS)
         goto out;
   }
   result = upload_kick(device, state->fs_bo, fs.num_regs, 0, &state->fs_kick);
   if (result == VK_SUCCESS)
      result = upload_varyings(device, state, &fs);
   if (result == VK_SUCCESS && dumping()) {
      dump16("vs usc", usc_offset(device, state->vs_bo), vs.code, vs.code_size / 2);
      dump16("fs usc", usc_offset(device, state->fs_bo), fs.code, fs.code_size / 2);
      dump_pds("vs kick", &state->vs_kick);
      dump_pds("fs kick", &state->fs_kick);
      dump_pds("varyings", &state->varyings);
   }

out:
   ralloc_free(mem_ctx);
   return result;
}

void
PVR_PER_ARCH(g9p_graphics_pipeline_destroy)(struct pvr_graphics_pipeline *pipeline)
{
   struct pvr_g9p_graphics_state *state = &pipeline->g9p;

   if (state->varyings.pvr_bo)
      pvr_bo_suballoc_free(state->varyings.pvr_bo);
   if (state->fs_kick.pvr_bo)
      pvr_bo_suballoc_free(state->fs_kick.pvr_bo);
   if (state->vs_kick.pvr_bo)
      pvr_bo_suballoc_free(state->vs_kick.pvr_bo);
   pvr_bo_suballoc_free(state->fs_bo);
   pvr_bo_suballoc_free(state->vs_bo);
}

/* --- Draws ---------------------------------------------------------------- */

static void
emit_words(struct pvr_csb *csb, const uint32_t *w, unsigned n)
{
   uint32_t *dst = pvr_arch_csb_alloc_dwords(csb, n);

   if (dst)
      memcpy(dst, w, 4 * n);
   dump32("vdm", 0, w, n);
}

/* PPP_STATE: word count and address bits 39:32, then bits 31:2 */
static void
emit_ppp(struct pvr_csb *csb, uint64_t addr, unsigned words)
{
   const uint32_t w[2] = { words << 8 | (uint32_t)(addr >> 32), (uint32_t)addr };

   emit_words(csb, w, 2);
}

static unsigned
hw_topology(VkPrimitiveTopology topology)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return 0;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      return 1;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      return 3;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
      return 9;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      return 11;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
   default:
      return 6;
   }
}

static VkResult upload_buffer_addrs(struct pvr_cmd_buffer *cmd_buffer, unsigned n,
                                    const unsigned (*sets)[2],
                                    enum pvr_stage_allocation stage, uint64_t *addr_out);
static VkResult buffer_addrs(struct pvr_cmd_buffer *cmd_buffer, unsigned n,
                             const unsigned (*sets)[2], enum pvr_stage_allocation stage,
                             uint64_t *addrs);

/*
 * The fragment shader's common store, DMA'd per draw by the program
 * TEXUNICODEBASE points at (its data by UNIFORMDATABASE): the first dwords
 * of each buffer the shader reads (slot k at G9P_FS_BUF_BASE +
 * G9P_FS_BUF_DWORDS k) and, per texture t (RENDER.md "Textures in Mesa"),
 * the texture state (4 dwords from the descriptor, 4 zeros) at dword 16t and
 * the sampler (2 dwords) at 16t + 8.
 */
static VkResult
upload_textures(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g9p_graphics_state *state,
                struct pvr_pds_upload *up)
{
   const struct pvr_g9p_gfx_programs *p = &cmd_buffer->device->g9p.gfx;
   const struct pvr_descriptor_state *desc = &cmd_buffer->state.gfx_desc_state;
   uint64_t src[G9P_MAX_BUFFERS + 3 * 4];
   uint32_t ctrl[G9P_MAX_BUFFERS + 3 * 4];
   uint32_t data[3 * (G9P_MAX_BUFFERS + 3 * 4) + 4] = { 0 };
   uint32_t code[G9P_MAX_BUFFERS + 3 * 4 + 1];
   unsigned n = 0, nc = 0;

   if (state->fs_num_buffers) {
      unsigned sets[G9P_MAX_BUFFERS][2];
      uint64_t addrs[G9P_MAX_BUFFERS];
      VkResult result;

      for (unsigned k = 0; k < state->fs_num_buffers; k++) {
         sets[k][0] = state->fs_buffers[k].set;
         sets[k][1] = state->fs_buffers[k].binding;
      }
      if (dumping())
         fprintf(stderr, "g9p: fs buffers of pipeline %u (%u dwords at set %u binding %u)\n",
                 state->id, state->fs_buffers[0].dwords, state->fs_buffers[0].set,
                 state->fs_buffers[0].binding);
      result = buffer_addrs(cmd_buffer, state->fs_num_buffers, sets,
                            PVR_STAGE_ALLOCATION_FRAGMENT, addrs);
      if (result != VK_SUCCESS)
         return result;
      for (unsigned k = 0; k < state->fs_num_buffers; k++) {
         if (!state->fs_buffers[k].dwords || !addrs[k])
            continue;
         src[n] = addrs[k];
         ctrl[n++] = doutd_ctrl(G9P_FS_BUF_BASE + G9P_FS_BUF_DWORDS * k,
                                state->fs_buffers[k].dwords, false);
      }
   }

   for (unsigned t = 0; t < state->num_textures; t++) {
      const struct pvr_descriptor_set *set = desc->sets[state->textures[t].set];
      uint64_t addr = 0;

      if (set) {
         const unsigned offset = set->layout->bindings[state->textures[t].binding].offset;

         addr = set->dev_addr.addr + offset;
         if (set->mapping)
            dump32("texture descriptor", addr, (const uint32_t *)((uint8_t *)set->mapping + offset),
                   (sizeof(struct pvr_image_descriptor) + 8) / 4);
      }
      for (unsigned k = 0; k < 3; k++) {
         src[n] = k == 0 ? addr : k == 1 ? p->zeros->dev_addr.addr
                                         : addr + sizeof(struct pvr_image_descriptor);
         ctrl[n++] = doutd_ctrl(G9P_FS_TEX_BASE + G9P_TEX_STRIDE * t + 4 * k, k == 2 ? 2 : 4,
                                false);
      }
   }

   /* a program needs a DMA: a zero dword when there is nothing to load */
   if (!n) {
      src[n] = p->zeros->dev_addr.addr;
      ctrl[n++] = doutd_ctrl(0, 1, false);
   }
   /* data: the source addresses, then the control words; one DOUTD each */
   ctrl[n - 1] |= BITFIELD_BIT(31);
   for (unsigned i = 0; i < n; i++) {
      data[2 * i] = src[i];
      data[2 * i + 1] = src[i] >> 32;
      data[2 * n + i] = ctrl[i];
      code[nc++] = g9p_pds_dout(G9P_DOUTD, i, G9P_PDS_DATA, 2 * n + i,
                                i + 1 == n ? G9P_PDS_END : 0);
   }
   code[nc++] = g9p_pds_halt();
   dump32("fs common store dma", 0, data, 3 * n);
   return pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, ALIGN_POT(3 * n, 4), 16, code, nc,
                                         16, 64, up);
}

/*
 * The address of the buffer a descriptor set binding holds: read from the
 * descriptor (its address comes first) or, for dynamic buffers, the set's
 * dynamic buffer plus its bound offset
 */
static uint64_t
descriptor_buffer_addr(const struct pvr_descriptor_set *ds, unsigned binding)
{
   const struct pvr_descriptor_set_layout_binding *b = &ds->layout->bindings[binding];

   if (b->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
       b->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
      const struct pvr_buffer_descriptor *d = &ds->dynamic_buffers[b->dynamic_buffer_idx];

      return d->addr + d->offset;
   }
   return ((const struct pvr_buffer_descriptor *)((const uint8_t *)ds->mapping + b->offset))
      ->addr;
}

/*
 * The addresses of a stage's buffers by common-store slot (vertex buffers
 * and strides, push constants, the viewport, descriptor set buffers)
 */
static VkResult
buffer_addrs(struct pvr_cmd_buffer *cmd_buffer, unsigned n, const unsigned (*sets)[2],
             enum pvr_stage_allocation stage, uint64_t *addrs)
{
   struct pvr_cmd_buffer_state *cs = &cmd_buffer->state;
   VkResult result;

   memset(addrs, 0, n * sizeof(*addrs));
   for (unsigned k = 0; k < n; k++) {
      const unsigned set = sets[k][0], binding = sets[k][1];

      if (set == G9P_VERTEX_BUFFERS) {
         const struct pvr_vertex_binding *vb = &cs->vertex_bindings[binding];

         if (vb->buffer)
            addrs[k] = vb->buffer->dev_addr.addr + vb->offset;
         if (vb->buffer && dumping()) {
            const uint32_t *v = (const uint32_t *)PVR_PER_ARCH(g9p_cpu_addr)(cmd_buffer->device,
                                                                            addrs[k]);

            if (v)
               dump32("vertex buffer", addrs[k], v, 16);
         }
      } else if (set == G9P_VERTEX_STRIDES) {
         addrs[k] = cmd_buffer->vk.dynamic_graphics_state.vi_binding_strides[binding];
      } else if (set == G9P_PUSH_CONSTANTS) {
         struct pvr_push_constants *pc = &cs->push_consts[stage];

         if (pc->dirty || !pc->dev_addr.addr) {
            struct pvr_suballoc_bo *pbo;

            result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, pc->data,
                                                        MAX2(pc->bytes_updated, 4), &pbo);
            if (result != VK_SUCCESS)
               return result;
            pc->dev_addr = pbo->dev_addr;
            pc->dirty = false;
         }
         addrs[k] = pc->dev_addr.addr;
      } else if (set == G9P_VIEWPORT) {
         /* scale/offset x, y, z of viewport 0 (gl_FragCoord's varying) */
         const struct vk_dynamic_graphics_state *dyn = &cmd_buffer->vk.dynamic_graphics_state;
         const VkViewport *vp = &dyn->vp.viewports[0];
         const float v[8] = { vp->width / 2, vp->x + vp->width / 2, vp->height / 2,
                              vp->y + vp->height / 2, vp->maxDepth - vp->minDepth,
                              vp->minDepth, 0, 0 };
         struct pvr_suballoc_bo *vbo;

         result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, v, sizeof(v), &vbo);
         if (result != VK_SUCCESS)
            return result;
         addrs[k] = vbo->dev_addr.addr;
      } else {
         const struct pvr_descriptor_set *ds = cs->gfx_desc_state.sets[set];

         if (ds && ds->layout)
            addrs[k] = descriptor_buffer_addr(ds, binding);
         if (addrs[k] && dumping()) {
            const uint32_t *v = (const uint32_t *)PVR_PER_ARCH(g9p_cpu_addr)(cmd_buffer->device,
                                                                            addrs[k]);

            /* G9P_DUMP_FULL: whole buffers (128 dwords), for replaying a draw */
            if (v)
               dump32(stage == PVR_STAGE_ALLOCATION_FRAGMENT ? "fs buffer" : "vs buffer",
                      addrs[k], v, getenv("G9P_DUMP_FULL") ? 128 : 16);
         }
      }
   }
   return VK_SUCCESS;
}

/*
 * The addresses of a stage's buffers by common-store slot, uploaded: the
 * DMA of the vertex shader's program copies them into the common store
 */
static VkResult
upload_buffer_addrs(struct pvr_cmd_buffer *cmd_buffer, unsigned n,
                    const unsigned (*sets)[2], enum pvr_stage_allocation stage,
                    uint64_t *addr_out)
{
   uint64_t addrs[G9P_MAX_BUFFERS];
   struct pvr_suballoc_bo *bo;
   VkResult result;

   result = buffer_addrs(cmd_buffer, n, sets, stage, addrs);
   if (result != VK_SUCCESS)
      return result;
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, addrs, MAX2(8 * n, 8), &bo);
   if (result != VK_SUCCESS)
      return result;
   dump32("buffer addresses", bo->dev_addr.addr, (const uint32_t *)addrs, 2 * n);
   *addr_out = bo->dev_addr.addr;
   return VK_SUCCESS;
}

/* The vertex shader's common store: its buffer addresses, by the draw's DMA */
static VkResult
upload_vs_uniforms(struct pvr_cmd_buffer *cmd_buffer,
                   const struct pvr_g9p_graphics_state *state, struct pvr_pds_upload *dma)
{
   unsigned sets[G9P_MAX_BUFFERS][2];
   uint32_t data[4];
   uint64_t addr;
   VkResult result;

   for (unsigned k = 0; k < state->num_buffers; k++) {
      sets[k][0] = state->buffers[k].set;
      sets[k][1] = state->buffers[k].binding;
   }
   result = upload_buffer_addrs(cmd_buffer, state->num_buffers, sets,
                                PVR_STAGE_ALLOCATION_VERTEX_GEOMETRY, &addr);
   if (result != VK_SUCCESS)
      return result;
   data[0] = addr;
   data[1] = addr >> 32;
   data[2] = doutd_ctrl(0, MAX2(2 * state->num_buffers, 2), true);
   data[3] = 0;
   return pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, ARRAY_SIZE(data), 16, NULL, 0,
                                         0, 64, dma);
}

/*
 * What a draw's words need besides the pipeline state: the vertex shader's
 * common-store DMA, the fragment shader's uniform/texture program (or none)
 * and its common-store size in dwords, and the viewport
 */
struct g9p_draw_programs {
   const struct pvr_pds_upload *dma, *tex;
   unsigned fs_common_dwords;
   float vp[6]; /* a0 m0 a1 m1 a2 m2 */
   /* the draw's entry in the render's scissor table, or -1 for none */
   int scissor;
   /*
    * Region clip in 32-pixel tiles, inclusive (left, top, right, bottom):
    * the tiler bins the draw into these tiles only. All zero: the whole
    * render area.
    */
   uint32_t region[4];
   /* the depth test (a depth attachment and the test enabled), its write and compare op */
   bool depth_test, depth_write;
   VkCompareOp depth_compare;
};

/*
 * A scissor table entry for the render (the scissor capture: {left << 16 |
 * right, top << 16 | bottom}, right and bottom exclusive), reusing the
 * previous one when equal; its index
 */
static int
scissor_index(struct pvr_cmd_buffer *cmd_buffer, VkRect2D r)
{
   struct util_dynarray *a = &cmd_buffer->scissor_array;
   const struct pvr_scissor_words w = {
      .w0 = (uint32_t)r.offset.x << 16 | (r.offset.x + r.extent.width),
      .w1 = (uint32_t)r.offset.y << 16 | (r.offset.y + r.extent.height),
   };
   const unsigned n = util_dynarray_num_elements(a, struct pvr_scissor_words);

   if (n && !memcmp(util_dynarray_element(a, struct pvr_scissor_words, n - 1), &w, sizeof(w)))
      return n - 1;
   util_dynarray_append_typed(a, struct pvr_scissor_words, w);
   return n;
}

static VkResult emit_draw(struct pvr_cmd_buffer *cmd_buffer,
                          const struct pvr_g9p_graphics_state *state,
                          const struct g9p_draw_programs *dp, const struct pvr_g9p_draw *draw,
                          VkPrimitiveTopology topology);

VkResult
PVR_PER_ARCH(g9p_cmd_draw)(struct pvr_cmd_buffer *cmd_buffer,
                           const struct pvr_g9p_draw *draw)
{
   struct pvr_cmd_buffer_state *cs = &cmd_buffer->state;
   const struct pvr_graphics_pipeline *pipeline = cs->gfx_pipeline;
   const struct pvr_g9p_graphics_state *state = &pipeline->g9p;
   const struct vk_dynamic_graphics_state *dyn = &cmd_buffer->vk.dynamic_graphics_state;
   const struct pvr_render_pass_info *info = &cs->render_pass_info;
   const uint32_t width = info->render_area.offset.x + info->render_area.extent.width;
   const uint32_t height = info->render_area.offset.y + info->render_area.extent.height;
   struct pvr_pds_upload dma, tex = { 0 };
   struct g9p_draw_programs dp = { .dma = &dma };
   VkResult result;

   result = pvr_arch_cmd_buffer_start_sub_cmd(cmd_buffer, PVR_SUB_CMD_TYPE_GRAPHICS);
   /* the scissor (all of the render area without one), within the render area */
   {
      const VkRect2D *ra = &info->render_area;
      VkRect2D sc = dyn->vp.scissor_count ? dyn->vp.scissors[0] : *ra;
      const int32_t x0 = MAX2(sc.offset.x, ra->offset.x), y0 = MAX2(sc.offset.y, ra->offset.y);
      const int32_t x1 = MIN2((int64_t)sc.offset.x + sc.extent.width, (int64_t)width);
      const int32_t y1 = MIN2((int64_t)sc.offset.y + sc.extent.height, (int64_t)height);

      if (x1 <= x0 || y1 <= y0)
         return VK_SUCCESS; /* nothing passes */
      sc = (VkRect2D){ { x0, y0 }, { x1 - x0, y1 - y0 } };
      dp.scissor = scissor_index(cmd_buffer, sc);
      dp.region[0] = x0 / 32;
      dp.region[1] = y0 / 32;
      dp.region[2] = (x1 - 1) / 32;
      dp.region[3] = (y1 - 1) / 32;
   }
   if (result != VK_SUCCESS)
      return result;
   {
      struct pvr_render_job *job = &cs->current_sub_cmd->gfx.job;

      uint32_t *t = job->g9p_draws.tiles;

      if (job->g9p_draws.num_draws < ARRAY_SIZE(job->g9p_draws.pipelines))
         job->g9p_draws.pipelines[job->g9p_draws.num_draws] = state->id;
      if (!job->g9p_draws.num_draws) {
         job->g9p_draws.render_area = info->render_area;
         memcpy(t, dp.region, sizeof(dp.region));
      } else {
         t[0] = MIN2(t[0], dp.region[0]);
         t[1] = MIN2(t[1], dp.region[1]);
         t[2] = MAX2(t[2], dp.region[2]);
         t[3] = MAX2(t[3], dp.region[3]);
      }
      job->g9p_draws.num_draws++;
   }

   if (cs->push_consts[PVR_STAGE_ALLOCATION_VERTEX_GEOMETRY].dirty) {
      struct pvr_push_constants *pc = &cs->push_consts[PVR_STAGE_ALLOCATION_VERTEX_GEOMETRY];
      struct pvr_suballoc_bo *bo;

      result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, pc->data,
                                                  MAX2(pc->bytes_updated, 4), &bo);
      if (result != VK_SUCCESS)
         return result;
      pc->dev_addr = bo->dev_addr;
      pc->dirty = false;
   }
   result = upload_vs_uniforms(cmd_buffer, state, &dma);
   if (result != VK_SUCCESS)
      return result;
   if (state->num_textures || state->fs_num_buffers) {
      result = upload_textures(cmd_buffer, state, &tex);
      if (result != VK_SUCCESS)
         return result;
      dp.tex = &tex;
      dp.fs_common_dwords = MAX2(state->fs_common_dwords, G9P_TEX_STRIDE * state->num_textures);
   }
   /* the viewport (a0 m0 a1 m1 a2 m2) */
   {
      const VkViewport *vp = &dyn->vp.viewports[0];
      const bool have_vp = dyn->vp.viewport_count > 0;
      const float w = have_vp ? vp->width : width, h = have_vp ? vp->height : height;
      const float x = have_vp ? vp->x : 0, y = have_vp ? vp->y : 0;
      const float zmin = have_vp ? vp->minDepth : 0, zmax = have_vp ? vp->maxDepth : 1;
      const float v[6] = { x + w / 2, w / 2, y + h / 2, h / 2, zmin, zmax - zmin };

      memcpy(dp.vp, v, sizeof(v));
   }
   /* the depth test: only with a depth attachment (D32_SFLOAT, the decoded one) */
   {
      const struct pvr_renderpass_hwsetup_render *hw = info->hw_render;
      const bool has_depth = hw && hw->ds_attach_idx != VK_ATTACHMENT_UNUSED &&
                             vk_format_has_depth(
                                info->attachments[hw->ds_attach_idx]->vk.format);

      dp.depth_test = has_depth && dyn->ds.depth.test_enable;
      dp.depth_write = dp.depth_test && dyn->ds.depth.write_enable;
      dp.depth_compare = dyn->ds.depth.compare_op;
   }
   return emit_draw(cmd_buffer, state, &dp, draw, dyn->ia.primitive_topology);
}

/*
 * A draw's words in the control stream: VDM state (the vertex kick), the
 * vertex shader's common-store DMA, the PPP state blocks and INDEX_LIST
 */
static VkResult
emit_draw(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g9p_graphics_state *state,
          const struct g9p_draw_programs *dp, const struct pvr_g9p_draw *draw,
          VkPrimitiveTopology topology)
{
   struct pvr_device *device = cmd_buffer->device;
   const struct pvr_g9p_gfx_programs *p = &device->g9p.gfx;
   struct pvr_cmd_buffer_state *cs = &cmd_buffer->state;
   const struct pvr_render_pass_info *info = &cs->render_pass_info;
   const uint32_t width = info->render_area.offset.x + info->render_area.extent.width;
   const uint32_t height = info->render_area.offset.y + info->render_area.extent.height;
   const struct pvr_pds_upload *dma = dp->dma, *tex = dp->tex;
   struct pvr_csb *csb = &cs->current_sub_cmd->gfx.control_stream;
   struct pvr_suballoc_bo *ppp_bo;
   uint32_t ppp[64], at = 0, blocks[12][2], nb = 0, pass;
   VkResult result;

   /*
    * PPP blocks, as iPadOS splits them (RENDER.md "Own programs"): each one
    * a header and its words, 16-byte aligned in one buffer
    */
#define PPP_BLOCK(...)                                                        \
   do {                                                                       \
      const uint32_t _w[] = { __VA_ARGS__ };                                  \
      blocks[nb][0] = at;                                                     \
      blocks[nb++][1] = ARRAY_SIZE(_w);                                       \
      memcpy(&ppp[at], _w, sizeof(_w));                                       \
      at += ALIGN_POT(ARRAY_SIZE(_w), 4);                                     \
   } while (0)

   /* 0: varying words (f32_linear = the varyings' components) */
   PPP_BLOCK(0x001c0000, 4 * state->num_varyings, 0, 0);
   /*
    * 1: PDS state pointers 0-1: shader base, texture/uniform program,
    * SIZEINFO1 (uniform data in 16-byte units, varying sizes), SIZEINFO2
    * (the USC common store, 64-byte units), varying base
    */
   PPP_BLOCK(0xc0, state->fs_kick.data_offset, tex ? tex->code_offset : p->halt.code_offset,
             (tex ? MAX2(tex->data_size / 4, 1) : 1) << 23 |
                (state->num_varyings
                    ? state->varying_units << 10 | (2 * state->num_varyings) << 4
                    : 0),
             DIV_ROUND_UP(4 * dp->fs_common_dwords, 64) << 23 |
                (state->uses_discard ? BITFIELD_BIT(14) : 0),
             state->num_varyings ? state->varyings.data_offset : p->halt.code_offset);
   /* 1b: PDS state pointer 3: the texture/buffer program's data */
   if (tex)
      PPP_BLOCK(0x200, tex->data_offset);
   /* 2: output selects: vertex size, rhw */
   PPP_BLOCK(0x00020000, state->vs_outputs << 24 | 0x00010000);
   /*
    * 3: ISP control and front/back words (no culling yet); pass type in
    * ISPA/BA bits 24-26: 1 translucent for shaders that read the tile
    * (blending, `blend` capture), else 4 punch-through for discarding
    * shaders (`discard` capture). The depth compare function is ISPA bits
    * 17-19 in VkCompareOp's order and bit 21 disables the depth write; with
    * a depth test the back words repeat the front ones and ISP_CTL bit 18
    * is clear (`depth` capture: compare less and write, 03c00000 00120f00
    * 0e000000 00120f00 0e000000). Without one: always, no write.
    */
   pass = state->uses_fbfetch ? 1u : state->uses_discard ? 4u : 0u;
   if (dp->depth_test) {
      const uint32_t a = 0x00100f00 | pass << 24 | (uint32_t)dp->depth_compare << 17 |
                         (dp->depth_write ? 0 : BITFIELD_BIT(21));

      PPP_BLOCK(0x1f, 0x03c00000 | (dp->scissor >= 0 ? BITFIELD_BIT(16) : 0), a, 0x0e000000, a,
                0x0e000000);
   } else {
      PPP_BLOCK(0x1f, 0x03c40000 | (dp->scissor >= 0 ? BITFIELD_BIT(16) : 0),
                0x003e0f00 | pass << 24, 0x0e000000, 0x00100f00 | pass << 24, 0);
   }
   /*
    * 4: region clip (32-pixel tiles, inclusive; mode OUTSIDE drops what lies
    * outside) and the viewport (a0 m0 a1 m1 a2 m2). The clip is the draw's
    * scissor, so the tiler bins a scissored draw (Hyprland's damage, its
    * blur passes) into its own tiles only, and a render that loads its
    * target skips the others (ISP_CTL in g9p_frag_stream())
    */
   {
      const bool all = !(dp->region[0] | dp->region[1] | dp->region[2] | dp->region[3]);
      const uint32_t right = all ? DIV_ROUND_UP(width, 32) - 1 : dp->region[2];
      const uint32_t bottom = all ? DIV_ROUND_UP(height, 32) - 1 : dp->region[3];

      PPP_BLOCK(0xc00, 0x80000000u | dp->region[0] << 16 | right, dp->region[1] << 16 | bottom,
                fui(dp->vp[0]), fui(dp->vp[1]), fui(dp->vp[2]), fui(dp->vp[3]),
                fui(dp->vp[4]), fui(dp->vp[5]));
   }
   /* 5: w clamp (1e-20) and PPP control */
   PPP_BLOCK(0x00210000, 0x1e3ce508, 0x4a0);
   /* 6: depth bias / scissor index (ISPCTL bit 16 enables the scissor) */
   PPP_BLOCK(0x20, dp->scissor >= 0 ? (uint32_t)dp->scissor : 0);
   /* 7: stream out (the data's address; HALT after it) */
   PPP_BLOCK(0x00c00000, 1, 0x4022, p->stream_out.data_offset);
#undef PPP_BLOCK

   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, ppp, 4 * at, &ppp_bo);
   if (result != VK_SUCCESS)
      return result;
   for (unsigned i = 0; i < nb; i++)
      dump32("ppp", ppp_bo->dev_addr.addr + 4 * blocks[i][0], &ppp[blocks[i][0]], blocks[i][1]);
   dump_pds("draw dma", dma);

   /*
    * VDM_STATE: vertex PDS code, output dwords, sizes: the vertex shader's
    * registers in bits 10-15, units of 2 (the kick's DOUTU temps do not
    * allocate them; ISA.md "Register allocation"), PDS temps 2, PDS data 1
    */
   /*
    * The draw's words move whole into the next buffer when the stream fills
    * up mid-draw (pvr_csb_buffer_extend(); ~150 draws in one render)
    */
   pvr_csb_set_relocation_mark(csb);
   {
      const uint32_t w[4] = { 0x44002f80, state->vs_kick.code_offset, state->vs_outputs,
                              DIV_ROUND_UP(MAX2(state->vs_regs, 8), 2) << 10 | 2 << 6 | 1 };

      emit_words(csb, w, 4);
   }
   emit_ppp(csb, ppp_bo->dev_addr.addr + 4 * blocks[0][0], blocks[0][1]);
   /*
    * PDS_STATE: the vertex-buffer DMA into the common store. Its size field
    * is not Rogue's 64-byte unit here: 1 hangs with 8 bytes loaded, 2 runs
    * (iPadOS: 2 for 24 bytes); 16-byte units, at least 2, until it is
    * measured further (pvr-tri TRI_PDS_STATE0).
    */
   pvr_csb_emit (csb, VDMCTRL_PDS_STATE0, s0) {
      s0.dm_target = ROGUE_VDMCTRL_DM_TARGET_VDM;
      s0.usc_target = ROGUE_VDMCTRL_USC_TARGET_ALL;
      s0.usc_common_size = MAX2(DIV_ROUND_UP(8 * state->num_buffers, 16), 2);
      s0.pds_data_size = DIV_ROUND_UP(PVR_DW_TO_BYTES(dma->data_size),
                                      ROGUE_VDMCTRL_PDS_STATE0_PDS_DATA_SIZE_UNIT_SIZE);
   }
   {
      const uint32_t w[2] = { dma->data_offset, p->dma.code_offset };

      emit_words(csb, w, 2);
   }
   {
      const uint32_t w[2] = { 0x48002f80, state->vs_kick.data_offset };

      emit_words(csb, w, 2);
   }
   for (unsigned i = 1; i < nb; i++)
      emit_ppp(csb, ppp_bo->dev_addr.addr + 4 * blocks[i][0], blocks[i][1]);

   /* INDEX_LIST: count, instances - 1, index offset, [address], split count */
   {
      uint32_t w[7], n = 0;

      w[n++] = 3u << 29 | BITFIELD_BIT(27) | BITFIELD_BIT(26) | BITFIELD_BIT(25) |
               BITFIELD_BIT(22) | hw_topology(topology) << 8 |
               (draw->indexed ? BITFIELD_BIT(28) | (uint32_t)(draw->index_addr >> 32) : 0) |
               (draw->indexed ? (draw->index_size == 2 ? 1u : 2u) : 2u) << 17;
      if (draw->indexed)
         w[n++] = (uint32_t)draw->index_addr;
      w[n++] = draw->count;
      w[n++] = MAX2(draw->instances, 1) - 1;
      w[n++] = draw->offset;
      w[n++] = 0xaa;
      emit_words(csb, w, n);
   }
   pvr_csb_clear_relocation_mark(csb);
   return csb->status;
}

/*
 * vkCmdClearAttachments: the rectangles of the first colour attachment as a
 * draw of the clear shader (the background object's, the colour through its
 * DMA), two triangles each. Depth/stencil clears do nothing yet (no
 * depth/stencil attachments either).
 */
VkResult
PVR_PER_ARCH(g9p_clear_attachments)(struct pvr_cmd_buffer *cmd_buffer, uint32_t attachment_count,
                                    const VkClearAttachment *attachments, uint32_t rect_count,
                                    const VkClearRect *rects)
{
   const struct pvr_g9p_gfx_programs *p = &cmd_buffer->device->g9p.gfx;
   const struct pvr_render_pass_info *info = &cmd_buffer->state.render_pass_info;
   const float w = info->render_area.offset.x + info->render_area.extent.width;
   const float h = info->render_area.offset.y + info->render_area.extent.height;
   const VkClearAttachment *colour = NULL;
   struct pvr_g9p_graphics_state state = {
      .vs_kick = p->clear_vs_kick,
      .fs_kick = p->clear_kick,
      .vs_outputs = 4,
      .vs_regs = 8,
      .num_buffers = 1,
   };
   struct pvr_pds_upload dma, colour_dma;
   struct g9p_draw_programs dp = {
      .dma = &dma, .tex = &colour_dma, .fs_common_dwords = 4,
      .vp = { w / 2, w / 2, h / 2, h / 2, 0, 1 },
      .scissor = -1,
   };
   struct pvr_suballoc_bo *verts_bo, *table_bo, *colour_bo;
   float *verts;
   VkResult result;

   for (uint32_t i = 0; i < attachment_count; i++) {
      if ((attachments[i].aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) &&
          attachments[i].colorAttachment == 0)
         colour = &attachments[i];
   }
   if (!colour || !rect_count)
      return VK_SUCCESS;

   result = pvr_arch_cmd_buffer_start_sub_cmd(cmd_buffer, PVR_SUB_CMD_TYPE_GRAPHICS);
   if (result != VK_SUCCESS)
      return result;

   /* window coordinates to clip space for the full-target viewport */
   verts = malloc(rect_count * 12 * sizeof(float));
   if (!verts)
      return vk_error(cmd_buffer, VK_ERROR_OUT_OF_HOST_MEMORY);
   for (uint32_t r = 0; r < rect_count; r++) {
      const VkRect2D *rc = &rects[r].rect;
      const float x0 = 2 * rc->offset.x / w - 1, y0 = 2 * rc->offset.y / h - 1;
      const float x1 = 2 * (rc->offset.x + rc->extent.width) / w - 1;
      const float y1 = 2 * (rc->offset.y + rc->extent.height) / h - 1;
      const float v[12] = { x0, y0, x1, y0, x0, y1, x1, y0, x1, y1, x0, y1 };

      memcpy(&verts[12 * r], v, sizeof(v));
   }
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, verts, rect_count * 12 * sizeof(float),
                                               &verts_bo);
   free(verts);
   if (result != VK_SUCCESS)
      return result;

   /* the vertex shader's common store: the vertices' address (a one-entry table) */
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, &verts_bo->dev_addr.addr, 8,
                                               &table_bo);
   if (result != VK_SUCCESS)
      return result;
   {
      const uint32_t data[4] = { table_bo->dev_addr.addr, table_bo->dev_addr.addr >> 32,
                                 doutd_ctrl(0, 2, true), 0 };

      result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, ARRAY_SIZE(data), 16, NULL, 0,
                                              0, 64, &dma);
      if (result != VK_SUCCESS)
         return result;
   }

   /* the clear shader's: the colour, by the DMA program the background uses */
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, colour->clearValue.color.float32,
                                               4 * sizeof(float), &colour_bo);
   if (result != VK_SUCCESS)
      return result;
   {
      const uint32_t data[4] = { colour_bo->dev_addr.addr, colour_bo->dev_addr.addr >> 32,
                                 doutd_ctrl(0, 4, true), 0 };

      result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, ARRAY_SIZE(data), 16, NULL, 0,
                                              0, 64, &colour_dma);
      if (result != VK_SUCCESS)
         return result;
      colour_dma.code_offset = p->dma.code_offset;
   }

   return emit_draw(cmd_buffer, &state, &dp,
                    &(const struct pvr_g9p_draw){ .count = 6 * rect_count, .instances = 1 },
                    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
}

/* --- Image and sampler states ---------------------------------------------- */

/*
 * The texture state of an image view's base level and layer (dwords 0-3;
 * the draw's DMA zeroes 4-7): the image state layout of render targets
 */
void
PVR_PER_ARCH(g9p_pack_image_view)(const struct pvr_image_view *iview, uint64_t words[2])
{
   const struct pvr_image *image = pvr_image_view_get_image(iview);
   const struct pvr_image_plane *plane = pvr_single_plane_const(image);
   const VkComponentMapping *sw = &iview->vk.swizzle;
   uint32_t fmt = g9p_image_format(iview->vk.format);
   const bool linear = image->memlayout == PVR_MEMLAYOUT_LINEAR;
   const uint64_t addr = image->dev_addr.addr + plane->offset +
                         plane->mip_levels[iview->vk.base_mip_level].offset +
                         iview->vk.base_array_layer * plane->layer_size;
   const uint32_t w = iview->vk.extent.width, h = iview->vk.extent.height;
   uint32_t d[4];

   if (!fmt)
      mesa_logw("G9P: texture format %s not decoded yet", vk_Format_to_str(iview->vk.format));
   /*
    * A view swapping red and blue (Zink's BGRA formats on RGBA images):
    * the 8-bit formats' other byte order. Other swizzles are not decoded
    * in the image state yet; alpha ONE (X formats) reads the stored alpha,
    * right for clients that write it opaque.
    */
   if (sw->r == VK_COMPONENT_SWIZZLE_B && sw->b == VK_COMPONENT_SWIZZLE_R &&
       (fmt == 0x0505 || fmt == 0x4405))
      fmt ^= 0x0505 ^ 0x4405;
   else if (sw->r != VK_COMPONENT_SWIZZLE_R || sw->g != VK_COMPONENT_SWIZZLE_G ||
            sw->b != VK_COMPONENT_SWIZZLE_B ||
            (sw->a != VK_COMPONENT_SWIZZLE_A && sw->a != VK_COMPONENT_SWIZZLE_ONE))
      mesa_logw("G9P: texture swizzle %u %u %u %u of %s not decoded yet", sw->r, sw->g, sw->b,
                sw->a, vk_Format_to_str(iview->vk.format));
   d[0] = 0x03000000u | (fmt ? fmt : 0x0505) << 8 | (linear ? 0x02 : 0x0a) | (w - 1) << 27;
   d[1] = (w - 1) >> 5 | (h - 1) << 9;
   d[2] = (uint32_t)(addr >> 1);
   d[3] = (uint32_t)(addr >> 33);
   if (linear)
      d[3] |= (plane->mip_levels[iview->vk.base_mip_level].pitch / 64 - 1) << 15;
   words[0] = d[0] | (uint64_t)d[1] << 32;
   words[1] = d[2] | (uint64_t)d[3] << 32;
}

/*
 * Sampler words: nearest/clamp as captured (000e0000 00c00780). iPadOS's
 * linear-filter-and-repeat sampler (`texfilter` capture) adds dword 0 bits
 * 23, 25 and 29 and dword 1 bit 0; on hardware (vk-tri with coordinates
 * past 1) bit 29 repeats u, dword 1 bit 0 repeats v, bit 23 makes
 * magnification linear (pvr-tri TRI_SAMPLER), which leaves bit 25 for
 * minification. Mirrored and border modes are not decoded: they clamp.
 */
uint64_t
PVR_PER_ARCH(g9p_pack_sampler)(const VkSamplerCreateInfo *info)
{
   uint32_t d0 = 0x000e0000u, d1 = 0x00c00780u;

   if (info->magFilter == VK_FILTER_LINEAR)
      d0 |= BITFIELD_BIT(23);
   if (info->minFilter == VK_FILTER_LINEAR)
      d0 |= BITFIELD_BIT(25);
   if (info->addressModeU == VK_SAMPLER_ADDRESS_MODE_REPEAT)
      d0 |= BITFIELD_BIT(29);
   if (info->addressModeV == VK_SAMPLER_ADDRESS_MODE_REPEAT)
      d1 |= BITFIELD_BIT(0);
   if ((info->addressModeU != VK_SAMPLER_ADDRESS_MODE_REPEAT &&
        info->addressModeU != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE) ||
       (info->addressModeV != VK_SAMPLER_ADDRESS_MODE_REPEAT &&
        info->addressModeV != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE))
      mesa_logw_once("G9P: sampler address modes %u/%u not decoded yet: clamping",
                     info->addressModeU, info->addressModeV);
   if (getenv("G9P_DEBUG"))
      fprintf(stderr, "g9p: sampler filter %u/%u address %u/%u/%u: %08x %08x\n",
              info->magFilter, info->minFilter, info->addressModeU, info->addressModeV,
              info->addressModeW, d0, d1);
   return d0 | (uint64_t)d1 << 32;
}

/* --- Render jobs ---------------------------------------------------------- */

/*
 * Image state of a BGRA8 render target (RENDER.md "A triangle from
 * scratch", "Render target captures"): format bits as captured, width - 1
 * at bit 27, height - 1 at bit 41 (the real size, also for twiddled NPOT
 * images), the address >> 1 with bit 63 set. Linear: layout bits 0x2
 * instead of 0xa, the row pitch in 64-byte units minus one at bit 111, and
 * dword 5 0xc0000000, as iPadOS writes them.
 */

static void
pack_image_state(uint32_t *w, uint64_t addr, uint32_t width, uint32_t height,
                 bool linear, uint32_t pitch, VkFormat format)
{
   const uint32_t fmt = g9p_image_format(format);

   memset(w, 0, 8 * 4);
   w[0] = 0x03000000u | (fmt ? fmt : 0x0505) << 8 | (linear ? 0x02 : 0x0a) | (width - 1) << 27;
   w[1] = (width - 1) >> 5 | (height - 1) << 9;
   w[2] = (uint32_t)(addr >> 1);
   w[3] = 0x80000000u | (uint32_t)(addr >> 33);
   if (linear) {
      w[3] |= (pitch / 64 - 1) << 15;
      w[5] = 0xc0000000u;
   }
}

VkResult
PVR_PER_ARCH(g9p_job_init)(struct pvr_cmd_buffer *cmd_buffer,
                           const struct pvr_renderpass_hwsetup_render *hw_render,
                           struct pvr_render_job *job)
{
   struct pvr_device *device = cmd_buffer->device;
   const struct pvr_g9p_gfx_programs *p = &device->g9p.gfx;
   const struct pvr_render_pass_info *info = &cmd_buffer->state.render_pass_info;
   float clear[4] = { 0, 0, 0, 0 };
   struct pvr_suballoc_bo *bo;
   struct pvr_pds_upload up;
   uint32_t image[8];
   VkResult result;

   memset(&job->g9p, 0, sizeof(job->g9p));
   job->g9p.enabled = true;

   if (hw_render->eot_surface_count) {
      const struct pvr_image_view *iview =
         info->attachments[hw_render->eot_surfaces[0].attachment_idx];
      const struct pvr_image *image_obj = pvr_image_view_get_image(iview);
      const struct pvr_image_plane *plane = pvr_single_plane_const(image_obj);
      const uint64_t addr = image_obj->dev_addr.addr + plane->offset +
                            plane->mip_levels[iview->vk.base_mip_level].offset +
                            iview->vk.base_array_layer * plane->layer_size;

      if (!g9p_image_format(iview->vk.format))
         mesa_logw_once("G9P: render target format %s stored as BGRA8",
                        vk_Format_to_str(iview->vk.format));
      const bool linear = image_obj->memlayout == PVR_MEMLAYOUT_LINEAR;
      const uint32_t pitch = plane->mip_levels[iview->vk.base_mip_level].pitch;

      if (linear && pitch % 64)
         mesa_logw("G9P: linear render target pitch %u is not a multiple of 64", pitch);
      job->g9p.target_addr = addr;
      job->g9p.target_pitch = pitch;
      job->g9p.target_linear = linear;
      pack_image_state(image, addr, iview->vk.extent.width, iview->vk.extent.height, linear,
                       pitch, iview->vk.format);
      job->g9p.width = iview->vk.extent.width;
      job->g9p.height = iview->vk.extent.height;
   } else {
      memset(image, 0, sizeof(image));
   }
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, image, 7 * 4, &bo);
   if (result != VK_SUCCESS)
      return result;
   {
      const uint32_t data[8] = { bo->dev_addr.addr, bo->dev_addr.addr >> 32,
                                 doutd_ctrl(0, 7, true), 0, usc_offset(device, p->eot_usc),
                                 usc_temps(8), 0, 0 };

      result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, ARRAY_SIZE(data), 16, NULL,
                                              0, 0, 64, &up);
      if (result != VK_SUCCESS)
         return result;
   }
   job->view_state.view[0].pds_pixel_event_data_offset = up.data_offset;
   job->g9p.eot_code = p->eot.code_offset;

   /*
    * Background object. Load op load: the load shader with the target's own
    * texture state (dwords 0-7), a nearest/clamp sampler (8-9) and its
    * coordinate scale (12-13) in the common store. Otherwise the clear
    * colour of the first colour attachment (dwords 0-3) and the clear
    * shader; load ops other than these clear to transparent black.
    */
   const bool load = hw_render->eot_surface_count && hw_render->color_init_count &&
                     hw_render->color_init[0].op == VK_ATTACHMENT_LOAD_OP_LOAD;
   uint32_t bg[16] = { 0 };
   unsigned bg_dwords = 4;

   if (load) {
      const struct pvr_image_view *iview =
         info->attachments[hw_render->eot_surfaces[0].attachment_idx];
      uint64_t tex[2];

      PVR_PER_ARCH(g9p_pack_image_view)(iview, tex);
      memcpy(bg, tex, sizeof(tex));
      bg[8] = 0x000e0000;
      bg[9] = 0x00c00780;
      bg[12] = fui(1.0f / (16 * iview->vk.extent.width));
      bg[13] = fui(1.0f / (16 * iview->vk.extent.height));
      bg_dwords = 16;
   } else if (hw_render->color_init_count &&
              hw_render->color_init[0].op == VK_ATTACHMENT_LOAD_OP_CLEAR &&
              hw_render->color_init[0].index < info->clear_value_count) {
      memcpy(clear, info->clear_values[hw_render->color_init[0].index].color.float32,
             sizeof(clear));
   }
   job->g9p.load = load;
   job->g9p.load_op = !hw_render->eot_surface_count ? VK_ATTACHMENT_LOAD_OP_CLEAR
                      : hw_render->color_init_count ? (int)hw_render->color_init[0].op
                                                    : G9P_LOAD_OP_NONE;
   if (!load)
      memcpy(bg, clear, sizeof(clear));
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, bg, 4 * bg_dwords, &bo);
   if (result != VK_SUCCESS)
      return result;
   {
      const uint32_t data[4] = { bo->dev_addr.addr, bo->dev_addr.addr >> 32,
                                 doutd_ctrl(0, bg_dwords, true), 0 };

      result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, ARRAY_SIZE(data), 16, NULL,
                                              0, 0, 64, &up);
      if (result != VK_SUCCESS)
         return result;
   }
   job->g9p.bgnd[0] = (uint64_t)p->dma.code_offset << 32 |
                      (load ? p->load_kick.data_offset : p->clear_kick.data_offset);
   job->g9p.bgnd[1] = 0;
   job->g9p.bgnd[2] = up.data_offset;
   job->g9p.bgnd[3] = 0x0080000000800000ull;
   dump32("image state", 0, image, 8);
   dump_pds("clear dma", &up);
   return VK_SUCCESS;
}

/* --- Job streams (the kernel's Apple extension, as pvr-tri builds them) --- */

struct g9p_stream {
   uint8_t *buf;
   uint32_t len, size;
};

static void
put(struct g9p_stream *s, const void *p, uint32_t size, uint32_t align)
{
   s->len = ALIGN_POT(s->len, align);
   assert(s->len + size <= s->size);
   memcpy(s->buf + s->len, p, size);
   s->len += size;
}

static void
put32(struct g9p_stream *s, uint32_t v)
{
   put(s, &v, 4, 4);
}

static void
put64(struct g9p_stream *s, uint64_t v)
{
   put(s, &v, 8, 8);
}

#define G9P_EXTHDR_APPLE (BITFIELD_BIT(29) | 1u)

/* A context store/resume pair as the registers take it: {sizes, code, data << 32 | sizes} */
static uint64_t
ctx_pds(uint64_t sizes, uint32_t data)
{
   return (uint64_t)data << 32 | sizes;
}

uint32_t
PVR_PER_ARCH(g9p_geom_stream)(const struct pvr_device *device,
                              const struct pvr_render_job *job, uint8_t *buf, uint32_t size)
{
   const struct pvr_g9p_gfx_programs *p = &device->g9p.gfx;
   const uint32_t halt = p->halt.code_offset, data = p->stream_out.data_offset;
   const uint64_t ctx = p->ctx_state->dev_addr.addr;
   struct g9p_stream s = { .buf = buf, .len = 8, .size = size };

   put64(&s, job->ctrl_stream_addr.addr); /* VDM control stream base */
   put64(&s, 0);                          /* border colour table */
   put32(&s, 2);                          /* PPP_CTRL */
   put32(&s, job->geometry_terminate ? 0x20000 : 0); /* TE_PSG: complete on terminate */
   put32(&s, 0);
   put32(&s, 0);
   *(uint32_t *)buf = s.len;

   put32(&s, G9P_EXTHDR_APPLE);
   put64(&s, 0x1e3ce508); /* 0xc90 */
   put64(&s, ctx + 0x200);
   put64(&s, ctx);
   /* context store / resume: HALT (they run only on a context switch) */
   put64(&s, 0x0003000524001002ull);
   put64(&s, halt);
   put64(&s, ctx_pds(0x4820, data));
   put64(&s, 0x0003002121001002ull);
   put64(&s, halt);
   put64(&s, ctx_pds(0x4820, data));
   put64(&s, p->vdm_end.data_offset); /* 0x470 */
   put64(&s, 0);                      /* 0x418 */
   put64(&s, ctx_pds(0x20000001, data | 5));
   put64(&s, halt);
   put64(&s, ctx_pds(0x20000001, data | 1));
   put64(&s, halt);
   put64(&s, 0);
   put64(&s, 0);
   for (unsigned i = 0; i < 6; i++)
      put64(&s, 0); /* cluster */
   put32(&s, 0x88); /* 0xc78 */
   dump32("geom stream", 0, (const uint32_t *)buf, s.len / 4);
   return s.len;
}

/*
 * ZLSCTL of a render with a D32_SFLOAT depth attachment: Rogue's fields
 * (enables, F32Z format) and the extent in tiles; 0 without one. The
 * hardware keeps the stored depth in its own (compressed) layout, which
 * its loads read back: a uniform 64x64 clear stores 96 of 4096 words.
 */
static uint64_t
g9p_zlsctl(const struct pvr_render_job *job)
{
   uint64_t v = 0;

   if (!job->has_depth_attachment)
      return 0;
   if (job->ds.load.d)
      v |= BITFIELD64_BIT(15) | BITFIELD64_BIT(1); /* zloaden, forcezload */
   if (job->ds.store.d)
      v |= BITFIELD64_BIT(19) | BITFIELD64_BIT(2); /* zstoreen, forcezstore */
   /*
    * Not twiddled: Rogue's storetwiddled (bit 17, set by iPadOS) hangs the
    * A10's fragment job here; loads and stores agree on the linear layout
    */
   v |= (uint64_t)(DIV_ROUND_UP(job->ds.physical_extent.width, 32) - 1) << 3;
   v |= (uint64_t)(DIV_ROUND_UP(job->ds.physical_extent.height, 32) - 1) << 27;
   return v;
}

uint32_t
PVR_PER_ARCH(g9p_frag_stream)(const struct pvr_device *device,
                              const struct pvr_render_job *job, uint8_t *buf, uint32_t size)
{
   const uint32_t w = MAX2(job->g9p.width, 32), h = MAX2(job->g9p.height, 32);
   struct g9p_stream s = { .buf = buf, .len = 8, .size = size };
   const float t15 = tanf(15.0f * M_PI / 180.0f), t60 = tanf(60.0f * M_PI / 180.0f);

   /*
    * ISP scissor table base, then the dbias and occlusion query bases,
    * ZLSCTL and the depth and stencil load/store bases (the kernel puts the
    * depth base into both 0xf50 and 0xf60). ZLSCTL as Rogue's (`depth`
    * capture: 0x000a2004, store twiddled and load twiddled, Z store and its
    * force bit, format F32Z) with the extent in 32-pixel tiles minus one,
    * which iPadOS's userspace passes as pixels (0x001f803f for 64x64).
    */
   put64(&s, job->scissor_table_addr.addr);
   put64(&s, 0);
   put64(&s, 0);
   put64(&s, g9p_zlsctl(job));
   put64(&s, job->has_depth_attachment ? job->ds.addr.addr : 0);
   /*
    * The stencil base: none yet (stencil formats are not decoded). The
    * kernel's Apple record takes the depth base for 0xf50/0xf58 (Z load and
    * store) and this one for 0xf60/0xf68; a stencil base set here breaks
    * the depth load even with stencil off in ZLSCTL.
    */
   put64(&s, 0);
   for (unsigned i = 0; i < 8 * 3; i++)
      put64(&s, 0); /* PBE words: in the image state instead */
   put64(&s, 0);
   for (unsigned i = 0; i < 6; i++)
      put64(&s, 0); /* Rogue background words: in the extension instead */
   for (unsigned i = 0; i < 8; i++)
      put32(&s, 0); /* USC clear registers */
   put32(&s, 0);
   put32(&s, job->has_depth_attachment ? fui(job->ds_clear_value.depth)
                                        : 0x3f800000); /* ISP_BGOBJDEPTH: the depth clear */
   put32(&s, 0x300);      /* ISP_BGOBJVALS */
   put32(&s, 0);
   /*
    * ISP_CTL: bit 17 (Rogue's process_empty_tiles) runs the background object
    * on tiles without geometry too; without it a render's clear reaches
    * only the tiles a primitive touched (pvr-tri TRI_NODRAW). A render that
    * loads its target leaves it off: tiles no draw reaches (outside every
    * draw's region clip) are neither loaded nor stored and keep their
    * memory, which is the load's result anyway. So does a render whose
    * target has no load op (don't care): its old contents are undefined,
    * and Hyprland's blur renders this way over the whole screen to draw a
    * small damaged part. G9P_ALL_TILES=1 keeps it on.
    */
   put32(&s, (job->g9p.load || job->g9p.load_op == G9P_LOAD_OP_NONE) && !getenv("G9P_ALL_TILES")
                ? 0x21000f
                : 0x23000f);
   put32(&s, 0x402);    /* EVENT_PIXEL_PDS_INFO */
   put32(&s, 0);
   put32(&s, job->view_state.view[0].pds_pixel_event_data_offset);
   put32(&s, 0);
   put32(&s, 0);
   *(uint32_t *)buf = s.len;

   put32(&s, G9P_EXTHDR_APPLE);
   put64(&s, 0x88); /* 0xfa8 sample positions */
   for (unsigned i = 0; i < 4; i++)
      put64(&s, job->g9p.bgnd[i]);
   for (unsigned i = 0; i < 6; i++)
      put64(&s, 0);     /* cluster 0 */
   for (unsigned i = 0; i < 6; i++)
      put64(&s, 0x1d3); /* cluster 1 */
   put64(&s, 0);        /* 0xfd0 */
   put32(&s, job->g9p.eot_code);
   /*
    * ISP_RGN: the dwords of one macrotile's region array: tiles per
    * macrotile (32x32 tiles, macrotiles of round_up(ceil(tiles / 4), 4) as
    * the kernel's HWRT dataset) x the region header size. iPadOS's 0x14 is
    * 4x4 tiles of 5 bytes; a fixed 0x14 hangs above 512 pixels.
    */
   {
      const uint32_t mx = ALIGN_POT(DIV_ROUND_UP(DIV_ROUND_UP(w, 32), 4), 4);
      const uint32_t my = ALIGN_POT(DIV_ROUND_UP(DIV_ROUND_UP(h, 32), 4), 4);

      put32(&s, DIV_ROUND_UP(mx * my *
                                rogue_get_region_header_size(&device->pdevice->dev_info),
                             4));
   }
   put32(&s, 0x1f); /* 0x6d0 */
   put32(&s, 0x408);
   /*
    * 0xfd8: ISP_ZLS_PIXELS, the depth buffer's size (width - 1, height - 1
    * << 15, `depth` capture 0x001f803f for 64x64; the store writes nothing
    * without it); iOS 10's 4 without depth
    */
   put32(&s, job->has_depth_attachment
                ? (job->ds.physical_extent.width - 1) | (job->ds.physical_extent.height - 1) << 15
                : 4);
   put32(&s, 0xf9);
   /* line rasterization (iOS 10): tan 15° / size, tan 60° / size, 16 x width */
   put32(&s, fui(t15 / w));
   put32(&s, fui(t15 / h));
   put32(&s, fui(t60 / w));
   put32(&s, fui(t60 / h));
   put32(&s, fui(16.0f * w));
   put32(&s, fui(t15 / w));
   dump32("frag stream", 0, (const uint32_t *)buf, s.len / 4);
   return s.len;
}
