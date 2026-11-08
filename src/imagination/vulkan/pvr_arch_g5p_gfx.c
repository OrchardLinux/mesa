/* SPDX-License-Identifier: MIT */
/*
 * Graphics on Apple's A9 GPU (G5P, has_apple_g5p), in the shape of the
 * G9P driver (pvr_arch_g9p_gfx.c): pipelines compiled by the G5P backend
 * (src/imagination/g5p, through pvr_g5p_nir.c), and draws as Rogue VDM/PPP
 * words with pdsc PDS programs (Apple's DOUT form). Every word is derived as
 * the A9 ICD's recorder derives it (g5p_record.c of the route-b tree:
 * place_pipeline(), g5p_job_draw()); its template, prologue, epilogue and
 * static arena are not used, so a draw is only what pvr_* records here plus
 * the render job of pvr_arch_g5p.c. Compiled only with
 * -Dimagination-g5p-experimental=true.
 *
 * Rung A of step 7 (docs: orchard-mesa-pr-plan.md "Step 7 design"): one
 * linear colour target, no depth/stencil, one vertex buffer, non-indexed
 * single-instance draws, scissors and partial render areas as the ICD clips
 * them (region clip, ISP scissor table); blending as
 * the G5P backend compiles it (pvr_g5p_blend.h); the fragment program's
 * texture, sampler, image-size and blend-constant uploads (no push
 * constants or buffers yet). Everything else is refused by name.
 */

#include "pvr_g5p_texel_buffer.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_log.h"
#include "vk_shader_module.h"

#include "util/ralloc.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"

#include "pco_g5p_general.h"
#include "pdsc.h"
#include "pdsc_internal.h"
#include "pdsc_isa_pack.h"
#include "pdsc_ops.h"
#include "pvr_bo.h"
#include "pvr_buffer.h"
#include "pvr_descriptor_set.h"
#include "pvr_cmd_buffer.h"
#include "pvr_csb.h"
#include "pvr_device.h"
#include "pvr_framebuffer.h"
#include "pvr_g5p.h"
#include "pvr_g5p_blend.h"
#include "pvr_g5p_formats.h"
#include "pvr_g5p_mrt.h"
#include "pvr_g5p_tex.h"
#include "pvr_g5p_tex_encode.h"
#include "pvr_g5p_upload.h"
#include "pvr_g5p_xfb.h"
#include "pvr_g5p_ds.h"
#include "pvr_g5p_msaa.h"
#include "pvr_g5p_vertex.h"
#include "pvr_image.h"
#include "pvr_pass.h"
#include "pvr_pipeline.h"

/* Named register values of a draw (the A9 ICD's g5p_record.c, from the
 * captured round6/round16 user draws; the G9P carries the same) */
/* VDM_STATE5: PDS temps 2, PDS data and unified store set per draw */
#define G5P_VS_STATE5 0x1082u
/* VDM_STATE0 cam_size of a vertex up to 32 dwords (output dwords + 5) */
#define G5P_VDM_CAM_SIZE 95
/* TA W clamp of every captured draw (~1e-20f; not Mesa's 1e-13f) */
#define G5P_WCLAMP 0x1e3ce508u
/* STATE_STREAM_OUT1: PDS temps 2, PDS data 2, and bit 14, which has no
 * csbgen field (the A9 captures and G9P's 0x4022) */
#define G5P_STREAM_OUT1 0x4022u
/* Region clip granularity in pixels */
#define G5P_TILE 32

/* PVR_G5P_WS_LOG: the words of every pipeline and draw (step7-compare.py reads them) */
static bool
g5p_logging(void)
{
   static int on = -1;
   if (on < 0)
      on = getenv("PVR_G5P_WS_LOG") != NULL;
   return on;
}

static void
g5p_dump(const char *who, const char *what, uint64_t addr, const uint32_t *w, unsigned n)
{
   if (!g5p_logging())
      return;
   fprintf(stderr, "g5p %s: %-16s @%#" PRIx64 ":", who, what, addr);
   for (unsigned i = 0; i < n; i++)
      fprintf(stderr, " %08x", w[i]);
   fprintf(stderr, "\n");
}

/* USC programs have byte sizes and need not end on a dword boundary. */
static void
g5p_dump_code(const char *what, uint64_t addr, const void *code, unsigned size)
{
   if (!g5p_logging())
      return;

   const uint8_t *bytes = code;
   fprintf(stderr, "g5p pipeline: %-16s @%#" PRIx64 ":", what, addr);
   for (unsigned offset = 0; offset < size;) {
      const unsigned count = MIN2(size - offset, sizeof(uint32_t));
      uint32_t word = 0;
      memcpy(&word, bytes + offset, count);
      fprintf(stderr, " %08x", word);
      offset += count;
   }
   fprintf(stderr, "\n");
}

/* A refusal, by name, on stderr too (vk_errorf() is silent in release builds) */
#define G5P_REFUSE(obj, ...)                                                                                       \
   (fprintf(stderr, "G5P refuses: " __VA_ARGS__), fprintf(stderr, "\n"),                                          \
    vk_errorf((obj), VK_ERROR_FEATURE_NOT_PRESENT, "G5P: " __VA_ARGS__))

/* ---- PDS programs (pdsc, Apple's DOUT form: the A9 ICD's) ---- */

static unsigned
pds_pack(pdsc_program *p, uint32_t *dest)
{
   const unsigned n = util_dynarray_num_elements(&p->instrs, pdsc_instr);
   for (unsigned i = 0; i < n; i++)
      dest[i] = pdsc_instr_pack(p, i, util_dynarray_element(&p->instrs, pdsc_instr, i)->op);
   return n;
}

static uint64_t
doutu_src0(uint32_t exe_off, unsigned temps)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-doutu");
   struct pdsc_doutu_src0 s = { .exe_off = exe_off, .temps = temps };
   const uint64_t v = pdsc_doutu_src0_pack(p, &s);
   ralloc_free(p);
   return v;
}

/* DOUTW control: one 32-bit half of t0:t1 into unified register `reg` */
static uint32_t
doutw_control(unsigned reg, bool upper, bool last)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-doutw");
   struct pdsc_doutw_src1 s = { .last_issue = last, .store_dest = PDSC_STORE_DEST_UNIFIED, .ao = reg,
                                .bsize = upper ? PDSC_DOUTW_BSIZE_UPPER : PDSC_DOUTW_BSIZE_LOWER };
   const uint32_t v = pdsc_doutw_src1_pack(p, &s);
   ralloc_free(p);
   return v;
}

static uint64_t
ddmad_src3(unsigned dwords, unsigned ao, bool last)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-ddmad");
   struct pdsc_ddmad_src3 s = { .last_issue = last, .bsize = dwords, .ao = ao };
   const uint64_t v = pdsc_ddmad_src3_pack(p, &s);
   ralloc_free(p);
   return v;
}

/* DOUTD control: `dwords` from the source address into common-store dword
 * `common` */
static uint32_t
doutd_control(unsigned common, unsigned dwords, bool last)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-doutd");
   struct pdsc_doutd_src1 s = { .last_issue = last, .store_dest = PDSC_STORE_DEST_COMMON, .ao = common,
                                .bsize = dwords };
   const uint32_t v = pdsc_doutd_src1_pack(p, &s);
   ralloc_free(p);
   return v;
}

/*
 * The fragment program's uniform/texture loader (the ICD's loader_program(),
 * iOS tex capture): DOUTD c(2i):c(2i+1) -> common store with the control word
 * in c(2n+2i), the last one ends, then HALT. Its data, per draw, is behind
 * the UNIFORMDATABASE pointer.
 */
static unsigned
loader_program(uint32_t *dest, unsigned n)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-fs-loader");
   for (unsigned i = 0; i < n; i++)
      pdsc_dout(p, pdsc_ref_const64(2 * i), pdsc_ref_const32(2 * n + 2 * i), pdsc_ref_null(),
                .dstdout = PDSC_DSTDOUT_D, .end = i == n - 1);
   pdsc_halt(p);
   const unsigned dwords = pds_pack(p, dest);
   ralloc_free(p);
   return dwords;
}

/*
 * Vertex fetch: per fetch f, DDMAD stride * t0 + base -> unified store, then
 * DOUTU (CC), HALT. Data per fetch: c(8f) base, c(8f+4) control, c(8f+6)
 * stride; the DOUTU word at c2:3 (c0:1 without fetches). The vertex and
 * instance indices are DOUTWs of t0:t1, their controls in a 16-byte unit
 * after the fetches. One fetch is the captured `e0180002 fc00000a d3000000`;
 * the LAST bit is set on the final DDMAD and the final DOUTW, as Apple does.
 */
static unsigned
fetch_doutu(unsigned fetches)
{
   return fetches ? 2 : 0;
}

static unsigned
fetch_ids(unsigned fetches)
{
   return fetches ? 8 * fetches : 2;
}

/* The base instance (Apple's g3b-basevertex: add32 t1, cN, t1 before the
 * DDMADs and the DOUTWs; t1 is the 0-based instance): fetch 0's free dword,
 * or after the index controls in a second unit without fetches */
static unsigned
fetch_base_instance(unsigned fetches)
{
   return fetches ? 7 : 4;
}

/* The distinct divisors > 1 of the fetches, div[k] (at most
 * PVR_G5P_MAX_DIVISORS, the layout's rule); slot[f]: fetch f's k */
static unsigned
fetch_divisors(const struct pvr_g5p_pipeline *g, uint32_t *div, unsigned *slot)
{
   unsigned n = 0;
   for (unsigned f = 0; f < g->fetch_count; f++) {
      if (g->fetch_list[f].rate != PVR_G5P_FETCH_INSTANCE_DIVISOR)
         continue;
      unsigned k = 0;
      while (k < n && div[k] != g->fetch_list[f].divisor)
         k++;
      if (k == n)
         div[n++] = g->fetch_list[f].divisor;
      slot[f] = k;
   }
   return n;
}

/* The divisors' multipliers: a unit after the index controls */
static unsigned
fetch_multipliers(unsigned fetches, unsigned id_mask)
{
   return 8 * fetches + (id_mask ? 4 : 0);
}

/* i / d = (i * m) >> (31 + s) for every 31-bit i, s = ceil(log2(d)) (upstream
 * Rogue's pvr_pds_vertex_shader() divide) */
static unsigned
divisor_shift(uint32_t d)
{
   return util_last_bit(d - 1);
}

static uint32_t
divisor_multiplier(uint32_t d)
{
   return (uint32_t)(((UINT64_C(1) << (31 + divisor_shift(d))) + d - 1) / d);
}

static unsigned
fetch_data_bytes(const struct pvr_g5p_pipeline *g, bool base_instance)
{
   uint32_t div[PVR_G5P_MAX_DIVISORS];
   unsigned slot[PVR_G5P_MAX_FETCH];
   const unsigned fetches = g->fetch_count;
   return fetches ? 32 * fetches + (g->vs_id_mask ? 16 : 0) + (fetch_divisors(g, div, slot) ? 16 : 0)
                  : base_instance ? 32 : 16;
}

/* The program reads the instance: its index, or a per-instance fetch */
static bool
fetch_reads_instance(const struct pvr_g5p_pipeline *g)
{
   bool r = g->vs_id_mask & 2;
   for (unsigned f = 0; f < g->fetch_count; f++)
      r |= g->fetch_list[f].rate == PVR_G5P_FETCH_INSTANCE || g->fetch_list[f].rate == PVR_G5P_FETCH_INSTANCE_DIVISOR;
   return r;
}

static unsigned
fetch_program(uint32_t *dest, const struct pvr_g5p_pipeline *g, bool base_instance, bool indirect)
{
   const struct pvr_g5p_fetch *list = g->fetch_list;
   const unsigned fetches = g->fetch_count, id_mask = g->vs_id_mask;
   uint32_t div[PVR_G5P_MAX_DIVISORS];
   unsigned slot[PVR_G5P_MAX_FETCH];
   const unsigned ndiv = fetch_divisors(g, div, slot);
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-vertex-fetch");
   /* A9's indirect base-instance variant reads persistent temporary 1
    * (encoded selector 0xc1), not a CPU-specialized data constant. */
   const pdsc_ref base = indirect ? pdsc_ref_ptemp32(1) : pdsc_ref_const32(fetch_base_instance(fetches));
   /* divisor k: t(2 + 2k) = t1 / d, then plus the base instance; t1 is
    * still the 0-based instance here. Apple's forms (our decode of
    * AGX::PDSEncoderGen1<>::EncodeVertexPDSProgram's step-rate path, which
    * divides into t2): a power of two is one shr32 by log2(d); another d is
    * limm t2 = 0, limm t3 = 0, mad t2:t3 = c(m) * t1 + t2:t3, one shr64 of
    * the pair by 31 + s; the base is add32 t2, c(base), t2. pdsc temp64
    * refs take the pair's first 32-bit temp. */
   for (unsigned k = 0; k < ndiv; k++) {
      const unsigned t = 2 + 2 * k;
      if (util_is_power_of_two_nonzero(div[k])) {
         pdsc_shr32(p, pdsc_ref_temp32(t), pdsc_ref_temp32(1), util_logbase2(div[k]));
      } else {
         const pdsc_ref pair = pdsc_ref_temp64(t);
         pdsc_limm(p, pdsc_ref_temp32(t), pdsc_ref_imm(0));
         pdsc_limm(p, pdsc_ref_temp32(t + 1), pdsc_ref_imm(0));
         pdsc_mad(p, pair, pdsc_ref_const32(fetch_multipliers(fetches, id_mask) + k), pdsc_ref_temp32(1), pair);
         pdsc_shr64(p, pair, pair, 31 + divisor_shift(div[k]));
      }
      if (base_instance)
         pdsc_add32(p, pdsc_ref_temp32(t), base, pdsc_ref_temp32(t));
   }
   if (base_instance)
      pdsc_add32(p, pdsc_ref_temp32(1), base, pdsc_ref_temp32(1));
   for (unsigned f = 0; f < fetches; f++) {
      const unsigned c = f * 8;
      /* a per-instance fetch multiplies by the instance (Apple's s1=t1), a
       * divided one by its divisor's pair */
      const unsigned t = list[f].rate == PVR_G5P_FETCH_INSTANCE           ? 1
                         : list[f].rate == PVR_G5P_FETCH_INSTANCE_DIVISOR ? 2 + 2 * slot[f]
                                                                          : 0;
      const pdsc_ref index = indirect && list[f].rate == PVR_G5P_FETCH_INSTANCE_DIVISOR0 ?
                               pdsc_ref_ptemp32(1) : pdsc_ref_temp32(t);
      pdsc_ddmad(p, pdsc_ref_const32(c + 6), index, pdsc_ref_const64(c), pdsc_ref_const64(c + 4));
   }
   unsigned k = 0;
   u_foreach_bit (i, id_mask)
      pdsc_dout(p, pdsc_ref_temp64(0), pdsc_ref_const32(fetch_ids(fetches) + k++), pdsc_ref_null(),
                .dstdout = PDSC_DSTDOUT_W);
   pdsc_dout(p, pdsc_ref_const64(fetch_doutu(fetches)), pdsc_ref_null(), pdsc_ref_null(), .dstdout = PDSC_DSTDOUT_U,
             .end = true, .cc = true);
   pdsc_halt(p);
   const unsigned n = pds_pack(p, dest);
   ralloc_free(p);
   return n;
}

/* The fragment launch: one DOUTU.end from c0:1, no CC and no HALT (the
 * captured form) */
static unsigned
launch_program(uint32_t *dest)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-fragment");
   pdsc_dout(p, pdsc_ref_const64(0), pdsc_ref_null(), pdsc_ref_null(), .dstdout = PDSC_DSTDOUT_U, .end = true);
   const unsigned n = pds_pack(p, dest);
   ralloc_free(p);
   return n;
}

/*
 * The coefficient (varying) program, as Apple's PDSEncoderGen1<G5>::
 * EncodeFragmentVaryingsPDSProgram builds it: one DOUTI per item, item i's
 * source in c(2i), the data first and padded to 16 bytes (the code starts
 * at VARYINGBASE + 16 * PDS_VARYINGSIZE). Items: W at coefficient set 0
 * when some varying is perspective-correct, then every varying the fragment
 * program reads (dest: its first set; offset: its vertex output dword in
 * the TA vertex, which drops x, y, z: dword d is TA dword d - 3, in 16-bit
 * units). Returns the data dwords; *code_dwords the code's.
 */
static unsigned
varying_program(const struct pco_g5p_program *fs, const struct pco_g5p_program *vs, uint32_t *data,
                uint32_t *code, unsigned *code_dwords)
{
   pdsc_init();
   pdsc_program *p = pdsc_program_create(NULL, PDSC_FEATURE_DOUT_EXT, "g5p-varying");
   struct pdsc_douti_src0 item[1 + PCO_G5P_MAX_FS_VARYINGS];
   unsigned items = 0;
   if (fs->w_set)
      item[items++] = (struct pdsc_douti_src0){ .shademodel = PDSC_DOUTI_SHADEMODEL_GOURAUD };
   for (unsigned i = 0; i < fs->num_varyings; i++) {
      const struct pco_g5p_varying *v = &fs->vary[i];
      const unsigned off = 2 * (vs->vout_dword[v->location] - 3);
      const bool flat = v->interp == PCO_G5P_INTERP_FLAT;
      item[items++] = (struct pdsc_douti_src0){
         .dest = v->set,
         /* PPP_CTRL flatshade_vtx is VERTEX_0 */
         .shademodel = flat ? PDSC_DOUTI_SHADEMODEL_FLAT_VERTEX0 : PDSC_DOUTI_SHADEMODEL_GOURAUD,
         .size = PDSC_DOUTI_SIZE_4D,
         .perspective = v->interp == PCO_G5P_INTERP_SMOOTH,
         .f32_offset = off,
         .f16_offset = off,
      };
   }
   item[items - 1].last_issue = true;
   for (unsigned i = 0; i < items; i++) {
      const uint64_t w = pdsc_douti_src0_pack(p, &item[i]);
      data[2 * i] = (uint32_t)w;
      data[2 * i + 1] = (uint32_t)(w >> 32);
   }
   const unsigned data_dwords = ALIGN_POT(2 * items, 4);
   memset(data + 2 * items, 0, 4 * (data_dwords - 2 * items));
   for (unsigned i = 0; i < items; i++)
      pdsc_dout(p, pdsc_ref_const64(i * 2), pdsc_ref_null(), pdsc_ref_null(), .dstdout = PDSC_DSTDOUT_I,
                .end = i == items - 1);
   *code_dwords = pds_pack(p, code);
   ralloc_free(p);
   return data_dwords;
}

/* USC offsets are from the USC heap base (the A9 kernel's usc_base == pds_base) */
static uint32_t
usc_off(const struct pvr_device *device, const struct pvr_suballoc_bo *bo)
{
   return (uint32_t)(bo->dev_addr.addr - device->heaps.usc_heap->base_addr.addr);
}

/* ---- pipelines ---- */

static bool
g5p_native_bresenham_line(const struct vk_graphics_pipeline_state *s)
{
   const char *mode = getenv("PVR_G5P_BRESENHAM");
   return (!mode || strcmp(mode, "0")) && s->rs && s->ia &&
      (s->ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
       s->ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP) &&
      s->rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM;
}

/* What rung A draws: refused by name otherwise (the ICD's validate_state()
 * with rung A's limits) */
static VkResult
g5p_check_state(struct pvr_device *device, const VkGraphicsPipelineCreateInfo *create_info,
                const struct vk_graphics_pipeline_state *s, struct pco_g5p_fs_key *key)
{
   /* the subpass's attachments: from the render pass object (pvr_* hands the
    * runtime only their aspects), or the dynamic rendering state */
   unsigned colors;
   VkFormat cf;
   VkFormat cfs[PVR_G5P_MRT_MAX_COLORS] = { VK_FORMAT_UNDEFINED };
   bool ds;
   if (create_info->renderPass) {
      VK_FROM_HANDLE(pvr_render_pass, pass, create_info->renderPass);
      const struct pvr_render_subpass *sub = &pass->subpasses[create_info->subpass];
      colors = 0;
      cf = VK_FORMAT_UNDEFINED;
      for (unsigned i = 0; i < sub->color_count; i++) {
         if (sub->color_attachments[i] == VK_ATTACHMENT_UNUSED)
            continue;
         if (colors < PVR_G5P_MRT_MAX_COLORS)
            cfs[colors] = pass->attachments[sub->color_attachments[i]].vk_format;
         if (!colors++)
            cf = pass->attachments[sub->color_attachments[i]].vk_format;
      }
      ds = sub->depth_stencil_attachment != VK_ATTACHMENT_UNUSED;
   } else if (s->rp) {
      colors = s->rp->color_attachment_count;
      cf = colors ? s->rp->color_attachment_formats[0] : VK_FORMAT_UNDEFINED;
      for (unsigned i = 0; i < MIN2(colors, PVR_G5P_MRT_MAX_COLORS); i++)
         cfs[i] = s->rp->color_attachment_formats[i];
      ds = s->rp->depth_attachment_format != VK_FORMAT_UNDEFINED ||
           s->rp->stencil_attachment_format != VK_FORMAT_UNDEFINED;
   } else {
      return G5P_REFUSE(device, "no render pass state");
   }

   if (s->shader_stages != (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT) || s->ts)
      return G5P_REFUSE(device, "shader stages %#x (a vertex and a fragment shader)", s->shader_stages);
   if (s->rs && (s->rs->depth_clamp_enable ||
                 (s->rs->polygon_mode != VK_POLYGON_MODE_FILL && !pvr_g5p_polygon_mode_enabled())))
      return G5P_REFUSE(device, "depth clamp or polygon mode %d", s->rs->polygon_mode);
   /* line and point fill (default; PVR_G5P_POLYGON_MODE=0 off): the qualified scope only */
   if (s->rs && s->rs->polygon_mode != VK_POLYGON_MODE_FILL) {
      if (s->rs->polygon_mode != VK_POLYGON_MODE_LINE && s->rs->polygon_mode != VK_POLYGON_MODE_POINT)
         return G5P_REFUSE(device, "polygon mode %d", s->rs->polygon_mode);
      /* an effective bias only (a non-zero factor): Zink enables depth bias in
       * every pipeline and sets zero factors when GL polygon offset is off.
       * Static factors are judged here, dynamic ones at the draw (g5p_cmd_draw). */
      if (s->rs->depth_bias.enable && !BITSET_TEST(s->dynamic, MESA_VK_DYNAMIC_RS_DEPTH_BIAS_ENABLE) &&
          !BITSET_TEST(s->dynamic, MESA_VK_DYNAMIC_RS_DEPTH_BIAS_FACTORS) &&
          (s->rs->depth_bias.constant_factor != 0.0f || s->rs->depth_bias.slope_factor != 0.0f))
         return G5P_REFUSE(device, "polygon offset with line or point fill is not qualified");
      if (BITSET_TEST(s->dynamic, MESA_VK_DYNAMIC_RS_POLYGON_MODE))
         return G5P_REFUSE(device, "dynamic polygon mode");
   }
   /* line/point fill: Zink asks for Bresenham polygon edges (bresenhamLines is
    * advertised, rectangularLines not); width-1 edges of the qualified shapes are the same pixels */
   const bool fill_bresenham = s->rs && s->rs->polygon_mode != VK_POLYGON_MODE_FILL &&
                               s->rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM;
   /* The ordinary LINE object is A9's native aliased-line path. Its
    * width-1, single-sample Bresenham coverage is qualified by the GL/GLES
    * c47 line-list and strip flight. Keep other widths and samples refused. */
   const bool bresenham_line = g5p_native_bresenham_line(s);
   if (s->rs && ((s->rs->line.mode != VK_LINE_RASTERIZATION_MODE_DEFAULT && !fill_bresenham && !bresenham_line) ||
                 s->rs->line.stipple.enable))
      return G5P_REFUSE(device, "line rasterization mode %d, stipple %d", s->rs->line.mode,
                        s->rs->line.stipple.enable);
   /* 1 or 4 samples (pvr_g5p_msaa.c); sample shading, alpha to coverage and a partial sample mask need the
    * program's coverage feedback, which is not built */
   if (s->ms && ((s->ms->rasterization_samples != VK_SAMPLE_COUNT_1_BIT &&
                  s->ms->rasterization_samples != VK_SAMPLE_COUNT_4_BIT) ||
                 s->ms->sample_shading_enable || s->ms->alpha_to_coverage_enable || s->ms->alpha_to_one_enable ||
                 (s->ms->sample_mask & BITFIELD_MASK(s->ms->rasterization_samples)) !=
                    BITFIELD_MASK(s->ms->rasterization_samples)))
      return G5P_REFUSE(device, "multisample state (1 or 4 samples, every sample, no sample shading or alpha to "
                        "coverage)");
   /* pvr_* gives a render pass without multiview the view mask 1 */
   if (s->mv && util_bitcount(s->mv->view_mask) > 1)
      return G5P_REFUSE(device, "multiview (view mask %#x)", s->mv->view_mask);
   /* several colour targets: 2 or 4 in one format (pvr_g5p_mrt.h) */
   if (colors > 1) {
      if (colors > PVR_G5P_MRT_MAX_COLORS)
         return G5P_REFUSE(device, "%u colour attachments (at most %u)", colors, PVR_G5P_MRT_MAX_COLORS);
      VkResult mrt = pvr_arch_g5p_mrt_pipeline_check(device, colors, cfs);
      if (mrt != VK_SUCCESS)
         return mrt;
   }
   /* the formats the two-word program renders: the A9 format table's (pvr_g5p_formats.h); no colour
    * attachment: a depth/stencil-only pass (its render stores the null target, pvr_arch_g5p.c) */
   uint32_t cf_key;
   bool cf_srgb;
   if (colors && !pvr_g5p_format_pbe(cf, &cf_key, &cf_srgb))
      return G5P_REFUSE(device, "colour attachment format %s", vk_Format_to_str(cf));
   /* a depth/stencil attachment: the ZLS formats (pvr_g5p_ds.c), no depth bounds */
   if (ds && s->ds && s->ds->depth.bounds_test.enable)
      return G5P_REFUSE(device, "a depth bounds test");
   /* blending and the write mask compile into the fragment program
    * (pvr_g5p_blend.h) */
   *key = PCO_G5P_FS_KEY_DEFAULT;
   /* Apple's source groups: 1 only on a register's last read. With group 1 on
    * every read, a register read again returns a wrong value on the A9: a
    * general sample's result read back for its outputs (texture() of an
    * array or cube, textureLod: red right, green to alpha scaled by a
    * coordinate), blending's destination term (labwc113). The A9 ICD ran
    * every device suite and the desktop with them (PVR_G5P_GROUPS=last-read,
    * automatic since Mesa117).
    */
   key->last_read_groups = true;
   const char *why = pvr_arch_g5p_blend_fs_key(s, colors, key);
   if (why)
      return G5P_REFUSE(device, "colour blend state: %s", why);
   /* three colour targets compile as four: the fourth (the pass's null target) is written zeros */
   if (colors == 3)
      key->color_targets = 4;
   /* 32-bit integer targets: the raw32 output (four raw words, pvr_g5p_raw32_kind()); the PBE stores the
    * target's own channels from them */
   const unsigned raw32 = colors ? pvr_g5p_raw32_kind(cf) : 0;
   if (colors && vk_format_is_int(cf) && !raw32 && !pvr_g5p_packed_uint_bits(cf))
      return G5P_REFUSE(device, "integer boundary qualification supports CLEAR only");
   if (pvr_g5p_packed_uint_bits(cf)) {
      key->color_output = PCO_G5P_COLOR_UINT32;
      key->packed_uint_bits = pvr_g5p_packed_uint_bits(cf);
      key->packed_sint = pvr_g5p_packed_is_signed(cf);
      key->packed_scalar8 = cf == VK_FORMAT_R8_UINT || cf == VK_FORMAT_R8_SINT;
      if (key->packed_sint)
         key->color_output = PCO_G5P_COLOR_SINT32;
   }
   if (raw32)
      key->color_output = raw32 == 2 ? PCO_G5P_COLOR_SINT32 : PCO_G5P_COLOR_UINT32;
   /* no colour attachment: the program writes no colour (the no-colour ISP form: depth/stencil only, a
    * discarding program still runs for its feedback) */
   if (!colors)
      key->write_mask = 0;
   return VK_SUCCESS;
}

static void
g5p_pipeline_free(struct pvr_device *device, struct pvr_g5p_pipeline *g)
{
   if (!g)
      return;
   pvr_bo_suballoc_free(g->vs_usc);
   pvr_bo_suballoc_free(g->fs_usc);
   pvr_bo_suballoc_free(g->fetch.pvr_bo);
   pvr_bo_suballoc_free(g->fetch_base.pvr_bo);
   pvr_bo_suballoc_free(g->fetch_indirect.pvr_bo);
   pvr_bo_suballoc_free(g->fs_launch.pvr_bo);
   pvr_bo_suballoc_free(g->varying.pvr_bo);
   pvr_bo_suballoc_free(g->fs_loader.pvr_bo);
   pvr_arch_g5p_vs_uploads_destroy(device, g->vs_uploads);
   pvr_bo_suballoc_free(g->static_vary);
   pvr_bo_suballoc_free(g->static_frag);
   g5p_pipeline_free(device, g->xfb);
   g5p_pipeline_free(device, g->punch);
   vk_free(&device->vk.alloc, g);
}

void
PVR_PER_ARCH(g5p_graphics_pipeline_destroy)(struct pvr_device *device, struct pvr_graphics_pipeline *pipeline)
{
   g5p_pipeline_free(device, pipeline->g5p);
   pipeline->g5p = NULL;
}

/* The programs and static state of a pipeline (the ICD's create_graphics()
 * and place_pipeline()) */
static VkResult
g5p_pipeline_build(struct pvr_device *device, const VkGraphicsPipelineCreateInfo *create_info,
                   const struct vk_graphics_pipeline_state *s, struct pvr_g5p_pipeline *g,
                   struct pco_g5p_program *vs, struct pco_g5p_program *fs)
{
   const struct pvr_g5p_programs *dp = &device->g5p->programs;
   const VkPipelineShaderStageCreateInfo *vs_info = NULL, *fs_info = NULL;
   struct pco_g5p_fs_key key;
   VkResult result = g5p_check_state(device, create_info, s, &key);
   if (result != VK_SUCCESS)
      return result;
   key.punch = g->punch_variant;
   g->samples = s->ms ? MAX2(s->ms->rasterization_samples, VK_SAMPLE_COUNT_1_BIT) : VK_SAMPLE_COUNT_1_BIT;
   g->bresenham_line_probe = g5p_native_bresenham_line(s);

   for (uint32_t i = 0; i < create_info->stageCount; i++) {
      const VkPipelineShaderStageCreateInfo *st = &create_info->pStages[i];
      if (st->stage == VK_SHADER_STAGE_VERTEX_BIT && !vs_info)
         vs_info = st;
      else if (st->stage == VK_SHADER_STAGE_FRAGMENT_BIT && !fs_info)
         fs_info = st;
      else
         return G5P_REFUSE(device, "shader stage %#x", st->stage);
   }
   if (!vs_info || !fs_info)
      return G5P_REFUSE(device, "a pipeline without a vertex and a fragment shader");

   /* the vertex input layout: the fetches, and the registers the vertex program reads them from
    * (pvr_g5p_vertex.h) */
   struct pvr_g5p_vertex_layout vertex;
   pvr_g5p_vertex_layout_init(&vertex, s->vi, BITSET_TEST(s->dynamic, MESA_VK_DYNAMIC_VI_BINDING_STRIDES));

   const char *depth_probe = getenv("PVR_G5P_DEPTH_EXPORT");
   key.depth_feedback = depth_probe && !strcmp(depth_probe, "probe");
   /* the fragment program first: the vertex program is compiled against the
    * varyings it reads (Apple's VertexProgramKey; Rogue's pco_rev_link_nir) */
   if (!pvr_g5p_compile_stage(fs_info, true, g->samples, NULL, &key, NULL, fs))
      return G5P_REFUSE(device, "fragment shader: %s", fs->refusal);
   struct pco_g5p_vs_link link = { .valid = true };
   /* points (Apple's g3b-points): the point size after the position; the
    * topology class is fixed per pipeline even when the topology is dynamic */
   link.point_size = s->ia && s->ia->primitive_topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   g->vs_point_size = link.point_size;
   g->polygon_mode = s->rs ? s->rs->polygon_mode : VK_POLYGON_MODE_FILL;
   for (unsigned i = 0; i < fs->num_varyings; i++)
      link.interp[fs->vary[i].location] = fs->vary[i].interp;
   /* the transform feedback variant (pvr_g5p_xfb.h): the capturing vertex program */
   if (g->xfb_variant ? !pvr_g5p_compile_vs_xfb(vs_info, &vertex, &link, &g->xfb_layout, vs)
                      : !pvr_g5p_compile_stage(vs_info, false, 1, &vertex, NULL, &link, vs))
      return G5P_REFUSE(device, "vertex shader%s: %s", g->xfb_variant ? " (transform feedback)" : "", vs->refusal);
   if (g->xfb_variant && !g->xfb_layout.buffers)
      return VK_SUCCESS; /* captures nothing: no variant */
   /* point fill rasterizes ISP points of one size per draw (pointlinewidth): a per-vertex
    * gl_PointSize would be ignored, so it is refused rather than drawn at 1.0 */
   if (g->polygon_mode == VK_POLYGON_MODE_POINT && vs->psize_dropped)
      return G5P_REFUSE(device, "polygon mode point with the vertex program's gl_PointSize");
   for (unsigned i = 0; i < fs->num_varyings; i++)
      if (!vs->vout_dword[fs->vary[i].location])
         return G5P_REFUSE(device, "the fragment program reads VAR%u, which the vertex program does not write",
                           fs->vary[i].location);
   /* the vertex program's common-store uploads and their loader (pvr_g5p_upload.h) */
   VkResult up_result = pvr_arch_g5p_vs_uploads_create(device, vs, &g->vs_uploads);
   if (up_result != VK_SUCCESS)
      return up_result;
   static_assert(PCO_G5P_MAX_UPLOADS <= PVR_G5P_MAX_FS_UPLOADS, "G5P upload list");
   for (unsigned i = 0; i < fs->num_uploads; i++) {
      const struct pco_g5p_upload *u = &fs->uploads[i];
      switch (u->kind) {
      case PCO_G5P_UPLOAD_IMAGE:
      case PCO_G5P_UPLOAD_SAMPLER:
      case PCO_G5P_UPLOAD_IMAGE_SIZE:
      case PCO_G5P_UPLOAD_BLEND_CONSTANTS:
         if (u->dwords > 8)
            return G5P_REFUSE(device, "fragment upload %u: %u dwords of descriptor words", i, u->dwords);
         break;
      case PCO_G5P_UPLOAD_PUSH:
      case PCO_G5P_UPLOAD_UBO:
      case PCO_G5P_UPLOAD_UBO_ADDR:
      case PCO_G5P_UPLOAD_SSBO_ADDR:
         /* read where they are (the vertex stage's uploads, pvr_arch_g5p_upload.c) */
         break;
      default:
         return G5P_REFUSE(device, "fragment upload kind %u", u->kind);
      }
      if (!u->dwords || u->dwords > UINT8_MAX || u->set > UINT8_MAX || u->binding > UINT8_MAX ||
          u->element > UINT16_MAX || u->offset > UINT16_MAX)
         return G5P_REFUSE(device, "fragment upload %u: set %u binding %u[%u] offset %u, %u dwords", i,
                           u->set, u->binding, u->element, u->offset, u->dwords);
      g->fs_uploads[i] = (struct pvr_g5p_upload){ .kind = u->kind, .set = u->set, .binding = u->binding,
                                                  .element = u->element, .dwords = u->dwords,
                                                  .offset = u->offset,
                                                  .common = u->common };
   }
   g->fs_num_uploads = fs->num_uploads;
   g->fs_common_dwords = fs->common_dwords;
   /* PDS has 128 constant dwords, four per transfer. Coalesce copied
    * descriptor intervals when the original upload list exceeds twelve.
    * GPU-written UBOs and push snapshots retain their own source addresses. */
   bool cpu[PCO_G5P_FS_COMMON_DWORDS] = {false};
   for (unsigned i = 0; i < fs->num_uploads; i++) {
      const struct pco_g5p_upload *u = &fs->uploads[i];
      bool copy = fs->num_uploads > 12 &&
                  (u->kind == PCO_G5P_UPLOAD_IMAGE || u->kind == PCO_G5P_UPLOAD_SAMPLER ||
                   u->kind == PCO_G5P_UPLOAD_IMAGE_SIZE || u->kind == PCO_G5P_UPLOAD_BLEND_CONSTANTS);
      if (u->common + u->dwords > ARRAY_SIZE(cpu))
         return G5P_REFUSE(device, "fragment upload outside common store");
      if (copy) {
         for (unsigned k = u->common; k < u->common + u->dwords; k++) cpu[k] = true;
      } else {
         if (g->fs_num_dmas == ARRAY_SIZE(g->fs_dma))
            return G5P_REFUSE(device, "more than 32 fragment PDS transfers");
         unsigned j = g->fs_num_dmas++;
         g->fs_dma[j].common = u->common;
         g->fs_dma[j].dwords = u->dwords;
         g->fs_dma[j].upload = i;
      }
   }
   for (unsigned k = 0; k < fs->common_dwords;) {
      if (!cpu[k]) { k++; continue; }
      unsigned first = k++;
      while (k < fs->common_dwords && cpu[k]) k++;
      if (g->fs_num_dmas == ARRAY_SIZE(g->fs_dma))
         return G5P_REFUSE(device, "more than 32 fragment PDS transfers");
      unsigned j = g->fs_num_dmas++;
      g->fs_dma[j].common = first;
      g->fs_dma[j].dwords = k - first;
      g->fs_dma[j].upload = -1;
   }
   if (vs->xfb_buffers & ~(g->xfb_variant ? g->xfb_layout.buffers : 0))
      return G5P_REFUSE(device, "transform feedback stores to buffers %#x outside the capture layout %#x",
                        vs->xfb_buffers, g->xfb_variant ? g->xfb_layout.buffers : 0);

   /* DOUTU temps (the ICD's vs_temps()): every register within the flown
    * envelope, else only the written ones (the inputs are in the unified store) */
   const unsigned bank1_temps = vs->temps <= PCO_G5P_VS_TEMPS_FLOWN ? vs->temps : vs->written_temps;
   if (bank1_temps > 32)
      return G5P_REFUSE(device, "vertex shader needs %u temps (> 32, the register file)", bank1_temps);
   /* A program with bank-0 temps declares them as Apple's do (vertex metadata
    * +140 = the DOUTU temps; docs/gpu/vs-two-bank-allocator.md); the unified
    * store keeps its bank-1 size. Without bank 0 every word is unchanged. */
   if (vs->bank0_temps > 32)
      return G5P_REFUSE(device, "vertex shader needs %u bank-0 temps (> 32)", vs->bank0_temps);
   g->vs_temps = MAX2(bank1_temps, vs->bank0_temps);
   g->vs_written_temps = vs->written_temps;
   g->fs_temps = fs->temps;
   g->vs_outputs = vs->outputs;
   g->vs_id_mask = vs->id_mask;
   memcpy(g->vs_id_reg, vs->id_reg, sizeof(g->vs_id_reg));
   g->fs_discard = (fs->evidence & PCO_G5P_EV_DISCARD) != 0;
   g->fs_discard_tokens = fs->discard_tokens;
   g->fs_discard_select = fs->discard_select;
   if (g->punch_variant && (fs->discard_tokens != 1 || fs->discard_select))
      return G5P_REFUSE(device, "punch-through variant needs exactly one real discard token");
   g->fs_side_effects = fs->side_effects;
   g->vs_side_effects = vs->side_effects;
   g->fs_depth_feedback = fs->depth_feedback;
   if (g->fs_depth_feedback && g->samples != 1)
      return G5P_REFUSE(device, "multisample fragment depth export is not qualified");
   g->fs_blend_mode = fs->blend_mode;
   g->fs_write_mask = key.write_mask;
   g->color_targets = key.color_targets;
   for (unsigned i = 0; i < fs->num_varyings; i++)
      g->fs_flat |= fs->vary[i].interp == PCO_G5P_INTERP_FLAT;

   g->fetch_count = vertex.fetch_count;
   memcpy(g->fetch_list, vertex.fetch_list, sizeof(g->fetch_list));
   if (!g->fetch_count && !(g->vs_id_mask & 1))
      return G5P_REFUSE(device, "a vertex shader without attributes or vertex index (no fetch-less PDS form)");

   /* The unified store holds every input dword: Apple's sizes fit 4 units
    * per 8 input dwords; above 8 temps or 8 input dwords it grows with the
    * program, 2 * ceil(max(temps, inputs + 2) / 4) (the ICD's g5p_job_draw()) */
   unsigned inputs = 0;
   for (unsigned f = 0; f < g->fetch_count; f++)
      inputs = MAX2(inputs, g->fetch_list[f].ao + g->fetch_list[f].dwords);
   u_foreach_bit (i, g->vs_id_mask)
      inputs = MAX2(inputs, g->vs_id_reg[i] + 1u);
   g->vs_unified = 4 * DIV_ROUND_UP(inputs, 8);
   if (bank1_temps > 8 || inputs > 8)
      g->vs_unified = MAX2(g->vs_unified, 2 * DIV_ROUND_UP(MAX2(bank1_temps, inputs + 2), 4));

   /* USC programs */
   g->vs_usc_size = vs->size;
   g->fs_usc_size = fs->size;
   result = pvr_gpu_upload_usc(device, vs->code, vs->size, 64, &g->vs_usc);
   if (result == VK_SUCCESS)
      result = pvr_gpu_upload_usc(device, fs->code, fs->size, 64, &g->fs_usc);
   if (result != VK_SUCCESS)
      return result;

   /* PDS: vertex fetch code */
   uint32_t code[96 + 4 * PVR_G5P_MAX_FS_UPLOADS], data[2 * (1 + PCO_G5P_MAX_FS_VARYINGS) + 4];
   unsigned n = fetch_program(code, g, false, false);
   result = pvr_gpu_upload_pds(device, NULL, 0, 0, code, n, 16, 64, &g->fetch);
   if (result != VK_SUCCESS)
      return result;
   if (fetch_reads_instance(g)) {
      n = fetch_program(code, g, true, false);
      result = pvr_gpu_upload_pds(device, NULL, 0, 0, code, n, 16, 64, &g->fetch_base);
      if (result != VK_SUCCESS)
         return result;
   }

   n = fetch_program(code, g, fetch_reads_instance(g), true);
   result = pvr_gpu_upload_pds(device, NULL, 0, 0, code, n, 16, 64, &g->fetch_indirect);
   if (result != VK_SUCCESS)
      return result;

   /* PDS: the fragment launch, data {DOUTU} and its code at +0x20 */
   {
      /* at 4x the fragment program runs at sample rate (pvr_g5p_msaa_doutu()) */
      const uint64_t doutu = pvr_g5p_msaa_doutu(doutu_src0(usc_off(device, g->fs_usc), g->fs_temps), g->samples);
      const uint32_t launch_data[4] = { (uint32_t)doutu, (uint32_t)(doutu >> 32), 0, 0 };
      n = launch_program(code);
      result = pvr_gpu_upload_pds(device, launch_data, ARRAY_SIZE(launch_data), 16, code, n, 0x20, 64,
                                  &g->fs_launch);
      if (result != VK_SUCCESS)
         return result;
      if (g->fs_launch.code_offset != g->fs_launch.data_offset + 0x20)
         return vk_errorf(device, VK_ERROR_INITIALIZATION_FAILED, "G5P: the fragment launch's code is at data + %#x",
                          g->fs_launch.code_offset - g->fs_launch.data_offset);
   }

   /* PDS: the coefficient program (the device HALT without varyings) */
   unsigned varying_data_dwords = 0;
   if (fs->num_varyings || fs->w_set) {
      unsigned code_dwords;
      varying_data_dwords = varying_program(fs, vs, data, code, &code_dwords);
      result = pvr_gpu_upload_pds(device, data, varying_data_dwords, 16, code, code_dwords, 16, 64, &g->varying);
      if (result != VK_SUCCESS)
         return result;
      if (g->varying.code_offset != g->varying.data_offset + 4 * varying_data_dwords)
         return vk_errorf(device, VK_ERROR_INITIALIZATION_FAILED, "G5P: the coefficient code is at data + %#x, not + %#x",
                          g->varying.code_offset - g->varying.data_offset, 4 * varying_data_dwords);
   }

   /* PDS: the uniform/texture loader (the device HALT without uploads) */
   if (g->fs_num_uploads) {
      n = loader_program(code, g->fs_num_dmas);
      result = pvr_gpu_upload_pds(device, NULL, 0, 0, code, n, 16, 64, &g->fs_loader);
      if (result != VK_SUCCESS)
         return result;
   }

   /* the static PPP blocks: the varying format (output dwords per
    * interpolation, in the vertex program's grouped order) */
   {
      uint32_t *w = g->static_vary_words;
      pvr_csb_pack (&w[0], TA_STATE_HEADER, h) {
         h.pres_varying_word0 = true;
         h.pres_varying_word1 = true;
         h.pres_varying_word2 = true;
      }
      pvr_csb_pack (&w[1], TA_STATE_VARYING0, v0) {
         for (unsigned l = 0; l < PCO_G5P_VARYING_SLOTS; l++) {
            if (!vs->vout_dword[l])
               continue;
            if (vs->vout_interp[l] == PCO_G5P_INTERP_FLAT)
               v0.f32_flat += 4;
            else if (vs->vout_interp[l] == PCO_G5P_INTERP_NOPERSPECTIVE)
               v0.f32_npc += 4;
            else
               v0.f32_linear += 4;
         }
      }
      w[2] = w[3] = 0;
      result = pvr_gpu_upload(device, device->heaps.general_heap, w, 16, 16, &g->static_vary);
      if (result != VK_SUCCESS)
         return result;
   }
   /* ... and the fragment pointers and sizes */
   {
      uint32_t *w = g->static_frag_words;
      pvr_csb_pack (&w[0], TA_STATE_HEADER, h) {
         h.pres_pds_state_ptr0 = true;
         h.pres_pds_state_ptr1 = true;
      }
      pvr_csb_pack (&w[1], TA_STATE_PDS_SHADERBASE, sb) {
         sb.addr = PVR_DEV_ADDR(g->fs_launch.data_offset);
      }
      /* the uniform/texture loader, or the device HALT without uploads */
      pvr_csb_pack (&w[2], TA_STATE_PDS_TEXUNICODEBASE, tb) {
         tb.addr = PVR_DEV_ADDR(g->fs_num_uploads ? g->fs_loader.code_offset : dp->halt.code_offset);
      }
      pvr_csb_pack (&w[3], TA_STATE_PDS_SIZEINFO1, s1) {
         s1.pds_uniformsize = g->fs_num_uploads ? DIV_ROUND_UP(4 * g->fs_num_dmas,
                                                               ROGUE_TA_STATE_PDS_SIZEINFO1_PDS_UNIFORMSIZE_UNIT_SIZE)
                                                : 1;
         if (fs->num_varyings || fs->w_set) {
            s1.pds_varyingsize = DIV_ROUND_UP(varying_data_dwords, 4);
            /* coefficient sets: W, then four per varying (4 dwords each) */
            s1.usc_varyingsize = DIV_ROUND_UP(4 * ((fs->w_set ? 1 : 0) + 4 * fs->num_varyings), 16);
         }
      }
      /* A discarding program runs punch-through with triangle merging off
       * (Apple's discard capture; the G9P; Rogue's pvr) */
      pvr_csb_pack (&w[4], TA_STATE_PDS_SIZEINFO2, s2) {
         s2.pds_tri_merge_disable = g->fs_discard;
         if (g->fs_num_uploads)
            s2.usc_sharedsize = DIV_ROUND_UP(g->fs_common_dwords,
                                             ROGUE_TA_STATE_PDS_SIZEINFO2_USC_SHAREDSIZE_UNIT_SIZE);
      }
      pvr_csb_pack (&w[5], TA_STATE_PDS_VARYINGBASE, vb) {
         vb.addr = PVR_DEV_ADDR(fs->num_varyings || fs->w_set ? g->varying.data_offset : dp->halt.code_offset);
      }
      result = pvr_gpu_upload(device, device->heaps.general_heap, w, 24, 16, &g->static_frag);
      if (result != VK_SUCCESS)
         return result;
   }

   if (g5p_logging()) {
      fprintf(stderr, "g5p pipeline: vs %u bytes temps %u (program %u, written %u) outputs %u unified %u ids %#x "
              "fetches %u; fs %u bytes temps %u varyings %u w_set %u discard %u\n",
              g->vs_usc_size, g->vs_temps, vs->temps, vs->written_temps, g->vs_outputs, g->vs_unified,
              g->vs_id_mask, g->fetch_count, g->fs_usc_size, g->fs_temps, fs->num_varyings, fs->w_set,
              g->fs_discard);
      g5p_dump_code("vs usc", usc_off(device, g->vs_usc), vs->code, vs->size);
      g5p_dump_code("fs usc", usc_off(device, g->fs_usc), fs->code, fs->size);
   }
   return VK_SUCCESS;
}


/* Port of the device-proven g5p-punch-ds@8150a8195df recipe. Unlike the
 * standalone USC-only swap, a full linked variant owns its matching vertex
 * program, varyings and upload layout; no old interface is reused by accident. */
static VkResult
g5p_pipeline_build_punch(struct pvr_device *device,
                        const VkGraphicsPipelineCreateInfo *create_info,
                        const struct vk_graphics_pipeline_state *state,
                        struct pvr_g5p_pipeline *g)
{
   if (g->fs_discard_tokens < 2 && !g->fs_discard_select)
      return VK_SUCCESS;
   struct pvr_g5p_pipeline *p =
      vk_zalloc(&device->vk.alloc, sizeof(*p), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!p)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   p->punch_variant = true;
   p->xfb_variant = g->xfb_variant;
   struct pco_g5p_program vs = { 0 }, fs = { 0 };
   VkResult result = g5p_pipeline_build(device, create_info, state, p, &vs, &fs);
   free(vs.code);
   free(fs.code);
   if (result == VK_SUCCESS) {
      g->punch = p;
   } else {
      g5p_pipeline_free(device, p);
      /* A normal colour-only shader remains valid. A draw that requires the
       * absent punch variant refuses by name before emitting any draw state. */
      if (result == VK_ERROR_FEATURE_NOT_PRESENT)
         return VK_SUCCESS;
   }
   return result;
}

/* Every refusal a draw could meet because of its programs is decided here,
 * at pipeline creation, never while recording: a refused draw fails the
 * whole command buffer (vkEndCommandBuffer), and Zink then drops the batch
 * (Aquarium on r39-6: no water/background, 3 fps). The draw selects the
 * punch variant from the dynamic depth/stencil state (g5p_draw()); this
 * takes every state the pipeline allows: a dynamic enable may be on, a
 * depth write needs the depth test, and a stencil test may write. */
static VkResult
g5p_pipeline_check_draw_variants(struct pvr_device *device,
                                 const struct vk_graphics_pipeline_state *state,
                                 const struct pvr_g5p_pipeline *g)
{
   if (state->rs && !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE) &&
       state->rs->rasterizer_discard_enable)
      return VK_SUCCESS;
   const bool depth = !state->rp || state->rp->depth_attachment_format != VK_FORMAT_UNDEFINED;
   const bool stencil = !state->rp || state->rp->stencil_attachment_format != VK_FORMAT_UNDEFINED;
   const struct vk_depth_stencil_state *ds = state->ds;
#define G5P_DYN_DS(x) BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_##x)
   const bool depth_test = depth && (G5P_DYN_DS(DEPTH_TEST_ENABLE) || (ds && ds->depth.test_enable));
   const bool depth_write = depth_test && (G5P_DYN_DS(DEPTH_WRITE_ENABLE) || (ds && ds->depth.write_enable));
   const bool stencil_test = stencil && (G5P_DYN_DS(STENCIL_TEST_ENABLE) || (ds && ds->stencil.test_enable));
#undef G5P_DYN_DS
   const bool testing = depth_test || stencil_test, writing = depth_write || stencil_test;
   const bool needs_punch = (testing && g->fs_discard_tokens > 1) || (writing && g->fs_discard_select);
   if (needs_punch && !g->punch)
      return G5P_REFUSE(device, "pipeline: discard under depth/stencil requires an unsupported punch-through variant");
   if (testing && (g->fs_discard || needs_punch) && g->samples != 1)
      return G5P_REFUSE(device, "pipeline: discard under multisample depth/stencil is not hardware-qualified");
   return VK_SUCCESS;
}

VkResult
PVR_PER_ARCH(g5p_graphics_pipeline_compile)(struct pvr_device *device,
                                            const VkGraphicsPipelineCreateInfo *create_info,
                                            struct pvr_graphics_pipeline *pipeline,
                                            const struct vk_graphics_pipeline_state *vk_state)
{
   struct pco_g5p_program vs = { 0 }, fs = { 0 };
   struct pvr_g5p_pipeline *g =
      vk_zalloc(&device->vk.alloc, sizeof(*g), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!g)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result = g5p_pipeline_build(device, create_info, vk_state, g, &vs, &fs);
   /* the program bytes are the backend's (malloc'd); uploaded or refused */
   free(vs.code);
   free(fs.code);
   if (result != VK_SUCCESS) {
      g5p_pipeline_free(device, g);
      return result;
   }
   result = g5p_pipeline_build_punch(device, create_info, vk_state, g);
   if (result == VK_SUCCESS)
      result = g5p_pipeline_check_draw_variants(device, vk_state, g);
   if (result != VK_SUCCESS) {
      g5p_pipeline_free(device, g);
      return result;
   }
   pipeline->g5p = g;

   /* transform feedback: the same pipeline with the capturing vertex
    * program, for draws while capturing (pvr_g5p_xfb.h) */
   const VkPipelineShaderStageCreateInfo *vs_info = NULL;
   for (uint32_t i = 0; i < create_info->stageCount; i++)
      if (create_info->pStages[i].stage == VK_SHADER_STAGE_VERTEX_BIT)
         vs_info = &create_info->pStages[i];
   if (vs_info && pvr_g5p_stage_has_xfb(vs_info)) {
      struct pco_g5p_program xvs = { 0 }, xfs = { 0 };
      struct pvr_g5p_pipeline *x =
         vk_zalloc(&device->vk.alloc, sizeof(*x), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!x)
         return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
      x->xfb_variant = true;
      result = g5p_pipeline_build(device, create_info, vk_state, x, &xvs, &xfs);
      free(xvs.code);
      free(xfs.code);
      if (result == VK_SUCCESS && x->xfb_layout.buffers)
         result = g5p_pipeline_build_punch(device, create_info, vk_state, x);
      if (result == VK_SUCCESS && x->xfb_layout.buffers)
         result = g5p_pipeline_check_draw_variants(device, vk_state, x);
      if (result == VK_SUCCESS && x->xfb_layout.buffers) {
         g->xfb = x;
      } else {
         g5p_pipeline_free(device, x);
         if (result != VK_SUCCESS) {
            g5p_pipeline_free(device, g);
            pipeline->g5p = NULL;
            return result;
         }
      }
   }
   return VK_SUCCESS;
}

/* ---- vkCmdClearAttachments: the clear quad ---- */

/* The clear quad's pipeline for a colour format and colour write mask
 * (pvr_g5p.h: made at first use, kept per device): the G5P stages built with
 * nir_builder (pvr_g5p_clear_nir()), one vec4 position per vertex (binding
 * 0, stride 16), a triangle list, the colour from fragment push constants
 * 0..15, no blending, viewport and scissor dynamic, dynamic rendering of the
 * one colour format. Write mask 0 (a depth/stencil-only clear) compiles the
 * colour output away, and the draw takes the no-colour ISP form (tag writes
 * disabled, user pass 0: depth/stencil updates only, pvr_g5p_blend.h). The
 * depth/stencil tests and writes are the draw's dynamic state. */
static VkResult
g5p_clear_pipeline(struct pvr_cmd_buffer *cmd_buffer, VkFormat format, uint8_t write_mask,
                   const struct pvr_graphics_pipeline **out)
{
   struct pvr_device *device = cmd_buffer->device;
   struct pvr_g5p_device *gd = device->g5p;
   const VkDevice vk_dev = pvr_device_to_handle(device);
   VkResult result = VK_SUCCESS;
   VkPipeline pipe = VK_NULL_HANDLE;

   simple_mtx_lock(&gd->clear_lock);
   for (unsigned i = 0; i < gd->clear_count; i++)
      if (gd->clear[i].format == format && gd->clear[i].write_mask == write_mask)
         pipe = gd->clear[i].pipeline;
   if (pipe)
      goto done;
   if (gd->clear_count == ARRAY_SIZE(gd->clear)) {
      result = G5P_REFUSE(cmd_buffer, "a clear quad of a %u-th colour format and write mask", gd->clear_count + 1);
      goto done;
   }
   if (!gd->clear_layout) {
      const VkPushConstantRange range = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16 };
      const VkPipelineLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                               .pushConstantRangeCount = 1, .pPushConstantRanges = &range };
      result = device->vk.dispatch_table.CreatePipelineLayout(vk_dev, &lci, NULL, &gd->clear_layout);
      if (result != VK_SUCCESS)
         goto done;
   }
   struct nir_shader *vs = pvr_g5p_clear_nir(true);
   struct nir_shader *fs = (pvr_g5p_raw32_kind(format) || pvr_g5p_packed_uint_bits(format)) ? pvr_g5p_clear_nir_raw(false, (pvr_g5p_raw32_kind(format) == 2 || pvr_g5p_packed_is_signed(format)))
                                                      : pvr_g5p_clear_nir(false);
   const VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vk_shader_module_handle_from_nir(vs), .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = vk_shader_module_handle_from_nir(fs), .pName = "main" },
   };
   const VkVertexInputBindingDescription binding = { 0, 16, VK_VERTEX_INPUT_RATE_VERTEX };
   const VkVertexInputAttributeDescription attribute = { 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
   const VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &binding, .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &attribute };
   const VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
   const VkPipelineViewportStateCreateInfo vp = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                                  .viewportCount = 1, .scissorCount = 1 };
   const VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
   const VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   const VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = write_mask };
   const VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = format != VK_FORMAT_UNDEFINED,
      .pAttachments = &cba };
   const VkDynamicState dynamic[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
   const VkPipelineDynamicStateCreateInfo dy = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                                 .dynamicStateCount = 2, .pDynamicStates = dynamic };
   const VkPipelineRenderingCreateInfo rendering = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                                     .colorAttachmentCount = format != VK_FORMAT_UNDEFINED,
                                                     .pColorAttachmentFormats = &format };
   const VkGraphicsPipelineCreateInfo gci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rendering, .stageCount = 2,
      .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vp,
      .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &dy,
      .layout = gd->clear_layout };
   result = device->vk.dispatch_table.CreateGraphicsPipelines(vk_dev, VK_NULL_HANDLE, 1, &gci, NULL, &pipe);
   pvr_g5p_clear_nir_free(vs);
   pvr_g5p_clear_nir_free(fs);
   if (result != VK_SUCCESS) {
      pipe = VK_NULL_HANDLE;
      goto done;
   }
   gd->clear[gd->clear_count].format = format;
   gd->clear[gd->clear_count].write_mask = write_mask;
   gd->clear[gd->clear_count++].pipeline = pipe;
done:
   simple_mtx_unlock(&gd->clear_lock);
   if (result == VK_SUCCESS) {
      VK_FROM_HANDLE(pvr_pipeline, pipeline, pipe);
      *out = to_pvr_graphics_pipeline(pipeline);
   }
   return result;
}

/* vkCmdClearAttachments (the hook in rogue/pvr_blit.c; Rogue's clear
 * programs and words do not run on the A9): the A9 ICD's clear quads
 * (g5p_cmd_gfx.c clear_quads()). Each rectangle is a draw of the clear
 * quad's pipeline in the current render, with its own state: the viewport
 * the whole attachment, one triangle covering it, the scissor the
 * rectangle within the render area (the ISP scissor table, as any clipped
 * draw), the colour from push constants. Depth and stencil go through the
 * ISP like any draw's: the depth aspect as a depth test ALWAYS with writes
 * on and the triangle at the clear depth (the viewport's depth range is
 * [0, 1], so the window depth is the vertex depth), the stencil aspect as a
 * stencil test ALWAYS whose pass op REPLACEs with the clear value as its reference
 * under a full write mask, on both faces. An aspect not cleared keeps its
 * test off and its writes off; colour not cleared takes write mask 0 (the
 * no-colour ISP form). The application's bound state is put back
 * afterwards. One colour target, one sample, layer 0. */
VkResult
PVR_PER_ARCH(g5p_clear_attachments)(struct pvr_cmd_buffer *cmd_buffer, uint32_t attachment_count,
                                    const VkClearAttachment *attachments, uint32_t rect_count,
                                    const VkClearRect *rects)
{
   struct pvr_cmd_buffer_state *cs = &cmd_buffer->state;
   const struct pvr_render_pass_info *info = &cs->render_pass_info;
   struct vk_dynamic_graphics_state *dyn = &cmd_buffer->vk.dynamic_graphics_state;

   if (!attachment_count || !rect_count)
      return VK_SUCCESS;
   if (!info->rstate || (!info->pass && !info->dr_info))
      return G5P_REFUSE(cmd_buffer, "vkCmdClearAttachments outside a render pass");
   const struct pvr_renderpass_hwsetup_render *hw_render =
      pvr_arch_pass_info_get_hw_render(info, info->current_hw_subpass);
   if (hw_render->eot_surface_count > 1)
      return G5P_REFUSE(cmd_buffer, "vkCmdClearAttachments with %u colour targets (one is built)",
                        hw_render->eot_surface_count);
   if (MAX2(hw_render->sample_count, 1) != 1)
      return G5P_REFUSE(cmd_buffer, "vkCmdClearAttachments at %u samples (one is built)", hw_render->sample_count);
   /* no colour attachment (a depth/stencil-only pass): depth/stencil rectangles alone, the clear pipeline
    * without a colour attachment */
   const VkFormat format = hw_render->eot_surface_count
                              ? info->attachments[hw_render->eot_surfaces[0].attachment_idx]->vk.format
                              : VK_FORMAT_UNDEFINED;

   /* what is cleared: the colour target, and the aspects of the render's depth/stencil attachment */
   VkImageAspectFlags ds_present = 0;
   if (hw_render->ds_attach_idx != VK_ATTACHMENT_UNUSED) {
      const VkFormat f = info->attachments[hw_render->ds_attach_idx]->vk.format;
      ds_present = (vk_format_has_depth(f) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
                   (vk_format_has_stencil(f) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
   }
   const VkClearColorValue *colour = NULL;
   VkImageAspectFlags ds_clear = 0;
   float depth = 0.0f;
   uint32_t stencil = 0;
   for (uint32_t i = 0; i < attachment_count; i++) {
      const VkImageAspectFlags a = attachments[i].aspectMask;
      if ((a & VK_IMAGE_ASPECT_COLOR_BIT) && attachments[i].colorAttachment == 0)
         colour = &attachments[i].clearValue.color;
      if (a & ds_present & VK_IMAGE_ASPECT_DEPTH_BIT)
         depth = attachments[i].clearValue.depthStencil.depth;
      if (a & ds_present & VK_IMAGE_ASPECT_STENCIL_BIT)
         stencil = attachments[i].clearValue.depthStencil.stencil;
      ds_clear |= a & ds_present & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
   }
   if (format == VK_FORMAT_UNDEFINED)
      colour = NULL; /* no colour attachment to clear */
   if (!colour && !ds_clear)
      return VK_SUCCESS;
   if (colour && vk_format_is_int(format) && !pvr_g5p_raw32_kind(format) && !pvr_g5p_packed_uint_bits(format))
      return G5P_REFUSE(cmd_buffer, "vkCmdClearAttachments of an unsupported integer target (%s)", vk_Format_to_str(format));
   /* the triangle's depth is the clear depth: inside the viewport's [0, 1] (VK_EXT_depth_range_unrestricted
    * is not exposed, so a valid clear depth is too) */
   if (!(depth >= 0.0f && depth <= 1.0f))
      return G5P_REFUSE(cmd_buffer, "vkCmdClearAttachments of depth %f outside [0, 1]", depth);
   for (uint32_t r = 0; r < rect_count; r++)
      if (rects[r].baseArrayLayer || rects[r].layerCount != 1)
         return G5P_REFUSE(cmd_buffer, "vkCmdClearAttachments of layers %u..%u (layer 0 is built)",
                           rects[r].baseArrayLayer, rects[r].baseArrayLayer + rects[r].layerCount - 1);

   const struct pvr_graphics_pipeline *pipeline;
   VkResult result = g5p_clear_pipeline(cmd_buffer, format, colour ? 0xf : 0, &pipeline);
   if (result != VK_SUCCESS)
      return result;
   /* one triangle covering the viewport (the whole attachment) at the clear depth: each rectangle is its scissor */
   const float z = (ds_clear & VK_IMAGE_ASPECT_DEPTH_BIT) ? depth : 0.0f;
   const float cover[12] = { -1, -1, z, 1, 3, -1, z, 1, -1, 3, z, 1 };
   struct pvr_suballoc_bo *verts;
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, cover, sizeof(cover), &verts);
   if (result != VK_SUCCESS)
      return result;
   struct pvr_buffer vbuf = { .dev_addr = verts->dev_addr };

   /* the application's state, put back below */
   const struct pvr_graphics_pipeline *saved_pipeline = cs->gfx_pipeline;
   const struct pvr_vertex_binding saved_binding = cs->vertex_bindings[0];
   struct pvr_push_constants *push = &cs->push_consts[PVR_STAGE_ALLOCATION_FRAGMENT];
   struct pvr_push_constants *saved_push = malloc(sizeof(*saved_push));
   struct vk_dynamic_graphics_state *saved_dyn = malloc(sizeof(*saved_dyn));
   if (!saved_push || !saved_dyn) {
      free(saved_push);
      free(saved_dyn);
      return vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   memcpy(saved_push, push, sizeof(*push));
   memcpy(saved_dyn, dyn, sizeof(*dyn));

   cs->gfx_pipeline = pipeline;
   cs->vertex_bindings[0] = (struct pvr_vertex_binding){ .buffer = &vbuf, .offset = 0, .size = sizeof(cover) };
   if (colour)
      memcpy(push->data, colour->float32, 16);
   else
      memset(push->data, 0, 16);
   push->bytes_updated = MAX2(push->bytes_updated, 16);
   push->dirty = true;
   dyn->vi_binding_strides[0] = 16;
   dyn->ia.primitive_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   dyn->ia.primitive_restart_enable = false;
   dyn->rs.rasterizer_discard_enable = false;
   dyn->rs.cull_mode = VK_CULL_MODE_NONE;
   dyn->rs.depth_bias.enable = false;
   dyn->rs.depth_clamp_enable = false;
   const bool clear_depth = ds_clear & VK_IMAGE_ASPECT_DEPTH_BIT, clear_stencil = ds_clear & VK_IMAGE_ASPECT_STENCIL_BIT;
   dyn->ds.depth.test_enable = clear_depth;
   dyn->ds.depth.write_enable = clear_depth;
   dyn->ds.depth.compare_op = VK_COMPARE_OP_ALWAYS;
   dyn->ds.depth.bounds_test.enable = false;
   dyn->ds.stencil.test_enable = clear_stencil;
   const struct vk_stencil_test_face_state face = {
      /* under ALWAYS (and a depth test that is ALWAYS or off) only the pass op is reachable */
      .op = { .fail = VK_STENCIL_OP_KEEP, .pass = VK_STENCIL_OP_REPLACE, .depth_fail = VK_STENCIL_OP_KEEP,
              .compare = VK_COMPARE_OP_ALWAYS },
      .compare_mask = 0xff, .write_mask = 0xff, .reference = (uint8_t)stencil,
   };
   dyn->ds.stencil.front = face;
   dyn->ds.stencil.back = face;
   dyn->vp.viewport_count = 1;
   dyn->vp.scissor_count = 1;
   dyn->vp.viewports[0] = (VkViewport){ 0, 0, info->rstate->width, info->rstate->height, 0, 1 };

   const struct pvr_g5p_draw draw = { .count = 3, .instances = 1 };
   for (uint32_t r = 0; r < rect_count && result == VK_SUCCESS; r++) {
      /* the scissor: the rectangle (within the render area: the draw intersects it) */
      dyn->vp.scissors[0] = rects[r].rect;
      result = pvr_arch_g5p_cmd_draw(cmd_buffer, &draw);
   }

   cs->gfx_pipeline = saved_pipeline;
   cs->vertex_bindings[0] = saved_binding;
   memcpy(push, saved_push, sizeof(*push));
   memcpy(dyn, saved_dyn, sizeof(*dyn));
   free(saved_push);
   free(saved_dyn);
   return result;
}

/* ---- draws ---- */

/* The draw's ISP scissor entry: the clip [x0, x1) x [y0, y1) as IPF words
 * (the A9 ICD's scissor_index(), upstream's form: max exclusive), appended to
 * the command buffer's table, which upstream uploads per render as
 * ISP_SCISSOR_BASE (the pass's scissor_va); the previous draw's entry when
 * the clip did not change */
static VkResult
g5p_scissor_index(struct pvr_cmd_buffer *cmd_buffer, int32_t x0, int32_t y0, int32_t x1, int32_t y1,
                  uint32_t *index)
{
   struct pvr_scissor_words s;
   pvr_csb_pack (&s.w0, IPF_SCISSOR_WORD_0, w0) {
      w0.scw0_xmin = x0;
      w0.scw0_xmax = x1;
   }
   pvr_csb_pack (&s.w1, IPF_SCISSOR_WORD_1, w1) {
      w1.scw1_ymin = y0;
      w1.scw1_ymax = y1;
   }
   g5p_dump("draw", "isp scissor", 0, (const uint32_t[2]){ s.w0, s.w1 }, 2);
   const unsigned n = util_dynarray_num_elements(&cmd_buffer->scissor_array, struct pvr_scissor_words);
   if (n && !memcmp(util_dynarray_top_ptr(&cmd_buffer->scissor_array, struct pvr_scissor_words), &s, sizeof(s))) {
      *index = n - 1;
      return VK_SUCCESS;
   }
   struct pvr_scissor_words *slot = util_dynarray_grow(&cmd_buffer->scissor_array, struct pvr_scissor_words, 1);
   if (!slot)
      return vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
   *slot = s;
   *index = n;
   return VK_SUCCESS;
}

/* Apple's PrimitiveTypeToVDMCTRLType and the captured split counts (170
 * list, 510 strip) */
static bool
g5p_topology(VkPrimitiveTopology t, uint32_t *hw, uint32_t *split, bool *lines)
{
   *lines = false;
   switch (t) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: /* split 512: Apple's g3b-points */
      *hw = ROGUE_VDMCTRL_PRIMITIVE_TOPOLOGY_POINT_LIST;
      *split = 512;
      return true;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      *hw = ROGUE_VDMCTRL_PRIMITIVE_TOPOLOGY_TRI_LIST;
      *split = 170;
      return true;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
      *hw = ROGUE_VDMCTRL_PRIMITIVE_TOPOLOGY_TRI_STRIP;
      *split = 510;
      return true;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      *hw = ROGUE_VDMCTRL_PRIMITIVE_TOPOLOGY_TRI_FAN;
      *split = 510;
      return true;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      *hw = ROGUE_VDMCTRL_PRIMITIVE_TOPOLOGY_LINE_LIST;
      *split = 255;
      *lines = true;
      return true;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      *hw = ROGUE_VDMCTRL_PRIMITIVE_TOPOLOGY_LINE_STRIP;
      *split = 510;
      *lines = true;
      return true;
   default:
      return false;
   }
}

static void
emit_words(struct pvr_csb *csb, const char *what, const uint32_t *w, unsigned n)
{
   uint32_t *dst = pvr_arch_csb_alloc_dwords(csb, n);
   if (dst)
      memcpy(dst, w, 4 * n);
   g5p_dump("draw", what, 0, w, n);
}

/* VDM PPP_STATE: a block of `words` at `addr` */
static void
emit_ppp(struct pvr_csb *csb, const char *what, uint64_t addr, const uint32_t *block, unsigned words)
{
   uint32_t w[2];
   pvr_csb_pack (&w[0], VDMCTRL_PPP_STATE0, s0) {
      s0.word_count = words;
      s0.addrmsb = PVR_DEV_ADDR(addr);
   }
   pvr_csb_pack (&w[1], VDMCTRL_PPP_STATE1, s1) {
      s1.addrlsb = PVR_DEV_ADDR(addr);
   }
   emit_words(csb, "ppp state", w, 2);
   g5p_dump("draw", what, addr, block, words);
}

/*
 * The fragment program's uploads of one draw (the ICD's loader_data()): the
 * words of each in one buffer, from the bound descriptor or the dynamic
 * state; the loader data, their addresses then the DOUTD controls; and the
 * PPP block that points at it (UNIFORMDATABASE). State the A9 has no words
 * for (zero in the descriptor, pvr_g5p_tex.h) is refused by name.
 */
static VkResult
fs_uploads(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g5p_pipeline *g, uint32_t block[2],
           uint64_t *block_va)
{
   const struct pvr_descriptor_state *ds = &cmd_buffer->state.gfx_desc_state;
   const struct vk_dynamic_graphics_state *dyn = &cmd_buffer->vk.dynamic_graphics_state;
   const struct pvr_push_constants *push = &cmd_buffer->state.push_consts[PVR_STAGE_ALLOCATION_FRAGMENT];
   const unsigned n = g->fs_num_uploads;
   /* descriptor words and blend constants are copied into one block (8 dwords per upload); push constants,
    * uniform buffers and buffer addresses are read where they are, as the vertex stage's uploads are
    * (pvr_arch_g5p_upload.c): each upload has its own DOUTD source address */
   uint32_t words[PVR_G5P_MAX_FS_UPLOADS][8] = { { 0 } }, data[4 * PVR_G5P_MAX_FS_UPLOADS];
   uint64_t src[PVR_G5P_MAX_FS_UPLOADS] = { 0 };
   bool copied[PVR_G5P_MAX_FS_UPLOADS] = { false };
   struct pvr_suballoc_bo *bo;
   struct pvr_pds_upload up;
   VkResult result;

   for (unsigned i = 0; i < n; i++) {
      const struct pvr_g5p_upload *u = &g->fs_uploads[i];
      const unsigned bytes = 4 * u->dwords;
      uint32_t *out = words[i];
      uint64_t addr, size;

      switch (u->kind) {
      case PCO_G5P_UPLOAD_BLEND_CONSTANTS: {
         float bc[4];
         if (u->dwords != 4)
            return G5P_REFUSE(cmd_buffer, "blend constants of %u dwords", u->dwords);
         pvr_g5p_blend_constants(dyn->cb.blend_constants, bc);
         memcpy(out, bc, sizeof(bc));
         copied[i] = true;
         continue;
      }
      case PCO_G5P_UPLOAD_PUSH:
         /* the bytes the program reads, as they are now (a copy per draw) */
         if (u->offset + bytes > PVR_MAX_PUSH_CONSTANTS_SIZE)
            return G5P_REFUSE(cmd_buffer, "fragment push constants [%u, %u) past %u bytes", u->offset,
                              u->offset + bytes, PVR_MAX_PUSH_CONSTANTS_SIZE);
         result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, push->data + u->offset, bytes, &bo);
         if (result != VK_SUCCESS)
            return result;
         src[i] = bo->dev_addr.addr;
         continue;
      case PCO_G5P_UPLOAD_UBO:
         /* the buffer itself: the DOUTD reads it when the job runs */
         result = pvr_arch_g5p_upload_buffer(cmd_buffer, u->set, u->binding, u->element, &addr, &size);
         if (result != VK_SUCCESS)
            return result;
         if (u->offset + bytes > size)
            return G5P_REFUSE(cmd_buffer, "fragment uniform buffer at set %u binding %u: [%u, %u) past its %" PRIu64
                              " bytes", u->set, u->binding, u->offset, u->offset + bytes, size);
         src[i] = addr + u->offset;
         continue;
      case PCO_G5P_UPLOAD_UBO_ADDR:
      case PCO_G5P_UPLOAD_SSBO_ADDR: {
         /* the 64-bit address a register-indexed or storage access adds to */
         result = pvr_arch_g5p_upload_buffer(cmd_buffer, u->set, u->binding, u->element, &addr, &size);
         if (result != VK_SUCCESS)
            return result;
         if (u->dwords != 2 || (u->kind == PCO_G5P_UPLOAD_UBO_ADDR && u->offset >= size))
            return G5P_REFUSE(cmd_buffer, "fragment buffer address: %u dwords at offset %u of %" PRIu64 " bytes",
                              u->dwords, u->offset, size);
         const uint64_t va = addr + (u->kind == PCO_G5P_UPLOAD_UBO_ADDR ? u->offset : 0);
         result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, &va, sizeof(va), &bo);
         if (result != VK_SUCCESS)
            return result;
         src[i] = bo->dev_addr.addr;
         continue;
      }
      default:
         break;
      }
      const struct pvr_descriptor_set *set = u->set < PVR_MAX_DESCRIPTOR_SETS ? ds->sets[u->set] : NULL;
      if (!set || !set->layout || u->binding >= set->layout->binding_count)
         return G5P_REFUSE(cmd_buffer, "fragment upload: set %u binding %u is not bound", u->set, u->binding);
      const struct pvr_descriptor_set_layout_binding *lb = &set->layout->bindings[u->binding];
      /* an arrayed binding: the element the program reads (a sampler array at a constant index) */
      if (u->element >= MAX2(lb->descriptor_count, 1))
         return G5P_REFUSE(cmd_buffer, "fragment upload: set %u binding %u element %u past its %u descriptors",
                           u->set, u->binding, u->element, lb->descriptor_count);
      const uint32_t *desc = (const uint32_t *)((const uint8_t *)set->mapping + lb->offset +
                                                (size_t)u->element * lb->stride);
      if (lb->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) {
         if (!pvr_g5p_tbo_upload(u->kind, u->offset, u->dwords, desc, out))
            return G5P_REFUSE(cmd_buffer, "unqualified texel-buffer upload at set %u binding %u element %u",
                              u->set, u->binding, u->element);
         copied[i] = true;
         continue;
      }
      switch (u->kind) {
      case PCO_G5P_UPLOAD_IMAGE:
         if (!pvr_g5p_tex_state_valid(desc))
            return G5P_REFUSE(cmd_buffer, "set %u binding %u: a view without A9 texture state", u->set,
                              u->binding);
         memcpy(out, desc, 4 * u->dwords);
         break;
      case PCO_G5P_UPLOAD_SAMPLER:
         /* a combined descriptor's sampler is at +32 (the upload's offset),
          * a standalone sampler's at +0 */
         if (lb->type != VK_DESCRIPTOR_TYPE_SAMPLER)
            desc += u->offset / 4;
         if (!pvr_g5p_sampler_valid(desc))
            return G5P_REFUSE(cmd_buffer, "set %u binding %u: a sampler without A9 words", u->set, u->binding);
         memcpy(out, desc, 4 * u->dwords);
         break;
      case PCO_G5P_UPLOAD_IMAGE_SIZE:
         if (u->dwords != 4 || !pvr_g5p_tex_size(desc, u->offset, out))
            return G5P_REFUSE(cmd_buffer, "set %u binding %u: an image size without A9 texture state", u->set,
                              u->binding);
         break;
      default:
         return G5P_REFUSE(cmd_buffer, "fragment upload kind %u", u->kind);
      }
      copied[i] = true;
   }
   bool any_copied = false;
   for (unsigned i = 0; i < n; i++)
      any_copied |= copied[i];
   if (any_copied) {
      result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, words, sizeof(words[0]) * n, &bo);
      if (result != VK_SUCCESS)
         return result;
      for (unsigned i = 0; i < n; i++)
         if (copied[i])
            src[i] = bo->dev_addr.addr + sizeof(words[0]) * i;
   }
   uint32_t staging[PCO_G5P_FS_COMMON_DWORDS] = {0};
   uint64_t staging_va = 0;
   if (n > 12 && any_copied) {
      for (unsigned i = 0; i < n; i++)
         if (copied[i])
            memcpy(staging + g->fs_uploads[i].common, words[i], 4 * g->fs_uploads[i].dwords);
      result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, staging, 4 * g->fs_common_dwords, &bo);
      if (result != VK_SUCCESS) return result;
      staging_va = bo->dev_addr.addr;
   }
   const unsigned dmas = g->fs_num_dmas;
   for (unsigned i = 0; i < dmas; i++) {
      const int upload = g->fs_dma[i].upload;
      uint64_t va = upload < 0 ? staging_va + 4 * g->fs_dma[i].common : src[upload];
      data[2 * i] = (uint32_t)va;
      data[2 * i + 1] = (uint32_t)(va >> 32);
      data[2 * dmas + 2 * i] = doutd_control(g->fs_dma[i].common, g->fs_dma[i].dwords, i == dmas - 1);
      data[2 * dmas + 2 * i + 1] = 0;
   }
   result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, data, 4 * dmas, 16, NULL, 0, 0, 16, &up);
   if (result != VK_SUCCESS)
      return result;
   pvr_csb_pack (&block[0], TA_STATE_HEADER, h) {
      h.pres_pds_state_ptr3 = true;
   }
   pvr_csb_pack (&block[1], TA_STATE_PDS_UNIFORMDATABASE, ub) {
      ub.addr = PVR_DEV_ADDR(up.data_offset);
   }
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, block, 8, &bo);
   if (result != VK_SUCCESS)
      return result;
   *block_va = bo->dev_addr.addr;
   for (unsigned i = 0; i < n; i++)
      g5p_dump("draw", "fs upload words", bo->dev_addr.addr, words[i], g->fs_uploads[i].dwords);
   g5p_dump("draw", "uniform data", up.data_offset, data, 4 * dmas);
   return VK_SUCCESS;
}

/* What the render's draws do with its depth/stencil attachment, for the job's ZLS load and store decisions
 * (pvr_arch_cmd_buffer.c, pvr_sub_cmd_gfx_job_init()): the flags Rogue's pvr_validate_draw_state() records,
 * which a G5P draw never reaches. Vulkan's rules: a depth test reads the attachment when it is enabled and its
 * compare is not ALWAYS, and writes only while the test is enabled; a stencil test reads it when enabled with a
 * compare other than ALWAYS on a face, and writes it when an op of a face is not KEEP under a non-zero write mask.
 * Without these, every render left depth_usage UNDEFINED and modifies_* false: its ZLS load always ran, and on a
 * tile-aligned render area a LOAD pass dropped the store of depth or stencil its draws wrote. */
static void
g5p_record_ds_usage(struct pvr_sub_cmd_gfx *sub, const struct vk_dynamic_graphics_state *dyn)
{
   if (dyn->rs.rasterizer_discard_enable)
      return; /* no fragment is tested */
   const bool dt = dyn->ds.depth.test_enable, st = dyn->ds.stencil.test_enable;
   if (dt && dyn->ds.depth.compare_op != VK_COMPARE_OP_ALWAYS && sub->depth_usage == PVR_DEPTH_STENCIL_USAGE_UNDEFINED)
      sub->depth_usage = PVR_DEPTH_STENCIL_USAGE_NEEDED;
   if (dt && dyn->ds.depth.write_enable)
      sub->modifies_depth = true;
   if (!st)
      return;
   const struct vk_stencil_test_face_state *face[2] = { &dyn->ds.stencil.front, &dyn->ds.stencil.back };
   for (unsigned i = 0; i < 2; i++) {
      if (face[i]->op.compare != VK_COMPARE_OP_ALWAYS && sub->stencil_usage == PVR_DEPTH_STENCIL_USAGE_UNDEFINED)
         sub->stencil_usage = PVR_DEPTH_STENCIL_USAGE_NEEDED;
      if (face[i]->write_mask && (face[i]->op.fail != VK_STENCIL_OP_KEEP || face[i]->op.pass != VK_STENCIL_OP_KEEP ||
                                  face[i]->op.depth_fail != VK_STENCIL_OP_KEEP))
         sub->modifies_stencil = true;
   }
}

VkResult
PVR_PER_ARCH(g5p_cmd_draw)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g5p_draw *draw)
{
   struct pvr_device *device = cmd_buffer->device;
   const struct pvr_g5p_programs *dp = &device->g5p->programs;
   struct pvr_cmd_buffer_state *cs = &cmd_buffer->state;
   const struct pvr_graphics_pipeline *pipeline = cs->gfx_pipeline;
   const struct pvr_g5p_pipeline *g = pipeline ? pipeline->g5p : NULL;
   const struct vk_dynamic_graphics_state *dyn = &cmd_buffer->vk.dynamic_graphics_state;
   const struct pvr_render_pass_info *info = &cs->render_pass_info;
   uint32_t hw_topology, split;
   bool lines;

   if (!g)
      return G5P_REFUSE(cmd_buffer, "a draw without a G5P graphics pipeline");
   /* transform feedback: while capturing, the pipeline's capturing variant
    * (pvr_g5p_xfb.h); a pipeline that captures nothing draws as it is */
   const bool xfb = g->xfb && pvr_arch_g5p_xfb_capturing(cmd_buffer);
   if (draw->indirect_addr && pvr_arch_g5p_xfb_capturing(cmd_buffer))
      return G5P_REFUSE(cmd_buffer, "indirect graphics transform-feedback accounting is not implemented");
   if (xfb) {
      VkResult xr = pvr_arch_g5p_xfb_draw_check(cmd_buffer, dyn->ia.primitive_topology);
      if (xr != VK_SUCCESS)
         return xr;
      g = g->xfb;
   }
   if (!draw->count || !draw->instances)
      return VK_SUCCESS;
   if (!g5p_topology(dyn->ia.primitive_topology, &hw_topology, &split, &lines))
      return G5P_REFUSE(cmd_buffer, "topology %d", dyn->ia.primitive_topology);
   VkResult result;
   if (g->bresenham_line_probe && (!lines || dyn->rs.line.width != 1.0f || g->samples != 1))
      return G5P_REFUSE(cmd_buffer, "Bresenham lines need line topology, width 1 and one sample");
   /* rasterizer discard: the vertex program runs (transform feedback), no
    * fragment is shaded or tested (pvr_g5p_xfb.h; the ISP words below); a
    * pipeline that discards has no viewport state: the render area stands
    * in for its viewport and scissor (our A9 driver's form) */
   bool discard = dyn->rs.rasterizer_discard_enable;
   if (!discard && (dyn->vp.viewport_count != 1 || dyn->vp.scissor_count != 1))
      return G5P_REFUSE(cmd_buffer, "%u viewports, %u scissors (one of each)", dyn->vp.viewport_count,
                        dyn->vp.scissor_count);
   if (dyn->rs.cull_mode == VK_CULL_MODE_FRONT_AND_BACK)
      return G5P_REFUSE(cmd_buffer, "culling front and back");
   if (!info->rstate)
      return G5P_REFUSE(cmd_buffer, "a draw outside a render pass");
   /* the pipeline rasterizes the pass's sample count (its launch's sample rate, the ISP user pass) */
   if (info->hw_render && MAX2(info->hw_render->sample_count, 1) != g->samples)
      return G5P_REFUSE(cmd_buffer, "a %u-sample pipeline in a %u-sample pass", g->samples,
                        info->hw_render->sample_count);
   /* one emitpix writes every target: the program's count is the pass's */
   VkResult mrt = pvr_arch_g5p_mrt_draw_check(cmd_buffer, g->color_targets);
   if (mrt != VK_SUCCESS)
      return mrt;

   result = pvr_arch_cmd_buffer_start_sub_cmd(cmd_buffer, PVR_SUB_CMD_TYPE_GRAPHICS);
   if (result != VK_SUCCESS)
      return result;
   /* Reserve one full CSB block for the next draw and end-of-pass words.
    * A G5P draw emits < 256 bytes of VDM (PPP payloads live elsewhere).
    * Counting full blocks overestimates the flattened length, including
    * relocation slack. This also provides old-kernel compatibility without
    * changing the application's render pass. */
   struct pvr_csb *stream = &cs->current_sub_cmd->gfx.control_stream;
   const uint64_t occupied = (uint64_t)list_length(&stream->pvr_bo_list) *
                             PVR_CMD_BUFFER_CSB_BO_SIZE;
   if (occupied + PVR_CMD_BUFFER_CSB_BO_SIZE > device->ws->max_vdm_bytes) {
      if (getenv("PVR_G5P_WS_LOG"))
         fprintf(stderr, "g5p vdm: continue with LOAD at limit %u\n", device->ws->max_vdm_bytes);
      result = pvr_cmd_buffer_insert_mid_frag_barrier_event(cmd_buffer,
                                                           PVR_PIPELINE_STAGE_FRAG_BIT,
                                                           PVR_PIPELINE_STAGE_GEOM_BIT);
      if (result != VK_SUCCESS)
         return result;
   }
   g5p_record_ds_usage(&cs->current_sub_cmd->gfx, dyn);
   /* Starting the sub-command establishes the actual DS attachment even
    * for a discarding first draw after CLEAR. Select before any shader-specific
    * vertex fetch, uploads, varyings or fragment state is emitted. */
   const VkImageAspectFlags punch_aspects = dyn->rs.rasterizer_discard_enable
                                              ? 0 : pvr_arch_g5p_ds_aspects(cmd_buffer);
   if (punch_aspects) {
      struct vk_depth_stencil_state ds = dyn->ds;
      vk_optimize_depth_stencil_state(&ds, punch_aspects, true);
      const bool testing = ds.depth.test_enable || ds.stencil.test_enable;
      const bool writing = ds.depth.write_enable || ds.stencil.write_enable;
      if ((testing && g->fs_discard_tokens > 1) || (writing && g->fs_discard_select)) {
         if (!g->punch)
            return G5P_REFUSE(cmd_buffer, "discard under depth/stencil requires an unsupported punch-through variant");
         g = g->punch;
      }
      if (testing && g->fs_discard && g->samples != 1)
         return G5P_REFUSE(cmd_buffer, "discard under multisample depth/stencil is not hardware-qualified");
   }
   result = pvr_arch_g5p_vertex_draw_check(cmd_buffer, g, draw);
   if (result != VK_SUCCESS)
      return result;
   if (g->fs_depth_feedback)
      cs->current_sub_cmd->gfx.has_depth_feedback = true;
   /* what the draw can read outside the arena, for the job's resource list (pvr_g5p_perf.h) */
   pvr_arch_g5p_perf_draw(cmd_buffer, draw->indexed);

   /* the clip: the render area within the attachment, cut to the scissor
    * and to the viewport. A clip smaller than the attachment is the A9 ICD's
    * (g5p_job_draw()): the TA region clip in whole tiles, and the ISP scissor
    * table entry the pixels are cut to. The tiles outside a partial render
    * area keep the LOAD background (pvr_arch_g5p.c), so they are stored back
    * unchanged. The PPP clips to a guard band, not to the viewport, so the
    * viewport bounds the pixels here: Apple's ClipRectContext::setViewports()
    * and updateClipRegions() (AGXMetalA9) take the floor of its low edge and
    * the ceiling of its high edge, for either sign of its extent, and
    * intersect it with the scissor; Rogue's driver intersects them too. */
   const uint32_t width = info->rstate->width, height = info->rstate->height;
   const VkRect2D *ra = &info->render_area;
   const int64_t ra_x0 = MAX2(ra->offset.x, 0), ra_y0 = MAX2(ra->offset.y, 0);
   const int64_t ra_x1 = MIN2((int64_t)ra->offset.x + ra->extent.width, (int64_t)width);
   const int64_t ra_y1 = MIN2((int64_t)ra->offset.y + ra->extent.height, (int64_t)height);
   int64_t cx0 = ra_x0, cy0 = ra_y0, cx1 = ra_x1, cy1 = ra_y1;
   if (!discard) {
      const VkRect2D *sc = &dyn->vp.scissors[0];
      const VkViewport *vp = &dyn->vp.viewports[0];
      const double vx0 = MIN2((double)vp->x, (double)vp->x + vp->width);
      const double vx1 = MAX2((double)vp->x, (double)vp->x + vp->width);
      const double vy0 = MIN2((double)vp->y, (double)vp->y + vp->height);
      const double vy1 = MAX2((double)vp->y, (double)vp->y + vp->height);
      cx0 = MAX3(cx0, (int64_t)sc->offset.x, (int64_t)CLAMP(floor(vx0), -1.0, (double)width));
      cy0 = MAX3(cy0, (int64_t)sc->offset.y, (int64_t)CLAMP(floor(vy0), -1.0, (double)height));
      cx1 = MIN3(cx1, (int64_t)sc->offset.x + sc->extent.width, (int64_t)CLAMP(ceil(vx1), -1.0, (double)width));
      cy1 = MIN3(cy1, (int64_t)sc->offset.y + sc->extent.height, (int64_t)CLAMP(ceil(vy1), -1.0, (double)height));
   }
   if (cx1 <= cx0 || cy1 <= cy0) {
      /* No pixel is left. Vulkan still runs the vertex program once per
       * vertex: when it writes memory or captures transform feedback, the
       * draw takes the rasterizer-discard form (no fragment is shaded or
       * tested; pvr_g5p_xfb.h) over the render area, else it emits nothing. */
      if (discard || !(xfb || g->vs_side_effects) || ra_x1 <= ra_x0 || ra_y1 <= ra_y0)
         return VK_SUCCESS;
      discard = true;
      cx0 = ra_x0, cy0 = ra_y0, cx1 = ra_x1, cy1 = ra_y1;
   }
   const int32_t x0 = cx0, y0 = cy0, x1 = cx1, y1 = cy1;
   /* The ISP shades whole 32-pixel tiles: without a scissor, the pixels of
    * an edge tile past the attachment (a 31x29 target's column 31 and rows
    * 29..31) run the fragment program as covered pixels. The PBE cuts their
    * colour to the attachment, so colour-only programs need no scissor
    * (Apple's updateClipRegions() enables it only for a clip inside the
    * attachment, as above). Their memory stores are not cut (hardware,
    * zink-fs-store: 116 such words, every one past the 31x29 attachment),
    * and Vulkan's scissor test removes those fragments before shading: a
    * program with side effects takes the ISP scissor unless the clip ends on
    * tile edges, as Rogue's driver scissors every draw
    * (pvr_arch_cmd_buffer.c). Under rasterizer discard no fragment runs: no
    * scissor (Rogue's scenable = !rasterizer_discard). */
   const bool side_effects_past_clip =
      !discard && g->fs_side_effects && ((x1 % G5P_TILE) || (y1 % G5P_TILE));
   const bool scissor = x0 || y0 || x1 != (int32_t)width || y1 != (int32_t)height || side_effects_past_clip;
   uint32_t scissor_index = 0;
   if (scissor) {
      result = g5p_scissor_index(cmd_buffer, x0, y0, x1, y1, &scissor_index);
      if (result != VK_SUCCESS)
         return result;
   }

   /* flat varyings take vertex 0; a fan's first vertex is the hardware's vertex 1 */
   if (dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN && g->fs_flat)
      return G5P_REFUSE(cmd_buffer, "flat varyings with a triangle fan");
   /* a depth/stencil attachment: its tests and bias (pvr_arch_g5p_ds.c) */
   /* rasterizer discard: no depth/stencil aspects (Rogue's rule), the colour-only ISP form */
   const VkImageAspectFlags ds_aspects = discard ? 0 : pvr_arch_g5p_ds_aspects(cmd_buffer);
   /* Zink keeps bias enabled with zero factors when polygon offset is off.
    * Only a potential nonzero offset needs the bias table or its refusals. */
   const bool dbias = ds_aspects && dyn->rs.depth_bias.enable &&
                      (dyn->rs.depth_bias.constant_factor != 0.0f || dyn->rs.depth_bias.slope_factor != 0.0f);
   if (g->fs_depth_feedback && dbias)
      return G5P_REFUSE(cmd_buffer, "fragment depth export with depth bias is not qualified");
   uint32_t dbias_index = 0;
   /* Real single-token feedback defers depth/stencil writes until the kill
    * result is known (g5p-punch-ds@8150a8195df); preserve all DS test/ops. */
   if ((ds_aspects & VK_IMAGE_ASPECT_DEPTH_BIT) && dyn->ds.depth.bounds_test.enable)
      return G5P_REFUSE(cmd_buffer, "a depth bounds test");
   /* line/point fill with an effective polygon offset (non-zero factors on a depth attachment): not qualified */
   if (dbias && g->polygon_mode != VK_POLYGON_MODE_FILL &&
       (dyn->rs.depth_bias.constant_factor != 0.0f || dyn->rs.depth_bias.slope_factor != 0.0f))
      return G5P_REFUSE(cmd_buffer, "polygon offset with line or point fill is not qualified");
   if (dbias) {
      result = pvr_arch_g5p_ds_bias(cmd_buffer, dyn, &dbias_index);
      if (result != VK_SUCCESS)
         return result;
   }

   /* 3. the vertex fetch data: bases, controls, strides, the DOUTU word */
   uint32_t fdata[4 * 2 * PVR_G5P_MAX_FETCH + 8] = { 0 };
   /* a first instance takes the program that adds it to t1 */
   const bool base_instance = draw->first_instance && fetch_reads_instance(g);
   /* The A9 indirect variant reads ptemp1, but its initializer is not yet
    * recovered/qualified. Keep base-zero indirect on the proven fetch. */
   const struct pvr_pds_upload *fetch_code = base_instance ? &g->fetch_base : &g->fetch;
   const unsigned fbytes = fetch_data_bytes(g, base_instance);
   if (base_instance)
      fdata[fetch_base_instance(g->fetch_count)] = draw->first_instance;
   {
      uint32_t div[PVR_G5P_MAX_DIVISORS];
      unsigned slot[PVR_G5P_MAX_FETCH];
      const unsigned ndiv = fetch_divisors(g, div, slot);
      for (unsigned k = 0; k < ndiv; k++)
         fdata[fetch_multipliers(g->fetch_count, g->vs_id_mask) + k] = divisor_multiplier(div[k]);
   }
   {
      const uint64_t doutu = doutu_src0(usc_off(device, g->vs_usc), g->vs_temps);
      fdata[fetch_doutu(g->fetch_count)] = (uint32_t)doutu;
      fdata[fetch_doutu(g->fetch_count) + 1] = (uint32_t)(doutu >> 32);
      unsigned k = 0;
      const unsigned last_id = g->vs_id_mask ? util_last_bit(g->vs_id_mask) - 1 : 0;
      u_foreach_bit (i, g->vs_id_mask)
         fdata[fetch_ids(g->fetch_count) + k++] = doutw_control(g->vs_id_reg[i], i == 1, i == last_id);
      for (unsigned f = 0; f < g->fetch_count; f++) {
         const struct pvr_g5p_fetch *fe = &g->fetch_list[f];
         const struct pvr_vertex_binding *vb = &cs->vertex_bindings[fe->binding];
         if (!vb->buffer)
            return G5P_REFUSE(cmd_buffer, "no vertex buffer bound at binding %u", fe->binding);
         const uint32_t stride = dyn->vi_binding_strides[fe->binding];
         /* divisor 0: every instance reads the first instance's element */
         const bool divisor0 = fe->rate == PVR_G5P_FETCH_INSTANCE_DIVISOR0;
         const uint64_t base = vb->buffer->dev_addr.addr + vb->offset + fe->offset +
                               (divisor0 ? (uint64_t)draw->first_instance * stride : 0);
         const uint64_t ctl = ddmad_src3(fe->dwords, fe->ao, f == g->fetch_count - 1);
         fdata[8 * f] = (uint32_t)base;
         fdata[8 * f + 1] = (uint32_t)(base >> 32);
         fdata[8 * f + 4] = (uint32_t)ctl;
         fdata[8 * f + 5] = (uint32_t)(ctl >> 32);
         fdata[8 * f + 6] = divisor0 ? 0 : stride;
      }
   }
   struct pvr_pds_upload fetch_data;
   result = pvr_arch_cmd_buffer_upload_pds(cmd_buffer, fdata, fbytes / 4, 16, NULL, 0, 0, 16, &fetch_data);
   if (result != VK_SUCCESS)
      return result;

   /* the draw's PPP blocks, each 16-byte aligned in one buffer (as the G9P) */
   uint32_t ppp[64] = { 0 }, at = 0, blocks[8][2], nb = 0;
#define PPP_BLOCK(words, n)                                                                                        \
   do {                                                                                                            \
      blocks[nb][0] = at;                                                                                          \
      blocks[nb++][1] = (n);                                                                                       \
      memcpy(&ppp[at], (words), 4 * (n));                                                                          \
      at += ALIGN_POT((n), 4);                                                                                     \
   } while (0)
   uint32_t w[16];
   /* 5. output select */
   pvr_csb_pack (&w[0], TA_STATE_HEADER, h) {
      h.pres_outselects = true;
   }
   pvr_csb_pack (&w[1], TA_OUTPUT_SEL, os) {
      os.vtxsize = g->vs_outputs;
      os.rhw_pres = true;
      /* points: the size follows the position (g3b-points) */
      os.psprite_size_pres = g->vs_point_size;
   }
   PPP_BLOCK(w, 2);
   /* 6. ISP: the colour-only form (no depth/stencil attachment); Apple's
    * pass rule (pvr_g5p_blend.h): a program that reads the tile runs
    * TRANSLUCENT, a discarding one punch-through, else OPAQUE */
   {
      struct pvr_g5p_blend_isp isp;
      pvr_arch_g5p_blend_isp(g->fs_blend_mode, g->fs_discard, g->fs_write_mask, discard, &isp);
      const enum ROGUE_TA_PASSTYPE pass = g->fs_depth_feedback ? ROGUE_TA_PASSTYPE_DEPTH_FEEDBACK : isp.pass_type;
      /* points rasterize as sprites (g3b-points: SPRITE_01UV, front and back) */
      /* triangles in line/point polygon mode: AGXMetalA9 encodeISPPrimitiveType
       * 0x207d66360 picks LINE_FILLED_TRIANGLE (fill mode 1) or
       * POINT_FILLED_TRIANGLE (2) for triangle primitives; its
       * encodeAndEmitRenderState 0x207d67814 sets pointlinewidth from the line
       * width, or for point fill from a point size, as width * 16 - 1 */
      const bool tri_fill_mode = !lines && !g->vs_point_size && g->polygon_mode != VK_POLYGON_MODE_FILL;
      const enum ROGUE_TA_OBJTYPE objtype = g->vs_point_size ? ROGUE_TA_OBJTYPE_SPRITE_01UV
                                            : lines         ? ROGUE_TA_OBJTYPE_LINE
                                            : !tri_fill_mode ? ROGUE_TA_OBJTYPE_TRIANGLE
                                            : g->polygon_mode == VK_POLYGON_MODE_LINE ? ROGUE_TA_OBJTYPE_LINE_FILLED_TRIANGLE
                                                                                       : ROGUE_TA_OBJTYPE_POINT_FILLED_TRIANGLE;
      /* point fill: the vertex program writes no gl_PointSize (refused at
       * pipeline creation), so the points are 1.0 (Vulkan's default) */
      const float width = tri_fill_mode && g->polygon_mode == VK_POLYGON_MODE_POINT ? 1.0f : dyn->rs.line.width;
      uint32_t lw = util_unsigned_fixed(width, 4);
      lw = MIN2(lw ? lw - 1 : 0, 255);
      pvr_csb_pack (&w[0], TA_STATE_HEADER, h) {
         h.pres_ispctl = true;
         h.pres_ispctl_fa = true;
         h.pres_ispctl_fb = true;
         h.pres_ispctl_ba = true;
         h.pres_ispctl_bb = true;
      }
      pvr_csb_pack (&w[1], TA_STATE_ISPCTL, ctl) {
         /* 0 for a program that writes no colour (pvr_g5p_blend.h), else the sample rate's user pass
          * (pvr_g5p_msaa.h: 15, or 7 at 4x) */
         ctl.upass = isp.upass ? pvr_g5p_msaa_upass(g->samples) : 0;
         ctl.bpres = true;
         ctl.scenable = scissor;
         ctl.tagwritedisable = isp.tag_write_disable;
      }
      pvr_csb_pack (&w[2], TA_STATE_ISPA, fa) {
         fa.objtype = objtype;
         fa.passtype = pass;
         fa.dwritedisable = true;
         fa.dfbztestenable = true;
         fa.dcmpmode = ROGUE_TA_CMPMODE_ALWAYS;
         fa.pointlinewidth = lw;
      }
      pvr_csb_pack (&w[3], TA_STATE_ISPB, fb) {
         fb.scmpmode = ROGUE_TA_CMPMODE_ALWAYS;
      }
      /* the captured back words: single-sided ISPCTL ignores them */
      pvr_csb_pack (&w[4], TA_STATE_ISPA, ba) {
         ba.objtype = objtype;
         ba.passtype = pass;
         ba.dfbztestenable = true;
         ba.pointlinewidth = lw;
      }
      /* points: the back B word is the front's (g3b-points) */
      w[5] = g->vs_point_size ? w[3] : 0;
      /* with a depth/stencil attachment: its tests, Apple's face form */
      if (ds_aspects)
         pvr_arch_g5p_ds_isp(dyn, ds_aspects, g->samples, lines, pass, scissor, dbias, false, &w[1]);
      /* DS packing owns tests/ops, while shader feedback owns tag/upass.
       * Mask-zero kill programs still need feedback, with upass0. */
      struct ROGUE_TA_STATE_ISPCTL ctl;
      ROGUE_TA_STATE_ISPCTL_unpack(&w[1], &ctl);
      ctl.upass = isp.upass ? pvr_g5p_msaa_upass(g->samples) : 0;
      ctl.tagwritedisable = isp.tag_write_disable;
      /* Rogue TA_STATE_ISPCTL: VISTEST enables accumulated sample counting;
       * VISREG selects the counter in ISP_OCLQRY_BASE. Match the generic
       * builder's discard rule, after DS has rebuilt the ISP control word. */
      if (!discard && cmd_buffer->state.vis_test_enabled) {
         ctl.vistest = true;
         ctl.visreg = cmd_buffer->state.vis_reg;
      }
      ROGUE_TA_STATE_ISPCTL_pack(&w[1], &ctl);
      /* ... with the sprite object type of a point pipeline, or a triangle's fill mode */
      if (ds_aspects && (g->vs_point_size || tri_fill_mode)) {
         for (unsigned k = 2; k <= 4; k += 2) {
            struct ROGUE_TA_STATE_ISPA a;
            ROGUE_TA_STATE_ISPA_unpack(&w[k], &a);
            a.objtype = objtype;
            if (tri_fill_mode)
               a.pointlinewidth = lw;
            ROGUE_TA_STATE_ISPA_pack(&w[k], &a);
         }
      }
      PPP_BLOCK(w, 6);
   }
   /* 7. region clip (32-pixel tiles) and the viewport */
   {
      pvr_csb_pack (&w[0], TA_STATE_HEADER, h) {
         h.pres_region_clip = true;
         h.pres_viewport = true;
      }
      pvr_csb_pack (&w[1], TA_REGION_CLIP0, c0) {
         c0.mode = ROGUE_TA_REGION_CLIP_MODE_OUTSIDE;
         c0.left = x0 / G5P_TILE;
         c0.right = DIV_ROUND_UP(x1, G5P_TILE) - 1;
      }
      pvr_csb_pack (&w[2], TA_REGION_CLIP1, c1) {
         c1.top = y0 / G5P_TILE;
         c1.bottom = DIV_ROUND_UP(y1, G5P_TILE) - 1;
      }
      const VkViewport area_vp = { ra->offset.x, ra->offset.y, ra->extent.width, ra->extent.height, 0.0f, 1.0f };
      const VkViewport *vp = discard ? &area_vp : &dyn->vp.viewports[0];
      const float f[6] = { vp->x + vp->width * 0.5f, vp->width * 0.5f, vp->y + vp->height * 0.5f,
                           vp->height * 0.5f, vp->minDepth, vp->maxDepth - vp->minDepth };
      memcpy(&w[3], f, sizeof(f));
      PPP_BLOCK(w, 9);
   }
   /* 8. W clamp and PPP control */
   {
      pvr_csb_pack (&w[0], TA_STATE_HEADER, h) {
         h.pres_wclamp = true;
         h.pres_ppp_ctrl = true;
      }
      w[1] = G5P_WCLAMP;
      pvr_csb_pack (&w[2], TA_STATE_PPP_CTRL, c) {
         c.clip_mode = ROGUE_TA_CLIP_MODE_FRONT_REAR;
         c.flatshade_vtx = ROGUE_TA_FLATSHADE_VTX_VERTEX_0;
         c.wclampen = true;
         if (dyn->rs.cull_mode == VK_CULL_MODE_BACK_BIT || dyn->rs.cull_mode == VK_CULL_MODE_FRONT_BIT) {
            /* Rogue's rule (pvr_setup_ppp_control) */
            const bool ccw = dyn->rs.front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;
            c.cullmode = (ccw ^ (dyn->rs.cull_mode == VK_CULL_MODE_FRONT_BIT)) ? ROGUE_TA_CULLMODE_CULL_CW
                                                                                : ROGUE_TA_CULLMODE_CULL_CCW;
         }
      }
      PPP_BLOCK(w, 3);
   }
   /* 9. depth bias and scissor indices: the draw's entries of the render's tables */
   {
      pvr_csb_pack (&w[0], TA_STATE_HEADER, h) {
         h.pres_ispctl_dbsc = true;
      }
      pvr_csb_pack (&w[1], TA_STATE_ISPDBSC, dbsc) {
         dbsc.dbindex = dbias_index;
         dbsc.scindex = scissor_index;
      }
      PPP_BLOCK(w, 2);
   }
   /* 10. stream out to the TA: the device HALT's 16 zero bytes of data, its
    * HALT at +0x10 (as the G9P stream_out) */
   {
      pvr_csb_pack (&w[0], TA_STATE_HEADER, h) {
         h.pres_stream_out_size = true;
         h.pres_stream_out_program = true;
      }
      pvr_csb_pack (&w[1], TA_STATE_STREAM_OUT0, s0) {
         s0.stream0_ta_output = true;
      }
      w[2] = G5P_STREAM_OUT1;
      pvr_csb_pack (&w[3], TA_STATE_STREAM_OUT2, s2) {
         s2.pds_data_addr = PVR_DEV_ADDR(dp->halt.data_offset);
      }
      PPP_BLOCK(w, 4);
   }
#undef PPP_BLOCK

   struct pvr_suballoc_bo *ppp_bo;
   result = pvr_arch_cmd_buffer_upload_general(cmd_buffer, ppp, 4 * at, &ppp_bo);
   if (result != VK_SUCCESS)
      return result;
   const uint64_t ppp_va = ppp_bo->dev_addr.addr;

   /* the fragment program's uploads: their data and PPP block */
   uint32_t uniform_block[2];
   uint64_t uniform_va = 0;
   if (g->fs_num_uploads) {
      result = fs_uploads(cmd_buffer, g, uniform_block, &uniform_va);
      if (result != VK_SUCCESS)
         return result;
   }

   /* the vertex program's uploads: their sources, loader data and PDS_STATE words */
   struct pvr_g5p_vs_upload_draw vs_up = { 0 };
   if (g->vs_uploads) {
      result = pvr_arch_g5p_vs_uploads_draw(cmd_buffer, g->vs_uploads, &vs_up);
      if (result != VK_SUCCESS)
         return result;
   }
   /* the captured draw's event: its block is filled when the job is submitted */
   if (xfb) {
      result = pvr_arch_g5p_xfb_draw(cmd_buffer, &g->xfb_layout, (uint8_t *)vs_up.xfb_block, draw->count,
                                     draw->instances, draw->first, dyn->ia.primitive_topology);
      if (result != VK_SUCCESS)
         return result;
   }

   /* the draw's words move whole into the next block when the stream fills
    * up mid-draw (pvr_csb's relocation mark) */
   struct pvr_csb *csb = &cs->current_sub_cmd->gfx.control_stream;
   pvr_csb_set_relocation_mark(csb);

   /* 1. VS code, outputs, allocation */
   {
      const unsigned vector = g->vs_outputs + 5;
      const unsigned cam = vector <= 32 ? G5P_VDM_CAM_SIZE : vector <= 48 ? 63 : vector <= 64 ? 31 : 15;
      pvr_csb_pack (&w[0], VDMCTRL_VDM_STATE0, v0) {
         v0.vs_other_present = true;
         v0.cam_size = cam;
      }
      pvr_csb_pack (&w[1], VDMCTRL_VDM_STATE3, v3) {
         v3.vs_pds_code_base_addr = PVR_DEV_ADDR(fetch_code->code_offset);
      }
      pvr_csb_pack (&w[2], VDMCTRL_VDM_STATE4, v4) {
         v4.vs_output_size = g->vs_outputs;
      }
      struct ROGUE_VDMCTRL_VDM_STATE5 v5;
      const uint32_t state5 = G5P_VS_STATE5;
      ROGUE_VDMCTRL_VDM_STATE5_unpack(&state5, &v5);
      v5.vs_pds_data_size = fbytes / 16;
      /* the fetch program's temps: t0:t1 and one pair per divisor; Apple's
       * VertexProgramVariantPDSState sizes them 2 * ceil(temps / 4)
       * (vertex-temps.md), which G5P_VS_STATE5's 2 covers up to t3 */
      {
         uint32_t div[PVR_G5P_MAX_DIVISORS];
         unsigned slot[PVR_G5P_MAX_FETCH];
         v5.vs_pds_temp_size = 2 * DIV_ROUND_UP(2 + 2 * fetch_divisors(g, div, slot), 4);
      }
      v5.vs_usc_unified_size = g->vs_unified;
      ROGUE_VDMCTRL_VDM_STATE5_pack(&w[3], &v5);
      /* a vertex above 64 output dwords: bit 30 (no csbgen field; AGXMetalA9) */
      if (vector > 64)
         w[3] |= BITFIELD_BIT(30);
      emit_words(csb, "vdm state vs", w, 4);
      g5p_dump("draw", "fetch code", fetch_code->code_offset,
               pvr_bo_suballoc_get_map_addr(fetch_code->pvr_bo), fetch_code->code_size);

      /* 2. the varying format */
      emit_ppp(csb, "ppp varying", g->static_vary->dev_addr.addr, g->static_vary_words, 4);

      /* 2b. the vertex program's common store: its loader (pvr_g5p_upload.h) */
      if (g->vs_uploads)
         emit_words(csb, "vdm pds state", vs_up.pds_state, 3);

      /* 3. the fetch data */
      pvr_csb_pack (&w[0], VDMCTRL_VDM_STATE0, v0) {
         v0.vs_data_addr_present = true;
         v0.cam_size = cam;
      }
      pvr_csb_pack (&w[1], VDMCTRL_VDM_STATE2, v2) {
         v2.vs_pds_data_base_addr = PVR_DEV_ADDR(fetch_data.data_offset);
      }
      emit_words(csb, "vdm state data", w, 2);
      g5p_dump("draw", "fetch data", fetch_data.data_offset, fdata, fbytes / 4);
   }

   /* 4. the fragment pointers and sizes */
   emit_ppp(csb, "ppp fragment", g->static_frag->dev_addr.addr, g->static_frag_words, 6);
   g5p_dump("draw", "fs launch data", g->fs_launch.data_offset, pvr_bo_suballoc_get_map_addr(g->fs_launch.pvr_bo), 4);
   g5p_dump("draw", "fs launch code", g->fs_launch.code_offset,
            (const uint32_t *)pvr_bo_suballoc_get_map_addr(g->fs_launch.pvr_bo) + 8, g->fs_launch.code_size);
   if (g->fs_num_uploads) {
      /* then, as in the ICD's draws, the per-draw loader data pointer */
      emit_ppp(csb, "ppp uniform data", uniform_va, uniform_block, 2);
      g5p_dump("draw", "uniform code", g->fs_loader.code_offset,
               (const uint32_t *)pvr_bo_suballoc_get_map_addr(g->fs_loader.pvr_bo), g->fs_loader.code_size);
   } else {
      g5p_dump("draw", "uniform code", dp->halt.code_offset,
               (const uint32_t *)pvr_bo_suballoc_get_map_addr(dp->halt.pvr_bo) + 4, 1);
   }
   if (g->varying.pvr_bo) {
      const uint32_t *m = pvr_bo_suballoc_get_map_addr(g->varying.pvr_bo);
      g5p_dump("draw", "varying data", g->varying.data_offset, m, g->varying.data_size);
      g5p_dump("draw", "varying code", g->varying.code_offset, m + g->varying.data_size, g->varying.code_size);
   } else {
      g5p_dump("draw", "varying code", dp->halt.code_offset,
               (const uint32_t *)pvr_bo_suballoc_get_map_addr(dp->halt.pvr_bo) + 4, 1);
   }

   /* 5-10. the per-draw blocks */
   static const char *const names[] = { "ppp outsel", "ppp isp", "ppp clip viewport", "ppp wclamp ctrl",
                                        "ppp dbsc", "ppp stream out" };
   for (unsigned i = 0; i < nb; i++)
      emit_ppp(csb, names[i], ppp_va + 4 * blocks[i][0], &ppp[blocks[i][0]], blocks[i][1]);
   g5p_dump("draw", "stream out data", dp->halt.data_offset, pvr_bo_suballoc_get_map_addr(dp->halt.pvr_bo), 5);

   /* 11. the index list: [the index buffer], count, instances - 1, first vertex or vertex offset, split
    * (pvr_arch_g5p_vertex.c) */
   if (draw->indirect_addr) {
      result = pvr_arch_g5p_vertex_indirect(cmd_buffer, draw, hw_topology, split, csb);
      if (result != VK_SUCCESS)
         return result;
   } else {
      unsigned n;
      result = pvr_arch_g5p_vertex_index_list(cmd_buffer, draw, hw_topology, split, w, &n);
      if (result != VK_SUCCESS)
         return result;
      emit_words(csb, "index list", w, n);
   }

   pvr_csb_clear_relocation_mark(csb);
   return csb->status;
}

/* vkCmdDrawIndexed: the same draw with an index list (pvr_arch_g5p_vertex.c) */
VkResult
PVR_PER_ARCH(g5p_cmd_draw_indexed)(struct pvr_cmd_buffer *cmd_buffer, uint32_t index_count, uint32_t instances,
                                   uint32_t first_index, int32_t vertex_offset, uint32_t first_instance)
{
   const struct pvr_g5p_draw draw = {
      .count = index_count,
      .instances = instances,
      .first = first_index,
      .first_instance = first_instance,
      .indexed = true,
      .vertex_offset = vertex_offset,
   };
   return pvr_arch_g5p_cmd_draw(cmd_buffer, &draw);
}
