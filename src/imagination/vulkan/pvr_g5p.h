/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P, has_apple_g5p): the G5P paths of the PowerVR driver.
 * Built only with -Dimagination-g5p-experimental=true (PVR_APPLE_G5P).
 *
 * G5P follows G9P's shape (pvr_g9p.h, the G9P driver): it keeps the pvr_*
 * object model (devices, command buffers, render passes, descriptors,
 * images, the queue, WSI, pdsc and csbgen) and replaces the pieces that
 * face the hardware through hooks, each in a
 * PVR_HAS_FEATURE(dev_info, apple_g5p) branch of its own. The programs it
 * uploads come from pdsc (PDS) and from the G5P backend in
 * src/imagination/g5p (USC); none is a captured Apple binary.
 */

#ifndef PVR_G5P_H
#define PVR_G5P_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan_core.h>

#include "pvr_common.h"
#include "pvr_macros.h"
#include "util/simple_mtx.h"
#include "pvr_g5p_xfb_core.h"
#include "pvr_g5p_perf.h"

struct pvr_cmd_buffer;
struct pvr_device;
struct pvr_render_job;
struct pvr_renderpass_hwsetup_render;
struct pvr_sub_cmd_gfx;
struct pvr_suballoc_bo;

/* Colour-target counts with programs: 1, 2 and 4, the counts with an Apple witness (others are refused) */
#define PVR_G5P_MRT_VARIANTS 3

/*
 * The device-level programs every G5P render uses, uploaded once per
 * device. PDS programs are built with pdsc, USC programs by the G5P
 * backend's generators (src/imagination/g5p/g5p_mrt_programs.h,
 * g5p_usc.h). A kick is the shape the hardware's program registers take: 16
 * bytes of data (the USC program's DOUTU word) and, at +0x20, DOUTU + HALT.
 */
struct pvr_g5p_programs {
   /* PDS: a lone HALT with 16 zero data bytes (the context store and resume
    * tasks, as on G9P, and the TA context program, ta_pds; route (b) on
    * hardware) */
   struct pvr_pds_upload halt;
   /* USC: background clear and end of tile, per target-count variant */
   struct pvr_suballoc_bo *clear_usc[PVR_G5P_MRT_VARIANTS];
   struct pvr_suballoc_bo *eot_usc[PVR_G5P_MRT_VARIANTS];
   /* PDS kicks of the clear programs (PDS_BGRND0's shader) */
   struct pvr_pds_upload clear_kick[PVR_G5P_MRT_VARIANTS];
   /* PDS code: the clear colour's DOUTD (PDS_BGRND0's texunicode; its data,
    * {floats, control}, is per render) and the end of tile, DOUTD of the PBE
    * words then DOUTU of the end-of-tile program (data per render) */
   struct pvr_pds_upload clear_dma, eot;
   /* the TA's end-of-stream program (the render's TA program slot): a
    * branch to the next 64-byte block, padding, end of task; and its kick */
   struct pvr_suballoc_bo *ta_usc;
   struct pvr_pds_upload ta_kick;
   /* the TA scratch the kernel's records point into, and the V2 tables of
    * a render that has none of its own (the template's sizes) */
   struct pvr_suballoc_bo *scratch, *scissor_pool, *dbias_pool;
   /* a render without colour attachments: the PBE target of its end of tile, a 16x16 R8 surface the PBE cuts
    * every pixel past to (pvr_arch_g5p.c g5p_job_init()) */
   struct pvr_suballoc_bo *null_rt;
   /* 4x (pvr_g5p_msaa.c): the end of tile of a one-target multisample
    * store (emitpix per sample), and the one-target clear's kick at sample
    * rate (DOUTU src0[34:33] 2) */
   struct pvr_suballoc_bo *eot_usc_ms;
   struct pvr_pds_upload clear_kick_ms;
   /* the background of a loaded colour attachment (loadOp LOAD): the
    * compiled load program (pvr_g5p_compile_load()), its kick, and the PDS
    * code of its uploads, one DOUTD each (data per render: the sources,
    * then the controls). NULL load_usc: the backend refused it, and a LOAD
    * render is refused with load_refusal. */
   struct pvr_suballoc_bo *load_usc;
   struct pvr_pds_upload load_kick, load_dma;
   unsigned load_temps, load_common_dwords, load_uploads;
   struct {
      unsigned kind; /* PCO_G5P_UPLOAD_IMAGE, _SAMPLER or _IMAGE_SIZE */
      unsigned dwords, common;
   } load_upload[3];
   char load_refusal[128];
   /* Lazy integer variants: float device allocation order remains identical. */
   struct pvr_g5p_programs *raw_load[2];
   struct pvr_suballoc_bo *raw_clear_usc[2];
   struct pvr_suballoc_bo *raw_eot_usc;
   struct pvr_pds_upload raw_clear_kick[2];
};

/* G5P's per-device state (pvr_device::g5p), allocated by g5p_device_init() */
struct pvr_g5p_device {
   struct pvr_g5p_programs programs;
   /* vkCmdClearAttachments' clear quad (pvr_arch_g5p_gfx.c): its pipeline
    * layout (fragment push constants 0..15) and a pipeline per colour
    * format and colour write mask (0xf, or 0 for a depth/stencil-only
    * clear), made at first use */
   simple_mtx_t clear_lock;
   VkPipelineLayout clear_layout;
   struct {
      VkFormat format;
      uint8_t write_mask;
      VkPipeline pipeline;
   } clear[8];
   unsigned clear_count;
};

/* What a G5P render job hands its fragment stream (pvr_render_job::g5p) */
struct pvr_g5p_job {
   uint64_t vdm_va;             /* the control stream, one block */
   uint32_t background[8];      /* PDS_BGRND0..3 (64-bit registers: 8-byte aligned) */
   uint32_t eot[4];             /* EVENT_PIXEL_PDS data, code, info */
   uint32_t vdm_bytes;          /* through STREAM_TERMINATE */
   uint32_t width, height;
   unsigned targets;            /* colour targets, 1, 2 or 4 */
   unsigned raw_words;          /* integer tile storage: two or four words */
   uint32_t dbias_count;        /* entries of upstream's depth-bias table, 0 without one */
   uint32_t scissor_count;      /* entries of upstream's scissor table, 0 without one */
   uint32_t samples;            /* 1 or 4 (hw_render->sample_count) */
   bool msaa_resolve;           /* 4x: the PBE resolves target 0 into a one-sample view */
   bool enabled;
   struct g5p_perf_tail perf;   /* what the job writes, for the winsys (pvr_g5p_perf.h) */
   /* last: what the pass's draws read outside the arena (pvr_arch_g5p_perf_draw()); job init keeps it */
   struct pvr_g5p_perf_reads reads;
};

/* ---- graphics (pvr_arch_g5p_gfx.c, pvr_g5p_nir.c) ---- */

struct pco_g5p_fs_key;
struct pco_g5p_program;
struct pco_g5p_vs_link;
struct pvr_graphics_pipeline;
struct vk_graphics_pipeline_state;

/* The most vertex attributes and fetches of a G5P pipeline (PCO_G5P_MAX_ATTRIBS) */
#define PVR_G5P_MAX_FETCH 16

/* The index a vertex fetch's DDMAD multiplies the stride by: the vertex
 * (t0), the instance (t1: per-instance rate, divisor 1; the base instance
 * is added to t1 first), none (divisor 0: the first instance's element,
 * the draw folds it into the base and fetches with stride 0), or the
 * instance divided by `divisor` (> 1) in a temp pair, plus the base */
enum pvr_g5p_fetch_rate {
   PVR_G5P_FETCH_VERTEX,
   PVR_G5P_FETCH_INSTANCE,
   PVR_G5P_FETCH_INSTANCE_DIVISOR0,
   PVR_G5P_FETCH_INSTANCE_DIVISOR,
};

/* Distinct divisors > 1 a pipeline's fetch divides by (PDS temps t2..t7) */
#define PVR_G5P_MAX_DIVISORS 3

/* One vertex fetch: a DDMAD of `dwords` from binding `binding` at `offset`
 * into the unified store at dword `ao` (the A9 ICD's g5p_fetch), by `rate`
 * (enum pvr_g5p_fetch_rate) */
struct pvr_g5p_fetch {
   uint8_t binding, dwords, ao, rate;
   uint32_t offset, divisor;
};

/* One common-store upload of the fragment program (the backend's
 * pco_g5p_upload): `kind` a pco_g5p_upload_kind, read at each draw from the
 * bound descriptor (set, binding) + offset bytes (the level of an image-size
 * upload), `dwords` into common-store dword `common` */
#define PVR_G5P_MAX_FS_UPLOADS 12
struct pvr_g5p_upload {
   uint8_t kind, set, binding, dwords;
   uint16_t element, offset, common;
};

/*
 * A G5P graphics pipeline (pvr_graphics_pipeline::g5p): the programs of its
 * draws, uploaded once, and what each draw's words derive from. The PDS
 * programs are pdsc's, the USC programs the G5P backend's.
 */
/* Line and point polygon modes (plan row R8, hardware-proven: polygon-mode-1
 * 13/13 on 2026-11-05) are on by default with fillModeNonSolid;
 * PVR_G5P_POLYGON_MODE=0 turns them off (A/B, the old named refusal). */
static inline bool
pvr_g5p_polygon_mode_enabled(void)
{
   const char *e = getenv("PVR_G5P_POLYGON_MODE");
   return !(e && !strcmp(e, "0"));
}

struct pvr_g5p_pipeline {
   struct pvr_suballoc_bo *vs_usc, *fs_usc;
   uint32_t vs_usc_size, fs_usc_size;
   /* PDS: the vertex fetch code (VDM_STATE3), the fragment launch (data
    * {DOUTU}, DOUTU.end at +0x20: the PPP shader base), the coefficient
    * program (data, then its code; none without varyings) */
   struct pvr_pds_upload fetch, fs_launch, varying;
   /* the vertex fetch code that adds the base instance to t1 first (Apple's
    * g3b-basevertex), for draws with a first instance: only when the
    * program reads the instance index or a fetch is per instance */
   struct pvr_pds_upload fetch_base;
   struct pvr_pds_upload fetch_indirect;
   /* static PPP blocks: the varying format (4 words) and the fragment
    * pointers and sizes (6 words) */
   struct pvr_suballoc_bo *static_vary, *static_frag;
   uint32_t static_vary_words[4], static_frag_words[6];
   /* the vertex program: DOUTU temps, output dwords, unified-store size,
    * system values the fetch writes (bit 0 vertex index, 1 instance index)
    * and their registers */
   unsigned vs_temps, vs_outputs, vs_unified, vs_written_temps;
   uint8_t vs_id_mask, vs_id_reg[2];
   unsigned fs_temps;
   unsigned fetch_count;
   struct pvr_g5p_fetch fetch_list[PVR_G5P_MAX_FETCH];
   /* the fragment program discards (punch-through, no triangle merging);
    * it reads a flat varying */
   bool fs_discard, fs_flat;
   unsigned fs_discard_tokens;
   bool fs_discard_select;
   /* Full linked variant: its uploads/varyings may differ after retaining a kill.
    * Original colour-only programs remain byte-identical. */
   struct pvr_g5p_pipeline *punch;
   bool punch_variant;
   /* the program writes memory (pco_g5p_program::side_effects): a fragment
    * program's draws bound the shaded pixels with the ISP scissor; a vertex
    * program's draws run even when no pixel is left (pvr_arch_g5p_gfx.c) */
   bool fs_side_effects, vs_side_effects;
   bool fs_depth_feedback;
   bool bresenham_line_probe;
   /* a point pipeline: the vertex program writes the point size after the
    * position (pco_g5p_vs_link::point_size) */
   bool vs_point_size;
   /* polygon mode of a triangle pipeline (on by default; PVR_G5P_POLYGON_MODE=0 off): the
    * ISP object type, as AGXMetalA9 encodeISPPrimitiveType picks it from
    * its triangle fill mode (desktop-gl-polygon-mode.md) */
   VkPolygonMode polygon_mode;
   /* the fragment program's tile read (pco_g5p_program::blend_mode: 0 none)
    * and colour write mask, for the ISP pass type (pvr_g5p_blend.h) */
   unsigned fs_blend_mode;
   uint8_t fs_write_mask;
   /* the fragment program's colour targets (pco_g5p_fs_key::color_targets:
    * 0 for one, else 2 or 4), checked against the pass per draw
    * (pvr_g5p_mrt.h) */
   unsigned color_targets;
   /* the vertex program's common-store uploads and their loader, NULL
    * without any (pvr_g5p_upload.h) */
   struct pvr_g5p_stage_uploads *vs_uploads;
   /* the fragment program's uploads (descriptor state, image sizes, blend
    * constants) and their loader, DOUTD per upload from the per-draw data
    * behind UNIFORMDATABASE (no uploads: the device HALT) */
   unsigned fs_num_uploads, fs_common_dwords;
   struct pvr_g5p_upload fs_uploads[PVR_G5P_MAX_FS_UPLOADS];
   struct pvr_pds_upload fs_loader;
   /* rasterization samples, 1 or 4: the fragment launch's sample rate and the draws' ISP user pass */
   uint32_t samples;
   /* transform feedback (pvr_g5p_xfb.h): the same pipeline with the vertex
    * program's capturing variant, used by draws while capturing (NULL: the
    * vertex shader captures nothing); in that variant, xfb_variant is set and
    * xfb_layout is what it stores */
   struct pvr_g5p_pipeline *xfb;
   bool xfb_variant;
   struct g5p_xfb_layout xfb_layout;
};

struct pvr_buffer;
struct pvr_csb;

/* A draw (CmdDraw, CmdDrawIndexed): `first` is the first vertex, or the
 * first index of an indexed draw, whose indices are offset by vertex_offset */
struct pvr_g5p_draw {
   uint32_t count, instances, first, first_instance;
   bool indexed;
   int32_t vertex_offset;
   /* Native VDM argument address; zero selects the ordinary direct packet. */
   uint64_t indirect_addr;
};

/* SPIR-V of one stage -> the G5P backend's program (pvr_g5p_nir.c), for a
 * pipeline of `samples` rasterization samples. The vertex stage takes its
 * vertex input layout (pvr_g5p_vertex.h; NULL for the fragment stage),
 * which also gives the backend its attribute components. false with
 * out->refusal set. */
struct pvr_g5p_vertex_layout;
bool pvr_g5p_compile_stage(const VkPipelineShaderStageCreateInfo *stage_info, bool half_output, uint32_t samples,
                           struct pvr_g5p_vertex_layout *vertex, const struct pco_g5p_fs_key *key,
                           const struct pco_g5p_vs_link *link, struct pco_g5p_program *out);
/* The background program of a loaded colour attachment (pvr_g5p_nir.c):
 * texture() of the attachment, bound as set 0 binding 0, at the fragment's
 * pixel centre. false with out->refusal set. */
bool pvr_g5p_compile_load(struct pco_g5p_program *out);
bool pvr_g5p_compile_load_raw(struct pco_g5p_program *out, bool is_signed);
/* The clear quad's stages, built with nir_builder (pvr_g5p_nir.c): the
 * vertex stage passes its vec4 position at location 0 through, the
 * fragment stage writes the push constant's colour (bytes 0..15) to colour
 * target 0. A pipeline takes them as internal NIR modules
 * (vk_shader_module_handle_from_nir()); pvr_g5p_clear_nir_free() frees
 * one. */
struct nir_shader;
struct nir_shader *pvr_g5p_clear_nir(bool vertex);
struct nir_shader *pvr_g5p_clear_nir_raw(bool vertex, bool is_signed);

/* The 32-bit integer colour targets the raw32 output renders (pco_g5p's PCO_G5P_COLOR_UINT32/SINT32: four raw
 * output words, no f16 conversion): 1 unsigned, 2 signed, 0 any other format. The PBE stores the target's own
 * channels (one, two or four 32-bit words of a pixel) from the first output words. */
static inline unsigned
pvr_g5p_raw32_kind(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R32_UINT:
   case VK_FORMAT_R32G32_UINT:
   case VK_FORMAT_R32G32B32A32_UINT:
      return 1;
   case VK_FORMAT_R32_SINT:
   case VK_FORMAT_R32G32_SINT:
   case VK_FORMAT_R32G32B32A32_SINT:
      return 2;
   default:
      return 0;
   }
}
void pvr_g5p_clear_nir_free(struct nir_shader *nir);
/* A vertex stage whose SPIR-V declares ExecutionMode Xfb */
bool pvr_g5p_stage_has_xfb(const VkPipelineShaderStageCreateInfo *stage_info);
/* The vertex shader's transform feedback variant (pvr_g5p_nir.c): true with
 * layout->buffers == 0 when it captures nothing (no variant); false with
 * out->refusal set. */
bool pvr_g5p_compile_vs_xfb(const VkPipelineShaderStageCreateInfo *stage_info, struct pvr_g5p_vertex_layout *vertex,
                            const struct pco_g5p_vs_link *link, struct g5p_xfb_layout *layout,
                            struct pco_g5p_program *out);

#ifdef PVR_PER_ARCH

VkResult PVR_PER_ARCH(g5p_graphics_pipeline_compile)(struct pvr_device *device,
                                                     const VkGraphicsPipelineCreateInfo *create_info,
                                                     struct pvr_graphics_pipeline *pipeline,
                                                     const struct vk_graphics_pipeline_state *vk_state);
void PVR_PER_ARCH(g5p_graphics_pipeline_destroy)(struct pvr_device *device,
                                                 struct pvr_graphics_pipeline *pipeline);
VkResult PVR_PER_ARCH(g5p_cmd_draw)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g5p_draw *draw);
VkResult PVR_PER_ARCH(g5p_cmd_draw_indirect)(struct pvr_cmd_buffer *cmd_buffer, struct pvr_buffer *buffer,
                                             VkDeviceSize offset, uint32_t count, uint32_t stride, bool indexed);
VkResult PVR_PER_ARCH(g5p_vertex_indirect)(struct pvr_cmd_buffer *cmd_buffer, const struct pvr_g5p_draw *draw,
                                           uint32_t topology, uint32_t split, struct pvr_csb *csb);
VkResult PVR_PER_ARCH(g5p_cmd_draw_indexed)(struct pvr_cmd_buffer *cmd_buffer, uint32_t index_count,
                                            uint32_t instances, uint32_t first_index, int32_t vertex_offset,
                                            uint32_t first_instance);

#   define pvr_arch_g5p_graphics_pipeline_compile PVR_PER_ARCH(g5p_graphics_pipeline_compile)
#   define pvr_arch_g5p_graphics_pipeline_destroy PVR_PER_ARCH(g5p_graphics_pipeline_destroy)
#   define pvr_arch_g5p_cmd_draw PVR_PER_ARCH(g5p_cmd_draw)
#   define pvr_arch_g5p_cmd_draw_indirect PVR_PER_ARCH(g5p_cmd_draw_indirect)
#   define pvr_arch_g5p_vertex_indirect PVR_PER_ARCH(g5p_vertex_indirect)
#   define pvr_arch_g5p_cmd_draw_indexed PVR_PER_ARCH(g5p_cmd_draw_indexed)

VkResult PVR_PER_ARCH(g5p_device_init)(struct pvr_device *device);
void PVR_PER_ARCH(g5p_device_finish)(struct pvr_device *device);
VkResult PVR_PER_ARCH(g5p_job_init)(struct pvr_cmd_buffer *cmd_buffer,
                                    struct pvr_sub_cmd_gfx *sub_cmd,
                                    const struct pvr_renderpass_hwsetup_render *hw_render,
                                    struct pvr_render_job *job);
uint32_t PVR_PER_ARCH(g5p_frag_stream)(const struct pvr_device *device,
                                       const struct pvr_render_job *job, uint8_t *buf,
                                       uint32_t size);

#   define pvr_arch_g5p_device_init PVR_PER_ARCH(g5p_device_init)
#   define pvr_arch_g5p_device_finish PVR_PER_ARCH(g5p_device_finish)
#   define pvr_arch_g5p_job_init PVR_PER_ARCH(g5p_job_init)
#   define pvr_arch_g5p_frag_stream PVR_PER_ARCH(g5p_frag_stream)

#endif /* PVR_PER_ARCH */

#endif /* PVR_G5P_H */
