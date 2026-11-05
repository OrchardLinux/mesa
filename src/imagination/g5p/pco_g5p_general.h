/* SPDX-License-Identifier: MIT
 * General NIR -> G5P USC backend (round 18), on top of the captured-shape
 * matcher in pco_g5p.c. Encodings come from g5p_usc.c (ISA round 18 field
 * encoders); register conventions per stage are the ones witnessed by the
 * host compiler (isa-g5p.md). Anything without such a witness is refused
 * with a reason naming the NIR operation.
 */
#ifndef PCO_G5P_GENERAL_H
#define PCO_G5P_GENERAL_H

#include "compiler/nir/nir.h"

#include <stdbool.h>
#include <stdint.h>

/* Runaway bound on one program (the backend has no size limit of its own; the
 * ICD's job arena is what bounds a draw's programs, see pco_g5p_general.c). */
#define PCO_G5P_PROGRAM_MAX (1u << 20)
#define PCO_G5P_MAX_UPLOADS 12
#define PCO_G5P_MAX_ATTRIBS 16
#define PCO_G5P_COMMON_DWORDS 64
/* Every vertex program that has flown needs <= 12 temps; 16 and 20 wedged
 * the TA (docs/gpu/experiments/xfb.md, device run 2). */
#define PCO_G5P_VS_TEMPS_FLOWN 12
/* Fragment programs: Apple's A9 compiler preloads constant blocks of up to
 * 256 dwords into the common store and reads them as common-store operands
 * up to dword 255 (tools/mesa-a9-hypr-re f_u256all; a 1024-dword block is
 * read from memory instead). */
#define PCO_G5P_FS_COMMON_DWORDS 256
/* Vertex programs: Apple's A9 compiler reads uniform blocks of up to 256
 * dwords from the common store in the vertex stage too, inline up to dword
 * 255 with the fragment's extension field (tools/mesa-a9-hyprland-gpu/
 * vsuni-probes.py v_u80all..v_u256all; bench-gaps.md). Reads past dword 31
 * keep this compiler's vertex form: a mov.shared into a register. */
#define PCO_G5P_VS_COMMON_DWORDS 256

/* Provenance of what a general program relies on (device qualification
 * order). HOST = Xcode metal-tt applegpu_g5p differential probes, never
 * executed; CAPTURE = iOS device capture of an executed pipeline. */
enum pco_g5p_evidence {
   PCO_G5P_EV_FS_ALU = 1u << 0,      /* HOST: fragment ALU/convert/literal fields */
   PCO_G5P_EV_VS_ALU = 1u << 1,      /* HOST: v_depth/v_add/v_mul/v_fma/v_base forms only */
   PCO_G5P_EV_TEX_CAPTURE = 1u << 2, /* CAPTURE: tex scene FS (coord iterator, 0x18 sample) */
   PCO_G5P_EV_TEX_F32 = 1u << 3,     /* HOST: f32 sample result consumed by ALU */
   PCO_G5P_EV_PACK = 1u << 4,        /* inferred composition: fmul.rh by 1.0 into output halves */
   PCO_G5P_EV_SHARED = 1u << 5,      /* HOST: mov.shared from the common store (FS) */
   PCO_G5P_EV_BUFFER_LOAD = 1u << 6, /* HOST: class-12 fixed-index buffer loads (FS) */
   PCO_G5P_EV_VARYINGS = 1u << 7,    /* ROGUE: more than one vec4 varying item */
   PCO_G5P_EV_ATTRIB_FILL = 1u << 8, /* attribute components beyond the format: (0,0,0,1) */
   PCO_G5P_EV_ITERATOR = 1u << 9,    /* HOST: f32 iterators + wait before ALU */
   /* round 19 (ISA round 2, isa-g5p.md; ds-isa-state.md) */
   PCO_G5P_EV_VS_SHARED = 1u << 10,  /* CAPTURE: v_mvp (tex scene) shared-source VS ALU + HOST v_mat3_* */
   PCO_G5P_EV_SELECT = 1u << 11,     /* HOST: compare/write + inverse MOV (strict select recipe) */
   PCO_G5P_EV_MATH = 1u << 12,       /* HOST: complete Apple lowerings (rcp/rsqrt/sqrt/sign/min/max/
                                      * ceil/trunc/rint/sin/cos), numerically unqualified */
   PCO_G5P_EV_DISCARD = 1u << 13,    /* HOST: discard test + 04 61 / 04 68 (helper-lane semantics open) */
   PCO_G5P_EV_FRAGCOORD = 1u << 14,  /* HOST: raster XY 1c..68, u2f32 x 1/16 */
   PCO_G5P_EV_BLEND = 1u << 15,      /* HOST: Apple premultiplied epilogue (124-byte reference) */
   PCO_G5P_EV_FLATTEN = 1u << 16,    /* if/else, phis and small loops flattened to selects (NIR) */
   PCO_G5P_EV_VS_CONTROL = 1u << 17, /* reserved: VS control tuple outside the witnessed table */
   PCO_G5P_EV_TEX_COMPUTED = 1u << 18, /* HOST: ALU-produced texture coordinate (21 probes) */
   PCO_G5P_EV_DYN_UBO = 1u << 19,  /* APPLE (dynubo.metal) + A10: register-indexed UBO load */
   PCO_G5P_EV_TEX_TYPES = 1u << 20, /* APPLE (textypes probes.metal): array/3D/cube, level/bias/
                                     * gradients/offsets, read, compare, integer results */
   /* transform feedback (xfb.md, tools/mesa-a9-xfb-re probes) */
   PCO_G5P_EV_VS_STORE = 1u << 21,   /* HOST: vertex address add + class-10 store, integer ALU */
   PCO_G5P_EV_VS_IDS = 1u << 22,     /* CAPTURE: vertex-index DOUTW (out/render-post); instance: ROGUE */
   PCO_G5P_EV_CF = 1u << 23,         /* HOST: if/else and loops on the execution-mask stack (G9P port) */
   /* g5p-hypr (bits 20..22 on that lane) */
   PCO_G5P_EV_COMMON_HIGH = 1u << 24, /* HOST: fragment common store past dword 63 (f_u80all, f_u256all) */
   PCO_G5P_EV_DERIV = 1u << 25,      /* HOST: dfdx/dfdy (f_ddx_*, f_ddy_*), fine or coarse unmeasured */
   PCO_G5P_EV_FS_STORE = 1u << 27, /* HOST: helper predicate + fragment class70/71 store */
   PCO_G5P_EV_HIGH_REGS = 1u << 26,  /* HOST: homes above r31 (G9P's moves; Apple's r32..r69 operands) */
};

/* Fragment program key (ds-isa-state.md §3.1, Apple FragmentProgramKey):
 * blend state, colour write mask and the draw-buffer format select a
 * compiled program; none of them is a hardware state word on this GPU.
 * Factors/ops use the VkBlendFactor/VkBlendOp numbering. */
struct pco_g5p_fs_key {
   bool blend;
   /* Real kill feedback for depth/stencil writes; merge conditions into one token. */
   bool punch;
   bool depth_feedback; /* qualification admission; false by default */
   uint8_t src_rgb, dst_rgb, src_a, dst_a, op_rgb, op_a;
   uint8_t write_mask; /* VkColorComponentFlags; 0xf = all */
   bool srgb;          /* _AGCDrawBuffer.gammaCorrection (bit location unmeasured: refused) */
   /* Source group bits by Apple's rule (1 only on a register's last read;
    * repeated reads get group 0 and the measured hint bits). Off: the
    * round-16..18 convention, group 1 on every fragment read (default, so
    * those scenes keep their bytes). PVR_G5P_GROUPS=last-read turns it on. */
   bool last_read_groups;
   /* Fuse an fmul and the fadds it feeds into ffmas (G9P's fusion), with
    * last_read_groups only. Off by default so proven programs keep their
    * bytes; PVR_G5P_FUSE=ffma turns it on. Captured shapes are not
    * affected. */
   bool fuse_ffma;
   /* Colour attachments written by the program: 0 = 1. 1, 2 and 4 are the
    * homogeneous 8-bit RGBA sets with an Apple witness (g5p_mrt_out.h). */
   uint8_t color_targets;
};
#define PCO_G5P_FS_KEY_DEFAULT ((struct pco_g5p_fs_key){.write_mask = 0xf})

enum pco_g5p_upload_kind {
   PCO_G5P_UPLOAD_SSBO_ADDR, /* 2 dwords: buffer VA (descriptor +0) */
   PCO_G5P_UPLOAD_IMAGE,     /* 8 dwords: image state (descriptor +0) */
   PCO_G5P_UPLOAD_SAMPLER,   /* 2 dwords: sampler words (descriptor +32) */
   PCO_G5P_UPLOAD_PUSH,      /* push-constant bytes [offset, offset+4*dwords) */
   PCO_G5P_UPLOAD_UBO,       /* UBO bytes [offset, offset+4*dwords) */
   PCO_G5P_UPLOAD_UBO_ADDR,  /* 2 dwords: UBO VA + offset (register-indexed loads) */
   /* textypes: 4 dwords of the view at level `offset`: max(1, w >> l), max(1, h >> l),
    * 3D max(1, d >> l) or the layer count, the level count (textureSize and
    * textureQueryLevels; Apple uploads a size block of its own too). */
   PCO_G5P_UPLOAD_IMAGE_SIZE,
   PCO_G5P_UPLOAD_XFB,       /* per-draw driver block, PCO_G5P_XFB_* dwords */
   PCO_G5P_UPLOAD_BLEND_CONSTANTS, /* 4 dwords: the dynamic blend constants (f32) */
};

/* Varyings (docs/gpu/experiments/varyings.md). Apple's A9 compiler takes at
 * most 60 interpolated components ("Number of varying components(64)
 * exceeds the limit (60)"): 15 vec4 fragment inputs. */
#define PCO_G5P_MAX_FS_VARYINGS 15
#define PCO_G5P_VARYING_SLOTS 32 /* VARYING_SLOT_VAR0..VAR31 */

enum pco_g5p_interp {
   PCO_G5P_INTERP_UNREAD = 0, /* VS link: the fragment program does not read it */
   PCO_G5P_INTERP_SMOOTH,     /* perspective-correct (STATE_VARYING0 f32_linear) */
   PCO_G5P_INTERP_FLAT,       /* provoking vertex (f32_flat) */
   PCO_G5P_INTERP_NOPERSPECTIVE, /* screen-linear (f32_npc) */
};

/* One fragment-read varying location: four coefficient sets from `set`
 * (W takes set 0 when any varying is perspective-correct), in location order. */
struct pco_g5p_varying {
   uint8_t location; /* VARn */
   uint8_t set;
   uint8_t interp;   /* enum pco_g5p_interp */
   bool centroid;
   uint8_t mask;     /* components read */
};

/* Vertex program link (Apple's VertexProgramKey carries the fragment
 * program's varyings; Rogue's pco_rev_link_nir): per VARn, how the
 * fragment program reads it. Outputs it does not read are not written. */
struct pco_g5p_vs_link {
   bool valid;
   uint8_t interp[PCO_G5P_VARYING_SLOTS];
   /* a point pipeline: gl_PointSize is the last output dword, after the
    * varyings (TA_OUTPUT_SEL psprite_size_pres; Apple's g3b-points) */
   bool point_size;
};

/* Per-draw transform feedback block (PCO_G5P_UPLOAD_XFB): the VA each
 * buffer's record 0 would have for this draw (dwords 2b, 2b+1), then the
 * draw's vertices per instance (nir load_num_vertices). */
#define PCO_G5P_XFB_ADDR(b) (2 * (b))
#define PCO_G5P_XFB_NUM_VERTICES 8
#define PCO_G5P_XFB_DWORDS 10

/* One DOUTD of the uniform loader (fragment: PPP UNIFORMDATABASE; vertex:
 * VDM PDS_STATE_UPDATE, round 19): dwords -> common[common..]. */
struct pco_g5p_upload {
   enum pco_g5p_upload_kind kind;
   unsigned set, binding;
   unsigned offset; /* bytes (push/UBO) */
   unsigned dwords;
   unsigned common;
};

struct pco_g5p_program {
   uint8_t *code; /* malloc'd */
   unsigned discard_tokens; /* Emitted fragment kill feedback tokens. */
   bool discard_select; /* Discard lowered to transparent black (colour-only safe). */
   unsigned blend_mode; /* 0 none, 1 premultiplied, 2 straight-over (pre-multiplied in f32), 4 general */
   size_t size;
   bool vertex;
   bool depth_feedback; /* native f32 fragment depth replacement */
   unsigned input_mask;                            /* VS generic locations read */
   uint8_t input_components[PCO_G5P_MAX_ATTRIBS];  /* VS components read */
   unsigned outputs;                               /* VS output dwords */
   unsigned varyings;                              /* vec4 varying items */
   unsigned temps;                                 /* registers used (DOUTU temps) */
   unsigned written_temps;                         /* VS: registers written, inputs excluded */
   unsigned common_dwords;                         /* common store used (FS; VS round 19) */
   unsigned num_uploads;
   struct pco_g5p_upload uploads[PCO_G5P_MAX_UPLOADS];
   /* FS: varyings read, location order (the coefficient program's items). */
   unsigned num_varyings;
   struct pco_g5p_varying vary[PCO_G5P_MAX_FS_VARYINGS];
   bool w_set; /* coefficient set 0 is W (some varying is perspective-correct) */
   /* VS: first output dword of each VARn (0 = not written) and its
    * interpolation; outputs are grouped linear, flat, npc (STATE_VARYING0). */
   uint8_t vout_dword[PCO_G5P_VARYING_SLOTS];
   uint8_t vout_interp[PCO_G5P_VARYING_SLOTS];
   uint32_t evidence;
   /* VS system values the vertex PDS writes with DOUTW (Rogue
    * pvr_pds_vertex_shader, Apple's vertex-index DOUTW): bit 0 vertex
    * index, bit 1 instance index, into dword id_reg[i]. */
   uint8_t id_mask;
   uint8_t id_reg[2];
   uint8_t xfb_buffers; /* VS: transform feedback buffers stored to */
   /* the program writes memory: a store or atomic to a buffer, image or
    * global address (has_side_effects()), transform feedback included.
    * A fragment program's draws bound the shaded pixels with the ISP
    * scissor; a vertex program's draws run their vertices even when no
    * pixel is left (pvr_arch_g5p_gfx.c). */
   bool side_effects;
   /* VS of a non-point pipeline: it wrote gl_PointSize, which was dropped
    * (set by pvr_g5p_nir.c after compilation; polygon mode POINT refuses it) */
   bool psize_dropped;
   char refusal[512];
};

/* A converted vertex input (8- and 16-bit vertex formats): a load_input
 * whose base is PCO_G5P_VS_CONVERT_BASE + 4 * format + byte reads byte
 * 0..3 (a 16-bit format: 0 or 2) of the fetched dword at its location and
 * component, and converts it to 32 bits by the format below; the program
 * converts each at its start, as Apple's vertex programs do. A load_input
 * with a lower base reads the dword itself. */
#define PCO_G5P_VS_CONVERT_BASE 0x100
enum pco_g5p_vs_format {
   /* to float: as enum g5p_small_format */
   PCO_G5P_VSF_U8, PCO_G5P_VSF_S8, PCO_G5P_VSF_UN8, PCO_G5P_VSF_SN8,
   PCO_G5P_VSF_U16, PCO_G5P_VSF_S16, PCO_G5P_VSF_UN16, PCO_G5P_VSF_SN16,
   PCO_G5P_VSF_F16,
   PCO_G5P_VSF_COUNT
};

/* attrib_components[L] (VS only, may be NULL): components the bound vertex
 * format of location L provides; components read beyond it are (0,0,0,1)
 * as Vulkan requires. half_output: FS colour is packed f16 (8-bit UNORM).
 * Returns false with out->refusal set. The shader is modified. */
bool pco_g5p_compile_general(nir_shader *nir, bool half_output, const uint8_t *attrib_components,
                             struct pco_g5p_program *out);
/* Round 19: the same with a fragment program key (NULL = default key). */
bool pco_g5p_compile_general_key(nir_shader *nir, bool half_output, const uint8_t *attrib_components,
                                 const struct pco_g5p_fs_key *key, struct pco_g5p_program *out);
/* The same with the vertex program's link to its fragment program (NULL or
 * !valid: every written VARn is a smooth output at 4 + 4 * n, unlinked). */
bool pco_g5p_compile_general_link(nir_shader *nir, bool half_output, const uint8_t *attrib_components,
                                  const struct pco_g5p_fs_key *key, const struct pco_g5p_vs_link *link,
                                  struct pco_g5p_program *out);
/* PVR_G5P_FS_DYN_UBO=1: fragment register-indexed UBO loads are compiled
 * (index clamped to the block) instead of refused. */
bool pco_g5p_fs_dyn_ubo(void);
bool pco_g5p_blend_general(void);

#endif
