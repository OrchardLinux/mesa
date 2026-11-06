/* SPDX-License-Identifier: MIT
 * Copyright 2026 Henrik Klee: the parts marked "From the G9P
 * compiler" (discards to a select, FMA fusion, control flow, registers above
 * r31), ported from OrchardLinux/mesa src/imagination/g9p.
 *
 * General NIR -> G5P USC backend (round 18).
 *
 * Straight-line vertex and fragment shaders are selected instruction by
 * instruction onto the ISA round-18 field encoders (g5p_usc.c). Values are
 * scalar after nir_lower_alu_to_scalar; vec/mov are register maps, fneg and
 * fabs fold into float source modifiers and fsat into its producer.
 *
 * Register conventions are only the witnessed ones (isa-g5p.md):
 *  - fragment: destination bank 0, header 1, source groups 1 (the round-4 /
 *    round-18 fragment dataflow model); iterators, shared moves and texture
 *    results land in the same bank; packed-f16 colour leaves from halves
 *    0..3 (`1c 21 00 08`) as in the iOS vc and tex captures;
 *  - vertex: bank-1 temps, attributes fetched by the PDS into r(4L+c); only
 *    the vertex forms of the v_depth/v_add/v_mul/v_fma/v_base probes (fadd
 *    reg-reg header 0x31 with A in group 0, f16-literal fadd/fmul header
 *    0x11, f16-literal ffma header 0x19, u2f32 header 0x11, MOVI). Every
 *    other vertex operation is refused instead of extrapolated.
 * Apple's allocation choices are reproduced where they are measurable: an
 * operand that dies is placed in source B (FMA: A) and the result reuses
 * its register; vertex outputs leave pass-through values first and computed
 * ones last, the final one as `5e` in the 64-byte block after a forward
 * branch. With these rules Zink's depth-remap VS equals Apple's 70-byte
 * v_depth, and Hyprland's passthru FS equals the iOS tex capture.
 */
#include "pco_g5p_general.h"
#include "../vulkan/pvr_g5p_dgl.h"
#include "g5p_vtex.h"
#include "g5p_usc.h"
#include "g5p_mrt_out.h"

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "g5p_texel_nir.h"
#include "g5p_texel_backend.h"
#include "util/half_float.h"
#include "util/macros.h"
#include "util/u_math.h"
#include "util/u_qsort.h"

#include <limits.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* No program size limit of the backend's own. Apple's A9 compiler emits
 * straight-line fragment programs far past 16 KiB (tools/mesa-a9-codesize-re:
 * 17546 to 1258052 bytes, one end, no jump), USC code is addressed by a 32-bit
 * offset from usc_base (code_window 4 GiB, render-uapi.md), and no program state
 * word holds a size. What bounds a program is where it is placed: the ICD puts a
 * draw's programs in one job's user arena and refuses a pipeline that cannot
 * fit there (g5p_pipeline.c). PCO_G5P_PROGRAM_MAX only stops runaway growth. The
 * buffer starts at CODE_INITIAL and doubles. */
#define CODE_INITIAL 16384
#define NUM_DWORDS 32
#define F32_ONE 0x3f800000u
#define LAST_END (INT_MAX - 1) /* read by the output block */
#define DISCARD_AT (INT_MAX - 2) /* read by the deferred discards, before the output */
#define G5P_MAX_LOOPS 32
#define G5P_MAX_DECLS 64
#define G5P_MAX_EXITS 64

enum val_state { VAL_NONE, VAL_REG, VAL_CONST, VAL_SHARED, VAL_HOME /* reg: r32..r127 */ };

struct val {
   uint8_t state;
   uint8_t reg;   /* dword: VAL_REG, or a materialized CONST/SHARED */
   bool held;     /* reg is allocated */
   bool input;    /* VS: unmodified attribute register */
   uint32_t imm;  /* CONST bits / SHARED common dword */
   int last;      /* last reading instruction (-1: unread) */
   unsigned uses; /* consumer references */
   bool body;     /* defined by an instruction of the body: may get a home */
};

struct ref {
   nir_def *def;
   unsigned comp;
   bool neg, abs;
};

struct tex_bind {
   unsigned set, binding, sset, sbinding;
   unsigned element, selement; /* array elements of the texture / sampler binding */
   unsigned image, sampler; /* common dwords */
   bool needs_sampler, needs_fetch_sampler;
};

struct range {
   enum pco_g5p_upload_kind kind;
   unsigned set, binding, element, lo, hi; /* bytes */
   unsigned common;
   bool memory; /* UBO range read through its address instead of preloaded */
};

enum { BLEND_NONE, BLEND_PREMULT, BLEND_STRAIGHT_OVER, BLEND_REPLACE, BLEND_GENERAL };

/* The general blend epilogue (g5p_blend.c, hyprland-gpu.md §11.2) for
 * settings without a measured epilogue. Premultiplied, straight-over and
 * replace keep their paths. By default only the settings of the device
 * blend ladder (linux-a9 blend-ladder.c, release7 2026-11-02: 45 rungs,
 * every pixel within 1 LSB) and of the bench ladder's three blend rungs
 * take their original byte-identical shape. All other core equations use
 * the same tile protocol with canonical native-f16 ALU lowering.
 * PVR_G5P_BLEND_GENERAL=1: explicitly enable (the default);
 * PVR_G5P_BLEND_GENERAL=0: none. */
bool pco_g5p_blend_general(void)
{
   const char *e = getenv("PVR_G5P_BLEND_GENERAL");
   return !e || strcmp(e, "0");
}

static bool blend_general_off(void)
{
   const char *e = getenv("PVR_G5P_BLEND_GENERAL");
   return e && !strcmp(e, "0");
}

/* VkBlendFactor/VkBlendOp: src_rgb, dst_rgb, op_rgb, src_a, dst_a, op_a. */
static const uint8_t blend_proven[][6] = {
   /* src_X: X/ZERO add on colour and alpha */
   {2, 0, 0, 2, 0, 0}, {3, 0, 0, 3, 0, 0}, {4, 0, 0, 4, 0, 0}, {5, 0, 0, 5, 0, 0}, {6, 0, 0, 6, 0, 0},
   {7, 0, 0, 7, 0, 0}, {8, 0, 0, 8, 0, 0}, {9, 0, 0, 9, 0, 0}, {10, 0, 0, 10, 0, 0}, {11, 0, 0, 11, 0, 0},
   {12, 0, 0, 12, 0, 0}, {13, 0, 0, 13, 0, 0}, {14, 0, 0, 14, 0, 0},
   /* dst_X: ZERO/X add */
   {0, 1, 0, 0, 1, 0}, {0, 2, 0, 0, 2, 0}, {0, 3, 0, 0, 3, 0}, {0, 4, 0, 0, 4, 0}, {0, 5, 0, 0, 5, 0},
   {0, 6, 0, 0, 6, 0}, {0, 7, 0, 0, 7, 0}, {0, 8, 0, 0, 8, 0}, {0, 9, 0, 0, 9, 0}, {0, 10, 0, 0, 10, 0},
   {0, 11, 0, 0, 11, 0}, {0, 12, 0, 0, 12, 0}, {0, 13, 0, 0, 13, 0},
   /* op_O: ONE/ONE with each op; op_O_sa: SRC_ALPHA/ONE_MINUS_SRC_ALPHA colour, ONE/ONE add alpha */
   {1, 1, 0, 1, 1, 0}, {1, 1, 1, 1, 1, 1}, {1, 1, 2, 1, 1, 2}, {1, 1, 3, 1, 1, 3}, {1, 1, 4, 1, 1, 4},
   {6, 7, 0, 1, 1, 0}, {6, 7, 1, 1, 1, 0}, {6, 7, 2, 1, 1, 0}, {6, 7, 3, 1, 1, 0}, {6, 7, 4, 1, 1, 0},
   /* qs_text (Qt's text), straight, qs_text_alpha */
   {10, 3, 0, 10, 3, 0}, {6, 7, 0, 6, 7, 0}, {10, 3, 0, 1, 7, 0},
   /* mixed settings outside Apple's 45 */
   {4, 7, 1, 4, 7, 1}, {12, 9, 2, 12, 9, 2}, {11, 2, 0, 13, 6, 0}, {14, 1, 0, 1, 1, 0}, {8, 4, 0, 8, 8, 0},
   /* linux-a9 bench-ladder (release8-pmprevshm, 2026-11-03, max error 1): blend_sa_one (glmark2 pulsar, terrain),
    * blend_sa_omsa_z1 (glmark2 desktop), blend_sa_omsa_10 (vkmark desktop) */
   {6, 1, 0, 6, 1, 0}, {6, 7, 0, 0, 1, 0}, {6, 7, 0, 1, 0, 0},
};

/* A channel's setting matches; min/max ignore the factors (Vulkan, and the generator). */
static bool blend_channel_eq(unsigned s, unsigned d, unsigned o, const uint8_t *p)
{
   if (o != p[2])
      return false;
   return o >= 3 || (s == p[0] && d == p[1]);
}

static bool blend_is_proven(const struct pco_g5p_fs_key *k)
{
   for (unsigned i = 0; i < ARRAY_SIZE(blend_proven); i++) {
      const uint8_t *p = blend_proven[i];
      if (blend_channel_eq(k->src_rgb, k->dst_rgb, k->op_rgb, p) &&
          blend_channel_eq(k->src_a, k->dst_a, k->op_a, p + 3))
         return true;
   }
   return false;
}

#define G5P_BLEND_CONST_BASE 0 /* Apple: the blend colour is common dwords 0..3 */

static struct g5p_blend_key general_key(const struct pco_g5p_fs_key *k)
{
   if (!k->blend)
      return (struct g5p_blend_key){1, 0, 0, 1, 0, 0};
   return (struct g5p_blend_key){k->src_rgb, k->dst_rgb, k->op_rgb, k->src_a, k->dst_a, k->op_a};
}

struct gctx {
   nir_shader *nir;
   nir_function_impl *impl;
   bool vs, half_output;
   const uint8_t *attrib_components;
   const struct pco_g5p_fs_key *key;
   bool *fused;        /* per def: compare evaluated at each select/b2f/discard use */
   nir_alu_instr *narrow_sat[4]; /* blend: fsat(varying) components of the saturating iterator */
   int zero_reg, ione_reg;
   nir_def *discards[16]; /* demote_if/terminate_if conditions, emitted before the output */
   unsigned ndiscards;
   bool discard_always;
   int fragcoord;      /* dword of the raw X (Y at +1), -1 when unused */
   bool remat;         /* register-pressure retry: constants/uniforms re-read per use */
   bool count_pass;    /* round 19: first codegen pass only records register reads */
   unsigned nput;      /* instructions emitted so far */
   int *lastput;       /* per value: index of the last instruction reading its register */
   int klast[3];       /* the same for the persistent 1.0 / 0 / 1 registers */
   uint64_t scratch;   /* registers of re-read values, free after the instruction */
   struct {            /* VS: store record bases, one address pair each */
      unsigned buffer;
      struct ref off;
      int pair;        /* first dword of the address pair, -1 until computed */
   } st_base[8];
   unsigned nst_base;
   unsigned blend;     /* pco_g5p_program.blend_mode */
   struct g5p_blend_info binfo; /* BLEND_GENERAL */
   struct pco_g5p_program *p;
   uint8_t *code;      /* CODE_INITIAL and doubling; kept by codegen_reset() */
   unsigned code_cap;
   unsigned size;
   jmp_buf fail;
   struct val *vals;
   unsigned nvals;
   bool *sat_folded; /* per def: fsat folded into its producer */
   bool *saturate;   /* per def: producer emitted with saturate */
   int *rh_half;     /* per def: emitted with result_half into this output half */
   uint64_t busy, reserved; /* vertex: bit VS_BANK0 + i is bank 0 r<i> */
   bool tex_bank0; /* explicit legacy texture route, separate from mixed-bank retry */
   unsigned max_dword, max_written;
   unsigned max_bank0; /* vertex: bank-0 registers used (the DOUTU temps) */
   bool alloc_out;     /* vertex bank-0 retry: the value being emitted is a vertex output (bank 1) */
   int now;
   int one_reg;
   /* fragment output: one entry per output half, 4 halves per colour target
    * (target t component k is half 4t+k, see g5p_mrt_out.h) */
   struct g5p_mrt_out_plan mrt;
   struct ref out[G5P_MRT_MAX_HALVES];
   bool out_written[G5P_MRT_MAX_HALVES];
   bool has_depth;
   struct ref depth;
   uint8_t slot_written[G5P_MRT_FS_SLOTS]; /* component mask stored per FRAG_RESULT_DATA slot */
   nir_tex_instr *narrow_tex;
   nir_intrinsic_instr *narrow_iter;
   int8_t vary_idx[PCO_G5P_VARYING_SLOTS]; /* FS: VARn -> c->p->vary[], -1 when not read */
   bool lazy_iter; /* FS pressure retry: each varying is read where the program loads it */
   /* vertex output */
   /* position, every VARn, point size; at: eager emission point, -1 = the end */
   struct { unsigned slot; struct ref r; int at; } vout[4 + 4 * PCO_G5P_VARYING_SLOTS + 4];
   unsigned nvout;
   const struct pco_g5p_vs_link *link;
   unsigned psize_dword; /* a point pipeline's point size: the last output dword */
   bool eager_vout; /* VS temps retry: each output written as soon as its value is final */
   bool vs_bank0;   /* VS pressure retry: temps in bank 0, outputs in bank 1 (vs-two-bank-allocator.md) */
   nir_intrinsic_instr *vend_store; /* eager: the output component written by the final vend */
   unsigned vend_comp;
   /* common store */
   struct tex_bind tex[16];
   unsigned ntex, fetch_sampler_common;
   bool cube_gradient; /* fixed native bank0 r0..r8 operand tuple */
   struct { unsigned set, binding, element, lod, common; } tsize[4]; /* textureSize/QueryLevels */
   unsigned nsize;
   struct range ranges[8];
   unsigned nranges;
   unsigned ssbo_set[2], ssbo_binding[2];
   unsigned nssbo;
   /* register-indexed UBO loads: one pointer (UBO VA + constant base) each.
    * The pairs sit at even common dwords below 32 (addr.add_shift's base
    * field), so at most 16; plan_common() refuses a pair past dword 30 and
    * add_upload() more than PCO_G5P_MAX_UPLOADS uploads. Zink puts every
    * default-block uniform in one UBO, so each constant index offset of an
    * array is its own base: Hyprland's border reads gradient[len - 1],
    * gradient[i] and gradient[i + 1] of two arrays (6 pointers). */
   struct { unsigned set, binding, element, common; uint32_t base; } uptr[16];
   unsigned nuptr;
   unsigned common_next;
   int xfb_common; /* VS: common dword of the PCO_G5P_UPLOAD_XFB block, -1 = none */
   /* control flow (the G9P port): ifs and loops left after flattening run
    * under the execution-mask stack. Set before codegen, kept by
    * codegen_reset(). */
   bool cf;
   struct { int start, end; } loops[G5P_MAX_LOOPS];
   unsigned nloops;
   nir_def *decls[G5P_MAX_DECLS]; /* NIR registers (out of SSA) */
   bool decl_out[G5P_MAX_DECLS];  /* vertex: a register whose value an output stores (bank 1) */
   int decl_first[G5P_MAX_DECLS];  /* their allocation point */
   unsigned ndecls;
   unsigned st;       /* mask-stack register (dword): the one after the program's own */
   int st_last;       /* count pass: index of the last stack operation */
   bool reserve_st31; /* pressure retry: keep r31 free for the mask stack */
   /* per codegen pass */
   bool decl_held[G5P_MAX_DECLS];
   bool stack_used;   /* a stack operation was put: pushes read the register */
   unsigned cf_depth; /* if/loop nesting of the instruction being put */
   unsigned pushes;   /* if levels pushed since the innermost loop began */
   unsigned exits[G5P_MAX_EXITS]; /* byte offsets of the breaks' exit jumps */
   unsigned nexits;
   /* Registers above r31 (G9P's homes, fragment only): per value, its
    * home register or -1. Chosen between codegen attempts; kept by
    * codegen_reset(). */
   int8_t *home;
   unsigned nhomes;
};

static _Noreturn void PRINTFLIKE(2, 3) refuse(struct gctx *c, const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   int n = snprintf(c->p->refusal, sizeof(c->p->refusal), "%s: ", c->vs ? "vertex" : "fragment");
   vsnprintf(c->p->refusal + n, sizeof(c->p->refusal) - n, fmt, ap);
   va_end(ap);
   longjmp(c->fail, 1);
}

static struct val *V(struct gctx *c, nir_def *d, unsigned comp)
{
   assert(d->index * 4 + comp < c->nvals);
   return &c->vals[d->index * 4 + comp];
}

static void put(struct gctx *c, const uint8_t *raw, unsigned n, const char *what)
{
   if (!n)
      refuse(c, "%s: encoder rejects the fields (%s)", what, g5p_usc_error ? g5p_usc_error : "?");
   if (c->size + n > c->code_cap) {
      if (c->size + n > PCO_G5P_PROGRAM_MAX)
         refuse(c, "program exceeds %u bytes", PCO_G5P_PROGRAM_MAX);
      unsigned cap = c->code_cap ? c->code_cap : CODE_INITIAL;
      while (cap < c->size + n)
         cap *= 2;
      uint8_t *code = realloc(c->code, cap);
      if (!code)
         refuse(c, "out of memory");
      c->code = code;
      c->code_cap = cap;
   }
   memcpy(c->code + c->size, raw, n);
   c->size += n;
   c->nput++;
}

/* ---- vertex register banks (docs/gpu/vs-two-bank-allocator.md) ----
 * A vertex register below VS_BANK0 is bank 1 (the attributes, outputs and
 * unified store; every vertex temp so far); VS_BANK0 + i is bank 0 r<i>, the
 * temps the DOUTU declares. An instruction's bank field is its destination's
 * bank, and the header marks register operands A, B and C in bank 1 with
 * 0x20, 0x10 and 0x08 (checked on 111 of Apple's vertex programs,
 * tools/g5p-1b-dgl/vs2bank/bank-check.py). Fragment registers are bank 0. */
#define VS_BANK0 32u
#define VS_A 0x20u
#define VS_B 0x10u
#define VS_C 0x08u

static inline unsigned vbank(const struct gctx *c, unsigned r)
{
   return c->vs && r < VS_BANK0;
}

/* The 16-bit half selector of register r's low half within its bank. */
static inline unsigned H(const struct gctx *c, unsigned r)
{
   return 2 * (c->vs && r >= VS_BANK0 ? r - VS_BANK0 : r);
}

/* A witnessed all-bank-1 vertex header with the bit of each bank-0 register
 * operand cleared; -1 for a slot that is not a register operand. The base
 * header must carry the bit of every register operand passed. */
static unsigned vhdr(const struct gctx *c, unsigned header, int a, int b, int cc)
{
   assert(c->vs);
   assert(a < 0 || (header & VS_A));
   assert(b < 0 || (header & VS_B));
   assert(cc < 0 || (header & VS_C));
   if (a >= 0 && !vbank(c, a))
      header &= ~VS_A;
   if (b >= 0 && !vbank(c, b))
      header &= ~VS_B;
   if (cc >= 0 && !vbank(c, cc))
      header &= ~VS_C;
   return header;
}

/* A vertex form witnessed only with every register in bank 1: refused by
 * name when an operand or the destination is in bank 0 (-1 ends the list). */
static void refuse(struct gctx *c, const char *fmt, ...);
static void vs_bank1_only(struct gctx *c, const char *what, int r0, int r1, int r2, int r3)
{
   const int r[4] = {r0, r1, r2, r3};
   for (unsigned i = 0; i < 4; i++)
      if (r[i] >= 0 && !vbank(c, r[i]))
         refuse(c, "%s with a register in vertex bank 0 (no witnessed mixed-bank form)", what);
}

/* ---- registers (dwords) ---- */
/* A vertex bank-0 register: the first free one, else -1. */
static int alloc_bank0(struct gctx *c)
{
   for (unsigned r = VS_BANK0; r < VS_BANK0 + NUM_DWORDS; r++)
      if (!((c->busy | c->reserved) & BITFIELD64_BIT(r))) {
         c->busy |= BITFIELD64_BIT(r);
         c->max_bank0 = MAX2(c->max_bank0, r - VS_BANK0 + 1);
         return r;
      }
   return -1;
}

static unsigned alloc_run(struct gctx *c, unsigned n, const char *why)
{
   if (c->vs && c->tex_bank0) {
      for (unsigned r = VS_BANK0; r + n <= VS_BANK0 + NUM_DWORDS; r++) {
         uint64_t mask = BITFIELD64_RANGE(r, n);
         if (!((c->busy | c->reserved) & mask)) {
            c->busy |= mask;
            c->max_bank0 = MAX2(c->max_bank0, r - VS_BANK0 + n);
            return r;
         }
      }
      refuse(c, "vertex texture bank0 register pressure: %s needs %u dwords", why, n);
   }
   /* The vertex bank-0 retry places registers as Apple's vertex programs do:
    * temps in bank 0, the values of vertex outputs in bank 1 (every one of
    * Apple's outputs reads bank 1). Runs (address pairs, loads) stay in
    * bank 1. */
   if (n == 1 && c->vs && c->vs_bank0 && !c->alloc_out) {
      int r = alloc_bank0(c);
      if (r >= 0)
         return r;
   }
   for (unsigned r = 0; r + n <= NUM_DWORDS; r++) {
      uint64_t mask = BITFIELD64_RANGE(r, n);
      if (!((c->busy | c->reserved) & mask)) {
         c->busy |= mask;
         c->max_dword = MAX2(c->max_dword, r + n);
         c->max_written = MAX2(c->max_written, r + n);
         return r;
      }
   }
   if (n == 1 && c->vs && c->vs_bank0) {
      int r = alloc_bank0(c);
      if (r >= 0)
         return r;
      refuse(c, "register pressure: %s needs %u dword(s), %u of %u in use in each vertex bank", why, n, NUM_DWORDS,
             NUM_DWORDS);
   }
   refuse(c, "register pressure: %s needs %u dword(s), %u of %u in use", why, n,
          util_bitcount64((c->busy | c->reserved) & BITFIELD64_MASK(NUM_DWORDS)), NUM_DWORDS);
}

/* The flown VS temps (g5p_gfx.h G5P_VS_TEMPS_FLOWN). A VS whose inputs
 * reach past them keeps its written registers below them where it can: the
 * DOUTU temps then cover only those, and the inputs stay in the unified
 * store (Apple's g3b-vtx16 writes only r4/r5 while reading up to r18). */
#define VS_WRITE_REGS 12

static unsigned alloc_reg(struct gctx *c, int prefer)
{
   /* A dying input above them is not reused in place while a lower register
    * is free. Every VS within the flown temps uses no register this high. */
   if (c->vs && prefer >= (int)VS_BANK0 && !c->tex_bank0 && !c->vs_bank0)
      prefer = -1; /* preserve the ordinary bank1 route */
   /* A vertex output's value is not computed in place over a bank-0 operand. */
   if (c->alloc_out && prefer >= (int)VS_BANK0)
      prefer = -1;
   if (c->vs && prefer >= VS_WRITE_REGS && prefer < (int)VS_BANK0 &&
       (~(c->busy | c->reserved) & BITFIELD64_MASK(prefer)))
      prefer = -1;
   if (prefer >= 0 && !((c->busy | c->reserved) & BITFIELD64_BIT(prefer))) {
      c->busy |= BITFIELD64_BIT(prefer);
      if (prefer >= (int)VS_BANK0) {
         c->max_bank0 = MAX2(c->max_bank0, (unsigned)prefer - VS_BANK0 + 1);
         return prefer;
      }
      c->max_dword = MAX2(c->max_dword, (unsigned)prefer + 1);
      c->max_written = MAX2(c->max_written, (unsigned)prefer + 1);
      return prefer;
   }
   return alloc_run(c, 1, "a value");
}

/* max_written: registers the program writes. Vertex inputs (fetched
 * attributes, index DOUTWs) are held, not written: the PDS puts them in the
 * unified store, which VDM_STATE5 sizes from the fetch. Apple's g3b-vtx16
 * reads r14/r15 with DOUTU temps 8 (inputs r0..r18, unified 12 units). */
static void hold(struct gctx *c, struct val *v, unsigned reg)
{
   v->reg = reg;
   v->held = true;
   c->busy |= BITFIELD64_BIT(reg);
   if (c->vs && reg >= VS_BANK0) {
      c->max_bank0 = MAX2(c->max_bank0, reg - VS_BANK0 + 1);
      return;
   }
   c->max_dword = MAX2(c->max_dword, reg + 1);
   if (!v->input)
      c->max_written = MAX2(c->max_written, reg + 1);
}

static void drop(struct gctx *c, struct val *v)
{
   if (v->held) {
      c->busy &= ~BITFIELD64_BIT(v->reg);
      v->held = false;
   }
}

/* Free every register whose value is not read after instruction idx. */
static void free_scratch(struct gctx *c)
{
   c->busy &= ~c->scratch;
   c->scratch = 0;
}

static void release(struct gctx *c, int idx)
{
   for (unsigned i = 0; i < c->nvals; i++)
      if (c->vals[i].held && c->vals[i].last <= idx)
         drop(c, &c->vals[i]);
   free_scratch(c);
}

/* ---- value resolution ---- */
static nir_alu_instr *alu_of(nir_def *d)
{
   nir_instr *i = nir_def_instr(d);
   return i->type == nir_instr_type_alu ? nir_instr_as_alu(i) : NULL;
}

static nir_intrinsic_instr *intrinsic_of(nir_def *d)
{
   nir_instr *i = nir_def_instr(d);
   return i->type == nir_instr_type_intrinsic ? nir_instr_as_intrinsic(i) : NULL;
}

/* Through mov/vec only. */
static struct ref resolve_plain(nir_def *d, unsigned comp)
{
   for (;;) {
      nir_alu_instr *a = alu_of(d);
      if (a && a->op == nir_op_mov) {
         comp = a->src[0].swizzle[comp];
         d = a->src[0].src.ssa;
      } else if (a && nir_op_is_vec(a->op)) {
         nir_alu_src *s = &a->src[comp];
         comp = s->swizzle[0];
         d = s->src.ssa;
      } else {
         return (struct ref){d, comp, false, false};
      }
   }
}

/* Any operand: mov/vec and a folded fsat (its value is the producer's). */
static struct ref resolve_value(struct gctx *c, nir_def *d, unsigned comp)
{
   for (;;) {
      struct ref r = resolve_plain(d, comp);
      nir_alu_instr *a = alu_of(r.def);
      if (!(a && a->op == nir_op_fsat && c->sat_folded[r.def->index]))
         return r;
      comp = a->src[0].swizzle[r.comp];
      d = a->src[0].src.ssa;
   }
}

/* Float operand: also through fneg/fabs and a folded fsat. */
static struct ref resolve_float(struct gctx *c, nir_def *d, unsigned comp)
{
   bool neg = false, abs = false;
   for (;;) {
      struct ref r = resolve_plain(d, comp);
      d = r.def;
      comp = r.comp;
      nir_alu_instr *a = alu_of(d);
      if (a && a->op == nir_op_fneg) {
         if (!abs)
            neg = !neg;
      } else if (a && a->op == nir_op_fabs) {
         abs = true;
      } else if (!(a && a->op == nir_op_fsat && c->sat_folded[d->index])) {
         return (struct ref){d, comp, neg, abs};
      }
      comp = a->src[0].swizzle[comp];
      d = a->src[0].src.ssa;
   }
}

static bool float_consumer(nir_op op)
{
   return op == nir_op_fadd || op == nir_op_fsub || op == nir_op_fmul || op == nir_op_ffma;
}

static bool alias_op(struct gctx *c, nir_alu_instr *a)
{
   return a->op == nir_op_mov || nir_op_is_vec(a->op) ||
          (a->op == nir_op_fsat && c->sat_folded[a->def.index]);
}

static struct ref alu_src(struct gctx *c, nir_alu_instr *a, unsigned i)
{
   return float_consumer(a->op) ? resolve_float(c, a->src[i].src.ssa, a->src[i].swizzle[0])
                                : resolve_value(c, a->src[i].src.ssa, a->src[i].swizzle[0]);
}

static uint32_t apply_mods(uint32_t bits, struct ref r)
{
   if (r.abs)
      bits &= 0x7fffffffu;
   if (r.neg)
      bits ^= 0x80000000u;
   return bits;
}

static bool f16_exact(uint32_t bits, uint16_t *h)
{
   float f;
   memcpy(&f, &bits, 4);
   if (f != f)
      return false;
   *h = _mesa_float_to_half(f);
   float back = _mesa_half_to_float(*h);
   uint32_t bb;
   memcpy(&bb, &back, 4);
   return bb == bits;
}

/* ---- support scan: every unsupported construct, not only the first ---- */
struct unsupported {
   char text[480];
   unsigned n;
};

static void note(struct unsupported *u, const char *what)
{
   if (strstr(u->text, what))
      return;
   size_t l = strlen(u->text);
   snprintf(u->text + l, sizeof(u->text) - l, "%s%s", u->n++ ? ", " : "", what);
}

/* Round 19: relations producing a 1-bit boolean (strict select recipe). */
static bool is_cmp(nir_op op)
{
   switch (op) {
   case nir_op_flt: case nir_op_fge: case nir_op_feq: case nir_op_fneu:
   case nir_op_ilt: case nir_op_ige: case nir_op_ieq: case nir_op_ine:
   case nir_op_ult: case nir_op_uge:
      return true;
   default:
      return false;
   }
}

/* Recipes whose vertex form is measured (vsops-probes.py: v_rcp, v_rsq,
 * v_min, v_max). */
static bool vs_recipe(nir_op op)
{
   return op == nir_op_frcp || op == nir_op_frsq || op == nir_op_fmin || op == nir_op_fmax || op == nir_op_fsin ||
          op == nir_op_fcos || op == nir_op_ftrunc || op == nir_op_fround_even || op == nir_op_fsign ||
          op == nir_op_fceil;
}

/* Ops with a measured complete Apple lowering (isa-g5p.md ISA round 2 §3). */
static bool is_recipe(nir_op op)
{
   switch (op) {
   case nir_op_fmin: case nir_op_fmax: case nir_op_frcp: case nir_op_frsq: case nir_op_fsqrt:
   case nir_op_fsign: case nir_op_fceil: case nir_op_ftrunc: case nir_op_fround_even:
   case nir_op_fsin: case nir_op_fcos:
      return true;
   default:
      return false;
   }
}

static bool is_deriv(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_ddx: case nir_intrinsic_ddx_fine: case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddy: case nir_intrinsic_ddy_fine: case nir_intrinsic_ddy_coarse:
      return true;
   default:
      return false;
   }
}

static bool alu_supported(struct gctx *c, nir_alu_instr *a)
{
   unsigned bits = a->def.bit_size;
   switch (a->op) {
   case nir_op_mov: case nir_op_vec2: case nir_op_vec3: case nir_op_vec4:
      return true;
   case nir_op_fadd: case nir_op_fmul: case nir_op_ffma: case nir_op_fneg:
   case nir_op_fabs: case nir_op_fsat: case nir_op_fsub: case nir_op_u2f32:
      return bits == 32;
   case nir_op_inot:
      /* vertex: v_inot32 (unary not, bank 1, header 0x21); a 1-bit one is the
       * literal XOR of a materialized 1.0/0.0 boolean in the vertex literal
       * form (v_bnot's binary literal: bank 1, header 0x11) */
      return bits == 32 || bits == 1;
   case nir_op_iand: case nir_op_ior: case nir_op_ixor:
      /* vertex: register pairs only (v_iand, v_ior, v_ixor: bank 1, header 0x31); a 1-bit one combines
       * materialized 1.0/0.0 booleans */
      return bits == 32 || bits == 1;
   /* Vertex forms of the selects, recipes, floor and i2f: Apple's own vertex
    * programs (tools/mesa-a9-hyprland-gpu/vsops-probes.py, hyprland-gpu.md
    * §10) are the fragment forms with destination bank 1 and the header's
    * bank bit per register operand (vs_bits). */
   case nir_op_bcsel:
      return bits == 32 || bits == 1;
   case nir_op_b2f32: case nir_op_b2i32:
      return true;
   case nir_op_iadd: case nir_op_imul:
      return bits == 32; /* VS: the XFB probe forms (vs_int) */
   case nir_op_ishl:
      return bits == 32; /* vertex, a dynamic amount: v_ishl_dyn (emit_shift) */
   case nir_op_imin: case nir_op_imax: case nir_op_umin:
   case nir_op_umax: case nir_op_ushr: case nir_op_ishr: case nir_op_isub:
   case nir_op_ineg:
   case nir_op_i2f32: case nir_op_ffloor:
      return bits == 32;
   case nir_op_flog2: case nir_op_fexp2: /* vertex: v_log2, v_exp2 (bank 1, header 0x21) */
      return bits == 32;
   case nir_op_f2i32: case nir_op_f2u32: /* vertex: v_f2i, v_f2u (convert, bank 1, header 0x11) */
      return bits == 32;
   case nir_op_bit_count: case nir_op_bitfield_reverse:
      return !c->vs && bits == 32;
   default:
      return (is_cmp(a->op) && bits == 1) ||
             (is_recipe(a->op) && bits == 32 && (!c->vs || vs_recipe(a->op)));
   }
}

/* Register-indexed UBO loads (defined with the common-store plan). */
static bool dyn_offset_alu(nir_alu_instr *a);
static bool dyn_ubo_load(nir_intrinsic_instr *in);
static bool dyn_ubo_offset(nir_intrinsic_instr *in, struct ref *idx, unsigned *shift, uint32_t *base);

/* Vertex iadd that only feeds UBO/push offsets folds into the upload plan. */
static bool offset_only(nir_def *d)
{
   nir_foreach_use(u, d) {
      nir_instr *i = nir_src_use_instr(u);
      if (i->type == nir_instr_type_alu && nir_instr_as_alu(i)->op == nir_op_iadd &&
          offset_only(&nir_instr_as_alu(i)->def))
         continue;
      if (i->type != nir_instr_type_intrinsic)
         return false;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(i);
      if (!((in->intrinsic == nir_intrinsic_load_ubo && u == &in->src[1]) ||
            (in->intrinsic == nir_intrinsic_load_push_constant && u == &in->src[0])))
         return false;
   }
   return true;
}

/* ---- textypes: texture shapes (docs/gpu/experiments/textypes.md §3) ---- */
/* The round-18 sample: 2D float, implicit LOD, coordinates only. */
static bool plain_tex(const nir_tex_instr *t)
{
   if (t->op != nir_texop_tex || t->sampler_dim != GLSL_SAMPLER_DIM_2D || t->is_array || t->is_shadow ||
       t->def.num_components != 4 || nir_alu_type_get_base_type(t->dest_type) != nir_type_float)
      return false;
   for (unsigned i = 0; i < t->num_srcs; i++)
      if (t->src[i].src_type != nir_tex_src_coord && t->src[i].src_type != nir_tex_src_texture_deref &&
          t->src[i].src_type != nir_tex_src_sampler_deref)
         return false;
   return true;
}

static bool size_query(const nir_tex_instr *t)
{
   return t->op == nir_texop_txs || t->op == nir_texop_query_levels;
}

/* Constant signed nibbles: x at0, y at4, and the qualified3D z at8. */
static int packed_offset_components(const nir_tex_instr *t, unsigned components)
{
   int i = nir_tex_instr_src_index(t, nir_tex_src_offset);
   if (i < 0 || !nir_src_is_const(t->src[i].src) || t->src[i].src.ssa->num_components != components)
      return -1;
   int packed = 0;
   for (unsigned k = 0; k < components; k++) {
      int v = nir_src_comp_as_int(t->src[i].src, k);
      if (v < -8 || v > 7)
         return -1;
      packed |= (v & 15) << (4*k);
   }
   return packed;
}

static int packed_offset(const nir_tex_instr *t)
{
   return packed_offset_components(t, 2);
}

/* Why Apple's probes do not cover this texture instruction, or NULL. */
static const char *tex_shape_refusal(const nir_tex_instr *t, bool vertex_gradient, bool vertex_xyz_offset, bool fragment)
{
   enum glsl_sampler_dim d = t->sampler_dim;
   if (d == GLSL_SAMPLER_DIM_BUF)
      return g5p_tbo_shape_refusal(t);
   nir_alu_type base = nir_alu_type_get_base_type(t->dest_type);
   if (d != GLSL_SAMPLER_DIM_2D && d != GLSL_SAMPLER_DIM_3D && d != GLSL_SAMPLER_DIM_CUBE)
      return "1D, buffer, rect, MS or subpass texture";
   if (t->is_array && d != GLSL_SAMPLER_DIM_2D)
      return "cube or 3D array (A9 has no cube arrays)";
   if (base != nir_type_float && base != nir_type_int && base != nir_type_uint)
      return "result type";
   if (size_query(t)) {
      int l = nir_tex_instr_src_index(t, nir_tex_src_lod);
      if (t->op == nir_texop_txs && (l < 0 || !nir_src_is_const(t->src[l].src)))
         return "textureSize with a non-constant level";
      return NULL;
   }
   switch (t->op) {
   case nir_texop_tex: case nir_texop_txb: case nir_texop_txl: case nir_texop_txd: case nir_texop_txf:
      break;
   default:
      return "texture op (not probed)";
   }
   if (t->def.num_components != (t->is_shadow ? 1u : 4u))
      return "result width (rgba, or one compare result)";
   if (t->op == nir_texop_txd && d != GLSL_SAMPLER_DIM_2D && !(fragment && (d == GLSL_SAMPLER_DIM_3D || d == GLSL_SAMPLER_DIM_CUBE)))
      return "gradients on 3D or cube (Apple's gradient layout is measured on 2D/arrays)";
   if (t->op == nir_texop_txf && (d == GLSL_SAMPLER_DIM_CUBE || t->is_shadow))
      return "texelFetch on a cube or shadow texture";
   if (t->is_shadow && (t->op == nir_texop_txb || t->op == nir_texop_txd || d == GLSL_SAMPLER_DIM_3D))
      return "compare with bias/gradients or on 3D (not probed)";
   bool offset = false;
   for (unsigned i = 0; i < t->num_srcs; i++) {
      switch (t->src[i].src_type) {
      case nir_tex_src_coord: case nir_tex_src_texture_deref: case nir_tex_src_sampler_deref:
      case nir_tex_src_comparator:
         break;
      case nir_tex_src_bias:
         if (t->op != nir_texop_txb)
            return "bias source";
         break;
      case nir_tex_src_lod:
         if (t->op != nir_texop_txl && t->op != nir_texop_txf)
            return "level source";
         break;
      case nir_tex_src_ddx: case nir_tex_src_ddy:
         break;
      case nir_tex_src_offset:
         offset = true;
         break;
      case nir_tex_src_projector: /* lower() divides it out; one left is a bug */
         return "texture source: projector left after nir_lower_tex";
      case nir_tex_src_min_lod:
         return "texture source: min-LOD clamp (textureClampARB / SPIR-V MinLod; no Apple form decoded)";
      case nir_tex_src_ms_index:
         return "texture source: sample index";
      case nir_tex_src_texture_offset: case nir_tex_src_sampler_offset:
         return "texture source: dynamically indexed sampler array";
      case nir_tex_src_texture_handle: case nir_tex_src_sampler_handle:
         return "texture source: bindless handle";
      default:
         return "texture source: other (not probed)";
      }
   }
   if (offset && vertex_xyz_offset && packed_offset_components(t, 3) < 0)
      return "3D offset requires constant XYZ components in -8..7";
   if (offset && ((vertex_xyz_offset ? packed_offset_components(t, 3) : packed_offset(t)) < 0 ||
                  (d != GLSL_SAMPLER_DIM_2D && !vertex_xyz_offset) || (t->op == nir_texop_txd && !vertex_gradient) ||
                  t->op == nir_texop_txf))
      return "texel offset: Apple's packed form is probed for constant 2D offsets in -8..7 on "
             "sample/level/bias/compare";
   return NULL;
}

/* Transform feedback record base from nir_lower_xfb_to_stores:
 * iadd(load_xfb_address(b), u2u64(off)), one 64-bit address add of a
 * shared pointer pair and a 32-bit offset (v_st_xfb, v_st_bytes). */
/* Ordinary bank0 texture forms only. No continuation is inferred from stage. */
static const char *vs_tex_refusal(const nir_tex_instr *t)
{
   const char *why = tex_shape_refusal(t, false, false, false);
   if (why || size_query(t)) return why;
   if (t->op != nir_texop_tex && t->op != nir_texop_txl)
      return "vertex fetch/bias/gradient requires separately qualified state";
   if (t->is_array || t->is_shadow ||
       (t->sampler_dim != GLSL_SAMPLER_DIM_2D && t->sampler_dim != GLSL_SAMPLER_DIM_CUBE))
      return "vertex array/3D/shadow texture not qualified on the ordinary path";
   if (nir_alu_type_get_base_type(t->dest_type) != nir_type_float)
      return "vertex sampled integer result not qualified";
   if (nir_tex_instr_src_index(t, nir_tex_src_offset) >= 0)
      return "vertex texture offset not qualified";
   return NULL;
}

static const char *vs_qualified_tex_refusal(const nir_tex_instr *t)
{
   if (t->sampler_dim == GLSL_SAMPLER_DIM_BUF)
      return g5p_tbo_shape_refusal(t);
   bool gradient = t->op == nir_texop_txd && g5p_vtex_gradient_probe() &&
                   t->sampler_dim == GLSL_SAMPLER_DIM_2D;
   bool xyz = g5p_vtex_xyz_offset_probe() && t->sampler_dim == GLSL_SAMPLER_DIM_3D &&
              (t->op == nir_texop_tex || t->op == nir_texop_txl);
   const char *why = tex_shape_refusal(t, gradient, xyz, false);
   if (why || size_query(t)) return why;
   if (t->op != nir_texop_tex && t->op != nir_texop_txl && t->op != nir_texop_txf && !gradient)
      return "vertex bias/gradient recipe not device-qualified";
   if ((t->is_shadow && (!g5p_vtex_shadow_probe() || t->sampler_dim != GLSL_SAMPLER_DIM_2D || t->is_array)) ||
       (t->sampler_dim == GLSL_SAMPLER_DIM_CUBE &&
                        (!g5p_vtex_cube_probe() || t->is_array)) ||
       (t->sampler_dim != GLSL_SAMPLER_DIM_2D && t->sampler_dim != GLSL_SAMPLER_DIM_3D &&
        t->sampler_dim != GLSL_SAMPLER_DIM_CUBE))
      return "vertex cube/shadow recipe not device-qualified";
   if (nir_tex_instr_src_index(t, nir_tex_src_offset) >= 0 && !g5p_vtex_fetch_offset_probe())
      return "vertex texture offset not device-qualified";
   /* Additive selectors retain baseline float cube/offset support without
    * admitting unqualified combinations of independently gated features. */
   if (g5p_vtex_integer_probe() && t->op != nir_texop_txf && nir_alu_type_get_base_type(t->dest_type) != nir_type_float &&
       (t->sampler_dim == GLSL_SAMPLER_DIM_CUBE ||
        (nir_tex_instr_src_index(t, nir_tex_src_offset) >= 0 && !g5p_vtex_xyz_offset_probe())))
      return "vertex integer cube/offset combination not qualified in this mode";
   /* Native float/sint/uint sample programs are byte-identical; image-view
    * format state determines the result bits, not a different USC recipe. */
   if (t->op != nir_texop_txf && nir_alu_type_get_base_type(t->dest_type) != nir_type_float &&
       !g5p_vtex_integer_probe())
      return "vertex sampled integer result not device-qualified";
   return NULL;
}

static bool xfb_base(struct ref r, unsigned *buffer, struct ref *off)
{
   nir_alu_instr *add = alu_of(r.def);
   if (!add || add->op != nir_op_iadd || add->def.bit_size != 64)
      return false;
   for (unsigned i = 0; i < 2; i++) {
      struct ref p = resolve_plain(add->src[i].src.ssa, add->src[i].swizzle[r.comp]);
      struct ref o = resolve_plain(add->src[1 - i].src.ssa, add->src[1 - i].swizzle[r.comp]);
      nir_intrinsic_instr *in = intrinsic_of(p.def);
      nir_alu_instr *cvt = alu_of(o.def);
      if (in && in->intrinsic == nir_intrinsic_load_xfb_address && nir_intrinsic_base(in) < 4 && cvt &&
          cvt->op == nir_op_u2u64 && cvt->src[0].src.ssa->bit_size == 32) {
         *buffer = nir_intrinsic_base(in);
         *off = resolve_plain(cvt->src[0].src.ssa, cvt->src[0].swizzle[o.comp]);
         return true;
      }
   }
   return false;
}

/* nir_lower_xfb_to_stores adds each output's byte offset to the 32-bit
 * index * stride before the 64-bit address add: an output at +4/+8/+12 has
 * the offset iadd(index * stride, k). Its record base is index * stride and
 * k the store's immediate dword offset, so the dwords of one buffer's record
 * share one address add (Apple folds +4/+8/+12 the same way: v_st_off12),
 * also when the varyings are packed one component per output. */
static void xfb_peel(struct ref *off, unsigned *dword)
{
   nir_alu_instr *add = alu_of(off->def);
   if (!add || add->op != nir_op_iadd || add->def.bit_size != 32)
      return;
   for (unsigned i = 0; i < 2; i++) {
      if (!nir_src_is_const(add->src[i].src))
         continue;
      uint64_t k = nir_src_comp_as_uint(add->src[i].src, add->src[i].swizzle[off->comp]);
      if (k % 4 || *dword + k / 4 > 3)
         return;
      *off = resolve_plain(add->src[1 - i].src.ssa, add->src[1 - i].swizzle[off->comp]);
      *dword += k / 4;
      return;
   }
}

/* A store address: the record base, or the base plus 4/8/12 for the later
 * dwords nir_lower_mem_access_bit_sizes split off. The constant becomes the
 * store's immediate dword offset, so a record needs one address add. */
static bool xfb_address(nir_def *addr, unsigned *buffer, struct ref *off, unsigned *dword)
{
   struct ref r = resolve_plain(addr, 0);
   *dword = 0;
   if (xfb_base(r, buffer, off)) {
      xfb_peel(off, dword);
      return true;
   }
   nir_alu_instr *add = alu_of(r.def);
   if (!add || add->op != nir_op_iadd || add->def.bit_size != 64)
      return false;
   for (unsigned i = 0; i < 2; i++) {
      if (!nir_src_is_const(add->src[i].src))
         continue;
      uint64_t k = nir_src_comp_as_uint(add->src[i].src, add->src[i].swizzle[r.comp]);
      struct ref b = resolve_plain(add->src[1 - i].src.ssa, add->src[1 - i].swizzle[r.comp]);
      if (k % 4 || k > 12 || !xfb_base(b, buffer, off))
         return false;
      *dword = k / 4;
      xfb_peel(off, dword);
      return true;
   }
   return false;
}

/* The record base of a store: the stores of one record share it. */
static int store_base(struct gctx *c, unsigned buffer, struct ref off, bool *first)
{
   for (unsigned i = 0; i < c->nst_base; i++) {
      if (c->st_base[i].buffer == buffer && c->st_base[i].off.def == off.def && c->st_base[i].off.comp == off.comp) {
         *first = false;
         return i;
      }
   }
   if (c->nst_base == ARRAY_SIZE(c->st_base))
      refuse(c, "more than %u transform feedback record bases", (unsigned)ARRAY_SIZE(c->st_base));
   *first = true;
   c->st_base[c->nst_base].buffer = buffer;
   c->st_base[c->nst_base].off = off;
   c->st_base[c->nst_base].pair = -1;
   return c->nst_base++;
}

/* 64-bit constants that only feed store addresses (the split's 4/8/12). */
static bool address_only(nir_alu_instr *a);
static bool address_const(nir_def *d)
{
   if (d->bit_size != 64 || nir_def_is_unused(d))
      return false;
   nir_foreach_use(u, d) {
      nir_instr *i = nir_src_use_instr(u);
      if (i->type != nir_instr_type_alu || !address_only(nir_instr_as_alu(i)))
         return false;
   }
   return true;
}

/* 64-bit ALU that only builds transform feedback store addresses. */
static bool address_only(nir_alu_instr *a)
{
   if (a->def.bit_size != 64 || (a->op != nir_op_iadd && a->op != nir_op_u2u64 && a->op != nir_op_mov))
      return false;
   nir_foreach_use(u, &a->def) {
      nir_instr *i = nir_src_use_instr(u);
      if (i->type == nir_instr_type_alu && address_only(nir_instr_as_alu(i)))
         continue;
      if (i->type != nir_instr_type_intrinsic)
         return false;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(i);
      unsigned b, k;
      struct ref o;
      if (in->intrinsic != nir_intrinsic_store_global || u != &in->src[1] ||
          !xfb_address(in->src[1].ssa, &b, &o, &k))
         return false;
   }
   return true;
}

static const char *store_refusal(nir_intrinsic_instr *in)
{
   unsigned b, k;
   struct ref o;
   if (!xfb_address(in->src[1].ssa, &b, &o, &k))
      return "store_global (address is not a transform feedback buffer + 32-bit offset)";
   /* Device run 2: the 16-byte vertex store is dropped, the 4-byte one lands. */
   if (in->src[0].ssa->bit_size != 32 || in->src[0].ssa->num_components != 1 || nir_intrinsic_write_mask(in) != 1)
      return "store_global (vertex tasks commit only 32-bit scalar stores)";
   return NULL;
}

static void scan_cf(struct gctx *c, struct exec_list *list, struct unsupported *u)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         if (!c->cf)
            note(u, "control flow (if) left after flattening");
         scan_cf(c, &nir_cf_node_as_if(node)->then_list, u);
         scan_cf(c, &nir_cf_node_as_if(node)->else_list, u);
         continue;
      }
      if (node->type == nir_cf_node_loop) {
         if (!c->cf)
            note(u, "control flow (loop) not unrolled");
         scan_cf(c, &nir_cf_node_as_loop(node)->body, u);
         continue;
      }
      nir_foreach_instr(ins, nir_cf_node_as_block(node)) {
         char buf[96];
         switch (ins->type) {
         case nir_instr_type_load_const:
         case nir_instr_type_undef:
         case nir_instr_type_deref:
            break;
         case nir_instr_type_alu: {
            nir_alu_instr *a = nir_instr_as_alu(ins);
            if (c->vs && a->op == nir_op_iadd && offset_only(&a->def))
               break; /* UBO offset arithmetic: folded by eval_offset */
            if (dyn_offset_alu(a))
               break; /* register-indexed UBO addressing: folded into addr.add_shift */
            if (c->vs && address_only(a))
               break; /* XFB store address: folded into the address add */
            if (!alu_supported(c, a)) {
               if (a->def.bit_size != 32 && a->def.bit_size != 1)
                  snprintf(buf, sizeof(buf), "%s (%u-bit)", nir_op_infos[a->op].name, a->def.bit_size);
               else if (c->vs && is_recipe(a->op))
                  snprintf(buf, sizeof(buf), "%s (vertex: not witnessed)", nir_op_infos[a->op].name);
               else
                  snprintf(buf, sizeof(buf), "%s", nir_op_infos[a->op].name);
               note(u, buf);
            }
            break;
         }
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(ins);
            bool ok;
            switch (in->intrinsic) {
            case nir_intrinsic_load_input: /* VS attributes; FS flat varyings */
               ok = true;
               break;
            case nir_intrinsic_load_interpolated_input:
            case nir_intrinsic_load_barycentric_pixel:
            case nir_intrinsic_load_barycentric_centroid:
            case nir_intrinsic_demote:
            case nir_intrinsic_demote_if:
            case nir_intrinsic_terminate:
            case nir_intrinsic_terminate_if:
               ok = !c->vs;
               break;
            case nir_intrinsic_load_frag_coord:
               ok = !c->vs;
               nir_foreach_use(use, &in->def) {
                  nir_instr *ui = nir_src_use_instr(use);
                  if (ui->type == nir_instr_type_alu) {
                     nir_alu_instr *ua = nir_instr_as_alu(ui);
                     for (unsigned k = 0; k < nir_op_infos[ua->op].num_inputs; k++)
                        if (&ua->src[k].src == use)
                           for (unsigned j = 0; j < nir_ssa_alu_instr_src_components(ua, k); j++)
                              if (ua->src[k].swizzle[j] > 1)
                                 ok = false;
                  } else {
                     ok = false;
                  }
               }
               if (!ok && !c->vs) {
                  note(u, "load_frag_coord .z/.w (moves user varying slots; only .xy is measured)");
                  continue;
               }
               break;
            case nir_intrinsic_store_output:
            case nir_intrinsic_vulkan_resource_index:
            case nir_intrinsic_load_vulkan_descriptor:
            case nir_intrinsic_load_push_constant:
            case nir_intrinsic_load_ubo:
               ok = true;
               break;
            case nir_intrinsic_store_ssbo:
               ok = in->src[0].ssa->bit_size==32 && in->num_components==1 && nir_intrinsic_write_mask(in)==1;
               break;
            case nir_intrinsic_load_ssbo:
               ok = !c->vs;
               break;
            case nir_intrinsic_load_vertex_id:
            case nir_intrinsic_load_instance_id:
            case nir_intrinsic_load_num_vertices:
            case nir_intrinsic_load_xfb_address:
               ok = c->vs;
               break;
            case nir_intrinsic_store_global:
               if (c->vs && store_refusal(in)) {
                  note(u, store_refusal(in));
                  continue;
               }
               ok = c->vs;
               break;
            case nir_intrinsic_ddx: case nir_intrinsic_ddx_fine: case nir_intrinsic_ddx_coarse:
            case nir_intrinsic_ddy: case nir_intrinsic_ddy_fine: case nir_intrinsic_ddy_coarse:
               ok = !c->vs && in->def.num_components == 1 && in->def.bit_size == 32;
               break;
            case nir_intrinsic_decl_reg:
            case nir_intrinsic_load_reg:
            case nir_intrinsic_store_reg:
               ok = c->cf;
               break;
            default:
               ok = false;
            }
            if (!ok)
               note(u, nir_intrinsic_infos[in->intrinsic].name);
            break;
         }
         case nir_instr_type_tex: {
            nir_tex_instr *t = nir_instr_as_tex(ins);
            const char *why = c->vs ? (g5p_vtex_qualified() ? vs_qualified_tex_refusal(t) : !g5p_vtex_probe() ? "vertex stage" : vs_tex_refusal(t)) :
                              plain_tex(t) ? NULL : tex_shape_refusal(t, true, t->sampler_dim == GLSL_SAMPLER_DIM_3D, true);
            if (why) {
               snprintf(buf, sizeof(buf), "tex(op %u, dim %u%s%s): %s", t->op, t->sampler_dim,
                        t->is_array ? ", array" : "", t->is_shadow ? ", shadow" : "", why);
               note(u, buf);
            }
            break;
         }
         case nir_instr_type_phi:
            note(u, "phi");
            break;
         case nir_instr_type_jump:
            if (c->cf && nir_instr_as_jump(ins)->type == nir_jump_break)
               break;
            if (c->cf && nir_instr_as_jump(ins)->type == nir_jump_continue)
               snprintf(buf, sizeof(buf), "continue");
            else
               snprintf(buf, sizeof(buf), "instruction type %u", ins->type);
            note(u, buf);
            break;
         default:
            snprintf(buf, sizeof(buf), "instruction type %u", ins->type);
            note(u, buf);
         }
      }
   }
}

/* Every instruction in program order: the one block of a straight-line
 * program, every block of a program with control flow. */
#define foreach_instr_all(ins, c)                                                                      \
   nir_foreach_block(ins##_block, (c)->impl)                                                           \
      nir_foreach_instr(ins, ins##_block)

/* ---- control flow: the loops and NIR registers ---- */
static int cf_ip_before(nir_cf_node *n)
{
   return nir_cf_node_as_block(nir_cf_node_prev(n))->end_ip;
}

static int cf_ip_after(nir_cf_node *n)
{
   return nir_cf_node_as_block(nir_cf_node_next(n))->start_ip;
}

static void plan_loops(struct gctx *c, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         plan_loops(c, &nir_cf_node_as_if(node)->then_list);
         plan_loops(c, &nir_cf_node_as_if(node)->else_list);
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *l = nir_cf_node_as_loop(node);
         if (c->nloops == G5P_MAX_LOOPS)
            refuse(c, "more than %u loops", G5P_MAX_LOOPS);
         c->loops[c->nloops].start = nir_loop_first_block(l)->start_ip;
         c->loops[c->nloops++].end = nir_loop_last_block(l)->end_ip;
         plan_loops(c, &l->body);
      }
   }
}

static int decl_slot(struct gctx *c, nir_def *d)
{
   for (unsigned i = 0; i < c->ndecls; i++)
      if (c->decls[i] == d)
         return i;
   return -1;
}

static bool vs_feeds_output(struct gctx *c, nir_def *d);
static void plan_decls(struct gctx *c)
{
   foreach_instr_all(ins, c) {
      if (ins->type != nir_instr_type_intrinsic ||
          nir_instr_as_intrinsic(ins)->intrinsic != nir_intrinsic_decl_reg)
         continue;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(ins);
      if (nir_intrinsic_num_components(in) != 1 || nir_intrinsic_num_array_elems(in) ||
          (nir_intrinsic_bit_size(in) != 32 && nir_intrinsic_bit_size(in) != 1))
         refuse(c, "NIR register: only scalar 32-bit and boolean registers");
      if (c->ndecls == G5P_MAX_DECLS)
         refuse(c, "more than %u NIR registers", G5P_MAX_DECLS);
      c->decl_first[c->ndecls] = INT_MAX;
      c->decl_out[c->ndecls] = false;
      c->decls[c->ndecls++] = &in->def;
   }
   /* vertex: a register whose value (a load_reg) an output stores, also
    * through another such register, lives in bank 1 */
   for (bool progress = c->vs; progress;) {
      progress = false;
      for (unsigned i = 0; i < c->ndecls; i++) {
         if (c->decl_out[i])
            continue;
         nir_foreach_use(u, c->decls[i]) {
            nir_instr *ui = nir_src_use_instr(u);
            if (ui->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(ui)->intrinsic == nir_intrinsic_load_reg &&
                vs_feeds_output(c, &nir_instr_as_intrinsic(ui)->def)) {
               c->decl_out[i] = progress = true;
               break;
            }
         }
      }
   }
}

/* ---- booleans (round 19) ----
 * A compare whose every use is a select condition, b2f32/b2i32 or a
 * discard condition is "fused": each consumer evaluates it with the strict
 * compare/write + inverse-MOV recipe. Any other boolean is materialized
 * as f32 0.0/1.0 in a register at its own position. With control flow an
 * if condition is a consumer too: the compare sets the stack's relation. */
static bool fusable_use(struct gctx *c, nir_alu_instr *cmp, nir_src *u)
{
   if (nir_src_is_if(u))
      return c->cf && cmp->src[0].src.ssa->bit_size == 32;
   nir_instr *i = nir_src_use_instr(u);
   if (i->type == nir_instr_type_alu) {
      nir_alu_instr *a = nir_instr_as_alu(i);
      return (a->op == nir_op_bcsel && u == &a->src[0].src) || a->op == nir_op_b2f32 ||
             a->op == nir_op_b2i32;
   }
   if (i->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(i);
      return in->intrinsic == nir_intrinsic_demote_if || in->intrinsic == nir_intrinsic_terminate_if;
   }
   return false;
}

static void plan_bools(struct gctx *c)
{
   foreach_instr_all(ins, c) {
      if (ins->type != nir_instr_type_alu || !is_cmp(nir_instr_as_alu(ins)->op))
         continue;
      nir_alu_instr *a = nir_instr_as_alu(ins);
      bool all = true;
      nir_foreach_use_including_if(u, &a->def)
         all &= fusable_use(c, a, u);
      c->fused[a->def.index] = all;
   }
}

static nir_alu_instr *fused_cmp(struct gctx *c, nir_def *d, unsigned comp)
{
   struct ref r = resolve_plain(d, comp);
   nir_alu_instr *a = alu_of(r.def);
   return a && is_cmp(a->op) && c->fused[a->def.index] ? a : NULL;
}

/* ---- varyings (varyings.md) ---- */
/* The barycentric source of an interpolated load: SMOOTH or NOPERSPECTIVE,
 * centre or centroid; anything else (sample, offset) has no witness. */
static bool varying_mode(nir_intrinsic_instr *in, uint8_t *interp, bool *centroid)
{
   if (in->intrinsic == nir_intrinsic_load_input) {
      *interp = PCO_G5P_INTERP_FLAT;
      *centroid = false;
      return true;
   }
   nir_intrinsic_instr *b = intrinsic_of(in->src[0].ssa);
   if (!b || (b->intrinsic != nir_intrinsic_load_barycentric_pixel &&
              b->intrinsic != nir_intrinsic_load_barycentric_centroid))
      return false;
   enum glsl_interp_mode m = nir_intrinsic_interp_mode(b);
   if (m != INTERP_MODE_NONE && m != INTERP_MODE_SMOOTH && m != INTERP_MODE_NOPERSPECTIVE)
      return false;
   *interp = m == INTERP_MODE_NOPERSPECTIVE ? PCO_G5P_INTERP_NOPERSPECTIVE : PCO_G5P_INTERP_SMOOTH;
   *centroid = b->intrinsic == nir_intrinsic_load_barycentric_centroid;
   return true;
}

/* A smooth pixel-centre varying: the only kind the coordinate and narrow
 * iterators are witnessed with. */
static bool smooth_centre(nir_intrinsic_instr *in)
{
   uint8_t interp;
   bool centroid;
   return in && in->intrinsic == nir_intrinsic_load_interpolated_input && varying_mode(in, &interp, &centroid) &&
          interp == PCO_G5P_INTERP_SMOOTH && !centroid;
}

/* First output dword of a vertex output location: 0 for the position, -1
 * when the location is not written (its fragment program does not read it),
 * -2 for a location without an output slot. */
static int vs_base(struct gctx *c, unsigned loc)
{
   if (loc == VARYING_SLOT_POS)
      return 0;
   if (loc == VARYING_SLOT_PSIZ && c->link && c->link->point_size)
      return c->psize_dword;
   if (loc < VARYING_SLOT_VAR0 || loc > VARYING_SLOT_VAR31)
      return -2;
   unsigned d = c->p->vout_dword[loc - VARYING_SLOT_VAR0];
   return d ? (int)d : -1;
}

/* Vertex output layout. Unlinked: VARn at 4 + 4n, smooth (the round-17..19
 * convention). Linked, as Apple's compiler and Rogue (pvr_alloc_vs_varyings)
 * lay it out: the position, then the read locations grouped by
 * interpolation (linear, flat, npc; STATE_VARYING0's three counts), each in
 * location order; locations the fragment program does not read are dropped. */
static void plan_vs_layout(struct gctx *c)
{
   uint32_t written = 0;
   nir_intrinsic_instr *last = NULL;
   unsigned last_comp = 0;
   foreach_instr_all(ins, c) {
      if (ins->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(ins);
      if (in->intrinsic != nir_intrinsic_store_output)
         continue;
      unsigned loc = nir_intrinsic_io_semantics(in).location;
      if (loc >= VARYING_SLOT_VAR0 && loc <= VARYING_SLOT_VAR31)
         written |= 1u << (loc - VARYING_SLOT_VAR0);
   }
   memset(c->p->vout_dword, 0, sizeof(c->p->vout_dword));
   memset(c->p->vout_interp, 0, sizeof(c->p->vout_interp));
   const unsigned first = 4;
   if (!c->link || !c->link->valid) {
      unsigned top = 0;
      u_foreach_bit(n, written) {
         c->p->vout_dword[n] = first + 4 * n;
         c->p->vout_interp[n] = PCO_G5P_INTERP_SMOOTH;
         top = n + 1;
      }
      c->p->varyings = top;
      c->p->outputs = first + 4 * top;
   } else {
      static const uint8_t order[] = {PCO_G5P_INTERP_SMOOTH, PCO_G5P_INTERP_FLAT, PCO_G5P_INTERP_NOPERSPECTIVE};
      unsigned next = first, kept = 0;
      for (unsigned m = 0; m < ARRAY_SIZE(order); m++)
         u_foreach_bit(n, written) {
            if (c->link->interp[n] != order[m])
               continue;
            c->p->vout_dword[n] = next;
            c->p->vout_interp[n] = order[m];
            next += 4;
            kept++;
         }
      c->p->varyings = kept;
      c->p->outputs = next;
   }
   /* a point pipeline: the point size is the last output dword (the TA reads
    * it there and leaves it out of the varyings; Apple's g3b-points has only
    * the position before it, and a varying in front of it is the device's
    * reading: with the size after the position, the TA took the colour's
    * alpha for the size) */
   if (c->link && c->link->point_size)
      c->psize_dword = c->p->outputs++;
   /* The A9 TA vertex holds 128 dwords, the position included
    * (varyings-capacity-1: 124 varying components exact, 128 wrong as a
    * whole with no GPU event); refuse past it by name, never draw it. */
   if (c->p->outputs > 128)
      refuse(c, "%u output dwords (the A9 TA vertex holds 128)", c->p->outputs);
   if (!c->eager_vout)
      return;
   /* Eager outputs: the program-order last written component ends the task. */
   foreach_instr_all(ins, c) {
      if (ins->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(ins);
      if (in->intrinsic != nir_intrinsic_store_output || vs_base(c, nir_intrinsic_io_semantics(in).location) < 0)
         continue;
      last = in;
      last_comp = util_last_bit(nir_intrinsic_write_mask(in)) - 1;
   }
   c->vend_store = last;
   c->vend_comp = last_comp;
}

/* Fragment varyings: every location read, in location order, four
 * coefficient sets each after W (set 0, only when some varying is
 * perspective-correct): Apple's layout (f_s*, f_n4, f_f4, f_m4). */
static void plan_varyings(struct gctx *c)
{
   uint8_t interp[PCO_G5P_VARYING_SLOTS] = {0}, mask[PCO_G5P_VARYING_SLOTS] = {0};
   bool centroid[PCO_G5P_VARYING_SLOTS] = {0};
   uint32_t read = 0;
   memset(c->vary_idx, -1, sizeof(c->vary_idx));
   foreach_instr_all(ins, c) {
      if (ins->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(ins);
      if (in->intrinsic != nir_intrinsic_load_interpolated_input && in->intrinsic != nir_intrinsic_load_input)
         continue;
      unsigned loc = nir_intrinsic_io_semantics(in).location;
      if (loc < VARYING_SLOT_VAR0 || loc > VARYING_SLOT_VAR31)
         refuse(c, "varying location %u is not a generic VARn slot", loc);
      unsigned src = in->intrinsic == nir_intrinsic_load_input ? 0 : 1;
      if (!nir_src_is_const(in->src[src]) || nir_src_as_uint(in->src[src]))
         refuse(c, "indirect varying");
      if (in->def.bit_size != 32)
         refuse(c, "%u-bit varying", in->def.bit_size);
      uint8_t m;
      bool cen;
      if (!varying_mode(in, &m, &cen))
         refuse(c, "varying interpolation other than smooth/flat/noperspective at the centre or centroid");
      unsigned n = loc - VARYING_SLOT_VAR0;
      if ((read & (1u << n)) && (interp[n] != m || centroid[n] != cen))
         refuse(c, "VAR%u is read with two interpolations", n);
      read |= 1u << n;
      interp[n] = m;
      centroid[n] = cen;
      mask[n] |= BITFIELD64_MASK(in->def.num_components) << nir_intrinsic_component(in);
   }
   if (util_bitcount(read) > PCO_G5P_MAX_FS_VARYINGS)
      refuse(c, "%u varying locations read (%u vec4: the reported 64 components)", util_bitcount(read),
             PCO_G5P_MAX_FS_VARYINGS);
   bool w = false;
   u_foreach_bit(n, read)
      w |= interp[n] == PCO_G5P_INTERP_SMOOTH;
   /* the last location's last component must fit the u7 iterator slot */
   if ((w ? 1 : 0) + 4 * util_bitcount(read) > 128)
      refuse(c, "%u varying locations with W exceed the u7 iterator slot", util_bitcount(read));
   unsigned i = 0;
   u_foreach_bit(n, read) {
      c->p->vary[i] = (struct pco_g5p_varying){.location = n, .set = (w ? 1 : 0) + 4 * i, .interp = interp[n],
                                               .centroid = centroid[n], .mask = mask[n]};
      c->vary_idx[n] = i++;
   }
   c->p->num_varyings = i;
   c->p->w_set = w;
}

static const struct pco_g5p_varying *vary_of(struct gctx *c, nir_intrinsic_instr *in)
{
   int i = c->vary_idx[nir_intrinsic_io_semantics(in).location - VARYING_SLOT_VAR0];
   assert(i >= 0);
   return &c->p->vary[i];
}

/* ---- liveness ---- */
/* Where a value gets its register: the varyings, gl_FragCoord and vertex
 * attributes at the program's start (hoist_iterators(), setup_inputs()),
 * whatever block NIR keeps their intrinsics in; everything else at its
 * instruction. */
static int def_ip(nir_def *d)
{
   nir_intrinsic_instr *in = intrinsic_of(d);
   if (in && (in->intrinsic == nir_intrinsic_load_frag_coord ||
              in->intrinsic == nir_intrinsic_load_interpolated_input || in->intrinsic == nir_intrinsic_load_input))
      return -1;
   return nir_def_instr(d)->index;
}

/* A value read in a loop it is defined outside of stays live to the loop's
 * end: the next iteration reads it again (G9P's use_scalar()). Defined
 * outside means where it gets its register (def_ip()): gl_FragCoord read in a
 * loop is NIR's instruction inside the loop, but its register is written
 * once, at the start (device run cf4: its register went to the loop counter
 * after the first iteration's read). */
static void mark(struct gctx *c, struct ref r, int idx)
{
   struct val *v = V(c, r.def, r.comp);
   int end = idx;
   for (unsigned l = 0; l < c->nloops; l++)
      if (c->loops[l].start <= idx && idx <= c->loops[l].end && def_ip(r.def) < c->loops[l].start)
         end = MAX2(end, c->loops[l].end);
   v->last = MAX2(v->last, end);
   v->uses++;
}

/* A NIR register access at idx: held from its first access (from the start
 * of a loop that accesses it) to its last (to that loop's end). */
static void mark_decl(struct gctx *c, nir_def *decl, int idx)
{
   int s = decl_slot(c, decl), first = idx;
   if (s < 0)
      refuse(c, "NIR register without a declaration");
   for (unsigned l = 0; l < c->nloops; l++)
      if (c->loops[l].start <= idx && idx <= c->loops[l].end)
         first = MIN2(first, c->loops[l].start);
   c->decl_first[s] = MIN2(c->decl_first[s], first);
   mark(c, (struct ref){decl, 0, false, false}, idx);
}

static void mark_bool(struct gctx *c, nir_def *d, unsigned comp, int idx)
{
   nir_alu_instr *cmp = fused_cmp(c, d, comp);
   if (cmp) {
      for (unsigned i = 0; i < 2; i++)
         mark(c, resolve_value(c, cmp->src[i].src.ssa, cmp->src[i].swizzle[0]), idx);
   } else {
      mark(c, resolve_value(c, d, comp), idx);
   }
}

/* The arm of an if is one block ending in a break: 1 if the break is all of
 * it, 2 if work precedes it; *side: 0 then, 1 else (G9P's emit_if()). */
static nir_block *single_block(struct exec_list *list)
{
   if (!exec_list_is_singular(list))
      return NULL;
   return nir_cf_node_as_block(exec_node_data(nir_cf_node, exec_list_get_head(list), node));
}

static unsigned break_shape(nir_if *nif, unsigned *side)
{
   for (unsigned s = 0; s < 2; s++) {
      nir_block *b = single_block(s ? &nif->else_list : &nif->then_list);
      nir_instr *last = b ? nir_block_last_instr(b) : NULL;
      if (!last || last->type != nir_instr_type_jump || nir_instr_as_jump(last)->type != nir_jump_break)
         continue;
      *side = s;
      return nir_block_first_instr(b) == last ? 1 : 2;
   }
   return 0;
}

/* If conditions are read before the push; one whose arm does work and then
 * breaks is compared again after the arms. */
static void liveness_cf(struct gctx *c, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         unsigned side;
         mark_bool(c, nif->condition.ssa, 0, cf_ip_before(node));
         if (break_shape(nif, &side) == 2)
            mark_bool(c, nif->condition.ssa, 0, cf_ip_after(node));
         liveness_cf(c, &nif->then_list);
         liveness_cf(c, &nif->else_list);
      } else if (node->type == nir_cf_node_loop) {
         liveness_cf(c, &nir_cf_node_as_loop(node)->body);
      }
   }
}

static void liveness(struct gctx *c)
{
   foreach_instr_all(ins, c) {
      int idx = ins->index;
      if (ins->type == nir_instr_type_alu) {
         nir_alu_instr *a = nir_instr_as_alu(ins);
         if (alias_op(c, a) || a->op == nir_op_fneg || a->op == nir_op_fabs || dyn_offset_alu(a))
            continue;
         if (c->vs && address_only(a))
            continue; /* read by the store's address add */
         if (is_cmp(a->op) && c->fused[a->def.index])
            continue; /* evaluated at its consumers */
         if (a->op == nir_op_fsat) { /* standalone: fmul.sat(x, 1.0) */
            mark(c, resolve_float(c, a->src[0].src.ssa, a->src[0].swizzle[0]), idx);
            continue;
         }
         if (a->op == nir_op_bcsel) {
            mark_bool(c, a->src[0].src.ssa, a->src[0].swizzle[0], idx);
            for (unsigned i = 1; i < 3; i++)
               mark(c, alu_src(c, a, i), idx);
            continue;
         }
         if (a->op == nir_op_b2f32 || a->op == nir_op_b2i32) {
            mark_bool(c, a->src[0].src.ssa, a->src[0].swizzle[0], idx);
            continue;
         }
         for (unsigned i = 0; i < nir_op_infos[a->op].num_inputs; i++)
            mark(c, alu_src(c, a, i), idx);
      } else if (ins->type == nir_instr_type_tex && !plain_tex(nir_instr_as_tex(ins))) {
         /* textypes: every component the general form reads goes through a register */
         nir_tex_instr *t = nir_instr_as_tex(ins);
         if (size_query(t))
            continue;
         for (unsigned i = 0; i < t->num_srcs; i++) {
            nir_tex_src_type ty = t->src[i].src_type;
            unsigned n = ty == nir_tex_src_backend1 && t->sampler_dim == GLSL_SAMPLER_DIM_BUF ? 2 :
                         ty == nir_tex_src_coord ? t->coord_components :
                         ty == nir_tex_src_ddx || ty == nir_tex_src_ddy ?
                            (t->sampler_dim == GLSL_SAMPLER_DIM_3D ||
                             t->sampler_dim == GLSL_SAMPLER_DIM_CUBE ? 3 : 2) :
                         ty == nir_tex_src_bias || ty == nir_tex_src_lod || ty == nir_tex_src_comparator ? 1 : 0;
            for (unsigned k = 0; k < n; k++)
               mark(c, resolve_value(c, t->src[i].src.ssa, k), idx);
         }
      } else if (ins->type == nir_instr_type_tex) {
         nir_tex_instr *t = nir_instr_as_tex(ins);
         nir_def *cd = nir_get_tex_src(t, nir_tex_src_coord);
         struct ref x = resolve_plain(cd, 0), y = resolve_plain(cd, 1);
         nir_intrinsic_instr *vin = intrinsic_of(x.def);
         if (!smooth_centre(vin) || x.def != y.def || y.comp != x.comp + 1) {
            mark(c, resolve_value(c, cd, 0), idx);
            mark(c, resolve_value(c, cd, 1), idx);
         }
      } else if (ins->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(ins);
         if (in->intrinsic == nir_intrinsic_demote_if || in->intrinsic == nir_intrinsic_terminate_if) {
            /* Discards are emitted after the body (no derivative work follows). */
            mark_bool(c, in->src[0].ssa, 0, DISCARD_AT);
            continue;
         }
         if (dyn_ubo_load(in)) {
            struct ref ix;
            unsigned shift;
            uint32_t base;
            if (!dyn_ubo_offset(in, &ix, &shift, &base))
               refuse(c, "load_ubo: register offset is not (index << s) + constant");
            mark(c, ix, idx);
            continue;
         }
         if (in->intrinsic == nir_intrinsic_store_global) {
            unsigned b, k;
            struct ref off;
            xfb_address(in->src[1].ssa, &b, &off, &k);
            /* The offset is read once, by the record's address add. */
            bool first;
            store_base(c, b, off, &first);
            if (first)
               mark(c, off, idx);
            mark(c, resolve_value(c, in->src[0].ssa, 0), idx);
            continue;
         }
         if (in->intrinsic == nir_intrinsic_store_ssbo) {
            mark(c,resolve_value(c,in->src[0].ssa,0),idx);
            mark(c,resolve_value(c,in->src[2].ssa,0),idx);
            continue;
         }
         if (in->intrinsic == nir_intrinsic_store_reg) {
            mark(c, resolve_value(c, in->src[0].ssa, 0), idx);
            mark_decl(c, in->src[1].ssa, idx);
            continue;
         }
         if (in->intrinsic == nir_intrinsic_load_reg) {
            mark_decl(c, in->src[0].ssa, idx);
            continue;
         }
         if (is_deriv(in->intrinsic)) {
            mark(c, resolve_value(c, in->src[0].ssa, 0), idx);
            continue;
         }
         if (in->intrinsic != nir_intrinsic_store_output)
            continue;
         if (in->instr.block->cf_node.parent->type != nir_cf_node_function)
            refuse(c, "store_output inside control flow");
         unsigned loc = nir_intrinsic_io_semantics(in).location;
         unsigned comp = nir_intrinsic_component(in);
         u_foreach_bit(k, nir_intrinsic_write_mask(in)) {
            if (c->vs) {
               int base = vs_base(c, loc);
               if (base == -1)
                  continue; /* not read by the fragment program: not written */
               bool now = c->eager_vout && base >= 0 && !(in == c->vend_store && k == c->vend_comp);
               struct ref r = resolve_value(c, in->src[0].ssa, k);
               mark(c, r, now ? (int)nir_def_instr(r.def)->index : LAST_END);
            } else if (loc == FRAG_RESULT_DEPTH) {
               if (!c->key->depth_feedback)
                  refuse(c, "fragment depth export awaits A9 qualification");
               if (c->nir->info.fs.early_fragment_tests)
                  refuse(c, "fragment depth export with early fragment tests is not qualified");
               if (comp + k || in->src[0].ssa->bit_size != 32)
                  refuse(c, "fragment depth export must be one f32 scalar");
               c->depth = resolve_float(c, in->src[0].ssa, k);
               if (c->depth.neg || c->depth.abs)
                  refuse(c, "fragment depth export source modifiers are not qualified");
               c->has_depth = true;
               mark(c, c->depth, LAST_END);
            } else if (loc < FRAG_RESULT_DATA0 || !g5p_mrt_slot_is_attached(loc - FRAG_RESULT_DATA0, c->mrt.targets) ||
                       comp + k >= 4) {
               refuse(c, "store_output to %s", gl_frag_result_name(loc));
            } else {
               unsigned slot = loc - FRAG_RESULT_DATA0, half = g5p_mrt_half(slot, comp + k);
               struct ref r = c->key->color_output != PCO_G5P_COLOR_F16
                                 ? resolve_value(c, in->src[0].ssa, k)
                                 : resolve_float(c, in->src[0].ssa, k);
               c->out[half] = r;
               c->out_written[half] = true;
               c->slot_written[slot] |= 1u << (comp + k);
               mark(c, r, LAST_END);
            }
         }
      }
   }
   /* Standalone fneg/fabs feed plain consumers; they read their operand. */
   foreach_instr_all(ins, c) {
      if (ins->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *a = nir_instr_as_alu(ins);
      if ((a->op == nir_op_fneg || a->op == nir_op_fabs) && V(c, &a->def, 0)->uses)
         mark(c, resolve_float(c, &a->def, 0), ins->index);
   }
   if (c->cf)
      liveness_cf(c, &c->impl->body);
}

/* ---- common-store plan (fragment) ---- */
static bool block_binding_element(nir_def *d, unsigned *set, unsigned *binding,
                                  unsigned *element)
{
   for (unsigned guard = 0; guard < 8; guard++) {
      struct ref r = resolve_plain(d, 0);
      nir_intrinsic_instr *in = intrinsic_of(r.def);
      if (!in)
         return false;
      if (in->intrinsic == nir_intrinsic_load_vulkan_descriptor) {
         d = in->src[0].ssa;
         continue;
      }
      if (in->intrinsic != nir_intrinsic_vulkan_resource_index || !nir_src_is_const(in->src[0]))
         return false;
      *set = nir_intrinsic_desc_set(in);
      *binding = nir_intrinsic_binding(in);
      *element = nir_src_as_uint(in->src[0]);
      return true;
   }
   return false;
}

static bool block_binding(nir_def *d, unsigned *set, unsigned *binding)
{
   unsigned element;
   return block_binding_element(d, set, binding, &element) && element == 0;
}

static struct range *range_for(struct gctx *c, enum pco_g5p_upload_kind kind, unsigned set,
                               unsigned binding, unsigned element)
{
   for (unsigned i = 0; i < c->nranges; i++)
      if (c->ranges[i].kind == kind && c->ranges[i].set == set && c->ranges[i].binding == binding &&
          c->ranges[i].element == element)
         return &c->ranges[i];
   if (c->nranges == ARRAY_SIZE(c->ranges))
      refuse(c, "more than %zu uniform blocks", ARRAY_SIZE(c->ranges));
   c->ranges[c->nranges] = (struct range){kind, set, binding, element, UINT_MAX, 0, 0};
   return &c->ranges[c->nranges++];
}

/* Constant byte offsets; the descriptor's offset channel is 0 (each
 * binding is uploaded from its own start). */
static bool eval_offset(nir_def *d, unsigned comp, uint32_t *v)
{
   struct ref r = resolve_plain(d, comp);
   nir_instr *i = nir_def_instr(r.def);
   if (i->type == nir_instr_type_load_const) {
      *v = nir_instr_as_load_const(i)->value[r.comp].u32;
      return true;
   }
   nir_intrinsic_instr *in = intrinsic_of(r.def);
   if (in && in->intrinsic == nir_intrinsic_load_vulkan_descriptor && r.comp == 1) {
      *v = 0;
      return true;
   }
   nir_alu_instr *a = alu_of(r.def);
   uint32_t x, y;
   if (a && a->op == nir_op_iadd && eval_offset(a->src[0].src.ssa, a->src[0].swizzle[0], &x) &&
       eval_offset(a->src[1].src.ssa, a->src[1].swizzle[0], &y)) {
      *v = x + y;
      return true;
   }
   return false;
}

static unsigned load_offset(struct gctx *c, nir_intrinsic_instr *in, unsigned src, unsigned base)
{
   uint32_t off;
   if (!eval_offset(in->src[src].ssa, 0, &off))
      refuse(c, "%s with a dynamic offset (pointer arithmetic is unresolved)",
             nir_intrinsic_infos[in->intrinsic].name);
   off += base;
   if (off % 4 || in->def.bit_size != 32)
      refuse(c, "%s: only aligned 32-bit components", nir_intrinsic_infos[in->intrinsic].name);
   return off;
}

/* ---- register-indexed UBO loads (gles3-tex.md §3) ----
 * vkcube's `ubuf.attr[gl_VertexIndex]` is load_ubo(offset = (index << 4) +
 * C). Apple's compiler builds a pointer pair = buffer pointer + (index <<
 * shift) and loads from it (dynubo.metal v_cexpr/v_cbase), as the A10 port
 * compiler does for every buffer load. Here the pointer is the UBO's VA + C
 * (a loader upload), so the offset arithmetic folds away entirely. */
static bool dyn_ubo_load(nir_intrinsic_instr *in)
{
   uint32_t v;
   return in->intrinsic == nir_intrinsic_load_ubo && !eval_offset(in->src[1].ssa, 0, &v);
}

/* offset = [ishl(idx, shift) | imul(idx, 1 << shift) | idx] + C (C constant) */
static bool dyn_ubo_offset(nir_intrinsic_instr *in, struct ref *idx, unsigned *shift, uint32_t *base)
{
   struct ref r = resolve_plain(in->src[1].ssa, 0);
   uint32_t x;
   *base = 0;
   *shift = 0;
   for (unsigned guard = 0; guard < 8; guard++) {
      nir_alu_instr *a = alu_of(r.def);
      if (!a || a->op != nir_op_iadd)
         break;
      unsigned s = eval_offset(a->src[0].src.ssa, a->src[0].swizzle[r.comp], &x) ? 0 :
                   eval_offset(a->src[1].src.ssa, a->src[1].swizzle[r.comp], &x) ? 1 : 2;
      if (s == 2)
         return false; /* two dynamic terms */
      *base += x;
      r = resolve_plain(a->src[1 - s].src.ssa, a->src[1 - s].swizzle[r.comp]);
   }
   nir_alu_instr *a = alu_of(r.def);
   if (a && (a->op == nir_op_ishl || a->op == nir_op_imul) &&
       eval_offset(a->src[1].src.ssa, a->src[1].swizzle[r.comp], &x)) {
      if (a->op == nir_op_imul) {
         if (!util_is_power_of_two_nonzero(x))
            return false;
         x = util_logbase2(x);
      }
      *shift = x;
      r = resolve_plain(a->src[0].src.ssa, a->src[0].swizzle[r.comp]);
   }
   *idx = r;
   return true;
}

/* iadd/ishl/imul whose every use is (the offset of) a register-indexed UBO
 * load: folded into the addressing, never emitted. */
static bool dyn_offset_alu(nir_alu_instr *a)
{
   if (a->op != nir_op_iadd && a->op != nir_op_ishl && a->op != nir_op_imul)
      return false;
   bool any = false;
   nir_foreach_use(u, &a->def) {
      nir_instr *i = nir_src_use_instr(u);
      if (i->type == nir_instr_type_alu && dyn_offset_alu(nir_instr_as_alu(i))) {
         any = true;
         continue;
      }
      if (i->type != nir_instr_type_intrinsic)
         return false;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(i);
      if (!dyn_ubo_load(in) || u != &in->src[1])
         return false;
      any = true;
   }
   return any && !nir_def_used_by_if(&a->def);
}

/* A texture or sampler deref: its variable and the array element it reads.
 * A variable, or one array level of a variable at a constant index (GLSL ES
 * 1.00 indexes sampler arrays with constant-index expressions, constants once
 * loops are unrolled); the element selects the descriptor the driver
 * uploads. A non-constant index (dynamically uniform indexing, ES 3.x /
 * GL 4) and bindless handles are refused by name: a dynamic index would
 * extend this with a per-draw descriptor table indexed in the program. */
static nir_variable *tex_deref(struct gctx *c, nir_deref_instr *d, unsigned *element)
{
   *element = 0;
   if (d && d->deref_type == nir_deref_type_array) {
      nir_deref_instr *parent = nir_deref_instr_parent(d);
      if (!parent || parent->deref_type != nir_deref_type_var)
         refuse(c, "tex: arrays of arrays of textures/samplers");
      if (!nir_src_is_const(d->arr.index))
         refuse(c, "tex: a texture/sampler array read at a non-constant index (dynamically uniform indexing)");
      *element = nir_src_as_uint(d->arr.index);
      return nir_deref_instr_get_variable(d);
   }
   if (!d || d->deref_type != nir_deref_type_var)
      refuse(c, "tex: bindless texture/sampler handles");
   return nir_deref_instr_get_variable(d);
}

static void tex_bindings(struct gctx *c, nir_tex_instr *t, struct tex_bind *b)
{
   nir_deref_instr *td = nir_get_tex_deref(t, nir_tex_src_texture_deref);
   nir_deref_instr *sd = nir_get_tex_deref(t, nir_tex_src_sampler_deref);
   unsigned element, selement;
   nir_variable *tv = tex_deref(c, td, &element);
   nir_variable *sv = sd ? tex_deref(c, sd, &selement) : tv;
   if (!sd)
      selement = element;
   *b = (struct tex_bind){tv->data.descriptor_set, tv->data.binding, sv->data.descriptor_set,
                          sv->data.binding, element, selement};
   b->needs_sampler = t->op != nir_texop_txf;
   b->needs_fetch_sampler = t->op == nir_texop_txf;
}

static struct tex_bind *tex_for(struct gctx *c, nir_tex_instr *t)
{
   struct tex_bind b;
   tex_bindings(c, t, &b);
   for (unsigned i = 0; i < c->ntex; i++)
      if (c->tex[i].set == b.set && c->tex[i].binding == b.binding && c->tex[i].sset == b.sset &&
          c->tex[i].sbinding == b.sbinding && c->tex[i].element == b.element && c->tex[i].selement == b.selement)
         { c->tex[i].needs_sampler |= b.needs_sampler;
           c->tex[i].needs_fetch_sampler |= b.needs_fetch_sampler;
           return &c->tex[i]; }
   if (c->ntex == (c->vs ? 4 : ARRAY_SIZE(c->tex)))
      refuse(c, "more than %u textures", c->vs ? 4u : (unsigned)ARRAY_SIZE(c->tex));
   c->tex[c->ntex] = b;
   return &c->tex[c->ntex++];
}

/* textypes: the size block of a texture binding at a constant level. */
static unsigned size_level(nir_tex_instr *t)
{
   int l = nir_tex_instr_src_index(t, nir_tex_src_lod);
   return t->op == nir_texop_txs && l >= 0 ? nir_src_as_uint(t->src[l].src) : 0;
}

static unsigned size_for(struct gctx *c, nir_tex_instr *t)
{
   unsigned element;
   nir_variable *tv = tex_deref(c, nir_get_tex_deref(t, nir_tex_src_texture_deref), &element);
   unsigned set = tv->data.descriptor_set, binding = tv->data.binding, lod = size_level(t);
   for (unsigned i = 0; i < c->nsize; i++)
      if (c->tsize[i].set == set && c->tsize[i].binding == binding && c->tsize[i].element == element &&
          c->tsize[i].lod == lod)
         return i;
   if (c->nsize == ARRAY_SIZE(c->tsize))
      refuse(c, "more than %zu texture size queries", ARRAY_SIZE(c->tsize));
   c->tsize[c->nsize].set = set;
   c->tsize[c->nsize].binding = binding;
   c->tsize[c->nsize].element = element;
   c->tsize[c->nsize].lod = lod;
   return c->nsize++;
}

static void add_upload(struct gctx *c, enum pco_g5p_upload_kind kind, unsigned set, unsigned binding,
                       unsigned element, unsigned offset, unsigned dwords, unsigned common)
{
   struct pco_g5p_program *p = c->p;
   if (p->num_uploads == PCO_G5P_MAX_UPLOADS)
      refuse(c, "more than %u common-store uploads", PCO_G5P_MAX_UPLOADS);
   p->uploads[p->num_uploads++] =
      (struct pco_g5p_upload){kind, set, binding, element, offset, dwords, common};
}

static void plan_common(struct gctx *c)
{
   bool xfb = false;
   c->xfb_common = -1;
   foreach_instr_all(ins, c) {
      if (ins->type == nir_instr_type_tex) {
         nir_tex_instr *t = nir_instr_as_tex(ins);
         if (size_query(t))
            size_for(c, t);
         else
            tex_for(c, t);
         continue;
      }
      if (ins->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(ins);
      unsigned set = 0, binding = 0, element = 0;
      if (in->intrinsic == nir_intrinsic_load_xfb_address || in->intrinsic == nir_intrinsic_load_num_vertices)
         xfb = true;
      if (in->intrinsic == nir_intrinsic_store_global) {
         unsigned b, k;
         struct ref off;
         if (xfb_address(in->src[1].ssa, &b, &off, &k))
            c->p->xfb_buffers |= 1u << b;
      }
      if (in->intrinsic == nir_intrinsic_load_push_constant) {
         unsigned off = load_offset(c, in, 0, nir_intrinsic_base(in));
         struct range *r = range_for(c, PCO_G5P_UPLOAD_PUSH, 0, 0, 0);
         r->lo = MIN2(r->lo, off);
         r->hi = MAX2(r->hi, off + 4 * in->def.num_components);
      } else if (dyn_ubo_load(in)) {
         if (!block_binding_element(in->src[0].ssa, &set, &binding, &element))
            refuse(c, "load_ubo: block index is not a constant descriptor binding");
         struct ref ix;
         unsigned shift, i;
         uint32_t base;
         dyn_ubo_offset(in, &ix, &shift, &base);
         for (i = 0; i < c->nuptr; i++)
            if (c->uptr[i].set == set && c->uptr[i].binding == binding &&
                c->uptr[i].element == element && c->uptr[i].base == base)
               break;
         if (i == c->nuptr) {
            if (c->nuptr == ARRAY_SIZE(c->uptr))
               refuse(c, "more than %zu register-indexed UBO arrays", ARRAY_SIZE(c->uptr));
            c->uptr[c->nuptr].set = set;
            c->uptr[c->nuptr].binding = binding;
            c->uptr[c->nuptr].element = element;
            c->uptr[c->nuptr++].base = base;
         }
      } else if (in->intrinsic == nir_intrinsic_load_ubo) {
         if (!block_binding_element(in->src[0].ssa, &set, &binding, &element))
            refuse(c, "load_ubo: block index is not a constant descriptor binding");
         unsigned off = load_offset(c, in, 1, 0);
         struct range *r = range_for(c, PCO_G5P_UPLOAD_UBO, set, binding, element);
         r->lo = MIN2(r->lo, off);
         r->hi = MAX2(r->hi, off + 4 * in->def.num_components);
      } else if (in->intrinsic == nir_intrinsic_load_ssbo || in->intrinsic == nir_intrinsic_store_ssbo) {
         unsigned source=in->intrinsic==nir_intrinsic_store_ssbo ? 1 : 0;
         if (!block_binding(in->src[source].ssa, &set, &binding))
            refuse(c, "load_ssbo: block index is not a constant descriptor binding");
         bool found = false;
         for (unsigned i = 0; i < c->nssbo; i++)
            found |= c->ssbo_set[i] == set && c->ssbo_binding[i] == binding;
         if (!found) {
            if (c->nssbo == 2)
               refuse(c, "load_ssbo: only pointer bases common0/common2 are witnessed (two buffers)");
            c->ssbo_set[c->nssbo] = set;
            c->ssbo_binding[c->nssbo++] = binding;
         }
      }
   }
   unsigned next = 0;
   if (!c->vs && c->blend == BLEND_GENERAL && c->binfo.uses_constants) {
      /* Apple's layout: the blend colour in common 0..3, everything else after it */
      if (c->nssbo)
         refuse(c, "blend constants with storage buffers (both want common dwords 0..3)");
      add_upload(c, PCO_G5P_UPLOAD_BLEND_CONSTANTS, 0, 0, 0, 0, 4, G5P_BLEND_CONST_BASE);
      next = 4;
   }
   for (unsigned i = 0; i < c->nssbo; i++)
      add_upload(c, PCO_G5P_UPLOAD_SSBO_ADDR, c->ssbo_set[i], c->ssbo_binding[i], 0, 0, 2, 2 * i);
   if (c->nssbo)
      next = MAX2(next, 4);
   bool qualified = c->vs && g5p_vtex_qualified();
   /* Preserve the flown <=4-texture layout. Larger fragment programs use
    * Apple's f16 capacity layout: all eight-dword images, then samplers.
    * Interleaving wastes six dwords per binding and exceeds the selectors. */
   bool compact = qualified || (!c->vs && c->ntex > 4);
   unsigned image_base = align(next, 8), sampler_base = image_base + 8 * c->ntex;
   unsigned sampler_count = 0, sampler_cursor = sampler_base;
   bool fetch_sampler = false;
   if (compact)
      for (unsigned i = 0; i < c->ntex; i++) {
         sampler_count += !qualified || c->tex[i].needs_sampler;
         fetch_sampler |= c->tex[i].needs_fetch_sampler;
      }
   c->fetch_sampler_common = sampler_base + 2 * sampler_count;
   for (unsigned i = 0; i < c->ntex; i++) {
      struct tex_bind *t = &c->tex[i];
      t->image = compact ? image_base + 8 * i : align(next, 8);
      t->sampler = compact ? ((!qualified || t->needs_sampler) ? sampler_cursor : c->fetch_sampler_common) : t->image + 8;
      sampler_cursor += compact && (!qualified || t->needs_sampler) ? 2 : 0;
      next = compact ? c->fetch_sampler_common + 2 * (qualified && fetch_sampler) : t->image + 10;
      if (t->image / 2 > (compact ? 120 : 28) || t->sampler / 2 > (compact ? 124 : 32))
         refuse(c, "texture descriptors beyond the bounded qualified selectors");
      add_upload(c, PCO_G5P_UPLOAD_IMAGE, t->set, t->binding, t->element, 0, 8, t->image);
      if (!qualified || t->needs_sampler)
         add_upload(c, PCO_G5P_UPLOAD_SAMPLER, t->sset, t->sbinding, t->selement, 32, 2, t->sampler);
   }
   if (qualified && fetch_sampler)
      add_upload(c, PCO_G5P_UPLOAD_FETCH_SAMPLER, 0, 0, 0, 0, 2, c->fetch_sampler_common);
   /* Pointer pairs of register-indexed UBO arrays: even dwords below 32 (the
    * addr.add_shift base field, 0x81 | 4 * dword). */
   for (unsigned i = 0; i < c->nuptr; i++) {
      c->uptr[i].common = next = align(next, 2);
      if (next + 2 > 32)
         refuse(c, "register-indexed UBO pointer beyond common dword 30");
      add_upload(c, PCO_G5P_UPLOAD_UBO_ADDR, c->uptr[i].set, c->uptr[i].binding,
                 c->uptr[i].element, c->uptr[i].base, 2, next);
      next += 2;
   }
   /* Shared moves are witnessed from common dword 2 upward. */
   next = MAX2(next, 2);
   for (unsigned i = 0; i < c->nsize; i++) {
      c->tsize[i].common = next;
      next += 4;
      add_upload(c, PCO_G5P_UPLOAD_IMAGE_SIZE, c->tsize[i].set, c->tsize[i].binding,
                 c->tsize[i].element, c->tsize[i].lod, 4, c->tsize[i].common);
   }
   unsigned limit = c->vs ? PCO_G5P_VS_COMMON_DWORDS : PCO_G5P_FS_COMMON_DWORDS;
   unsigned preloaded = next;
   for (unsigned i = 0; i < c->nranges; i++)
      preloaded += (c->ranges[i].hi - c->ranges[i].lo) / 4;

   /* Apple's f_u1024all stops preloading a block that would overflow the
    * 256-dword common-store operand field. It uploads the buffer address and
    * uses ordinary memory loads instead. Keep the small-program byte shape,
    * but make the same choice for all UBO ranges when their combined span no
    * longer fits. Push constants remain preloaded. */
   if (preloaded > limit) {
      for (unsigned i = 0; i < c->nranges; i++) {
         struct range *r = &c->ranges[i];
         if (r->kind != PCO_G5P_UPLOAD_UBO)
            continue;
         r->memory = true;
         r->common = next = align(next, 2);
         if (next + 2 > 32)
            refuse(c, "uniform-buffer address beyond common dword 30");
         add_upload(c, PCO_G5P_UPLOAD_UBO_ADDR, r->set, r->binding,
                    r->element, r->lo, 2, next);
         next += 2;
      }
   }
   for (unsigned i = 0; i < c->nranges; i++) {
      struct range *r = &c->ranges[i];
      if (r->memory)
         continue;
      r->common = next;
      next += (r->hi - r->lo) / 4;
      if (next > limit)
         refuse(c, "uniforms exceed %u common-store dwords", limit);
      add_upload(c, r->kind, r->set, r->binding, r->element, r->lo,
                 (r->hi - r->lo) / 4, r->common);
   }
   /* Transform feedback block: pointer pairs at even dwords (address add). */
   if (xfb) {
      c->xfb_common = align(next, 2);
      next = c->xfb_common + PCO_G5P_XFB_DWORDS;
      if (next > PCO_G5P_COMMON_DWORDS)
         refuse(c, "uniforms and the XFB block exceed %u common-store dwords", PCO_G5P_COMMON_DWORDS);
      add_upload(c, PCO_G5P_UPLOAD_XFB, 0, 0, 0, 0, PCO_G5P_XFB_DWORDS, c->xfb_common);
   }
   if (next > PCO_G5P_COMMON_DWORDS)
      c->p->evidence |= PCO_G5P_EV_COMMON_HIGH;
   c->common_next = c->p->num_uploads ? next : 0;
}

/* ---- registers above r31 ----
 * From the G9P compiler (OrchardLinux/mesa g9p_compile.c,
 * d1ab45daed3: park_def(), src_reg(), the victim rule in g9p_compile();
 * MIT): when a fragment program runs out of r0..r31, the value live at that
 * point with the furthest last read gets a home above r31 and the program is
 * generated again. A value with a home is moved there right after its
 * definition and its register freed; each read moves it back into a
 * register for that instruction only. The moves are his: an OR of the
 * value with itself, in the two-source form whose operands above r31 take
 * the extension word (g5p_usc_alu2_wide), as Apple's A9 compiler writes it
 * for r32..r69 (control-flow.md §2.3). Apple allocates r32 and up for
 * every form; this keeps every other encoder on r0..r31. */
#define G5P_OP_OR 0xce
#define G5P_MAX_HOMES 96

static void park(struct gctx *c, nir_instr *ins)
{
   nir_def *d = nir_instr_def(ins);
   if (!d || c->vs)
      return;
   for (unsigned k = 0; k < d->num_components; k++) {
      struct val *v = V(c, d, k);
      if (!v->held || v->state != VAL_REG)
         continue;
      v->body = true;
      int h = c->home ? c->home[d->index * 4 + k] : -1;
      if (h < 0)
         continue;
      uint8_t raw[G5P_USC_MAX];
      put(c, raw, g5p_usc_alu2_wide(raw, G5P_OP_OR, h, v->reg, 0, v->reg, 0), "move to a home above r31");
      drop(c, v);
      v->state = VAL_HOME;
      v->reg = h;
   }
}

/* After a pass that ran out of registers at c->now: the value held across
 * that point with the furthest last read (G9P's rule for SSA values),
 * or -1. */
static int pick_home(struct gctx *c)
{
   int best = -1;
   for (unsigned i = 0; i < c->nvals; i++) {
      struct val *v = &c->vals[i];
      if (!v->held || v->state != VAL_REG || !v->body || v->last <= c->now || c->home[i] >= 0 ||
          (c->reserved & BITFIELD64_BIT(v->reg)))
         continue;
      if (best < 0 || v->last > c->vals[best].last)
         best = i;
   }
   return best;
}

/* ---- operands ---- */
static unsigned reg_of(struct gctx *c, struct ref r, const char *what)
{
   struct val *v = V(c, r.def, r.comp);
   if (v->held)
      return v->reg;
   uint8_t raw[G5P_USC_MAX];
   /* Pressure retry: a constant or uniform read again later is re-read
    * per instruction instead of occupying a register in between. */
   /* Inside control flow too: only the lanes that run would get it. */
   bool scratch = (c->cf_depth > 0 || (c->remat && v->last > c->now)) &&
                  (v->state == VAL_CONST || v->state == VAL_SHARED);
   switch (v->state) {
   case VAL_CONST: {
      unsigned reg = alloc_reg(c, -1);
      put(c, raw, g5p_usc_immediate(raw, H(c, reg), v->imm, vbank(c, reg), -1), "movi");
      if (scratch)
         c->scratch |= BITFIELD64_BIT(reg);
      else
         hold(c, v, reg);
      return reg;
   }
   case VAL_HOME: {
      /* parked above r31: back into a register for this instruction */
      unsigned reg = alloc_reg(c, -1);
      put(c, raw, g5p_usc_alu2_wide(raw, G5P_OP_OR, reg, v->reg, 0, v->reg, 0), "move from a home above r31");
      c->scratch |= BITFIELD64_BIT(reg);
      return reg;
   }
   case VAL_SHARED: {
      unsigned reg = alloc_reg(c, -1);
      put(c, raw, g5p_usc_mov_shared(raw, H(c, reg), v->imm, vbank(c, reg)), "mov.shared");
      c->p->evidence |= c->vs ? PCO_G5P_EV_VS_SHARED : PCO_G5P_EV_SHARED;
      if (scratch)
         c->scratch |= BITFIELD64_BIT(reg);
      else
         hold(c, v, reg);
      return reg;
   }
   default:
      refuse(c, "%s: operand has no register (produced by an unsupported form)", what);
   }
}

static bool dies(struct gctx *c, struct ref r)
{
   struct val *v = V(c, r.def, r.comp);
   return v->held && v->last == c->now;
}

static bool is_const(struct gctx *c, struct ref r)
{
   return V(c, r.def, r.comp)->state == VAL_CONST;
}

static unsigned scratch_imm(struct gctx *c, uint32_t value)
{
   uint8_t raw[G5P_USC_MAX];
   unsigned reg = alloc_reg(c, -1);
   put(c, raw, g5p_usc_immediate(raw, H(c, reg), value, vbank(c, reg), -1), "movi");
   c->scratch |= BITFIELD64_BIT(reg);
   return reg;
}

static unsigned one_reg(struct gctx *c)
{
   if (c->remat || c->cf_depth > 0)
      return scratch_imm(c, F32_ONE);
   if (c->one_reg < 0) {
      uint8_t raw[G5P_USC_MAX];
      c->one_reg = alloc_reg(c, -1);
      put(c, raw, g5p_usc_immediate(raw, H(c, c->one_reg), F32_ONE, vbank(c, c->one_reg), -1), "movi 1.0");
   }
   return c->one_reg;
}

/* Dying operands are released before the destination is chosen, so the
 * result can take over one of their registers. */
static void free_dying(struct gctx *c, const struct ref *r, unsigned n)
{
   for (unsigned i = 0; i < n; i++)
      if (dies(c, r[i]))
         drop(c, V(c, r[i].def, r[i].comp));
}

static void define(struct gctx *c, nir_def *d, unsigned reg)
{
   struct val *v = V(c, d, 0);
   v->state = VAL_REG;
   hold(c, v, reg);
}

static struct g5p_src gsrc(const struct gctx *c, unsigned reg, struct ref r, unsigned group)
{
   return (struct g5p_src){.half = H(c, reg), .group = group, .neg = r.neg, .abs = r.abs};
}

/* ---- round 19: Apple's fragment source hints ----
 * In the 980 decodable programs of the ISA round-2 corpus a compact
 * source's group bit is 1 exactly on the register's last read (0 of 2635
 * group-1 reads is read again; tools/gpu-userspace/group-bits-r19.py). A
 * read the program repeats has group 0 and, by form, the class extension
 * (A), with header 0x40 for B and 0x80 for C (binary, FMA, compare/write,
 * literal B, convert; unary and predicated move: the extension only).
 * Rounds 16-18 put group 1 on every fragment read; their scenes read each
 * register once, so they keep their bytes. A first codegen pass counts the
 * reads; the emitting pass sets the bit on the last one.
 *
 * The vertex stage always follows it (vpds round 4, vertex-temps.md "The
 * group bit"): a group-1 read ends a value an instruction wrote, and a later
 * read sees the register-file contents from before (x1, x1b, im). Its
 * integer, address and u2f32 reads go through vgrp; the store data, the
 * outputs and the round-18 float forms (group from dies()) are not counted
 * here, so a vertex read is also last only if no later NIR instruction reads
 * the value. Vertex reads use group 0 without class-extension or header hints
 * (Apple's plain forms: v_st_two's second address add, v_st_off12's shift,
 * v_base's u2f32, v_st_cond_xfb2's not). */
enum { K_ONE, K_ZERO, K_IONE };

/* Called before the instruction reading r is put: c->nput is its index. */
static unsigned vgrp(struct gctx *c, struct ref r)
{
   if (!c->vs && !(c->key->last_read_groups || c->cf))
      return 1;
   struct val *v = V(c, r.def, r.comp);
   if (!v->held && (v->state == VAL_CONST || v->state == VAL_SHARED || v->state == VAL_HOME))
      return 1; /* re-read into a scratch register (pressure retry, home) */
   /* Control flow: a value read again later in program order, or in the
    * next iteration of a loop around this read (mark()), is not at its
    * last read here. */
   if (c->cf && v->last > c->now)
      return 0;
   unsigned i = r.def->index * 4 + r.comp;
   if (c->count_pass) {
      c->lastput[i] = c->nput;
      return 1;
   }
   return c->lastput[i] == (int)c->nput && (!c->vs || dies(c, r));
}

static unsigned kgrp(struct gctx *c, unsigned k)
{
   /* Vertex: group 1 only on the constant register's last read, as for any
    * value (vgrp). Marking every read last broke a re-read on hardware
    * (qs-ladder b2f: lanes x and w both select one_reg; w read a wrong
    * value). */
   if (c->remat || c->cf_depth > 0 || (!c->vs && !(c->key->last_read_groups || c->cf)))
      return 1;
   if (c->count_pass) {
      c->klast[k] = c->nput;
      return 1;
   }
   return c->klast[k] == (int)c->nput;
}

/* Class extension and header bits for repeated reads (binary/FMA forms). */
static void alu_hint(struct gctx *c, struct g5p_alu *f, bool three)
{
   if (c->vs || f->result_half)
      return;
   bool a = !f->a.group, b = !f->b.group, cc = three && !f->c.group;
   if (a || b || cc)
      f->source_extension = 4;
   f->header |= (b ? 0x40 : 0) | (cc ? 0x80 : 0);
}

/* x * 1.0 into a fresh register (or result half): standalone modifiers,
 * saturate and output packing. */
static unsigned vs_dst(struct gctx *c, const struct ref *r, unsigned n);

static void emit_times_one(struct gctx *c, struct ref x, bool sat, int rh, nir_def *def)
{
   if (c->vs) {
      /* vsops-probes.py v_fneg/v_fabs/v_fsat (and _rr): -0.0 + B with B's
       * modifiers or saturate, header 0x11, B's group its last read */
      if (rh >= 0)
         refuse(c, "standalone fneg/fabs/fsat into a result half (vertex)");
      unsigned xr = reg_of(c, x, "fneg/fabs/fsat");
      unsigned g = vgrp(c, x);
      unsigned dst = vs_dst(c, &x, 1);
      uint8_t raw[G5P_USC_MAX];
      put(c, raw, g5p_usc_fadd_literal_mods(raw, H(c, dst), 0x8000, H(c, xr), vbank(c, dst), g,
                                             vhdr(c, 0x11, -1, xr, -1), x.neg, x.abs, sat),
          "vertex fneg/fabs/fsat");
      c->p->evidence |= PCO_G5P_EV_VS_ALU;
      if (def)
         define(c, def, dst);
      return;
   }
   unsigned xr = reg_of(c, x, "fmul");
   unsigned one = one_reg(c);
   unsigned gx = vgrp(c, x), g1 = kgrp(c, K_ONE);
   struct ref rr[1] = {x};
   free_dying(c, rr, 1);
   struct g5p_alu f = G5P_ALU(0);
   unsigned dst = 0;
   if (rh >= 0) {
      f.dst_half = rh;
      f.result_half = true;
   } else {
      dst = alloc_reg(c, dies(c, x) ? (int)xr : -1);
      f.dst_half = 2 * dst;
   }
   f.a = gsrc(c, xr, x, gx);
   f.b = (struct g5p_src){.half = 2 * one, .group = g1};
   f.saturate = sat;
   alu_hint(c, &f, false);
   uint8_t raw[G5P_USC_MAX];
   put(c, raw, g5p_usc_binary(raw, G5P_BIN_FMUL, &f), "fmul by 1.0");
   if (def)
      define(c, def, dst);
}

/* ---- round 19: constants used by selects ---- */
static unsigned zero_reg(struct gctx *c)
{
   if (c->remat || c->cf_depth > 0)
      return scratch_imm(c, 0);
   if (c->zero_reg < 0) {
      uint8_t raw[G5P_USC_MAX];
      c->zero_reg = alloc_reg(c, -1);
      put(c, raw, g5p_usc_immediate(raw, H(c, c->zero_reg), 0, vbank(c, c->zero_reg), -1), "movi 0");
   }
   return c->zero_reg;
}

static unsigned ione_reg(struct gctx *c)
{
   if (c->remat || c->cf_depth > 0)
      return scratch_imm(c, 1);
   if (c->ione_reg < 0) {
      uint8_t raw[G5P_USC_MAX];
      c->ione_reg = alloc_reg(c, -1);
      put(c, raw, g5p_usc_immediate(raw, H(c, c->ione_reg), 1, vbank(c, c->ione_reg), -1), "movi 1");
   }
   return c->ione_reg;
}

static void put_recipe(struct gctx *c, enum g5p_recipe r, unsigned d, unsigned x, unsigned y, unsigned t,
                       unsigned flags)
{
   uint8_t buf[256];
   const char *err = NULL;
   /* The vertex recipes are witnessed with every operand and temp in bank 1
    * (g5p_recipes.c); a bank-0 one has no witnessed form yet. */
   if (c->vs && !(vbank(c, d) && vbank(c, x) && vbank(c, y) && vbank(c, t)))
      refuse(c, "%s lowering: an operand in vertex bank 0 (no witnessed mixed-bank form)", g5p_recipe_name[r]);
   unsigned n = g5p_recipe(r, buf, sizeof(buf), H(c, d), H(c, x), H(c, y), H(c, t),
                           flags | (c->vs ? G5P_RC_VERTEX : 0), &err);
   if (!n)
      refuse(c, "%s lowering: %s (%s)", g5p_recipe_name[r], err ? err : "?", g5p_usc_error ? g5p_usc_error : "?");
   /* The whole lowering is one put (one read index for its operands). */
   put(c, buf, n, g5p_recipe_name[r]);
   c->p->evidence |= PCO_G5P_EV_MATH;
}

/* A register holding x that the caller may overwrite: x's own register
 * when x dies here, else a copy. */
static unsigned consumable(struct gctx *c, struct ref x, const char *what)
{
   unsigned xr = reg_of(c, x, what);
   if (dies(c, x)) {
      drop(c, V(c, x.def, x.comp));
      c->busy |= BITFIELD64_BIT(xr);
      return xr;
   }
   unsigned d = alloc_reg(c, -1);
   uint8_t raw[G5P_USC_MAX];
   /* vertex: the class-98 single-source form (floor, minmax merge) with bank 1, header 0x21 */
   put(c, raw, g5p_usc_move(raw, H(c, d), H(c, xr), vbank(c, d), vgrp(c, x), c->vs ? vhdr(c, 0x21, xr, -1, -1) : 1),
       "copy before an in-place lowering");
   return d;
}

static void emit_recipe(struct gctx *c, nir_alu_instr *a)
{
   enum g5p_recipe r;
   switch (a->op) {
   case nir_op_frcp: r = G5P_RC_RCP; break;
   case nir_op_frsq: r = G5P_RC_RSQ; break;
   case nir_op_fsqrt: r = G5P_RC_SQRT; break;
   case nir_op_fsign: r = G5P_RC_SIGN; break;
   case nir_op_fmin: r = G5P_RC_MIN; break;
   case nir_op_fmax: r = G5P_RC_MAX; break;
   case nir_op_fceil: r = G5P_RC_CEIL; break;
   case nir_op_ftrunc: r = G5P_RC_TRUNC; break;
   case nir_op_fround_even: r = G5P_RC_RINT; break;
   case nir_op_fsin: r = G5P_RC_SIN; break;
   default: r = G5P_RC_COS; break;
   }
   struct ref x = alu_src(c, a, 0);
   if (x.neg || x.abs)
      refuse(c, "%s of a negated/absolute operand (no modifier in the measured lowering)",
             nir_op_infos[a->op].name);
   /* vertex trunc (v_trunc_rr): x kept, the destination is the round trip's temporary */
   if (r == G5P_RC_TRUNC && c->vs) {
      unsigned xr = reg_of(c, x, "trunc");
      unsigned d = alloc_reg(c, -1);
      put_recipe(c, r, d, xr, 0, 0, 0);
      free_dying(c, &x, 1);
      define(c, &a->def, d);
      return;
   }
   if (r == G5P_RC_RSQ) {
      unsigned xr = reg_of(c, x, "rsqrt");
      unsigned d = alloc_reg(c, -1); /* never the input: the -0 test writes first */
      put_recipe(c, r, d, xr, 0, 0, vgrp(c, x) ? 0 : G5P_RC_X_LIVE);
      free_dying(c, &x, 1);
      define(c, &a->def, d);
      return;
   }
   unsigned y = 0;
   struct ref yr = {0};
   if (r == G5P_RC_MIN || r == G5P_RC_MAX) {
      yr = alu_src(c, a, 1);
      if (yr.neg || yr.abs)
         refuse(c, "%s of a negated/absolute operand", nir_op_infos[a->op].name);
      /* Commutative: overwrite whichever operand dies. */
      if (!dies(c, x) && dies(c, yr)) {
         struct ref tmp = x;
         x = yr;
         yr = tmp;
      }
      y = reg_of(c, yr, "min/max");
   }
   /* Record the recipe's reads (the whole lowering is one put): x in place
    * when it dies here (else consumable() copies it and records the copy's
    * read), and min/max's y. */
   reg_of(c, x, nir_op_infos[a->op].name);
   if (dies(c, x))
      vgrp(c, x);
   unsigned d = consumable(c, x, nir_op_infos[a->op].name);
   unsigned t = 0;
   bool scratch = r != G5P_RC_MIN && r != G5P_RC_MAX && r != G5P_RC_CEIL;
   if (scratch)
      t = alloc_reg(c, -1);
   unsigned flags = 0;
   if ((r == G5P_RC_MIN || r == G5P_RC_MAX) && !vgrp(c, yr))
      flags |= G5P_RC_Y_LIVE;
   put_recipe(c, r, d, d, y, t, flags);
   if (scratch)
      c->busy &= ~BITFIELD64_BIT(t);
   if (r == G5P_RC_MIN || r == G5P_RC_MAX)
      free_dying(c, &yr, 1);
   c->busy &= ~BITFIELD64_BIT(d);
   define(c, &a->def, d);
}

#define S(h, g) ((struct g5p_src){.half = (h), .group = (g)})

/* ---- round 19: relations and selections (assemble_compare_select,
 * strict pattern: float ge uses relation control 10, float ne swaps the
 * selected values; int/uint ge and ne swap) ---- */
struct gsrc4 { unsigned a, b, t, f; };

/* assemble_compare_select (strict): float ne and int/uint ge/ne write the
 * false value first. */
static bool cmp_swaps(nir_op op)
{
   return op == nir_op_fneu || op == nir_op_ige || op == nir_op_ine || op == nir_op_uge;
}

static void cmp_select_regs(struct gctx *c, nir_op op, unsigned a, unsigned b, unsigned t, unsigned f, unsigned dst,
                            struct gsrc4 g)
{
   enum g5p_cmp code;
   bool fl = false, ge = false, ne = false;
   switch (op) {
   case nir_op_flt: code = G5P_FLT; fl = true; break;
   case nir_op_fge: code = G5P_FLT; fl = ge = true; break;
   case nir_op_feq: code = G5P_FEQ; fl = true; break;
   case nir_op_fneu: code = G5P_FEQ; fl = ne = true; break;
   case nir_op_ilt: code = G5P_ILT; break;
   case nir_op_ige: code = G5P_ILT; ge = true; break;
   case nir_op_ieq: code = G5P_IEQ; break;
   case nir_op_ine: code = G5P_IEQ; ne = true; break;
   case nir_op_ult: code = G5P_ULT; break;
   default: code = G5P_ULT; ge = true; break; /* uge */
   }
   unsigned rc = fl && ge ? 0x10 : 0;
   bool swap = fl ? ne : (ge || ne);
   if (swap) {
      unsigned tmp = t;
      t = f;
      f = tmp;
      tmp = g.t;
      g.t = g.f;
      g.f = tmp;
   }
   /* Repeated reads: compare/write B -> ext 4 + 0x40, C -> ext 4 + 0x80
    * (A alone: group only); inverse move -> ext 4 (Apple's selects). */
   struct g5p_ctl kw = G5P_CTL, km = G5P_CTL;
   if (c->vs) {
      /* vertex (v_flt_sel, v_fneu_sel, v_ilt_sel): A, B, C and the move's
       * source are temps in bank 1; a read that is not the last one keeps
       * group 0 without the fragment's repeated-read controls (Apple's B and
       * C there are read again by the outputs: header 0x39, extension 0) */
      kw.bank = km.bank = vbank(c, dst);
      kw.header |= (vbank(c, a) ? VS_A : 0) | (vbank(c, b) ? VS_B : 0) | (vbank(c, t) ? VS_C : 0);
      km.header |= vbank(c, f) ? VS_A : 0;
   } else {
      if (!g.b || !g.t)
         kw.source_extension = 4;
      kw.header |= (!g.b ? 0x40 : 0) | (!g.t ? 0x80 : 0);
      if (!g.f)
         km.source_extension = 4;
   }
   uint8_t raw[G5P_USC_MAX];
   put(c, raw, g5p_usc_compare_write(raw, code, H(c, dst), S(H(c, a), g.a), S(H(c, b), g.b), S(H(c, t), g.t), kw,
                                     rc),
       "compare/write");
   put(c, raw, g5p_usc_predicated_move(raw, H(c, dst), S(H(c, f), g.f), km), "inverse move");
   c->p->evidence |= PCO_G5P_EV_SELECT;
}

/* Condition operands: a fused compare's sources, or a materialized
 * boolean compared against 0.0 (fneu). */
struct cond {
   nir_op op;
   struct ref a, b;
   bool zero_b; /* b is the 0.0 constant register */
};

static struct cond cond_of(struct gctx *c, nir_def *d, unsigned comp)
{
   nir_alu_instr *cmp = fused_cmp(c, d, comp);
   if (cmp)
      return (struct cond){cmp->op, resolve_value(c, cmp->src[0].src.ssa, cmp->src[0].swizzle[0]),
                           resolve_value(c, cmp->src[1].src.ssa, cmp->src[1].swizzle[0]), false};
   return (struct cond){nir_op_fneu, resolve_value(c, d, comp), {0}, true};
}

/* dst = cond ? t : f, t/f given as refs or as persistent constant
 * registers (kt/kf: K_ONE, K_ZERO, K_IONE). */
static unsigned select_into(struct gctx *c, struct cond k, struct ref *tv, unsigned kt, struct ref *fv,
                            unsigned kf)
{
   static unsigned (*const kreg[3])(struct gctx *) = {one_reg, zero_reg, ione_reg};
   unsigned ar = reg_of(c, k.a, "compare");
   unsigned br = k.zero_b ? zero_reg(c) : reg_of(c, k.b, "compare");
   unsigned treg = tv ? reg_of(c, *tv, "select") : kreg[kt](c);
   unsigned freg = fv ? reg_of(c, *fv, "select") : kreg[kf](c);
   /* Reads in instruction order: compare/write (a, b, then t or f as C),
    * the inverse move the other one (cmp_select_regs swaps for ne/ge). */
   bool swap = cmp_swaps(k.op);
   struct gsrc4 g = {vgrp(c, k.a), k.zero_b ? kgrp(c, K_ZERO) : vgrp(c, k.b), 0, 0};
   if (!swap)
      g.t = tv ? vgrp(c, *tv) : kgrp(c, kt);
   else
      g.f = fv ? vgrp(c, *fv) : kgrp(c, kf);
   /* Apple's pairs (the 292 compare/write + inverse-move pairs of the ISA
    * round-2 corpus, fast and strict) never write B, C or the inverse
    * move's source; the destination takes A's register when A dies (89 of
    * them), else a free one. The compare/write writes its destination before
    * the inverse move reads its source (hypr-a9 h1 on hardware: a pair whose
    * destination was its own inverse source read back C), so every other
    * dying operand stays held until the instruction's release(). */
   int prefer = -1;
   if (dies(c, k.a) && ar != br && ar != treg && ar != freg) {
      prefer = ar;
      drop(c, V(c, k.a.def, k.a.comp));
   }
   unsigned dst = alloc_reg(c, prefer);
   /* The inverse move is the next instruction: its read is counted there. */
   c->nput++;
   if (!swap)
      g.f = fv ? vgrp(c, *fv) : kgrp(c, kf);
   else
      g.t = tv ? vgrp(c, *tv) : kgrp(c, kt);
   c->nput--;
   cmp_select_regs(c, k.op, ar, br, treg, freg, dst, g);
   return dst;
}

static void emit_bool_alu(struct gctx *c, nir_alu_instr *a)
{
   uint8_t raw[G5P_USC_MAX];
   if (is_cmp(a->op)) { /* materialized: 1.0 / 0.0 */
      if (c->fused[a->def.index])
         return;
      struct cond k = {a->op, alu_src(c, a, 0), alu_src(c, a, 1), false};
      define(c, &a->def, select_into(c, k, NULL, K_ONE, NULL, K_ZERO));
      return;
   }
   if (a->op == nir_op_bcsel) {
      struct cond k = cond_of(c, a->src[0].src.ssa, a->src[0].swizzle[0]);
      struct ref t = alu_src(c, a, 1), f = alu_src(c, a, 2);
      define(c, &a->def, select_into(c, k, &t, 0, &f, 0));
      return;
   }
   if (a->op == nir_op_b2f32 || a->op == nir_op_b2i32) {
      struct cond k = cond_of(c, a->src[0].src.ssa, a->src[0].swizzle[0]);
      if (a->op == nir_op_b2f32 && k.zero_b && !c->vs) { /* already 0.0 / 1.0 */
         unsigned xr = reg_of(c, k.a, "b2f32");
         bool d = dies(c, k.a);
         unsigned g = d ? 1 : vgrp(c, k.a);
         free_dying(c, &k.a, 1);
         unsigned dst = alloc_reg(c, d ? (int)xr : -1);
         if (dst != xr)
            put(c, raw, g5p_usc_move(raw, 2 * dst, 2 * xr, 0, g, 1), "b2f32");
         define(c, &a->def, dst);
         return;
      }
      define(c, &a->def, select_into(c, k, NULL, a->op == nir_op_b2f32 ? K_ONE : K_IONE, NULL, K_ZERO));
      return;
   }
   /* inot of a 0.0/1.0 boolean: xor with the 1.0 bit pattern (literal XOR). */
   struct ref x = alu_src(c, a, 0);
   unsigned xr = reg_of(c, x, "inot");
   bool d = dies(c, x);
   unsigned g = vgrp(c, x);
   free_dying(c, &x, 1);
   unsigned dst = alloc_reg(c, d ? (int)xr : -1);
   /* vertex: Apple's v_ixor_big / v_ixor_big_rr (x ^ 0x3f800000): bank 1, header 0x11, the source group alone
    * marks a repeated read (no extension) */
   if (c->vs)
      put(c, raw, g5p_usc_binary_literal2(raw, G5P_BIN_XOR, H(c, dst), F32_ONE, H(c, xr), vbank(c, dst), g,
                                          vhdr(c, 0x11, -1, xr, -1), G5P_LIT_U32, -1, 0, 0, false), "boolean not");
   else
      put(c, raw, g5p_usc_binary_literal2(raw, G5P_BIN_XOR, 2 * dst, F32_ONE, 2 * xr, 0, g, g ? 1 : 0x41, G5P_LIT_U32,
                                          -1, g ? 0 : 4, 0, false), "boolean not");
   define(c, &a->def, dst);
}

/* ---- round 19: vertex ALU with common-store (uniform) sources ----
 * Measured vertex convention (v_mvp, executed in the iOS tex capture, and
 * the round-2 v_mat3 and v_uniform probes): every temp is bank 1, the
 * header carries one bank bit per register operand (A 0x20, B 0x10,
 * C 0x08), and a source group bit is 1 exactly on the operand's last read.
 * Only control tuples present in the corpus are emitted (vs_tuple below);
 * an operand that is read again is never marked as a last read. */
enum vkind { VK_SHARED, VK_REG };

/* Inline shared ALU sources are witnessed only below dword 32. The high index
 * bit (byte 2) is witnessed on the shared MOV (sources 2..33) and on texture
 * LOD sources, never on an ALU source; the first vertex shader that read
 * sh32/sh33 inline wedged the TA (docs/gpu/experiments/vertex-temps.md, vk).
 * A higher dword is moved into a register first. */
static enum vkind vkind(struct gctx *c, struct ref r)
{
   struct val *v = V(c, r.def, r.comp);
   return v->state == VAL_SHARED && !v->held && v->imm < 32 ? VK_SHARED : VK_REG;
}

static unsigned vs_dst(struct gctx *c, const struct ref *r, unsigned n)
{
   int prefer = -1;
   for (unsigned i = 0; i < n && prefer < 0; i++)
      if (vkind(c, r[i]) == VK_REG && dies(c, r[i]))
         prefer = V(c, r[i].def, r[i].comp)->reg;
   free_dying(c, r, n);
   return alloc_reg(c, prefer);
}

static void vs_fbin(struct gctx *c, nir_alu_instr *a)
{
   enum g5p_binop op = a->op == nir_op_fmul ? G5P_BIN_FMUL : G5P_BIN_FADD;
   struct ref r[2] = {alu_src(c, a, 0), alu_src(c, a, 1)};
   if (a->op == nir_op_fsub)
      r[1].neg = !r[1].neg;
   uint8_t raw[G5P_USC_MAX];
   if (c->saturate[a->def.index])
      refuse(c, "%s with saturate (no witnessed vertex form)", nir_op_infos[a->op].name);
   /* Round 18: f16 literal in A, register in B (header 0x11, B read last). */
   for (unsigned i = 0; i < 2; i++) {
      struct ref lit = r[i], x = r[1 - i];
      if (!is_const(c, lit) || is_const(c, x))
         continue;
      uint32_t bits = apply_mods(V(c, lit.def, lit.comp)->imm, lit);
      if (op == G5P_BIN_FMUL && x.neg && !x.abs) {
         bits ^= 0x80000000u;
         x.neg = false;
      }
      uint16_t h;
      if (x.neg || x.abs || !f16_exact(bits, &h) || vkind(c, x) != VK_REG)
         continue;
      unsigned xr = reg_of(c, x, "literal ALU");
      if (!dies(c, x))
         continue; /* B is read again: no witnessed group-0 literal form */
      unsigned dst = vs_dst(c, &x, 1);
      put(c, raw, g5p_usc_binary_literal(raw, op, H(c, dst), h, H(c, xr), vbank(c, dst), 1, vhdr(c, 0x11, -1, xr, -1),
                                         true, -1), "f16-literal ALU");
      c->p->evidence |= PCO_G5P_EV_VS_ALU;
      define(c, &a->def, dst);
      return;
   }
   if (op == G5P_BIN_FMUL && (r[0].neg || r[0].abs || r[1].neg || r[1].abs) && vkind(c, r[0]) == VK_REG &&
       vkind(c, r[1]) == VK_REG && !is_const(c, r[0]) && !is_const(c, r[1])) {
      /* vsalu-probes.py v_fmul_neg(_rr), v_fmul_abs(_b), v_fmul_negabs: the two-register fmul {h31 x0},
       * operands' groups their last reads. Apple puts an operand with abs in A and the product's sign on A,
       * or on B when A has abs (-|x| * y: abs A, negate B). Both abs and a sign has no witnessed form. */
      unsigned ia = !r[0].abs && r[1].abs ? 1 : 0, ib = 1 - ia;
      struct ref ra = r[ia], rb = r[ib];
      bool neg = ra.neg != rb.neg;
      ra.neg = rb.neg = false;
      if (neg && !ra.abs)
         ra.neg = true;
      else if (neg && !rb.abs)
         rb.neg = true;
      else if (neg)
         refuse(c, "fmul of two absolute operands with a sign (no witnessed vertex form)");
      unsigned rega = reg_of(c, r[ia], "fmul"), regb = reg_of(c, r[ib], "fmul");
      struct g5p_alu f = G5P_ALU(0);
      f.header = vhdr(c, 0x31, rega, regb, -1);
      f.a = gsrc(c, rega, ra, vgrp(c, r[ia]));
      f.b = gsrc(c, regb, rb, vgrp(c, r[ib]));
      unsigned dst = vs_dst(c, r, 2);
      f.bank = vbank(c, dst);
      f.dst_half = H(c, dst);
      put(c, raw, g5p_usc_binary(raw, op, &f), "fmul with source modifiers");
      c->p->evidence |= PCO_G5P_EV_VS_ALU;
      define(c, &a->def, dst);
      return;
   }
   if (r[0].neg || r[0].abs || r[1].neg || r[1].abs) {
      if (op != G5P_BIN_FADD)
         refuse(c, "%s with a source modifier (no witnessed vertex form)", nir_op_infos[a->op].name);
      /* vsops-probes.py v_fsub, v_fsub_rr, v_fadd_abs, v_fadd_negabs: the two-register fadd
       * {h31 x0 dc0} with each operand's negate/abs and group (its last read), operands in order */
      unsigned ra = reg_of(c, r[0], "fadd"), rb = reg_of(c, r[1], "fadd");
      struct g5p_alu f = G5P_ALU(0);
      f.a = gsrc(c, ra, r[0], vgrp(c, r[0]));
      f.b = gsrc(c, rb, r[1], vgrp(c, r[1]));
      unsigned dst = vs_dst(c, r, 2);
      f.bank = vbank(c, dst);
      f.header = vhdr(c, 0x31, ra, rb, -1);
      f.dst_half = H(c, dst);
      put(c, raw, g5p_usc_binary(raw, op, &f), nir_op_infos[a->op].name);
      c->p->evidence |= PCO_G5P_EV_VS_ALU;
      define(c, &a->def, dst);
      return;
   }
   if (vkind(c, r[0]) == VK_SHARED && vkind(c, r[1]) == VK_SHARED)
      reg_of(c, r[1], "uniform operand"); /* mov.shared to a temp */
   if (vkind(c, r[1]) == VK_SHARED) {
      struct ref t = r[0];
      r[0] = r[1];
      r[1] = t;
   }
   if (vkind(c, r[0]) == VK_SHARED && (op == G5P_BIN_FMUL || dies(c, r[1]) || vkind(c, r[1]) != VK_REG)) {
      /* binary.shared: fmul {h11 x0 dc0 bg0|bg1}, fadd {h11 x0 dc0 bg1} */
      unsigned sd = V(c, r[0].def, r[0].comp)->imm;
      unsigned br = reg_of(c, r[1], "uniform ALU");
      unsigned bg = dies(c, r[1]);
      unsigned dst = vs_dst(c, r, 2);
      vs_bank1_only(c, "uniform ALU", dst, br, -1, -1);
      put(c, raw, g5p_usc_alu_shared(raw, op, H(c, dst), sd, H(c, br), 1, bg, 0x11, 0, 0, false, -1, 1),
          "uniform ALU");
      c->p->evidence |= PCO_G5P_EV_VS_SHARED;
      define(c, &a->def, dst);
      return;
   }
   /* Two registers (a uniform read again becomes a temp first). */
   unsigned reg[2] = {reg_of(c, r[0], "ALU"), reg_of(c, r[1], "ALU")};
   bool d[2] = {dies(c, r[0]), dies(c, r[1])};
   struct g5p_alu f = G5P_ALU(0);
   unsigned ia = 0, ib = 1; /* the operands in A and B */
   if (!d[0] && !d[1]) {
      /* both read again (vsops-probes.py v_fmul_rr / v_fadd_rr, fast and strict): {h31 x0 dc0 ag0 bg0} */
      f.header = 0x31;
      f.a = gsrc(c, reg[0], r[0], 0);
      f.b = gsrc(c, reg[1], r[1], 0);
   } else if (op == G5P_BIN_FADD) {
      /* round 18 / v_depth: {h31 x0 dc0, A group 0, B read last} */
      ia = d[0] ? 1 : 0;
      ib = 1 - ia;
      f.header = 0x31;
      f.a = gsrc(c, reg[ia], r[ia], 0);
      f.b = gsrc(c, reg[ib], r[ib], 1);
   } else {
      /* v_uniform_live_*: {h31 x0 dc128 ag1 bg1} or {h71 x4 dc128 ag1 bg0} */
      ia = d[0] ? 0 : 1;
      ib = 1 - ia;
      f.header = d[ib] ? 0x31 : 0x71;
      f.source_extension = d[ib] ? 0 : 4;
      f.dst_control = 0x80;
      f.a = gsrc(c, reg[ia], r[ia], 1);
      f.b = gsrc(c, reg[ib], r[ib], d[ib]);
   }
   unsigned dst = vs_dst(c, r, 2);
   f.bank = vbank(c, dst);
   f.header = vhdr(c, f.header, reg[ia], reg[ib], -1);
   f.dst_half = H(c, dst);
   put(c, raw, g5p_usc_binary(raw, op, &f), nir_op_infos[a->op].name);
   c->p->evidence |= PCO_G5P_EV_VS_ALU;
   define(c, &a->def, dst);
}

static void vs_ffma(struct gctx *c, nir_alu_instr *a)
{
   struct ref r[3] = {alu_src(c, a, 0), alu_src(c, a, 1), alu_src(c, a, 2)};
   uint8_t raw[G5P_USC_MAX];
   if (c->saturate[a->def.index])
      refuse(c, "ffma with saturate (no witnessed vertex form)");
   /* Round 18: f16 literal multiplicand (header 0x19, B and C read last). */
   if (!r[2].neg && !r[2].abs) {
      for (unsigned i = 0; i < 2; i++) {
         struct ref lit = r[i], x = r[1 - i];
         if (!is_const(c, lit) || is_const(c, x))
            continue;
         uint32_t bits = apply_mods(V(c, lit.def, lit.comp)->imm, lit);
         if (x.neg && !x.abs) {
            bits ^= 0x80000000u;
            x.neg = false;
         }
         uint16_t h;
         if (x.abs || !f16_exact(bits, &h) || vkind(c, x) != VK_REG || vkind(c, r[2]) != VK_REG)
            continue;
         unsigned xr = reg_of(c, x, "ffma literal"), cr = reg_of(c, r[2], "ffma literal");
         if (!dies(c, x) || !dies(c, r[2]))
            continue;
         struct ref both[2] = {x, r[2]};
         unsigned dst = vs_dst(c, both, 2);
         put(c, raw, g5p_usc_fma_literal(raw, H(c, dst), h, H(c, xr), H(c, cr), vbank(c, dst), 1, 1,
                                         vhdr(c, 0x19, -1, xr, cr)), "f16-literal ffma");
         c->p->evidence |= PCO_G5P_EV_VS_ALU;
         define(c, &a->def, dst);
         return;
      }
   }
   /* Source modifiers on the register FMA: vsalu-probes.py v_ffma_nega/negc/absa/absc/negabsa and
    * v_ffma_negb_rr/negc_rr, Apple's {h39} with each operand's negate/abs in place, operands in order. The
    * common-store forms have no witnessed modifier. */
   for (unsigned i = 0; i < 3; i++)
      if ((r[i].neg || r[i].abs) && (vkind(c, r[0]) == VK_SHARED || vkind(c, r[1]) == VK_SHARED ||
                                     vkind(c, r[2]) == VK_SHARED))
         refuse(c, "ffma with a source modifier and a uniform operand (no witnessed vertex form)");
   if (vkind(c, r[0]) == VK_SHARED && vkind(c, r[1]) == VK_SHARED)
      reg_of(c, r[1], "uniform operand");
   if (vkind(c, r[1]) == VK_SHARED) {
      struct ref t = r[0];
      r[0] = r[1];
      r[1] = t;
   }
   bool sa = vkind(c, r[0]) == VK_SHARED, sc = vkind(c, r[2]) == VK_SHARED;
   unsigned dst;
   if (sa && sc) {
      /* fma.shared_ac route70 (A*B+C): {h11 x0 dc0 bg1|bg0} */
      unsigned br = reg_of(c, r[1], "uniform FMA");
      unsigned bg = dies(c, r[1]);
      dst = vs_dst(c, r, 3);
      vs_bank1_only(c, "uniform FMA (shared A/C)", dst, br, -1, -1);
      put(c, raw, g5p_usc_fma_shared_ac(raw, H(c, dst), V(c, r[0].def, r[0].comp)->imm, H(c, br),
                                        V(c, r[2].def, r[2].comp)->imm, 1, bg, 0x11, 0, 0, 0x70, 1, 1),
          "uniform FMA (shared A/C)");
      c->p->evidence |= PCO_G5P_EV_VS_SHARED;
   } else if (sa) {
      /* fma.shared_a (A*B+C): {h19 x0 dc0 bg1 cg1} or {h59 x4 dc0 bg0 cg1} */
      unsigned br = reg_of(c, r[1], "uniform FMA"), cr = reg_of(c, r[2], "uniform FMA");
      bool db = dies(c, r[1]), dc = dies(c, r[2]);
      if (!dc)
         refuse(c, "ffma: uniform x register + register addend that is read again "
                   "(no witnessed vertex control tuple)");
      dst = vs_dst(c, r, 3);
      vs_bank1_only(c, "uniform FMA (shared A)", dst, br, cr, -1);
      put(c, raw, g5p_usc_fma_shared_a(raw, H(c, dst), V(c, r[0].def, r[0].comp)->imm, H(c, br), H(c, cr), 1, db, 1,
                                       db ? 0x19 : 0x59, db ? 0 : 4, 0, -1, 1), "uniform FMA (shared A)");
      c->p->evidence |= PCO_G5P_EV_VS_SHARED;
   } else if (sc) {
      /* fma.shared_a route58 (B*C + shared): {h19 x0 dc0 bg1 cg1} or {h59 x4 dc0 bg0 cg0} */
      unsigned br = reg_of(c, r[0], "uniform FMA"), cr = reg_of(c, r[1], "uniform FMA");
      bool last = dies(c, r[0]) && dies(c, r[1]);
      dst = vs_dst(c, r, 3);
      vs_bank1_only(c, "uniform FMA (shared addend)", dst, br, cr, -1);
      put(c, raw, g5p_usc_fma_shared_a(raw, H(c, dst), V(c, r[2].def, r[2].comp)->imm, H(c, br), H(c, cr), 1, last,
                                       last, last ? 0x19 : 0x59, last ? 0 : 4, 0, 0x58, 1),
          "uniform FMA (shared addend)");
      c->p->evidence |= PCO_G5P_EV_VS_SHARED;
   } else {
      /* v_uniform_live_*: {h39 x0 ag1 bg1 cg1} or {h79 x4 ag1 bg0 cg1} */
      unsigned reg[3] = {reg_of(c, r[0], "ffma"), reg_of(c, r[1], "ffma"), reg_of(c, r[2], "ffma")};
      bool d[3] = {dies(c, r[0]), dies(c, r[1]), dies(c, r[2])};
      struct g5p_alu f = G5P_ALU(0);
      unsigned ia = 0, ib = 1;
      const bool mods = r[0].neg || r[0].abs || r[1].neg || r[1].abs || r[2].neg || r[2].abs;
      if ((!d[0] && !d[1]) || !d[2] || mods) {
         /* vsalu-probes.py v_ffma_rr..rr4 and the modifier forms: operands in order, each group its last read
          * and its negate/abs in place, {h39 x0} */
         f.header = 0x39;
         f.a = gsrc(c, reg[0], r[0], d[0]);
         f.b = gsrc(c, reg[1], r[1], d[1]);
         f.c = gsrc(c, reg[2], r[2], d[2]);
      } else {
         ia = d[0] ? 0 : 1;
         ib = 1 - ia;
         f.header = d[ib] ? 0x39 : 0x79;
         f.source_extension = d[ib] ? 0 : 4;
         f.a = gsrc(c, reg[ia], r[ia], 1);
         f.b = gsrc(c, reg[ib], r[ib], d[ib]);
         f.c = gsrc(c, reg[2], r[2], 1);
      }
      dst = vs_dst(c, r, 3);
      f.bank = vbank(c, dst);
      f.header = vhdr(c, f.header, reg[ia], reg[ib], reg[2]);
      f.dst_half = H(c, dst);
      put(c, raw, g5p_usc_fma(raw, &f), "ffma");
      c->p->evidence |= PCO_G5P_EV_VS_ALU;
   }
   define(c, &a->def, dst);
}

static void emit_fbin(struct gctx *c, nir_alu_instr *a)
{
   if (c->vs) {
      vs_fbin(c, a);
      return;
   }
   enum g5p_binop op = a->op == nir_op_fmul ? G5P_BIN_FMUL : G5P_BIN_FADD;
   struct ref r[2] = {alu_src(c, a, 0), alu_src(c, a, 1)};
   if (a->op == nir_op_fsub)
      r[1].neg = !r[1].neg;
   bool sat = c->saturate[a->def.index];
   int rh = c->rh_half[a->def.index];
   uint8_t raw[G5P_USC_MAX];
   /* f16 literal in source A, register in B (promoted literal form). */
   if (!sat && rh < 0) {
      for (unsigned i = 0; i < 2; i++) {
         struct ref lit = r[i], x = r[1 - i];
         if (!is_const(c, lit) || is_const(c, x))
            continue;
         uint32_t bits = apply_mods(V(c, lit.def, lit.comp)->imm, lit);
         if (op == G5P_BIN_FMUL && x.neg && !x.abs) {
            bits ^= 0x80000000u;
            x.neg = false;
         }
         uint16_t h;
         if (x.neg || x.abs || !f16_exact(bits, &h))
            continue;
         unsigned xr = reg_of(c, x, "literal ALU");
         bool d = dies(c, x);
         unsigned gx = vgrp(c, x);
         free_dying(c, &x, 1);
         unsigned dst = alloc_reg(c, d ? (int)xr : -1);
         put(c, raw, g5p_usc_binary_literal2(raw, op, 2 * dst, h, 2 * xr, c->vs, gx, (c->vs ? 0x11 : 1) | (gx ? 0 : 0x40),
                                             G5P_LIT_F16, -1, gx ? 0 : 4, 0, false), "f16-literal ALU");
         c->p->evidence |= c->vs ? PCO_G5P_EV_VS_ALU : PCO_G5P_EV_FS_ALU;
         define(c, &a->def, dst);
         return;
      }
   }
   if (c->vs && (op != G5P_BIN_FADD || sat || rh >= 0 || r[0].neg || r[0].abs || r[1].neg || r[1].abs))
      refuse(c, "%s with two register operands%s: only v_depth's plain fadd (header 0x31) is witnessed "
             "in the vertex stage", nir_op_infos[a->op].name, r[0].neg || r[1].neg ? " and a negate" : "");
   unsigned reg[2] = {reg_of(c, r[0], "ALU"), reg_of(c, r[1], "ALU")};
   bool d0 = dies(c, r[0]), d1 = dies(c, r[1]);
   unsigned ia = 0, ib = 1;
   if (d0) {
      ia = 1;
      ib = 0;
   }
   unsigned ga = vgrp(c, r[ia]), gb = vgrp(c, r[ib]);
   free_dying(c, r, 2);
   int prefer = d0 ? (int)reg[0] : d1 ? (int)reg[1] : -1;
   struct g5p_alu f = G5P_ALU(0);
   unsigned dst = 0;
   if (rh >= 0) {
      f.dst_half = rh;
      f.result_half = true;
   } else {
      dst = alloc_reg(c, prefer);
      f.dst_half = 2 * dst;
   }
   f.bank = c->vs;
   f.header = c->vs ? 0x31 : 1;
   f.a = gsrc(c, reg[ia], r[ia], c->vs ? 0 : ga);
   f.b = gsrc(c, reg[ib], r[ib], gb);
   f.saturate = sat;
   alu_hint(c, &f, false);
   put(c, raw, g5p_usc_binary(raw, op, &f), nir_op_infos[a->op].name);
   c->p->evidence |= c->vs ? PCO_G5P_EV_VS_ALU : PCO_G5P_EV_FS_ALU;
   if (rh < 0)
      define(c, &a->def, dst);
}

static void emit_ffma(struct gctx *c, nir_alu_instr *a)
{
   if (c->vs) {
      vs_ffma(c, a);
      return;
   }
   struct ref r[3] = {alu_src(c, a, 0), alu_src(c, a, 1), alu_src(c, a, 2)};
   bool sat = c->saturate[a->def.index];
   int rh = c->rh_half[a->def.index];
   uint8_t raw[G5P_USC_MAX];
   if (!sat && rh < 0 && !r[2].neg && !r[2].abs) {
      for (unsigned i = 0; i < 2; i++) {
         struct ref lit = r[i], x = r[1 - i];
         if (!is_const(c, lit) || is_const(c, x))
            continue;
         uint32_t bits = apply_mods(V(c, lit.def, lit.comp)->imm, lit);
         if (x.neg && !x.abs) {
            bits ^= 0x80000000u;
            x.neg = false;
         }
         uint16_t h;
         if (!x.abs && !f16_exact(bits, &h) && is_const(c, r[2]) &&
             (!getenv("PVR_G5P_FMA_LITERALS") || strcmp(getenv("PVR_G5P_FMA_LITERALS"), "0"))) {
            uint32_t addend = V(c, r[2].def, r[2].comp)->imm;
            if (!(addend & 0x200)) {
               unsigned xr = reg_of(c, x, "full-f32 literal ffma");
               bool dx = dies(c, x);
               struct ref operands[3] = {lit, x, r[2]};
               free_dying(c, operands, 3);
               unsigned dst = alloc_reg(c, dx ? (int)xr : -1);
               put(c, raw, g5p_usc_fma_f32_literals(raw, 2 * dst, 2 * xr, bits, addend, 0),
                   "full-f32 literal ffma");
               c->p->evidence |= PCO_G5P_EV_FS_ALU;
               define(c, &a->def, dst);
               return;
            }
         }
         if (x.abs || !f16_exact(bits, &h))
            continue;
         unsigned xr = reg_of(c, x, "ffma literal"), cr = reg_of(c, r[2], "ffma literal");
         bool dx = dies(c, x), dc = dies(c, r[2]);
         unsigned gx = vgrp(c, x), gc = vgrp(c, r[2]);
         struct ref both[2] = {x, r[2]};
         free_dying(c, both, 2);
         unsigned dst = alloc_reg(c, dx ? (int)xr : dc ? (int)cr : -1);
         put(c, raw, g5p_usc_fma_literal(raw, 2 * dst, h, 2 * xr, 2 * cr, c->vs, gx, gc, c->vs ? 0x19 : 1),
             "f16-literal ffma");
         c->p->evidence |= c->vs ? PCO_G5P_EV_VS_ALU : PCO_G5P_EV_FS_ALU;
         define(c, &a->def, dst);
         return;
      }
   }
   if (c->vs)
      refuse(c, "ffma without an f16-exact literal multiplicand: no witnessed vertex-stage register FMA");
   unsigned reg[3] = {reg_of(c, r[0], "ffma"), reg_of(c, r[1], "ffma"), reg_of(c, r[2], "ffma")};
   bool d[3] = {dies(c, r[0]), dies(c, r[1]), dies(c, r[2])};
   unsigned ia = 0, ib = 1;
   if (!d[0] && d[1]) {
      ia = 1;
      ib = 0;
   }
   unsigned ga = vgrp(c, r[ia]), gb = vgrp(c, r[ib]), gc = vgrp(c, r[2]);
   free_dying(c, r, 3);
   int prefer = d[ia] ? (int)reg[ia] : d[2] ? (int)reg[2] : d[ib] ? (int)reg[ib] : -1;
   struct g5p_alu f = G5P_ALU(0);
   unsigned dst = 0;
   if (rh >= 0) {
      f.dst_half = rh;
      f.result_half = true;
   } else {
      dst = alloc_reg(c, prefer);
      f.dst_half = 2 * dst;
   }
   f.a = gsrc(c, reg[ia], r[ia], ga);
   f.b = gsrc(c, reg[ib], r[ib], gb);
   f.c = gsrc(c, reg[2], r[2], gc);
   f.saturate = sat;
   alu_hint(c, &f, true);
   put(c, raw, g5p_usc_fma(raw, &f), "ffma");
   c->p->evidence |= PCO_G5P_EV_FS_ALU;
   if (rh < 0)
      define(c, &a->def, dst);
}

/* Commutative integer binary/min/max (fragment only). */
static void emit_ibin(struct gctx *c, nir_alu_instr *a)
{
   struct ref r[2] = {alu_src(c, a, 0), alu_src(c, a, 1)};
   uint8_t raw[G5P_USC_MAX];
   if (a->op == nir_op_ixor) { /* 140 witnessed literal XORs */
      for (unsigned i = 0; i < 2; i++) {
         if (!is_const(c, r[i]) || is_const(c, r[1 - i]))
            continue;
         struct ref x = r[1 - i];
         unsigned xr = reg_of(c, x, "xor literal");
         bool d = dies(c, x);
         unsigned gx = vgrp(c, x);
         free_dying(c, &x, 1);
         unsigned dst = alloc_reg(c, d ? (int)xr : -1);
         put(c, raw, g5p_usc_binary_literal(raw, G5P_BIN_XOR, 2 * dst, V(c, r[i].def, r[i].comp)->imm, 2 * xr, 0,
                                            gx, 1, false, -1), "xor literal");
         define(c, &a->def, dst);
         return;
      }
   }
   unsigned reg[2] = {reg_of(c, r[0], "integer ALU"), reg_of(c, r[1], "integer ALU")};
   bool d0 = dies(c, r[0]), d1 = dies(c, r[1]);
   unsigned ia = d0 ? 1 : 0, ib = d0 ? 0 : 1;
   unsigned ga = vgrp(c, r[ia]), gb = vgrp(c, r[ib]);
   free_dying(c, r, 2);
   unsigned dst = alloc_reg(c, d0 ? (int)reg[0] : d1 ? (int)reg[1] : -1);
   struct g5p_alu f = G5P_ALU(2 * dst);
   f.a = gsrc(c, reg[ia], (struct ref){0}, ga);
   f.b = gsrc(c, reg[ib], (struct ref){0}, gb);
   unsigned n;
   switch (a->op) {
   case nir_op_iadd: n = g5p_usc_binary(raw, G5P_BIN_IADD, &f); break;
   case nir_op_imul: n = g5p_usc_binary(raw, G5P_BIN_IMUL, &f); break;
   case nir_op_iand: n = g5p_usc_binary(raw, G5P_BIN_AND, &f); break;
   case nir_op_ior: n = g5p_usc_binary(raw, G5P_BIN_OR, &f); break;
   case nir_op_ixor: n = g5p_usc_binary(raw, G5P_BIN_XOR, &f); break;
   case nir_op_umin: n = g5p_usc_minmax(raw, G5P_UMIN, &f); break;
   case nir_op_umax: n = g5p_usc_minmax(raw, G5P_UMAX, &f); break;
   case nir_op_imin: n = g5p_usc_minmax(raw, G5P_IMIN, &f); break;
   default: n = g5p_usc_minmax(raw, G5P_IMAX, &f); break;
   }
   put(c, raw, n, nir_op_infos[a->op].name);
   c->p->evidence |= PCO_G5P_EV_FS_ALU;
   define(c, &a->def, dst);
}

/* a - b: Apple's two-register integer subtract (op a5, f_udiv_reg; G9P's
 * G9P_ISUB, OrchardLinux/mesa g9p_isa.h), operands in their order;
 * ineg is 0 - x, as his emit_alu_op() does it. */
static void emit_isub(struct gctx *c, nir_alu_instr *a)
{
   bool neg = a->op == nir_op_ineg;
   struct ref r[2] = {neg ? (struct ref){0} : alu_src(c, a, 0), alu_src(c, a, neg ? 0 : 1)};
   unsigned ra = neg ? zero_reg(c) : reg_of(c, r[0], "isub");
   unsigned rb = reg_of(c, r[1], "isub");
   unsigned ga = neg ? kgrp(c, K_ZERO) : vgrp(c, r[0]), gb = vgrp(c, r[1]);
   int prefer = !neg && dies(c, r[0]) ? (int)ra : dies(c, r[1]) ? (int)rb : -1;
   free_dying(c, r + neg, 2 - neg);
   unsigned dst = alloc_reg(c, prefer);
   struct g5p_alu f = G5P_ALU(2 * dst);
   f.a = (struct g5p_src){.half = 2 * ra, .group = ga};
   f.b = (struct g5p_src){.half = 2 * rb, .group = gb};
   alu_hint(c, &f, false);
   uint8_t raw[G5P_USC_MAX];
   put(c, raw, g5p_usc_binary(raw, G5P_BIN_ISUB, &f), nir_op_infos[a->op].name);
   c->p->evidence |= PCO_G5P_EV_FS_ALU;
   define(c, &a->def, dst);
}

/* Shifts. A vertex program takes Apple's vertex form of the literal shifts
 * (bank 1, header 0x11; our decode of Apple's compile of the vertex probes
 * v_ishr2, v_ishr24, v_ushr2, v_ushr8, v_ushr24, v_shl8 in
 * tools/mesa-a9-vtx-re/probes.metal, byte-equal to these encoders). The
 * fragment form (bank 0, header 1) in a vertex program reads and writes
 * other registers: x >> 2 then read again gave x on hardware (vertex-input's
 * gl_VertexIndex rungs). Dynamic amounts have no vertex probe. */
static void emit_shift(struct gctx *c, nir_alu_instr *a)
{
   struct ref x = alu_src(c, a, 0), amt = alu_src(c, a, 1);
   uint8_t raw[G5P_USC_MAX];
   unsigned xr = reg_of(c, x, nir_op_infos[a->op].name);
   if (is_const(c, amt)) {
      unsigned s = V(c, amt.def, amt.comp)->imm & 31;
      bool d = dies(c, x);
      unsigned gx = vgrp(c, x);
      free_dying(c, &x, 1);
      unsigned dst = alloc_reg(c, d ? (int)xr : -1);
      const unsigned bank = vbank(c, dst), header = c->vs ? vhdr(c, 0x11, -1, xr, -1) : 1;
      if (a->op == nir_op_ishr)
         put(c, raw, g5p_usc_binary_literal(raw, G5P_BIN_ISHR, H(c, dst), s, H(c, xr), bank, gx, header, false, -1),
             "ishr literal");
      else
         put(c, raw, g5p_usc_shift(raw, a->op == nir_op_ishl, H(c, dst), H(c, xr), s, true, bank, gx, 1, header),
             "immediate shift");
      define(c, &a->def, dst);
      return;
   }
   if (c->vs) {
      /* Apple's v_ishl_dyn(_rr), v_ushr_dyn, v_ishr_dyn (tools/g5p-1b-dgl/vs-canary/vsalu-probes.py): the
       * amount & 31 (the u32 binary literal, the amount in B, header 0x11), then the register-amount shift
       * (value in B, the masked amount in C: header 0x19, bank 1) or, for ishr, the two-register binary (value
       * in A, amount in B: header 0x31). Each source group is 1 on its last read. */
      unsigned ar = reg_of(c, amt, "shift amount");
      unsigned ga = vgrp(c, amt);
      unsigned m = alloc_reg(c, -1);
      put(c, raw, g5p_usc_binary_literal2(raw, G5P_BIN_AND, H(c, m), 31, H(c, ar), vbank(c, m), ga,
                                          vhdr(c, 0x11, -1, ar, -1), G5P_LIT_U32, -1, 0, 0, false),
          "vertex amount & 31");
      struct ref both[2] = {x, amt};
      bool d = dies(c, x);
      unsigned gx = vgrp(c, x);
      free_dying(c, both, 2);
      unsigned dst = alloc_reg(c, d ? (int)xr : -1);
      if (a->op == nir_op_ishr) {
         struct g5p_alu f = G5P_ALU(H(c, dst));
         f.bank = vbank(c, dst);
         f.header = vhdr(c, 0x31, xr, m, -1);
         f.a = (struct g5p_src){.half = H(c, xr), .group = gx};
         f.b = (struct g5p_src){.half = H(c, m), .group = 1};
         put(c, raw, g5p_usc_binary(raw, G5P_BIN_ISHR, &f), "vertex dynamic ishr");
      } else {
         put(c, raw, g5p_usc_shift(raw, a->op == nir_op_ishl, H(c, dst), H(c, xr), H(c, m), false, vbank(c, dst), gx,
                                   1, vhdr(c, 0x19, -1, xr, m)),
             "vertex dynamic shift");
      }
      c->busy &= ~BITFIELD64_BIT(m);
      c->p->evidence |= PCO_G5P_EV_VS_ALU;
      define(c, &a->def, dst);
      return;
   }
   /* Apple's f_ishl_dyn, f_ushr_dyn and f_ishr_dyn
    * (tools/g5p-1b-dgl/vs-canary/vsalu-probes.py) first mask the amount
    * with a u32 literal 31.  Signed right shift then uses the ordinary
    * two-register ishr; the other two use the register-amount shift form. */
   unsigned ar = reg_of(c, amt, "shift amount");
   unsigned ga = vgrp(c, amt);
   unsigned mask = alloc_reg(c, -1);
   put(c,
       raw,
       g5p_usc_binary_literal2(raw,
                               G5P_BIN_AND,
                               2 * mask,
                               31,
                               2 * ar,
                               0,
                               ga,
                               1,
                               G5P_LIT_U32,
                               -1,
                               0,
                               0,
                               false),
       "amount & 31");
   struct ref both[2] = {x, amt};
   bool d = dies(c, x);
   unsigned gx = vgrp(c, x);
   free_dying(c, both, 2);
   unsigned dst = alloc_reg(c, d ? (int)xr : -1);
   if (a->op == nir_op_ishr) {
      struct g5p_alu f = G5P_ALU(2 * dst);
      f.a = (struct g5p_src){.half = 2 * xr, .group = gx};
      f.b = (struct g5p_src){.half = 2 * mask, .group = 1};
      put(c, raw, g5p_usc_binary(raw, G5P_BIN_ISHR, &f), "dynamic ishr");
   } else {
      put(c, raw, g5p_usc_shift(raw, a->op == nir_op_ishl, 2 * dst, 2 * xr, 2 * mask, false, 0, gx, 1, 1),
          "dynamic shift");
   }
   c->busy &= ~BITFIELD64_BIT(mask);
   c->p->evidence |= PCO_G5P_EV_FS_ALU;
   define(c, &a->def, dst);
}

static void emit_unary_like(struct gctx *c, nir_alu_instr *a)
{
   struct ref x = alu_src(c, a, 0);
   uint8_t raw[G5P_USC_MAX];
   unsigned xr = reg_of(c, x, nir_op_infos[a->op].name);
   bool d = dies(c, x);
   unsigned g = vgrp(c, x);
   free_dying(c, &x, 1);
   unsigned dst = alloc_reg(c, d ? (int)xr : -1);
   unsigned n;
   /* Repeated reads: convert (B slot) ext 4 + header 0x40, unary (A) ext 4. */
   unsigned cx = g ? 0 : 4, ch = g ? 1 : 0x41;
   /* the vertex unary form: the source in A */
   const unsigned uh = c->vs ? vhdr(c, 0x21, xr, -1, -1) : 1;
   switch (a->op) {
   /* vertex (v_f2i, v_f2u, v_f2i_rr: tools/g5p-1b-dgl/vs-canary/vsops2-probes.py): bank 1, header 0x11, the
    * source group 1 on the last read, no extension, as v_base's u2f32 */
   case nir_op_f2i32:
      n = c->vs ? g5p_usc_convert(raw, G5P_F2I32, H(c, dst), H(c, xr), vbank(c, dst), g, vhdr(c, 0x11, -1, xr, -1))
                : g5p_usc_convert2(raw, G5P_F2I32, H(c, dst), H(c, xr), 0, g, ch, cx, 0);
      break;
   case nir_op_f2u32:
      n = c->vs ? g5p_usc_convert(raw, G5P_F2U32, H(c, dst), H(c, xr), vbank(c, dst), g, vhdr(c, 0x11, -1, xr, -1))
                : g5p_usc_convert2(raw, G5P_F2U32, H(c, dst), H(c, xr), 0, g, ch, cx, 0);
      break;
   case nir_op_i2f32: /* vertex (v_i2f): bank 1, header 0x10 (B) */
      n = g5p_usc_convert2(raw, G5P_I2F32, H(c, dst), H(c, xr), vbank(c, dst), g, ch | (vbank(c, xr) ? VS_B : 0),
                           cx, 0);
      break;
   case nir_op_u2f32:
      /* v_base: u2f32 in the vertex stage uses header 0x11. */
      n = c->vs ? g5p_usc_convert(raw, G5P_U2F32, H(c, dst), H(c, xr), vbank(c, dst), g, vhdr(c, 0x11, -1, xr, -1))
                : g5p_usc_convert2(raw, G5P_U2F32, H(c, dst), H(c, xr), 0, g, ch, cx, 0);
      break;
   case nir_op_ffloor: /* vertex (v_floor): bank 1, header 0x20 (A) */
      n = g5p_usc_unary2(raw, G5P_FLOOR, H(c, dst), H(c, xr), vbank(c, dst), g, uh, false, cx, 0, false);
      break;
   case nir_op_flog2: /* vertex (v_log2): bank 1, header 0x20 (A) */
      n = g5p_usc_unary2(raw, G5P_LOG2, H(c, dst), H(c, xr), vbank(c, dst), g, uh, false, cx, 0, false);
      break;
   case nir_op_fexp2: /* vertex (v_exp2) */
      n = g5p_usc_unary2(raw, G5P_EXP2, H(c, dst), H(c, xr), vbank(c, dst), g, uh, false, cx, 0, false);
      break;
   case nir_op_inot: /* vertex (v_inot32): bank 1, header 0x20 (A) */
      n = g5p_usc_unary2(raw, G5P_NOT, H(c, dst), H(c, xr), vbank(c, dst), g, uh, false, cx, 0, false);
      break;
   case nir_op_bit_count: n = g5p_usc_unary2(raw, G5P_POPCOUNT, 2 * dst, 2 * xr, 0, g, 1, false, cx, 0, false); break;
   default: n = g5p_usc_unary2(raw, G5P_BITREVERSE, 2 * dst, 2 * xr, 0, g, 1, false, cx, 0, false); break;
   }
   put(c, raw, n, nir_op_infos[a->op].name);
   c->p->evidence |= c->vs ? PCO_G5P_EV_VS_ALU : PCO_G5P_EV_FS_ALU;
   define(c, &a->def, dst);
}

static void vs_int(struct gctx *c, nir_alu_instr *a);
static void vs_bitwise(struct gctx *c, nir_alu_instr *a);
static void vs_minsub(struct gctx *c, nir_alu_instr *a);

/* A value a vertex output stores, directly, through a mov/vec, or as the
 * value of a NIR register an output stores (control flow: Apple's v_loop
 * keeps such an accumulator in bank 1). */
static bool vs_feeds_output(struct gctx *c, nir_def *d)
{
   nir_foreach_use(u, d) {
      nir_instr *i = nir_src_use_instr(u);
      if (i->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(i);
         if (in->intrinsic == nir_intrinsic_store_output)
            return true;
         if (in->intrinsic == nir_intrinsic_store_reg && u == &in->src[0])
            for (unsigned k = 0; k < c->ndecls; k++)
               if (c->decls[k] == in->src[1].ssa && c->decl_out[k])
                  return true;
      }
      if (i->type == nir_instr_type_alu) {
         nir_alu_instr *m = nir_instr_as_alu(i);
         if (nir_op_is_vec_or_mov(m->op) && vs_feeds_output(c, &m->def))
            return true;
      }
   }
   return false;
}

static void emit_alu_body(struct gctx *c, nir_alu_instr *a);

static void emit_alu(struct gctx *c, nir_alu_instr *a)
{
   c->alloc_out = c->vs && c->vs_bank0 && vs_feeds_output(c, &a->def);
   emit_alu_body(c, a);
   c->alloc_out = false;
}

static void emit_alu_body(struct gctx *c, nir_alu_instr *a)
{
   if (alias_op(c, a))
      return;
   if (c->vs && address_only(a))
      return;
   for (unsigned k = 0; k < 4; k++)
      if (c->narrow_sat[k] == a)
         return; /* the saturating narrow iterator */
   if (!V(c, &a->def, 0)->uses && c->rh_half[a->def.index] < 0)
      return; /* only folded into float consumers */
   if (is_cmp(a->op) || a->op == nir_op_bcsel || a->op == nir_op_b2f32 || a->op == nir_op_b2i32 ||
       (a->op == nir_op_inot && a->def.bit_size == 1)) {
      emit_bool_alu(c, a);
      return;
   }
   if (is_recipe(a->op)) {
      emit_recipe(c, a);
      return;
   }
   if (c->vs && (a->op == nir_op_iand || a->op == nir_op_ior || a->op == nir_op_ixor)) {
      vs_bitwise(c, a);
      return;
   }
   if (c->vs && (a->op == nir_op_iadd || a->op == nir_op_imul ||
                 (a->op == nir_op_ishl && nir_src_is_const(a->src[1].src)) || a->op == nir_op_inot)) {
      vs_int(c, a);
      return;
   }
   if (c->vs && (a->op == nir_op_imin || a->op == nir_op_imax || a->op == nir_op_umin ||
                 a->op == nir_op_umax || a->op == nir_op_isub || a->op == nir_op_ineg)) {
      vs_minsub(c, a);
      return;
   }
   switch (a->op) {
   case nir_op_fneg:
   case nir_op_fabs:
      emit_times_one(c, resolve_float(c, &a->def, 0), false, -1, &a->def);
      break;
   case nir_op_fsat:
      emit_times_one(c, resolve_float(c, a->src[0].src.ssa, a->src[0].swizzle[0]), true, -1, &a->def);
      break;
   case nir_op_fadd:
   case nir_op_fsub:
   case nir_op_fmul:
      emit_fbin(c, a);
      break;
   case nir_op_ffma:
      emit_ffma(c, a);
      break;
   case nir_op_iadd: case nir_op_imul: case nir_op_iand: case nir_op_ior: case nir_op_ixor:
   case nir_op_imin: case nir_op_imax: case nir_op_umin: case nir_op_umax:
      emit_ibin(c, a);
      break;
   case nir_op_ishl: case nir_op_ushr: case nir_op_ishr:
      emit_shift(c, a);
      break;
   case nir_op_isub: case nir_op_ineg:
      emit_isub(c, a);
      break;
   default:
      emit_unary_like(c, a);
   }
}

/* Vertex integer ALU, the transform feedback address arithmetic. Only the
 * forms of Apple's vertex probes (tools/mesa-a9-xfb-re): register pairs
 * with header 0x31 (v_iadd_rr, v_imul_rr), a shared operand in A with
 * header 0x11 (v_iadd_k, v_imul_k), a literal addend (v_iadd_lit*), a
 * multiply by 2..16 or a shift by 1..4 as Apple's shift-add (v_shl4) and
 * NOT (v_not). Other literal multipliers are shift/add chains
 * (vs_mul_literal); only huge ones are materialized by MOVI. As in the other
 * vertex forms, a dying operand sits in B and gives its register. */
/* x * k for a literal k outside the single shift-add (2..16). Apple's compiler
 * builds such products from shift-adds (v_imul_lit: x·5 << 2, v_imul_lit48:
 * x·3 << 4); a MOVI multiplier in a register-pair IMUL gave 0 on hardware
 * (vertex-temps.md, x1: the XFB stride-32 offset). Horner over k's bits with
 * flown forms only: shifts of 1..4 (v_shl4) and register-pair adds
 * (v_iadd_rr), x in A (read again) and the dying partial product in B. */
static bool vs_mul_literal(struct gctx *c, nir_alu_instr *a, struct ref x, uint32_t k)
{
   if (k < 2 || util_bitcount(k) + DIV_ROUND_UP(util_last_bit(k) - 1, 4) > 8)
      return false;
   uint8_t raw[G5P_USC_MAX];
   unsigned xr = reg_of(c, x, "multiply"), acc = xr, pend = 0;
   bool acc_is_x = true;
   for (int b = util_last_bit(k) - 2; b >= -1; b--) {
      if (b >= 0) {
         pend++;
         if (!(k & (1u << b)))
            continue;
      }
      while (pend) {
         unsigned sh = MIN2(pend, 4), dst = acc_is_x ? alloc_reg(c, -1) : acc;
         /* x is read again by the adds; the partial product is rewritten in place. */
         unsigned g = acc_is_x ? vgrp(c, x) : 1;
         vs_bank1_only(c, "vertex shift-add", acc, -1, -1, -1);
         put(c, raw, g5p_usc_shl_literal(raw, H(c, dst), H(c, acc), vbank(c, dst), g, sh), "shift-add");
         acc = dst;
         acc_is_x = false;
         pend -= sh;
      }
      if (b >= 0) {
         struct g5p_alu f = G5P_ALU(H(c, acc));
         f.bank = vbank(c, acc);
         f.header = vhdr(c, 0x31, xr, acc, -1);
         f.a = gsrc(c, xr, (struct ref){0}, vgrp(c, x));
         f.b = gsrc(c, acc, (struct ref){0}, 1);
         put(c, raw, g5p_usc_binary(raw, G5P_BIN_IADD, &f), "iadd");
      }
   }
   free_dying(c, &x, 1);
   define(c, &a->def, acc);
   return true;
}

/* Vertex iand/ior/ixor (bench-gaps.md): Apple's v_iand, v_ior, v_ixor are the
 * register-pair form of v_iadd_rr (bank 1, header 0x31). A 1-bit one combines
 * materialized 1.0/0.0 booleans (glmark2 ideas: `a == 0.0 && b == 0.0`). */
static void vs_bitwise(struct gctx *c, nir_alu_instr *a)
{
   uint8_t raw[G5P_USC_MAX];
   struct ref r[2] = {alu_src(c, a, 0), alu_src(c, a, 1)};
   enum g5p_binop bop = a->op == nir_op_iand ? G5P_BIN_AND : a->op == nir_op_ior ? G5P_BIN_OR : G5P_BIN_XOR;
   /* A 32-bit literal operand: Apple's v_iand_lit, v_iand_big, v_ior_big, v_ixor_big(_rr) (tools/g5p-1b-dgl/
    * vs-canary/vsops2-probes.py): the u32 binary literal with bank 1, header 0x11, the register in B, its source
    * group alone marking a repeated read; short or long by the value. */
   for (unsigned i = 0; i < 2 && a->def.bit_size == 32; i++) {
      if (!is_const(c, r[i]) || is_const(c, r[1 - i]) || vkind(c, r[1 - i]) != VK_REG)
         continue;
      uint32_t k = V(c, r[i].def, r[i].comp)->imm;
      struct ref x = r[1 - i];
      unsigned xr = reg_of(c, x, "vertex bitwise literal");
      unsigned g = vgrp(c, x);
      unsigned dst = vs_dst(c, &x, 1);
      put(c, raw, g5p_usc_binary_literal2(raw, bop, H(c, dst), k, H(c, xr), vbank(c, dst), g, vhdr(c, 0x11, -1, xr, -1),
                                          G5P_LIT_U32, -1, 0, 0, false),
          "vertex bitwise literal");
      c->p->evidence |= PCO_G5P_EV_VS_ALU;
      define(c, &a->def, dst);
      return;
   }
   /* v_iand, v_ior, v_ixor: the register-pair form of v_iadd_rr. A
    * common-store operand has no probed vertex form. */
   if (vkind(c, r[0]) != VK_REG || vkind(c, r[1]) != VK_REG || is_const(c, r[0]) || is_const(c, r[1]))
      refuse(c, "vertex %s with a uniform operand or a 1-bit literal (the register-pair and 32-bit literal forms "
             "are probed)", nir_op_infos[a->op].name);
   unsigned reg[2] = {reg_of(c, r[0], "vertex bitwise"), reg_of(c, r[1], "vertex bitwise")};
   bool d0 = dies(c, r[0]), d1 = dies(c, r[1]);
   unsigned ia = d0 ? 1 : 0, ib = d0 ? 0 : 1;
   unsigned ga = vgrp(c, r[ia]), gb = vgrp(c, r[ib]);
   free_dying(c, r, 2);
   unsigned dst = alloc_reg(c, d0 ? (int)reg[0] : d1 ? (int)reg[1] : -1);
   struct g5p_alu f = G5P_ALU(H(c, dst));
   f.bank = vbank(c, dst);
   f.header = vhdr(c, 0x31, reg[ia], reg[ib], -1);
   f.a = gsrc(c, reg[ia], (struct ref){0}, ga);
   f.b = gsrc(c, reg[ib], (struct ref){0}, gb);
   put(c, raw, g5p_usc_binary(raw, bop, &f), nir_op_infos[a->op].name);
   c->p->evidence |= PCO_G5P_EV_VS_ALU;
   define(c, &a->def, dst);
   return;
}

/* Vertex integer min/max, subtract and negate in Apple's vertex forms, our
 * decode of the m_* probes in tools/mesa-a9-vtx-re/probes.metal (Apple's
 * applegpu_g5p compile, byte-equal to these encoders): register pairs with
 * bank 1, header 0x31 (m_imin_rr, m_isub_rr); a literal in A with header 0x11
 * (m_imin_lit, m_imax_neg, m_isub_lit, m_isub_big; ineg is 0 - x, m_ineg);
 * a common-store operand in A (m_imin_u; m_isub_u is u - x, m_isub_u2 x - u
 * by routing 0x70). The fragment form (bank 0, header 1) in a vertex program
 * reads other registers (the vertex shift, device run 1). x - k is an iadd of
 * -k (Apple's v_iadd_lit form). */
static void vs_minsub(struct gctx *c, nir_alu_instr *a)
{
   uint8_t raw[G5P_USC_MAX];
   const bool neg = a->op == nir_op_ineg, sub = neg || a->op == nir_op_isub;
   const enum g5p_minmax mop = a->op == nir_op_umin ? G5P_UMIN : a->op == nir_op_umax ? G5P_UMAX
                              : a->op == nir_op_imin ? G5P_IMIN : G5P_IMAX;
   const char *name = nir_op_infos[a->op].name;
   c->p->evidence |= PCO_G5P_EV_VS_ALU;
   if (neg) {
      struct ref x = alu_src(c, a, 0);
      unsigned xr = reg_of(c, x, name);
      unsigned g = vgrp(c, x);
      unsigned dst = vs_dst(c, &x, 1);
      put(c, raw, g5p_usc_isub_literal(raw, H(c, dst), 0, H(c, xr), vbank(c, dst), g, vhdr(c, 0x11, -1, xr, -1)),
          "ineg");
      define(c, &a->def, dst);
      return;
   }
   struct ref r[2] = {alu_src(c, a, 0), alu_src(c, a, 1)};
   for (unsigned i = 0; i < 2; i++) {
      if (!is_const(c, r[i]) || is_const(c, r[1 - i]))
         continue;
      uint32_t k = V(c, r[i].def, r[i].comp)->imm;
      struct ref x = r[1 - i];
      unsigned xr = reg_of(c, x, name);
      unsigned g = vgrp(c, x);
      unsigned dst = vs_dst(c, &x, 1);
      unsigned n;
      if (sub && i == 1) {
         n = g5p_usc_iadd_literal2(raw, H(c, dst), -k, H(c, xr), vbank(c, dst), g, vhdr(c, 0x11, -1, xr, -1));
      }
      else if (sub)
         n = g5p_usc_isub_literal(raw, H(c, dst), k, H(c, xr), vbank(c, dst), g, vhdr(c, 0x11, -1, xr, -1));
      else
         n = g5p_usc_minmax_literal(raw, mop, H(c, dst), k, H(c, xr), vbank(c, dst), g, vhdr(c, 0x11, -1, xr, -1));
      put(c, raw, n, sub && i == 1 ? "isub literal (iadd)" : sub ? "isub literal" : "min/max literal");
      define(c, &a->def, dst);
      return;
   }
   for (unsigned i = 0; i < 2; i++) {
      if (vkind(c, r[i]) != VK_SHARED || vkind(c, r[1 - i]) != VK_REG)
         continue;
      unsigned sd = V(c, r[i].def, r[i].comp)->imm;
      struct ref x = r[1 - i];
      unsigned xr = reg_of(c, x, name);
      unsigned g = vgrp(c, x);
      unsigned dst = vs_dst(c, &x, 1);
      vs_bank1_only(c, sub ? "vertex shared isub" : "vertex shared min/max", dst, xr, -1, -1);
      unsigned n = sub ? g5p_usc_alu_shared(raw, G5P_BIN_ISUB, H(c, dst), sd, H(c, xr), vbank(c, dst), g,
                                            vhdr(c, 0x11, -1, xr, -1), 0, 0, false,
                                            i ? 0x70 : -1, 1)
                       : g5p_usc_minmax_shared(raw, mop, H(c, dst), sd, H(c, xr), vbank(c, dst), g,
                                               vhdr(c, 0x11, -1, xr, -1), 1);
      put(c, raw, n, sub ? "shared isub" : "shared min/max");
      define(c, &a->def, dst);
      return;
   }
   unsigned reg[2] = {reg_of(c, r[0], name), reg_of(c, r[1], name)};
   bool d0 = dies(c, r[0]), d1 = dies(c, r[1]);
   /* min/max: a dying operand in B, as the other vertex pairs; a - b keeps its order */
   unsigned ia = !sub && d0 ? 1 : 0, ib = 1 - ia;
   unsigned ga = vgrp(c, r[ia]), gb = vgrp(c, r[ib]);
   free_dying(c, r, 2);
   unsigned dst = alloc_reg(c, d0 ? (int)reg[0] : d1 ? (int)reg[1] : -1);
   struct g5p_alu f = G5P_ALU(H(c, dst));
   f.bank = vbank(c, dst);
   f.header = vhdr(c, 0x31, reg[ia], reg[ib], -1);
   f.a = gsrc(c, reg[ia], (struct ref){0}, ga);
   f.b = gsrc(c, reg[ib], (struct ref){0}, gb);
   put(c, raw, sub ? g5p_usc_binary(raw, G5P_BIN_ISUB, &f) : g5p_usc_minmax(raw, mop, &f), name);
   define(c, &a->def, dst);
}

static void vs_int(struct gctx *c, nir_alu_instr *a)
{
   uint8_t raw[G5P_USC_MAX];
   struct ref r[2] = {alu_src(c, a, 0), {0}};
   c->p->evidence |= PCO_G5P_EV_VS_STORE;
   if (a->op == nir_op_inot) {
      unsigned xr = reg_of(c, r[0], "not");
      unsigned g = vgrp(c, r[0]);
      unsigned dst = vs_dst(c, r, 1);
      put(c, raw, g5p_usc_unary2(raw, G5P_NOT, H(c, dst), H(c, xr), vbank(c, dst), g, vhdr(c, 0x21, xr, -1, -1), false,
                                 0, 0, false), "not");
      define(c, &a->def, dst);
      return;
   }
   r[1] = alu_src(c, a, 1);
   enum g5p_binop op = a->op == nir_op_iadd ? G5P_BIN_IADD : G5P_BIN_IMUL;
   int shift = -1;
   if (a->op == nir_op_ishl) {
      unsigned s = V(c, r[1].def, r[1].comp)->imm & 31;
      if (s >= 1 && s <= 4)
         shift = s;
      else
         r[1] = (struct ref){0}; /* multiply by 1 << s below */
   }
   for (unsigned i = 0; i < 2 && a->op != nir_op_ishl; i++) {
      if (!is_const(c, r[i]) || is_const(c, r[1 - i]))
         continue;
      uint32_t k = V(c, r[i].def, r[i].comp)->imm;
      if (a->op == nir_op_iadd) {
         struct ref x = r[1 - i];
         unsigned xr = reg_of(c, x, "iadd literal");
         unsigned g = vgrp(c, x);
         unsigned dst = vs_dst(c, &x, 1);
         /* bank 0 too: Apple's v_loop counter increment (header 0x01) */
         put(c, raw, g5p_usc_iadd_literal2(raw, H(c, dst), k, H(c, xr), vbank(c, dst), g, vhdr(c, 0x11, -1, xr, -1)),
             "iadd literal");
         define(c, &a->def, dst);
         return;
      }
      if (util_is_power_of_two_nonzero(k) && util_logbase2(k) >= 1 && util_logbase2(k) <= 4) {
         shift = util_logbase2(k);
         r[0] = r[1 - i];
      } else if (vs_mul_literal(c, a, r[1 - i], k)) {
         return;
      }
      break;
   }
   if (shift > 0) {
      struct ref x = r[0];
      unsigned xr = reg_of(c, x, "shift");
      unsigned g = vgrp(c, x);
      unsigned dst = vs_dst(c, &x, 1);
      vs_bank1_only(c, "vertex shift-add", xr, -1, -1, -1);
      put(c, raw, g5p_usc_shl_literal(raw, H(c, dst), H(c, xr), vbank(c, dst), g, shift), "shift-add");
      define(c, &a->def, dst);
      return;
   }
   if (a->op == nir_op_ishl) {
      /* x << s, s outside 1..4: shifts of up to 4, else x * (1 << s) with the
       * multiplier in a register. */
      unsigned s = V(c, alu_src(c, a, 1).def, alu_src(c, a, 1).comp)->imm & 31;
      if (s && vs_mul_literal(c, a, r[0], 1u << s))
         return;
      unsigned xr = reg_of(c, r[0], "shift");
      unsigned kr = scratch_imm(c, 1u << s);
      unsigned g = vgrp(c, r[0]);
      unsigned dst = vs_dst(c, r, 1);
      struct g5p_alu f = G5P_ALU(H(c, dst));
      f.bank = vbank(c, dst);
      f.header = vhdr(c, 0x31, kr, xr, -1);
      f.a = gsrc(c, kr, (struct ref){0}, 1);
      f.b = gsrc(c, xr, r[0], g);
      put(c, raw, g5p_usc_binary(raw, G5P_BIN_IMUL, &f), "imul");
      define(c, &a->def, dst);
      return;
   }
   for (unsigned i = 0; i < 2; i++) {
      if (vkind(c, r[i]) != VK_SHARED || vkind(c, r[1 - i]) != VK_REG)
         continue;
      struct ref x = r[1 - i];
      unsigned sd = V(c, r[i].def, r[i].comp)->imm;
      unsigned xr = reg_of(c, x, "shared integer ALU");
      unsigned g = vgrp(c, x);
      unsigned dst = vs_dst(c, &x, 1);
      vs_bank1_only(c, "vertex shared integer ALU", dst, xr, -1, -1);
      put(c, raw, g5p_usc_alu_shared(raw, op, H(c, dst), sd, H(c, xr), vbank(c, dst), g, vhdr(c, 0x11, -1, xr, -1), 0,
                                     0, false, -1, 1),
          "shared integer ALU");
      define(c, &a->def, dst);
      return;
   }
   unsigned reg[2] = {reg_of(c, r[0], "integer ALU"), reg_of(c, r[1], "integer ALU")};
   bool d0 = dies(c, r[0]), d1 = dies(c, r[1]);
   unsigned ia = d0 ? 1 : 0, ib = d0 ? 0 : 1;
   unsigned ga = vgrp(c, r[ia]), gb = vgrp(c, r[ib]);
   free_dying(c, r, 2);
   unsigned dst = alloc_reg(c, d0 ? (int)reg[0] : d1 ? (int)reg[1] : -1);
   struct g5p_alu f = G5P_ALU(H(c, dst));
   f.bank = vbank(c, dst);
   f.header = vhdr(c, 0x31, reg[ia], reg[ib], -1);
   f.a = gsrc(c, reg[ia], (struct ref){0}, ga);
   f.b = gsrc(c, reg[ib], (struct ref){0}, gb);
   put(c, raw, g5p_usc_binary(raw, op, &f), nir_op_infos[a->op].name);
   define(c, &a->def, dst);
}

/* Transform feedback store, one dword: vertex tasks commit only the 4-byte
 * form (xfb.md, device run 2), so lower() splits every record. The record's
 * first store adds the buffer's pointer pair and the record offset; its
 * other dwords reuse that pair with the store's immediate dword offset, as
 * Apple's compiler folds +4/+8/+12 (v_st_bytes, v_st_stride12, v_st_off12).
 * Apple never rewrites a store's data or address registers after the store
 * without a csync between; the stores are the program's last instructions
 * (sink_vs_stores) and their registers stay reserved to the end. */
static void vs_store(struct gctx *c, nir_intrinsic_instr *in)
{
   uint8_t raw[G5P_USC_MAX];
   unsigned b, k;
   struct ref off;
   if (store_refusal(in))
      refuse(c, "%s", store_refusal(in));
   xfb_address(in->src[1].ssa, &b, &off, &k);
   bool first;
   int i = store_base(c, b, off, &first);
   if (c->st_base[i].pair < 0) {
      unsigned oreg = reg_of(c, off, "store offset");
      unsigned og = vgrp(c, off);
      unsigned addr = alloc_run(c, 2, "store address");
      vs_bank1_only(c, "transform feedback address add", addr, addr + 1, oreg, -1);
      put(c, raw, g5p_usc_address_add(raw, H(c, addr), c->xfb_common + PCO_G5P_XFB_ADDR(b), 1, H(c, oreg), og, 0),
          "address add");
      c->st_base[i].pair = addr;
      c->reserved |= BITFIELD64_RANGE(addr, 2);
   }
   struct ref r = resolve_value(c, in->src[0].ssa, 0);
   unsigned data = reg_of(c, r, "store data");
   vs_bank1_only(c, "transform feedback store", data, -1, -1, -1);
   put(c, raw, g5p_usc_global_store(raw, 4, H(c, data), H(c, c->st_base[i].pair), k), "store");
   c->reserved |= BITFIELD64_BIT(data);
   c->p->evidence |= PCO_G5P_EV_VS_STORE;
}

/* ---- intrinsics and textures ---- */
/* Coefficient set of a varying load's first component. */
static unsigned varying_slot(struct gctx *c, nir_intrinsic_instr *in)
{
   return vary_of(c, in)->set + nir_intrinsic_component(in);
}

/* Dependent-read counters (Apple's applegpu_g5p output, varyings.md): a
 * batch issues up to 8 coefficient reads on counter 0, then up to 8 with b0
 * bit 0 (counter 1), and waits for each counter used (`4d00`, `4d02`)
 * before any result is read. */
struct drc {
   unsigned n[2];
};

static void drc_flush(struct gctx *c, struct drc *d)
{
   uint8_t raw[G5P_USC_MAX];
   if (d->n[0])
      put(c, raw, g5p_usc_wait(raw, 0), "wait");
   if (d->n[1])
      put(c, raw, g5p_usc_wait(raw, 2), "wait (counter 1)");
   d->n[0] = d->n[1] = 0;
}

/* Counter of the next read (true: counter 1); a full batch is waited for first. */
static bool drc_take(struct gctx *c, struct drc *d)
{
   if (d->n[0] == 8 && d->n[1] == 8)
      drc_flush(c, d);
   if (d->n[0] < 8) {
      d->n[0]++;
      return false;
   }
   d->n[1]++;
   return true;
}

/* Iterate (smooth/npc) or load (flat) the used components of one varying
 * load into registers. */
static void read_varying(struct gctx *c, nir_intrinsic_instr *in, struct drc *d)
{
   uint8_t raw[G5P_USC_MAX];
   const struct pco_g5p_varying *v = vary_of(c, in);
   unsigned n = in->def.num_components, slot = varying_slot(c, in);
   bool any = false;
   for (unsigned j = 0; j < n; j++)
      any |= V(c, &in->def, j)->uses > 0;
   if (!any)
      return;
   /* A flat set past the u8 C-term load (coefficient dword >= 256: the
    * 16th location's last components) is iterated without perspective: the
    * flat DOUTI shade model gives a constant plane (A = B = 0), so
    * A*x + B*y + C is exactly C (varyings-capacity-1: c124f, 31 flat
    * locations, exact on the A9). */
   const bool flat_iterated = v->interp == PCO_G5P_INTERP_FLAT && 4 * (slot + n - 1) + 2 >= 256;
   if (v->interp == PCO_G5P_INTERP_FLAT && !flat_iterated) {
      /* The C term of each component's coefficient set (Apple's `14 23`). */
      for (unsigned j = 0; j < n; j++) {
         struct val *val = V(c, &in->def, j);
         if (!val->uses)
            continue;
         bool drc1 = drc_take(c, d);
         unsigned dst = alloc_run(c, 1, "flat varying");
         put(c, raw, g5p_usc_coeff_load(raw, 2 * dst, 4 * (slot + j) + 2, drc1), "flat varying");
         val->state = VAL_REG;
         val->reg = dst;
         val->held = true;
      }
      c->p->evidence |= PCO_G5P_EV_ITERATOR;
      return;
   }
   unsigned flags = (v->interp == PCO_G5P_INTERP_NOPERSPECTIVE || flat_iterated ? G5P_ITER_LINEAR : 0) |
                    (v->centroid && !flat_iterated ? G5P_ITER_CENTROID : 0);
   if (drc_take(c, d))
      flags |= G5P_ITER_DRC1;
   unsigned dst = alloc_run(c, n, "varying");
   put(c, raw, g5p_usc_iterator_flags(raw, 2 * dst, slot, n, false, false, flags), "iterator");
   for (unsigned j = 0; j < n; j++) {
      struct val *val = V(c, &in->def, j);
      val->state = VAL_REG;
      val->reg = dst + j;
      val->held = true;
   }
   c->p->evidence |= PCO_G5P_EV_ITERATOR;
}

/* pair = UBO pointer + (index << shift); load 4 or 16 bytes from it. */
/* Fragment register-indexed UBO loads compile with their index clamped to
 * the block (g5p_nir.c clamp_dyn_ubo); PVR_G5P_FS_DYN_UBO=0 refuses them
 * (the stage-1 behaviour, for A/B). */
bool pco_g5p_fs_dyn_ubo(void)
{
   const char *e = getenv("PVR_G5P_FS_DYN_UBO");
   return !(e && !strcmp(e, "0"));
}

/* The index is clamp_dyn_ubo's umin(i, last), a constant bound inside the block. */
static bool clamped_index(struct ref ix)
{
   nir_alu_instr *a = alu_of(ix.def);
   if (!a || a->op != nir_op_umin)
      return false;
   return nir_src_is_const(a->src[0].src) || nir_src_is_const(a->src[1].src);
}

static void emit_dyn_ubo(struct gctx *c, nir_intrinsic_instr *in)
{
   uint8_t raw[G5P_USC_MAX];
   unsigned set, binding, element, shift, n = in->def.num_components;
   struct ref ix;
   uint32_t base;
   block_binding_element(in->src[0].ssa, &set, &binding, &element);
   dyn_ubo_offset(in, &ix, &shift, &base);
   unsigned bytes = 4 * n;
   /* Witnessed: 16-byte loads (Apple's v_cexpr, vec4 elements at shift 4) and
    * 4-byte loads (G9P's, at any byte offset); the shift field is
    * corpus-verified 0-4. vec2/vec3 (8-byte, 12-byte) loads are not. */
   if (in->def.bit_size != 32 || !((n == 4 && shift == 4) || (n == 1 && shift != 1)) || base % bytes)
      refuse(c, "load_ubo: register-indexed %u x %u-bit at shift %u, base %u (witnessed: vec4 at shift 4, "
                "scalar at shift 0/2-4, element-aligned base)", n, in->def.bit_size, shift, base);
   /* Apple's form for an unsigned index is this one: umin, then addr.add_shift
    * of the pointer pair and a 16-byte load (Metal probe f_ubo_uint_idx,
    * linux-a9 tools/mesa-a9-hyprland-gpu/ubo-probes.py). add_shift does not
    * sign-extend: a negative index (Hyprland's border, floor(progress *
    * (len - 1)) in helper lanes) read 4 GiB past the pointer and faulted the
    * 3D phase (stage 1 job 30). Apple's signed form is a 32-bit offset load
    * we do not encode, so every fragment index must be clamp_dyn_ubo's
    * umin(i, last): any value, negative included, then stays inside the
    * block (robust access). An index the clamp could not bound is refused. */
   if (!c->vs && !pco_g5p_fs_dyn_ubo())
      refuse(c, "register-indexed UBO load (PVR_G5P_FS_DYN_UBO=0)");
   if (!c->vs && !clamped_index(ix))
      refuse(c, "register-indexed UBO load with an index not clamped to its block (a negative index would "
                "fault: add_shift does not sign-extend)");
   unsigned common = 0;
   for (unsigned i = 0; i < c->nuptr; i++)
      if (c->uptr[i].set == set && c->uptr[i].binding == binding &&
          c->uptr[i].element == element && c->uptr[i].base == base)
         common = c->uptr[i].common;
   unsigned index = reg_of(c, ix, "register-indexed UBO load index");
   if (c->vs)
      vs_bank1_only(c, "vertex register-indexed UBO load", index, -1, -1, -1);
   bool kill = dies(c, ix);
   /* Registers are read at issue: like Apple's v_cexpr (index r0 -> pair
    * r0:r1 -> data r0..r3), the pair may take a dying index's register and
    * the data the pair's (the VS budget is 12 temps, G5P_VS_TEMPS_FLOWN). */
   if (kill)
      drop(c, V(c, ix.def, ix.comp));
   unsigned pair = 0;
   for (pair = 0; pair + 2 <= NUM_DWORDS; pair += 2)
      if (!((c->busy | c->reserved) & BITFIELD64_RANGE(pair, 2)))
         break;
   if (pair + 2 > NUM_DWORDS)
      refuse(c, "register pressure: an address pair");
   c->max_dword = MAX2(c->max_dword, pair + 2);
   put(c, raw, g5p_usc_address_add_shift(raw, 2 * pair, common, 2 * index, shift, kill, c->vs), "addr.add_shift");
   unsigned dst = 0;
   for (dst = 0; dst + n <= NUM_DWORDS; dst += n)
      if (!((c->busy | c->reserved) & BITFIELD64_RANGE(dst, n)))
         break;
   if (dst + n > NUM_DWORDS)
      refuse(c, "register pressure: load_ubo needs %u aligned dword(s)", n);
   c->busy |= BITFIELD64_RANGE(dst, n);
   c->max_dword = MAX2(c->max_dword, dst + n);
   if (c->vs)
      put(c, raw, g5p_usc_vertex_address_load(raw, 2 * dst, bytes, 2 * pair), "vertex address load");
   else
      put(c, raw, g5p_usc_address_load(raw, 2 * dst, bytes, 2 * pair, 0), "fragment address load");
   put(c, raw, g5p_usc_wait(raw, 0), "wait");
   for (unsigned k = 0; k < n; k++) {
      struct val *v = V(c, &in->def, k);
      v->state = VAL_REG;
      v->reg = dst + k;
      v->held = true;
   }
   c->p->evidence |= PCO_G5P_EV_DYN_UBO;
}

/* A constant-offset load from a UBO range whose contiguous span would not
 * fit in the common store. The pointer uploaded at r->common already includes
 * r->lo. Materialise the remaining byte offset, add it to that pointer and
 * issue the same addressed load used by Apple's dynamic UBO programs. */
static void emit_memory_ubo(struct gctx *c, nir_intrinsic_instr *in,
                            const struct range *r, unsigned off)
{
   const unsigned n = in->def.num_components;
   if (in->def.bit_size != 32 || !n || n > 4)
      refuse(c, "memory UBO load: only 1..4 aligned 32-bit components");

   const unsigned dst = alloc_run(c, n, "memory UBO load");
   if (c->vs)
      vs_bank1_only(c, "vertex memory UBO load", dst, -1, -1, -1);
   const unsigned first = off - r->lo;
   const bool vector = n == 4 && !(first & 15);
   const unsigned loads = vector ? 1 : n;
   uint8_t raw[G5P_USC_MAX];

   for (unsigned i = 0; i < loads; i++) {
      const unsigned bytes = vector ? 16 : 4;
      const unsigned at = first + (vector ? 0 : 4 * i);
      const unsigned index = scratch_imm(c, at);
      unsigned pair;

      for (pair = 0; pair + 2 <= NUM_DWORDS; pair += 2)
         if (!((c->busy | c->reserved) & BITFIELD64_RANGE(pair, 2)))
            break;
      if (pair + 2 > NUM_DWORDS)
         refuse(c, "memory UBO load: register pressure for address pair");
      c->busy |= BITFIELD64_RANGE(pair, 2);
      c->max_dword = MAX2(c->max_dword, pair + 2);
      c->max_written = MAX2(c->max_written, pair + 2);

      put(c, raw,
          g5p_usc_address_add_shift(raw, 2 * pair, r->common, 2 * index,
                                    0, true, c->vs),
          "constant UBO address");
      if (c->vs)
         put(c, raw,
             g5p_usc_vertex_address_load(raw, 2 * (dst + (vector ? 0 : i)),
                                         bytes, 2 * pair),
             "vertex constant UBO load");
      else
         put(c, raw,
             g5p_usc_address_load(raw, 2 * (dst + (vector ? 0 : i)), bytes,
                                  2 * pair, 0),
             "fragment constant UBO load");
      put(c, raw, g5p_usc_wait(raw, 0), "wait");
      c->busy &= ~BITFIELD64_RANGE(pair, 2);
      free_scratch(c);
   }

   for (unsigned k = 0; k < n; k++) {
      struct val *v = V(c, &in->def, k);
      v->state = VAL_REG;
      v->reg = dst + k;
      v->held = true;
   }
   c->p->evidence |= PCO_G5P_EV_DYN_UBO;
}

/* Apple's derivatives (f_ddx_*, f_ddy_*): one instruction, the unary
 * hints for a source read again. Straight-line only: under the mask stack a
 * neighbouring lane may not have run. */
static void emit_deriv(struct gctx *c, nir_intrinsic_instr *in)
{
   if (c->cf_depth)
      refuse(c, "%s inside control flow (no Apple witness)", nir_intrinsic_infos[in->intrinsic].name);
   bool y = in->intrinsic == nir_intrinsic_ddy || in->intrinsic == nir_intrinsic_ddy_fine ||
            in->intrinsic == nir_intrinsic_ddy_coarse;
   struct ref x = resolve_value(c, in->src[0].ssa, 0);
   unsigned xr = reg_of(c, x, "derivative");
   bool d = dies(c, x);
   unsigned g = vgrp(c, x);
   free_dying(c, &x, 1);
   unsigned dst = alloc_reg(c, d ? (int)xr : -1);
   uint8_t raw[G5P_USC_MAX];
   put(c, raw, g5p_usc_derivative(raw, y, 2 * dst, 2 * xr, g, 1, g ? 0 : 4, 0),
       nir_intrinsic_infos[in->intrinsic].name);
   c->p->evidence |= PCO_G5P_EV_DERIV;
   define(c, &in->def, dst);
}

static void stage_ssbo_store(struct gctx *c,nir_intrinsic_instr *in)
{
   if(in->src[0].ssa->bit_size!=32 || in->num_components!=1 || nir_intrinsic_write_mask(in)!=1 ||
      nir_intrinsic_offset_shift(in)!=0 || nir_intrinsic_align_mul(in)<4 || nir_intrinsic_align_offset(in)%4)
      refuse(c,"store_ssbo: aligned scalar32-bit byte offsets only");
   unsigned set,binding,base=0;block_binding(in->src[1].ssa,&set,&binding);
   for(unsigned i=0;i<c->nssbo;i++)if(c->ssbo_set[i]==set && c->ssbo_binding[i]==binding)base=2*i;
   struct ref offset=resolve_value(c,in->src[2].ssa,0),value=resolve_value(c,in->src[0].ssa,0);
   unsigned index=reg_of(c,offset,"store_ssbo byte offset"),data=reg_of(c,value,"store_ssbo value");
   if(c->vs)vs_bank1_only(c,"vertex SSBO store",index,data,-1,-1);
   unsigned pair;
   for(pair=0;pair+2<=NUM_DWORDS;pair+=2)
      if(!((c->busy|c->reserved)&BITFIELD64_RANGE(pair,2)))break;
   if(pair+2>NUM_DWORDS)refuse(c,"store_ssbo: register pressure for address pair");
   c->busy|=BITFIELD64_RANGE(pair,2);
   c->max_dword=MAX2(c->max_dword,pair+2);c->max_written=MAX2(c->max_written,pair+2);
   uint8_t raw[G5P_USC_MAX];
   put(c,raw,g5p_usc_address_add_shift(raw,2*pair,base,2*index,0,false,c->vs),"SSBO address");
   if(c->vs)put(c,raw,g5p_usc_global_store(raw,4,2*data,2*pair,0),"vertex SSBO store");
   else {
      put(c,raw,g5p_usc_fs_store_predicate(raw),"fragment store predicate");
      put(c,raw,g5p_usc_fs_store(raw,4,2*data,2*pair,1),"fragment SSBO store");
   }
   put(c,raw,g5p_usc_wait(raw,0),"SSBO store completion");
   c->busy&=~BITFIELD64_RANGE(pair,2);free_scratch(c);
   c->p->evidence|=c->vs?PCO_G5P_EV_VS_STORE:PCO_G5P_EV_FS_STORE;
}

static void emit_intrinsic(struct gctx *c, nir_intrinsic_instr *in)
{
   uint8_t raw[G5P_USC_MAX];
   if (is_deriv(in->intrinsic)) {
      emit_deriv(c, in);
      return;
   }
   switch (in->intrinsic) {
   case nir_intrinsic_load_vulkan_descriptor: {
      unsigned set, binding, element;
      /* Each accepted binding is uploaded from its own start. The offset
       * channel is zero, as eval_offset() already assumes for constant
       * loads. Dynamic SSBO store arithmetic must be able to read it too. */
      if (in->def.num_components == 2 &&
          block_binding_element(in->src[0].ssa, &set, &binding, &element)) {
         struct val *offset = V(c, &in->def, 1);
         offset->state = VAL_CONST;
         offset->imm = 0;
      }
      break;
   }
   case nir_intrinsic_load_input: /* VS: values were bound in setup_inputs() */
   case nir_intrinsic_load_interpolated_input:
      /* FS: bound in hoist_iterators(), or read here on the pressure retry. */
      if (!c->vs && c->lazy_iter && in != c->narrow_iter) {
         struct drc d = {{0}};
         read_varying(c, in, &d);
         drc_flush(c, &d);
      }
      break;
   case nir_intrinsic_load_frag_coord: /* bound in hoist_iterators() */
   case nir_intrinsic_load_vertex_id:  /* VS: PDS DOUTW registers, setup_inputs() */
   case nir_intrinsic_load_instance_id:
   case nir_intrinsic_load_xfb_address: /* read by the store's address add */
      break;
   case nir_intrinsic_load_num_vertices: {
      struct val *v = V(c, &in->def, 0);
      v->state = VAL_SHARED;
      v->imm = c->xfb_common + PCO_G5P_XFB_NUM_VERTICES;
      break;
   }
   case nir_intrinsic_store_ssbo:
      stage_ssbo_store(c,in);
      break;
   case nir_intrinsic_store_global:
      vs_store(c, in);
      break;
   case nir_intrinsic_demote:
   case nir_intrinsic_terminate:
      c->discard_always = true;
      break;
   case nir_intrinsic_demote_if:
   case nir_intrinsic_terminate_if:
      if (c->ndiscards == ARRAY_SIZE(c->discards))
         refuse(c, "more than %zu conditional discards", ARRAY_SIZE(c->discards));
      c->discards[c->ndiscards++] = in->src[0].ssa;
      break;
   case nir_intrinsic_load_ubo:
      if (dyn_ubo_load(in)) {
         emit_dyn_ubo(c, in);
         break;
      }
      FALLTHROUGH;
   case nir_intrinsic_load_push_constant: {
      unsigned set = 0, binding = 0, element = 0, off;
      enum pco_g5p_upload_kind kind = PCO_G5P_UPLOAD_PUSH;
      if (in->intrinsic == nir_intrinsic_load_ubo) {
         block_binding_element(in->src[0].ssa, &set, &binding, &element);
         off = load_offset(c, in, 1, 0);
         kind = PCO_G5P_UPLOAD_UBO;
      } else {
         off = load_offset(c, in, 0, nir_intrinsic_base(in));
      }
      struct range *r = range_for(c, kind, set, binding, element);
      if (r->memory) {
         emit_memory_ubo(c, in, r, off);
         break;
      }
      for (unsigned k = 0; k < in->def.num_components; k++) {
         struct val *v = V(c, &in->def, k);
         v->state = VAL_SHARED;
         v->imm = r->common + (off - r->lo) / 4 + k;
      }
      break;
   }
   case nir_intrinsic_load_ssbo: {
      unsigned set, binding;
      block_binding(in->src[0].ssa, &set, &binding);
      unsigned base = 0;
      for (unsigned i = 0; i < c->nssbo; i++)
         if (c->ssbo_set[i] == set && c->ssbo_binding[i] == binding)
            base = 2 * i;
      unsigned off = load_offset(c, in, 1, 0);
      unsigned n = in->def.num_components, bytes = n == 1 ? 4 : n == 4 ? 16 : 0;
      if (!bytes || off % bytes || off / bytes > 255)
         refuse(c, "load_ssbo: only 4/16-byte element loads at a constant element index <= 255 are witnessed");
      unsigned dst = alloc_run(c, n, "load_ssbo");
      put(c, raw, g5p_usc_buffer_load(raw, 2 * dst, bytes, base, off / bytes, 0), "fixed buffer load");
      put(c, raw, g5p_usc_wait(raw, 0), "wait");
      for (unsigned k = 0; k < n; k++) {
         struct val *v = V(c, &in->def, k);
         v->state = VAL_REG;
         v->reg = dst + k;
         v->held = true;
      }
      c->p->evidence |= PCO_G5P_EV_BUFFER_LOAD;
      break;
   }
   case nir_intrinsic_load_reg: { /* a copy: nir_trivialize_registers adds movs, which are register maps here */
      nir_def *decl = in->src[0].ssa;
      struct val *dv = V(c, decl, 0);
      if (!dv->held)
         refuse(c, "load_reg: NIR register without a register");
      if (!V(c, &in->def, 0)->uses)
         break;
      unsigned g = vgrp(c, (struct ref){decl, 0, false, false});
      /* vertex: a read that an output stores goes to bank 1 */
      c->alloc_out = c->vs && c->vs_bank0 && vs_feeds_output(c, &in->def);
      unsigned dst = alloc_reg(c, -1);
      c->alloc_out = false;
      put(c, raw, g5p_usc_move(raw, H(c, dst), H(c, dv->reg), vbank(c, dst), g,
                               c->vs ? vhdr(c, 0x21, dv->reg, -1, -1) : 1), "register read");
      define(c, &in->def, dst);
      break;
   }
   case nir_intrinsic_store_reg: {
      struct val *dv = V(c, in->src[1].ssa, 0);
      if (!dv->held)
         refuse(c, "store_reg: NIR register without a register");
      struct ref r = resolve_value(c, in->src[0].ssa, 0);
      struct val *v = V(c, r.def, r.comp);
      if (v->state == VAL_CONST && !v->held) {
         put(c, raw, g5p_usc_immediate(raw, H(c, dv->reg), v->imm, vbank(c, dv->reg), -1), "register write");
         break;
      }
      unsigned src = reg_of(c, r, "store_reg");
      if (src != dv->reg)
         put(c, raw, g5p_usc_move(raw, H(c, dv->reg), H(c, src), vbank(c, dv->reg), vgrp(c, r),
                                  c->vs ? vhdr(c, 0x21, src, -1, -1) : 1), "register write");
      free_scratch(c);
      break;
   }
   default:
      break;
   }
}

/* Coordinate from two consecutive components of one smooth varying: the
 * slot of its coordinate iterator, else -1 (computed coordinate). */
static int varying_coord_slot(struct gctx *c, nir_tex_instr *t)
{
   nir_def *coord = nir_get_tex_src(t, nir_tex_src_coord);
   struct ref x = resolve_plain(coord, 0), y = resolve_plain(coord, 1);
   nir_intrinsic_instr *in = intrinsic_of(x.def);
   if (!smooth_centre(in) || x.def != y.def || y.comp != x.comp + 1)
      return -1;
   return varying_slot(c, in) + x.comp;
}

/* textypes: textureSize / textureQueryLevels read the size block the ICD
 * uploads (PCO_G5P_UPLOAD_IMAGE_SIZE): the results are common-store values,
 * moved into registers by their consumers (mov.shared). */
static void emit_size(struct gctx *c, nir_tex_instr *t)
{
   unsigned common = c->tsize[size_for(c, t)].common;
   for (unsigned k = 0; k < t->def.num_components; k++) {
      struct val *v = V(c, &t->def, k);
      v->state = VAL_SHARED;
      v->imm = common + (t->op == nir_texop_query_levels ? 3 : k);
   }
   c->p->evidence |= PCO_G5P_EV_TEX_TYPES;
}

/* A source component moved into dword dst (group bit by Apple's last-read rule). */
static void move_into(struct gctx *c, unsigned dst, nir_def *d, unsigned k, const char *what)
{
   uint8_t raw[G5P_USC_MAX];
   struct ref r = resolve_value(c, d, k);
   unsigned rr = reg_of(c, r, what);
   if (c->vs && (r.neg || r.abs))
      refuse(c, "vertex texture coordinate modifier not qualified");
   put(c, raw, g5p_usc_move(raw, H(c, dst), H(c, rr), 0, vgrp(c, r),
                           c->vs ? vhdr(c, 0x21, rr, -1, -1) : 1), what);
}

/* textypes: Apple's general sample/read (probes.metal, textypes.md §3).
 * Operand blocks are scratch registers laid out as Apple's programs lay
 * them out:
 *  - coordinates x, y [, z] as f32 (read: u32); a 2D array's layer is a u16
 *    half after y (sample: f32 + 0.5 truncated by `89 .. 09 ..`, Apple's
 *    uint(); read: the integer's low half), the texel offset in the half
 *    after it;
 *  - level/bias as f16 in the low half of a dword (fmul by 1.0 into a
 *    result half, Apple's `89 .. 48 02`); a read's level as an integer;
 *  - gradients f32 as [dPdx.x, dPdy.x, dPdx.y, dPdy.y];
 *  - extras: the compare reference f32, then the packed offset half.
 * One result for a compare, four otherwise; integer formats return their
 * integers in the same registers (the instruction is format-blind). */
/* Apple's cube gradient conversion has a coupled register interface, not
 * independently relocatable derivative operands. Every native uniform and
 * mixed-texture control uses r0..5 derivatives, r6..8 direction; conversion
 * produces coordinates r0..2 and gradients r3..8. Reserve this exact tuple
 * for the shader, and keep all SSA values outside it. */
static void emit_fragment_cube_gradient(struct gctx *c, nir_tex_instr *t)
{
   uint8_t raw[G5P_USC_MAX];
   struct tex_bind *b = tex_for(c, t);
   nir_def *dx = nir_get_tex_src(t, nir_tex_src_ddx);
   nir_def *dy = nir_get_tex_src(t, nir_tex_src_ddy);
   nir_def *coord = nir_get_tex_src(t, nir_tex_src_coord);
   for (unsigned k = 0; k < 3; k++) {
      move_into(c, 2*k, dx, k, "cube dPdx");
      move_into(c, 2*k + 1, dy, k, "cube dPdy");
      move_into(c, 6 + k, coord, k, "cube direction");
   }
   put(c, raw, g5p_usc_cube_gradient_tuple(raw), "cube gradient conversion r0..8");
   unsigned dst = alloc_run(c, 4, "cube gradient result");
   struct g5p_tex_ext x = {
      .dst_half = 2 * dst, .image_qword = b->image / 2,
      .sampler_qword = b->sampler / 2, .coord_dword = 0,
      .dim = G5P_TEX_CUBE, .mask = 15, .lod = G5P_TEX_LOD_GRAD,
      .lod_dword = 3,
   };
   put(c, raw, g5p_usc_sample_ext(raw, &x), "cube gradient sample");
   put(c, raw, g5p_usc_wait(raw, 0), "wait");
   for (unsigned k = 0; k < 4; k++) {
      struct val *v = V(c, &t->def, k);
      v->state = VAL_REG; v->reg = dst + k; v->held = true;
   }
   c->p->evidence |= PCO_G5P_EV_TEX_TYPES;
}

static void emit_tex_ext(struct gctx *c, nir_tex_instr *t)
{
   if (!c->vs && t->op == nir_texop_txd && t->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
      emit_fragment_cube_gradient(c, t);
      return;
   }
   bool saved_bank = c->tex_bank0;
   c->tex_bank0 = c->vs;
   uint8_t raw[G5P_USC_MAX];
   struct tex_bind *b = tex_for(c, t);
   enum glsl_sampler_dim sd = t->sampler_dim;
   bool read = t->op == nir_texop_txf, array = t->is_array;
   bool third = array || sd == GLSL_SAMPLER_DIM_3D || sd == GLSL_SAMPLER_DIM_CUBE;
   int packed = packed_offset_components(t, sd == GLSL_SAMPLER_DIM_3D ? 3 : 2);
   struct g5p_tex_ext x = {
      .image_qword = b->image / 2, .sampler_qword = b->sampler / 2, .read = read,
      .dim = sd == GLSL_SAMPLER_DIM_CUBE ? G5P_TEX_CUBE : sd == GLSL_SAMPLER_DIM_3D ? G5P_TEX_3D :
             array ? G5P_TEX_2D_ARRAY : G5P_TEX_2D,
      .mask = t->is_shadow ? 1 : 0xf, .compare = t->is_shadow, .offset = packed >= 0,
   };
   uint64_t scratch = 0;
   nir_def *cd = nir_get_tex_src(t, sd == GLSL_SAMPLER_DIM_BUF ? nir_tex_src_backend1 : nir_tex_src_coord);
   /* a cube's block has a fourth scratch dword: the cube-coordinate op's
    * output width is not measured */
   unsigned nblock = 2 + third + (sd == GLSL_SAMPLER_DIM_CUBE);
   unsigned block = alloc_run(c, nblock, "texture coordinates");
   scratch |= BITFIELD64_RANGE(block, nblock);
   x.coord_dword = H(c, block) / 2;
   move_into(c, block, cd, 0, "texture coordinate x");
   move_into(c, block + 1, cd, 1, "texture coordinate y");
   if (array && !read) {
      /* layer = u16(z + 0.5) */
      unsigned tmp = alloc_reg(c, -1);
      scratch |= BITFIELD64_BIT(tmp);
      move_into(c, tmp, cd, 2, "array layer");
      put(c, raw, g5p_usc_binary_literal(raw, G5P_BIN_FADD, 2 * tmp, 0x3800, 2 * tmp, 0, 1, 1, true, -1),
          "layer + 0.5");
      put(c, raw, g5p_usc_f2u16_half(raw, 2 * (block + 2), 2 * tmp, 1), "layer f32 -> u16");
   } else if (third) {
      move_into(c, block + 2, cd, 2, array ? "array layer" : "texture coordinate z");
   }
   /* level / bias / gradients */
   int li = nir_tex_instr_src_index(t, t->op == nir_texop_txb ? nir_tex_src_bias : nir_tex_src_lod);
   if (t->op == nir_texop_txd) {
      unsigned axes = sd == GLSL_SAMPLER_DIM_3D ? 3 : 2;
      unsigned g = alloc_run(c, 2 * axes, "texture gradients");
      scratch |= BITFIELD64_RANGE(g, 2 * axes);
      nir_def *dx = nir_get_tex_src(t, nir_tex_src_ddx), *dy = nir_get_tex_src(t, nir_tex_src_ddy);
      /* Apple f_3d_grad_*: [dx.x,dy.x,dx.y,dy.y,dx.z,dy.z]. */
      for (unsigned k = 0; k < axes; k++) {
         move_into(c, g + 2*k, dx, k, "dPdx");
         move_into(c, g + 2*k + 1, dy, k, "dPdy");
      }
      x.lod = G5P_TEX_LOD_GRAD;
      x.lod_dword = g;
   } else if (read && sd == GLSL_SAMPLER_DIM_BUF) {
      /* The size upload reserves an explicit common-store level zero. */
      x.lod_shared = true;
      x.lod_dword = c->tsize[size_for(c, t)].common + 1;
      x.lod = G5P_TEX_LOD_LEVEL;
   } else if (read) {
      struct ref r = li >= 0 ? resolve_value(c, t->src[li].src.ssa, 0) : (struct ref){0};
      x.lod_dword = li >= 0 ? reg_of(c, r, "texel level") : zero_reg(c);
      if (li >= 0)
         vgrp(c, r); /* read by the sample instruction, the next one put */
      else
         kgrp(c, K_ZERO);
      x.lod = G5P_TEX_LOD_LEVEL;
   } else if (c->vs) {
      /* Narrow in bank1 with the native vertex literal conversion, then
       * copy its low half to bank0 with the witnessed A-source move.
       * No texture continuation bits or bank0-to-bank1 copy is needed. */
      c->tex_bank0 = false;
      unsigned l1 = alloc_reg(c, -1);
      unsigned src = l1, group = 1;
      if (li >= 0) {
         struct ref r = resolve_value(c, t->src[li].src.ssa, 0);
         struct val *v = V(c, r.def, r.comp);
         if (v->state == VAL_CONST) {
            put(c, raw, g5p_usc_immediate(raw, H(c, l1), apply_mods(v->imm, r), 1, -1), "vertex LOD constant");
         } else if (v->state == VAL_SHARED && !r.neg && !r.abs) {
            put(c, raw, g5p_usc_mov_shared(raw, H(c, l1), v->imm, 1), "vertex LOD shared");
         } else {
            if (r.neg || r.abs) refuse(c, "vertex texture LOD modifier not qualified");
            src = reg_of(c, r, "vertex texture LOD");
            vs_bank1_only(c, "vertex texture LOD source", src, -1, -1, -1);
            group = vgrp(c, r);
         }
      } else {
         put(c, raw, g5p_usc_immediate(raw, H(c, l1), 0, 1, -1), "vertex LOD zero");
      }
      put(c, raw, g5p_usc_vertex_texture_level(raw, H(c, l1), H(c, src), group), "vertex LOD f32 to f16");
      c->tex_bank0 = true;
      unsigned l0 = alloc_reg(c, -1);
      put(c, raw, g5p_usc_move(raw, H(c, l0), H(c, l1), 0, 1, 0x21), "vertex LOD into bank0");
      c->busy &= ~BITFIELD64_BIT(l1);
      scratch |= BITFIELD64_BIT(l0);
      x.lod = G5P_TEX_LOD_LEVEL;
      x.lod_dword = H(c, l0) / 2;
   } else if (li >= 0) {
      struct ref r = resolve_value(c, t->src[li].src.ssa, 0);
      unsigned lr = reg_of(c, r, "texture level");
      unsigned one = one_reg(c);
      unsigned gl = vgrp(c, r), g1 = kgrp(c, K_ONE);
      unsigned l = alloc_reg(c, -1);
      scratch |= BITFIELD64_BIT(l);
      struct g5p_alu f = G5P_ALU(2 * l);
      f.result_half = true;
      f.a = gsrc(c, lr, r, gl);
      f.b = (struct g5p_src){.half = 2 * one, .group = g1};
      alu_hint(c, &f, false);
      put(c, raw, g5p_usc_binary(raw, G5P_BIN_FMUL, &f), "level/bias f32 -> f16");
      x.lod = t->op == nir_texop_txb ? G5P_TEX_LOD_BIAS : G5P_TEX_LOD_LEVEL;
      x.lod_dword = l;
   }
   /* extras: [compare reference][offset] */
   unsigned off_half = 0;
   if (t->is_shadow) {
      unsigned e = alloc_run(c, 1 + x.offset, "compare reference");
      scratch |= BITFIELD64_RANGE(e, 1 + x.offset);
      move_into(c, e, nir_get_tex_src(t, nir_tex_src_comparator), 0, "compare reference");
      x.extras_half = 2 * e;
      off_half = 2 * (e + 1);
   } else if (x.offset && array) {
      off_half = x.extras_half = 2 * (block + 2) + 1;
   } else if (x.offset) {
      unsigned e = alloc_reg(c, -1);
      scratch |= BITFIELD64_BIT(e);
      off_half = x.extras_half = 2 * e;
   }
   if (x.offset)
      put(c, raw, sd == GLSL_SAMPLER_DIM_3D ?
             g5p_usc_half_immediate_xyz(raw, off_half, packed, 0) :
             g5p_usc_half_immediate(raw, off_half, packed), "texel offset");
   /* coordinate reads by the sample instruction itself are counted in move_into */
   /* cube: the direction becomes the cube coordinates in place, right
    * before the sample (Apple's f_cube_alu, f_cube_alu_lvl, f_cmp_cube_alu) */
   if (x.dim == G5P_TEX_CUBE)
      put(c, raw, g5p_usc_cube_coords(raw, H(c, block), H(c, block)), "cube coordinates");
   unsigned n = t->is_shadow ? 1 : 4;
   unsigned dst = alloc_run(c, n, "texture result");
   x.dst_half = H(c, dst);
   put(c, raw, g5p_usc_sample_ext(raw, &x), read ? "texel read" : "texture sample");
   put(c, raw, g5p_usc_wait(raw, 0), "wait");
   c->scratch |= scratch;
   for (unsigned k = 0; k < n; k++) {
      struct val *v = V(c, &t->def, k);
      v->state = VAL_REG;
      v->reg = dst + k;
      v->held = true;
   }
   c->tex_bank0 = saved_bank;
   c->p->evidence |= PCO_G5P_EV_TEX_TYPES;
}

static void vertex_qualified_copy(struct gctx *c, unsigned dst, nir_def *def, unsigned component)
{
   struct ref r = resolve_value(c, def, component);
   if (r.neg || r.abs) refuse(c, "qualified vertex texture source modifier not witnessed");
   unsigned src = reg_of(c, r, "qualified vertex texture source");
   vs_bank1_only(c, "qualified vertex texture copy", dst, src, -1, -1);
   uint8_t raw[G5P_USC_MAX];
   put(c, raw, g5p_usc_move(raw, H(c, dst), H(c, src), 1, vgrp(c, r), 0x21), "qualified vertex texture copy");
}

static void emit_vertex_qualified_texture(struct gctx *c, nir_tex_instr *t)
{
   uint8_t raw[G5P_USC_MAX];
   struct tex_bind *b = tex_for(c, t);
   bool read = t->op == nir_texop_txf, array = t->is_array;
   bool compare = t->is_shadow;
   bool cube = t->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   bool third = array || t->sampler_dim == GLSL_SAMPLER_DIM_3D || cube;
   bool buffer = t->sampler_dim == GLSL_SAMPLER_DIM_BUF;
   nir_def *coord = nir_get_tex_src(t, buffer ? nir_tex_src_backend1 : nir_tex_src_coord);
   unsigned direction = 0;
   uint64_t scratch = 0;
   if (cube) {
      /* Native uniform and attribute cube probes both write the direction
       * in bank0. The helper's output-bank bit does not change its source. */
      bool saved_bank = c->tex_bank0;
      c->tex_bank0 = true;
      direction = alloc_run(c, 3, "native cube direction in bank0");
      scratch |= BITFIELD64_RANGE(direction, 3);
      for (unsigned k = 0; k < 3; k++)
         move_into(c, direction + k, coord, k, "cube direction into bank0");
      c->tex_bank0 = saved_bank;
   }
   unsigned block = alloc_run(c, 2 + third, "qualified vertex texture coordinates");
   scratch |= BITFIELD64_RANGE(block, 2 + third);
   if (!cube) {
      vertex_qualified_copy(c, block, coord, 0);
      vertex_qualified_copy(c, block + 1, coord, 1);
   }
   if (array && !read) {
      unsigned tmp = alloc_reg(c, -1);
      scratch |= BITFIELD64_BIT(tmp);
      vertex_qualified_copy(c, tmp, coord, 2);
      put(c, raw, g5p_usc_binary_literal(raw, G5P_BIN_FADD, H(c,tmp), 0x3800, H(c,tmp), 1, 1, 0x11, true, -1),
          "qualified array layer + 0.5");
      put(c, raw, g5p_usc_vertex_array_layer(raw, H(c,block+2), H(c,tmp)), "qualified array layer to u16");
   } else if (third && !cube) vertex_qualified_copy(c, block + 2, coord, 2);
   bool gradient = t->op == nir_texop_txd;
   int li = nir_tex_instr_src_index(t, nir_tex_src_lod);
   unsigned level = buffer ? 0 : gradient ? alloc_run(c, 4, "qualified vertex derivatives") : alloc_reg(c, -1);
   if (!buffer) scratch |= BITFIELD64_RANGE(level, gradient ? 4 : 1);
   if (gradient) {
      /* Native distinct literal gradients independently witness this order:
       * dx.x, dy.x, dx.y, dy.y, all f32 in consecutive bank1 dwords. */
      nir_def *dx = nir_get_tex_src(t, nir_tex_src_ddx), *dy = nir_get_tex_src(t, nir_tex_src_ddy);
      for (unsigned k = 0; k < 2; k++) {
         vertex_qualified_copy(c, level + 2*k, dx, k);
         vertex_qualified_copy(c, level + 2*k + 1, dy, k);
      }
   } else if (!buffer) {
      if (li >= 0) vertex_qualified_copy(c, level, t->src[li].src.ssa, 0);
      else put(c, raw, g5p_usc_immediate(raw, H(c,level), 0, 1, -1), "qualified vertex level zero");
      if (!read)
         put(c, raw, g5p_usc_vertex_texture_level(raw, H(c,level), H(c,level), 1), "qualified vertex level to f16");
   }
   if (cube)
      put(c, raw, g5p_usc_cube_coords_bank1(raw, H(c,block), H(c,direction)),
          "native bank0 direction to bank1 cube coordinates");
   bool xyz = g5p_vtex_xyz_offset_probe() && t->sampler_dim == GLSL_SAMPLER_DIM_3D;
   int packed = xyz ? packed_offset_components(t, 3) : packed_offset(t);
   /* Match the native producers: an array's offset uses the layer's high
    * half; a 2D offset uses the f16 level's high half. The low half survives. */
   unsigned extras = H(c,array ? block + 2 : level) + 1;
   unsigned offset_half = extras;
   if (compare) {
      unsigned e = alloc_run(c, 1 + (packed >= 0), "qualified vertex compare reference/offset");
      scratch |= BITFIELD64_RANGE(e, 1 + (packed >= 0));
      vertex_qualified_copy(c, e, nir_get_tex_src(t, nir_tex_src_comparator), 0);
      extras = H(c,e);
      offset_half = extras + 2;
   }
   if (packed >= 0 && gradient && !array) {
      /* Derivatives occupy full dwords; their high halves are not spare. */
      unsigned e = alloc_reg(c, -1);
      scratch |= BITFIELD64_BIT(e);
      extras = H(c,e);
      offset_half = extras;
   }
   if (packed >= 0)
      put(c, raw, xyz ? g5p_usc_half_immediate_xyz(raw, offset_half, packed, 1) :
                       g5p_usc_vertex_half_immediate(raw, offset_half, packed), "qualified vertex packed offset");
   unsigned nresult = compare ? 1 : 4;
   unsigned dst = alloc_run(c, nresult, "qualified vertex texture result");
   struct g5p_tex_ext x = {
      .dst_half=H(c,dst), .image_qword=b->image/2,
      .sampler_qword=(read ? c->fetch_sampler_common : b->sampler)/2,
      .coord_dword=H(c,block)/2,
      .dim=cube ? G5P_TEX_CUBE : t->sampler_dim==GLSL_SAMPLER_DIM_3D ? G5P_TEX_3D : array ? G5P_TEX_2D_ARRAY : G5P_TEX_2D,
      .mask=compare ? 1 : 15, .read=read, .compare=compare,
      .lod=gradient ? G5P_TEX_LOD_GRAD : G5P_TEX_LOD_LEVEL,
      .lod_dword=buffer ? c->tsize[size_for(c,t)].common + 1 : H(c,level)/2, .lod_shared=buffer,
      .offset=packed >= 0, .extras_half=compare || packed >= 0 ? extras : 0,
      .tuple=buffer ? G5P_TEX_TUPLE_COUPLED_SHARED_LEVEL :
             compare ? (packed >= 0 ? G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL_COMPARE_OFFSET : G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL_COMPARE) :
             xyz && packed >= 0 ? G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL_OFFSET_3D :
             gradient ? (packed >= 0 ? G5P_TEX_TUPLE_COUPLED_GRAD_OFFSET : G5P_TEX_TUPLE_COUPLED_GRAD) :
             packed >= 0 ? G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL_OFFSET : G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL,
   };
   put(c, raw, g5p_usc_sample_ext(raw,&x), "qualified vertex sample/read recipe");
   put(c, raw, g5p_usc_wait(raw,0), "wait");
   c->scratch |= scratch;
   for (unsigned k=0;k<nresult;k++) {
      struct val *v=V(c,&t->def,k);v->state=VAL_REG;v->reg=dst+k;v->held=true;
   }
   c->p->evidence |= PCO_G5P_EV_TEX_TYPES;
}

static void emit_tex(struct gctx *c, nir_tex_instr *t)
{
   if (size_query(t)) {
      emit_size(c, t);
      return;
   }
   if (c->vs && g5p_vtex_qualified()) {
      emit_vertex_qualified_texture(c, t);
      return;
   }
   if (c->vs || !plain_tex(t)) {
      emit_tex_ext(c, t);
      return;
   }
   uint8_t raw[G5P_USC_MAX];
   struct tex_bind *b = tex_for(c, t);
   int slot = varying_coord_slot(c, t);
   bool narrow = c->narrow_tex == t;
   unsigned coord;
   if (slot >= 0) {
      coord = narrow ? 0 : alloc_run(c, 2, "texture coordinate");
      put(c, raw, g5p_usc_coord_iterator(raw, 2 * coord, slot, 2), "coordinate iterator");
   } else {
      /* ISA round 2 §4: an ALU-produced coordinate uses the same sample
       * encoding with the temp base (21 Apple computed-coordinate probes). */
      nir_def *cd = nir_get_tex_src(t, nir_tex_src_coord);
      struct ref x = resolve_value(c, cd, 0), y = resolve_value(c, cd, 1);
      unsigned xr = reg_of(c, x, "texture coordinate"), yr = reg_of(c, y, "texture coordinate");
      if (yr == xr + 1) {
         coord = xr;
         /* Read by the sample instruction, the next one put. */
         vgrp(c, x);
         vgrp(c, y);
      } else {
         coord = alloc_run(c, 2, "texture coordinate");
         put(c, raw, g5p_usc_move(raw, 2 * coord, 2 * xr, 0, vgrp(c, x), 1), "coordinate x");
         put(c, raw, g5p_usc_move(raw, 2 * coord + 2, 2 * yr, 0, vgrp(c, y), 1), "coordinate y");
         c->scratch |= BITFIELD64_RANGE(coord, 2);
      }
      c->p->evidence |= PCO_G5P_EV_TEX_COMPUTED;
   }
   struct g5p_tex s = {.image_qword = b->image / 2, .sampler_qword = b->sampler / 2, .coord_dword = coord,
                       .coord_control = true, .dim = G5P_TEX_2D, .narrow = narrow};
   unsigned dst = 0;
   if (!narrow) {
      if (slot >= 0)
         c->busy &= ~BITFIELD64_RANGE(coord, 2);
      dst = alloc_run(c, 4, "texture result");
   }
   s.dst_half = 2 * dst;
   put(c, raw, g5p_usc_sample(raw, &s), "texture sample");
   put(c, raw, g5p_usc_wait(raw, 0), "wait");
   c->p->evidence |= narrow ? PCO_G5P_EV_TEX_CAPTURE : PCO_G5P_EV_TEX_F32;
   if (narrow)
      return;
   for (unsigned k = 0; k < 4; k++) {
      struct val *v = V(c, &t->def, k);
      v->state = VAL_REG;
      v->reg = dst + k;
      v->held = true;
   }
}

/* ---- stage setup and output ---- */
struct vs_convert {
   struct val *v;
   unsigned reg, fmt, byte;
};

static int vs_convert_order(const void *x, const void *y)
{
   const struct vs_convert *a = x, *b = y;
   if (a->reg != b->reg)
      return a->reg < b->reg ? -1 : 1;
   return a->byte < b->byte ? -1 : a->byte > b->byte;
}

/* The converted vertex inputs (PCO_G5P_VS_CONVERT_BASE), in Apple's form
 * (our decode of the c_* and v_unorm4x8 vertex probes): one convert per
 * component at the program's start, from the fetched dword's half (byte
 * select within it), into a new register; the dword's last convert writes
 * it in place. A half read again later takes class 9d, header 0x51 and
 * group 0 (v_unorm4x8 reads each half twice); its last read group 1, header
 * 0x11. An f16 source is read once (v_half2). */
static void vs_converts(struct gctx *c, struct vs_convert *cv, unsigned n)
{
   uint8_t raw[G5P_USC_MAX];
   c->alloc_out = true; /* converted attributes stay in bank 1, where they are fetched */
   qsort(cv, n, sizeof(*cv), vs_convert_order);
   for (unsigned i = 0; i < n; i++) {
      const unsigned half = 2 * cv[i].reg + (cv[i].byte >> 1);
      bool last_half = true, last_reg = true;
      for (unsigned k = i + 1; k < n; k++) {
         last_half &= 2 * cv[k].reg + (cv[k].byte >> 1) != half;
         last_reg &= cv[k].reg != cv[i].reg;
      }
      if (last_reg)
         c->busy &= ~BITFIELD64_BIT(cv[i].reg); /* the dword's last read: in place */
      unsigned dst = alloc_reg(c, last_reg ? (int)cv[i].reg : -1);
      unsigned m;
      if (cv[i].fmt == PCO_G5P_VSF_F16) {
         if (!last_half)
            refuse(c, "an f16 vertex input half read twice (no probe)");
         m = g5p_usc_half_to_float(raw, H(c, dst), half, vbank(c, dst), 1, 0x11);
      } else {
         m = g5p_usc_convert_small(raw, cv[i].fmt, H(c, dst), half, cv[i].byte & 1, vbank(c, dst), last_half,
                                   last_half ? 0x11 : 0x51, last_half ? 0 : 4);
      }
      put(c, raw, m, "vertex input convert");
      hold(c, cv[i].v, dst);
   }
   c->alloc_out = false;
   c->p->evidence |= PCO_G5P_EV_VS_ALU;
}

static void setup_inputs(struct gctx *c)
{
   struct vs_convert cv[4 * PCO_G5P_MAX_ATTRIBS];
   unsigned ncv = 0;
   uint32_t raw_regs = 0;
   foreach_instr_all(ins, c) {
      if (ins->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(ins);
      if (in->intrinsic != nir_intrinsic_load_input)
         continue;
      unsigned loc = nir_intrinsic_io_semantics(in).location;
      if (loc < VERT_ATTRIB_GENERIC0 || loc >= VERT_ATTRIB_GENERIC0 + PCO_G5P_MAX_ATTRIBS ||
          !nir_src_is_const(in->src[0]) || nir_src_as_uint(in->src[0]) || in->def.bit_size != 32)
         refuse(c, "load_input: only direct 32-bit generic attributes");
      unsigned l = loc - VERT_ATTRIB_GENERIC0, first = nir_intrinsic_component(in);
      if (4 * l + 4 > NUM_DWORDS)
         refuse(c, "vertex input location %u: its fetch lands beyond r%u", l, NUM_DWORDS - 1);
      c->p->input_mask |= 1u << l;
      for (unsigned k = 0; k < in->def.num_components; k++) {
         unsigned comp = first + k;
         struct val *v = V(c, &in->def, k);
         c->p->input_components[l] = MAX2(c->p->input_components[l], comp + 1);
         if (c->attrib_components && comp >= c->attrib_components[l]) {
            /* Vulkan fills missing components with (0, 0, 0, 1): integer 1
             * for an integer input (vk10: 32-bit integer vertex formats). */
            v->state = VAL_CONST;
            v->imm = comp != 3 ? 0 : nir_alu_type_get_base_type(nir_intrinsic_dest_type(in)) == nir_type_float ? F32_ONE
                                                                                                    : 1;
            c->p->evidence |= PCO_G5P_EV_ATTRIB_FILL;
            continue;
         }
         v->state = VAL_REG;
         const unsigned base = nir_intrinsic_base(in);
         if (base >= PCO_G5P_VS_CONVERT_BASE) {
            const unsigned fmt = (base - PCO_G5P_VS_CONVERT_BASE) / 4, byte = (base - PCO_G5P_VS_CONVERT_BASE) % 4;
            if (fmt >= PCO_G5P_VSF_COUNT || in->def.num_components != 1 ||
                (fmt >= PCO_G5P_VSF_U16 && (byte & 1)))
               refuse(c, "vertex input location %u: converted load %#x", l, base);
            raw_regs |= BITFIELD64_BIT(4 * l + comp);
            if (v->uses) {
               if (ncv == ARRAY_SIZE(cv))
                  refuse(c, "more than %u converted vertex input components", (unsigned)ARRAY_SIZE(cv));
               cv[ncv++] = (struct vs_convert){v, 4 * l + comp, fmt, byte};
            }
            continue;
         }
         v->input = true;
         if (v->uses)
            hold(c, v, 4 * l + comp);
      }
   }
   /* the fetched dwords of converted inputs: in the unified store (not
    * written), busy until their converts */
   c->busy |= raw_regs;
   if (raw_regs)
      c->max_dword = MAX2(c->max_dword, util_last_bit(raw_regs));
   /* Vertex and instance index: DOUTW of the PDS input temp pair into the
    * dwords after the last fetched attribute (Apple: r0 / r1 without
    * attributes), vertex index first. */
   nir_def *ids[2] = {NULL, NULL};
   foreach_instr_all(ins, c) {
      if (ins->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(ins);
      int i = in->intrinsic == nir_intrinsic_load_vertex_id ? 0 :
              in->intrinsic == nir_intrinsic_load_instance_id ? 1 : -1;
      if (i < 0 || !V(c, &in->def, 0)->uses)
         continue;
      if (ids[i])
         refuse(c, "%s read twice (not CSE'd)", nir_intrinsic_infos[in->intrinsic].name);
      ids[i] = &in->def;
   }
   unsigned next = c->p->input_mask ? 4 * util_last_bit(c->p->input_mask) : 0;
   for (unsigned i = 0; i < 2; i++) {
      if (!ids[i])
         continue;
      if (next >= NUM_DWORDS)
         refuse(c, "%s index: no register after the last attribute", i ? "instance" : "vertex");
      struct val *v = V(c, ids[i], 0);
      v->state = VAL_REG;
      v->input = true;
      hold(c, v, next);
      c->p->id_mask |= 1u << i;
      c->p->id_reg[i] = next++;
      c->p->evidence |= PCO_G5P_EV_VS_IDS;
   }
   if (ncv)
      vs_converts(c, cv, ncv);
   /* a converted dword none of whose components is read */
   for (unsigned i = 0; i < ncv; i++)
      raw_regs &= ~BITFIELD64_BIT(cv[i].reg);
   c->busy &= ~raw_regs;
}

static bool is_varying_load(nir_intrinsic_instr *in)
{
   return in->intrinsic == nir_intrinsic_load_interpolated_input || in->intrinsic == nir_intrinsic_load_input;
}

/* Coefficient reads a varying load issues (flat: one per used component). */
static unsigned varying_reads(struct gctx *c, nir_intrinsic_instr *in)
{
   unsigned n = 0;
   for (unsigned j = 0; j < in->def.num_components; j++)
      n += V(c, &in->def, j)->uses > 0;
   if (in == c->narrow_iter)
      return 1;
   if (!n)
      return 0;
   /* flat: one C-term load per used component, or one iterator past dword 255 (read_varying()) */
   const bool loaded = vary_of(c, in)->interp == PCO_G5P_INTERP_FLAT &&
                       4 * (varying_slot(c, in) + in->def.num_components - 1) + 2 < 256;
   return loaded ? n : 1;
}

/* Fragment varyings (and gl_FragCoord.xy) read ahead of the body in program
 * order, in dependent-read batches, then waited for. On the register
 * pressure retry (lazy_iter) each varying is read where the program loads
 * it instead (emit_intrinsic). */
static void hoist_iterators(struct gctx *c)
{
   uint8_t raw[G5P_USC_MAX];
   struct drc d = {{0}};
   /* gl_FragCoord.xy: raster X/Y (unsigned fixed point, 1/16 pixel) read with
    * the iterators, converted after the wait like f_position_*. */
   nir_intrinsic_instr *fc = NULL;
   foreach_instr_all(ins, c)
      if (ins->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(ins)->intrinsic == nir_intrinsic_load_frag_coord)
         fc = nir_instr_as_intrinsic(ins);
   bool fc_used = fc && (V(c, &fc->def, 0)->uses || V(c, &fc->def, 1)->uses);
   unsigned reads = 0;
   foreach_instr_all(ins, c)
      if (ins->type == nir_instr_type_intrinsic && is_varying_load(nir_instr_as_intrinsic(ins)) &&
          (!c->lazy_iter || nir_instr_as_intrinsic(ins) == c->narrow_iter))
         reads += varying_reads(c, nir_instr_as_intrinsic(ins));
   /* The raster position read has no witnessed counter-1 form: it goes
    * first when the batch spills onto counter 1, else last (round 18). */
   bool fc_first = fc_used && reads + 1 > 8;
   if (fc_first) {
      drc_take(c, &d);
      c->fragcoord = alloc_run(c, 2, "gl_FragCoord");
      put(c, raw, g5p_usc_position_xy(raw, 2 * c->fragcoord, 2, 0), "raster position");
   }
   foreach_instr_all(ins, c) {
      if (ins->type != nir_instr_type_intrinsic || !is_varying_load(nir_instr_as_intrinsic(ins)))
         continue;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(ins);
      if (in == c->narrow_iter) {
         unsigned flags = drc_take(c, &d) ? G5P_ITER_DRC1 : 0;
         put(c, raw, g5p_usc_iterator_flags(raw, 0, varying_slot(c, in), 4, true, c->narrow_sat[0] != NULL, flags),
             "narrow iterator");
         continue;
      }
      if (!c->lazy_iter)
         read_varying(c, in, &d);
   }
   if (fc_used && !fc_first) {
      drc_take(c, &d);
      c->fragcoord = alloc_run(c, 2, "gl_FragCoord");
      put(c, raw, g5p_usc_position_xy(raw, 2 * c->fragcoord, 2, 0), "raster position");
   }
   drc_flush(c, &d);
   if (fc_used) {
      for (unsigned k = 0; k < 2; k++) {
         unsigned r = c->fragcoord + k;
         put(c, raw, g5p_usc_convert(raw, G5P_U2F32, 2 * r, 2 * r, 0, 1, 1), "position u2f32");
         put(c, raw, g5p_usc_binary_literal(raw, G5P_BIN_FMUL, 2 * r, 0x2c00, 2 * r, 0, 1, 1, true, -1),
             "position / 16");
         struct val *v = V(c, &fc->def, k);
         v->state = VAL_REG;
         v->reg = r;
         v->held = true;
      }
      c->p->evidence |= PCO_G5P_EV_FRAGCOORD;
   }
}

static bool out_const(struct gctx *c, unsigned k, uint16_t *h)
{
   if (!c->out_written[k]) {
      *h = 0;
      return true;
   }
   struct ref r = c->out[k];
   if (!is_const(c, r))
      return false;
   uint32_t bits = apply_mods(V(c, r.def, r.comp)->imm, r);
   float f;
   memcpy(&f, &bits, 4);
   *h = _mesa_float_to_half(f);
   return true;
}

/* Output-plan decisions that need use counts. */
static void plan_fs_output(struct gctx *c)
{
   if (c->key->color_output != PCO_G5P_COLOR_F16) {
      /* four raw words per target: 4, or 8 for two targets (emitpix field 0, Apple's 4-target width) */
      if (c->mrt.targets > 1) { /* one emitpix writes both targets: both need their stores */
         enum g5p_mrt_refusal mr = g5p_mrt_check_writes(&c->mrt, c->slot_written);
         if (mr != G5P_MRT_OK)
            refuse(c, "%s", g5p_mrt_refusal_str(mr));
      }
      c->mrt.words = c->mrt.pixel_output_width = 4 * c->mrt.targets;
      c->mrt.emit_field = g5p_mrt_emit_field(c->mrt.words);
      return; /* No narrow f16 loads, output ALU fusion, or f16 constants. */
   }
   enum g5p_mrt_refusal mr = g5p_mrt_check_writes(&c->mrt, c->slot_written);
   if (mr != G5P_MRT_OK)
      refuse(c, "%s", g5p_mrt_refusal_str(mr));
   /* The narrow tex/iterator forms write output halves 0..3 directly: one target only. */
   bool all = c->mrt.targets == 1 && c->out_written[0] && c->out_written[1] && c->out_written[2] &&
              c->out_written[3];
   bool plain = all;
   for (unsigned k = 0; k < 4 && plain; k++)
      plain = !c->out[k].neg && !c->out[k].abs && c->out[k].comp == k && c->out[k].def == c->out[0].def &&
              V(c, c->out[k].def, k)->uses == 1;
   if (plain) {
      nir_instr *p = nir_def_instr(c->out[0].def);
      if (p->type == nir_instr_type_tex && plain_tex(nir_instr_as_tex(p))) {
         c->narrow_tex = nir_instr_as_tex(p);
         return;
      }
      nir_intrinsic_instr *in = intrinsic_of(c->out[0].def);
      if (smooth_centre(in) && in->def.num_components == 4 && !nir_intrinsic_component(in)) {
         c->narrow_iter = in;
         return;
      }
   }
   /* Blending clamps the source: fsat(varying.xyzw) is the saturating
    * narrow iterator `0a 03 01 02` (ds-isa-state.md §3.2). */
   nir_intrinsic_instr *src = NULL;
   bool sat4 = all;
   for (unsigned k = 0; k < 4 && sat4; k++) {
      nir_alu_instr *a = alu_of(c->out[k].def);
      sat4 = a && a->op == nir_op_fsat && !c->out[k].neg && !c->out[k].abs && V(c, &a->def, 0)->uses == 1 &&
             !c->sat_folded[a->def.index];
      if (!sat4)
         break;
      struct ref r = resolve_plain(a->src[0].src.ssa, a->src[0].swizzle[0]);
      nir_intrinsic_instr *in = intrinsic_of(r.def);
      sat4 = smooth_centre(in) && in->def.num_components == 4 &&
             !nir_intrinsic_component(in) && r.comp == k && (!src || src == in) &&
             V(c, r.def, k)->uses == 1;
      src = in;
      c->narrow_sat[k] = a;
   }
   if (sat4) {
      c->narrow_iter = src;
      return;
   }
   memset(c->narrow_sat, 0, sizeof(c->narrow_sat));
   for (unsigned k = 0; k < G5P_MRT_HALVES_PER_TARGET * c->mrt.targets; k++) {
      if (!c->out_written[k] || c->out[k].neg || c->out[k].abs)
         continue;
      nir_alu_instr *a = alu_of(c->out[k].def);
      if (!a || (a->op != nir_op_fadd && a->op != nir_op_fsub && a->op != nir_op_fmul && a->op != nir_op_ffma) ||
          V(c, &a->def, 0)->uses != 1)
         continue;
      c->rh_half[a->def.index] = k;
   }
}

static bool any_color(struct gctx *c)
{
   for (unsigned k = 0; k < G5P_MRT_HALVES_PER_TARGET * c->mrt.targets; k++)
      if (c->out_written[k])
         return true;
   return false;
}

static void fs_prologue(struct gctx *c)
{
   if (c->key->color_output != PCO_G5P_COLOR_F16)
      return;
   uint8_t raw[G5P_USC_MAX];
   if (c->narrow_tex || c->narrow_iter || !any_color(c))
      return;
   for (unsigned d = 0; d < c->mrt.words; d++) {
      uint16_t h0 = 0, h1 = 0;
      bool c0 = out_const(c, 2 * d, &h0), c1 = out_const(c, 2 * d + 1, &h1);
      if (!c0 && !c1)
         continue;
      if (!c0)
         h0 = 0;
      if (!c1)
         h1 = 0;
      put(c, raw, g5p_usc_immediate(raw, 2 * d, h0 | (uint32_t)h1 << 16, 0, -1), "packed constant output");
   }
}

/* Discards run after the body (no derivative work follows them): a test
 * of the 0.0/1.0 condition against 0.0 feeding `04 61`, or `04 68`. */
static void fs_discards(struct gctx *c)
{
   uint8_t raw[G5P_USC_MAX];
   c->p->discard_tokens = c->ndiscards + c->discard_always;
   for (unsigned i = 0; i < c->ndiscards; i++) {
      struct cond k = cond_of(c, c->discards[i], 0);
      unsigned b = k.zero_b ? reg_of(c, k.a, "discard condition")
                            : select_into(c, k, NULL, K_ONE, NULL, K_ZERO);
      unsigned t = alloc_reg(c, -1);
      unsigned z = zero_reg(c);
      unsigned gb = k.zero_b ? vgrp(c, k.a) : 1, gz = kgrp(c, K_ZERO);
      put(c, raw, g5p_usc_discard_test(raw, G5P_FEQ, 2 * t, S(2 * b, gb), S(2 * z, gz), 0), "discard test");
      put(c, raw, g5p_usc_discard(raw, 2 * t), "discard");
      c->busy &= ~BITFIELD64_BIT(t);
      free_scratch(c);
      c->p->evidence |= PCO_G5P_EV_DISCARD;
   }
   if (c->discard_always) {
      put(c, raw, g5p_usc_discard(raw, -1), "discard");
      c->p->evidence |= PCO_G5P_EV_DISCARD;
   }
}

static void fs_output(struct gctx *c)
{
   uint8_t raw[G5P_USC_MAX];
   c->now = DISCARD_AT;
   if (c->has_depth) {
      if (c->ndiscards || c->discard_always || c->blend != BLEND_NONE ||
          c->mrt.targets != 1 || c->key->write_mask != 15)
         refuse(c, "fragment depth export with discard, blend, MRT or partial colour mask is not qualified");
      const unsigned reg = reg_of(c, c->depth, "fragment depth export");
      put(c, raw, g5p_usc_depth_feedback(raw, 2 * reg), "fragment depth feedback");
      c->p->depth_feedback = true;
   }
   fs_discards(c);
   c->now = LAST_END;
   if (c->key->color_output != PCO_G5P_COLOR_F16) {
      /* target t's channels: output words 4t..4t+3 (c->out[] is indexed 4 * slot + component) */
      const unsigned words = 4 * c->mrt.targets;
      struct pvr_g5p_dgl_source sources[8] = {0};
      for (unsigned k = 0; k < words; k++) {
         if (!(c->key->write_mask & (1u << (k % 4))) || !c->out_written[k]) {
            sources[k].immediate = true; /* undefined component: choose zero */
            continue;
         }
         struct ref r = c->out[k];
         struct val *v = V(c, r.def, r.comp);
         sources[k].immediate = is_const(c, r);
         if (sources[k].immediate) {
            sources[k].value = v->imm;
         } else {
            sources[k].value = reg_of(c, r, "raw integer output");
            sources[k].group = vgrp(c, r);
            /* A repeated component must survive earlier output moves. */
            for (unsigned j = k + 1; j < words; j++)
               if (c->out_written[j] && c->out[j].def == r.def && c->out[j].comp == r.comp)
                  sources[k].group = 0;
         }
      }
      uint8_t code[10 * G5P_USC_MAX];
      size_t size = pvr_g5p_dgl_output_words(code, sizeof(code), sources, words, c->key->write_mask);
      if (!size)
         refuse(c, "raw integer output source cannot be encoded");
      put(c, code, size, "raw integer output and emitpix");
      free_scratch(c);
      return;
   }
   if (!any_color(c)) { /* colour write mask 0: Apple's f_mask_none is `00 80` */
      put(c, raw, g5p_usc_special(raw, true), "end");
      return;
   }
   if (!c->narrow_tex && !c->narrow_iter) {
      for (unsigned k = 0; k < G5P_MRT_HALVES_PER_TARGET * c->mrt.targets; k++) {
         uint16_t h;
         if (out_const(c, k, &h))
            continue;
         struct ref r = c->out[k];
         nir_alu_instr *a = alu_of(r.def);
         if (a && c->rh_half[a->def.index] == (int)k)
            continue;
         /* Every half is written at LAST_END, where a value dies at its first
          * read: one that a later half also reads (o1 = v.yxzw after o0 = v)
          * must keep its register until then. Only shapes that were refused
          * before take this, so every other program keeps its bytes. */
         bool later = false;
         for (unsigned k2 = k + 1; k2 < G5P_MRT_HALVES_PER_TARGET * c->mrt.targets && !later; k2++)
            later = c->out_written[k2] && c->out[k2].def == r.def && c->out[k2].comp == r.comp;
         if (later)
            c->now = LAST_END - 1;
         emit_times_one(c, r, false, k, NULL);
         c->now = LAST_END;
         free_scratch(c);
         c->p->evidence |= PCO_G5P_EV_PACK;
      }
   }
   if (c->blend == 1 || c->blend == 2) {
      uint8_t buf[160];
      unsigned n = g5p_blend_epilogue(buf, sizeof(buf));
      if (!n)
         refuse(c, "blend epilogue encoder rejected its fields (%s)", g5p_usc_error ? g5p_usc_error : "?");
      put(c, buf, n, "premultiplied blend epilogue");
      c->max_dword = MAX2(c->max_dword, 7); /* destination halves 4..7, 1-alpha in half 12 */
      c->p->evidence |= PCO_G5P_EV_BLEND;
   } else if (c->blend == BLEND_GENERAL) {
      /* No hardware discard in a destination-reading program: the select of
       * discards_to_select() only holds for source-over, and a hardware
       * discard in the TRANSLUCENT pass lost other objects' colour in 32x32
       * tiles on A10. */
      if (c->ndiscards || c->discard_always)
         refuse(c, "discard in a program with a general blend epilogue");
      uint8_t buf[512];
      const char *err;
      struct g5p_blend_key bk = general_key(c->key);
      unsigned n = g5p_blend_masked(buf, sizeof(buf), &bk, G5P_BLEND_CONST_BASE, c->key->write_mask,
                                   &c->binfo, &err);
      if (err)
         refuse(c, "general blend epilogue: %s", err);
      if (n) {
         put(c, buf, n, "general blend epilogue");
         c->max_dword = MAX2(c->max_dword, c->binfo.max_half / 2);
      }
      c->p->evidence |= PCO_G5P_EV_BLEND;
   }
   /* One emitpix carries the total word count over all targets (2/4/8). */
   put(c, raw, g5p_usc_emit_mrt(raw, c->mrt.words, 0), "emit");
   put(c, raw, g5p_usc_special(raw, true), "end");
}

static void collect_vs_outputs(struct gctx *c)
{
   foreach_instr_all(ins, c) {
      if (ins->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(ins);
      if (in->intrinsic != nir_intrinsic_store_output)
         continue;
      unsigned loc = nir_intrinsic_io_semantics(in).location;
      int base = vs_base(c, loc);
      if (base == -2)
         refuse(c, "store_output to %s (only position and generic varyings have output slots)",
                gl_varying_slot_name_for_stage(loc, MESA_SHADER_VERTEX));
      if (!nir_src_is_const(in->src[1]) || nir_src_as_uint(in->src[1]))
         refuse(c, "indirect store_output");
      if (base == -1)
         continue; /* not read by the fragment program */
      u_foreach_bit(k, nir_intrinsic_write_mask(in)) {
         unsigned slot = base + nir_intrinsic_component(in) + k;
         unsigned i;
         for (i = 0; i < c->nvout && c->vout[i].slot != slot; i++)
            ;
         if (i == ARRAY_SIZE(c->vout))
            refuse(c, "too many output slots");
         c->vout[i].slot = slot;
         c->vout[i].r = resolve_value(c, in->src[0].ssa, k);
         /* Eager: written right after the value's producer (outputs are
          * stored at the end of the NIR program, after every producer). */
         c->vout[i].at = c->eager_vout && !(in == c->vend_store && k == c->vend_comp)
                            ? (int)nir_def_instr(c->vout[i].r.def)->index : -1;
         c->nvout = MAX2(c->nvout, i + 1);
      }
   }
}

static int vout_order(const void *pa, const void *pb, void *data)
{
   struct gctx *c = data;
   const unsigned *a = pa, *b = pb;
   bool ia = V(c, c->vout[*a].r.def, c->vout[*a].r.comp)->input;
   bool ib = V(c, c->vout[*b].r.def, c->vout[*b].r.comp)->input;
   if (ia != ib)
      return ia ? -1 : 1;
   return (int)c->vout[*a].slot - (int)c->vout[*b].slot;
}

/* A vertex output's register, in bank 1 (every output of Apple's reads bank
 * 1): a constant or uniform held in bank 0 is written again into bank 1. */
static unsigned vs_out_reg(struct gctx *c, struct ref r)
{
   c->alloc_out = true;
   unsigned reg = reg_of(c, r, "vertex output");
   struct val *v = V(c, r.def, r.comp);
   if (!vbank(c, reg) && (v->state == VAL_CONST || v->state == VAL_SHARED)) {
      uint8_t raw[G5P_USC_MAX];
      unsigned b1 = alloc_reg(c, -1);
      if (v->state == VAL_CONST)
         put(c, raw, g5p_usc_immediate(raw, H(c, b1), v->imm, vbank(c, b1), -1), "movi (output)");
      else
         put(c, raw, g5p_usc_mov_shared(raw, H(c, b1), v->imm, vbank(c, b1)), "mov.shared (output)");
      c->scratch |= BITFIELD64_BIT(b1);
      reg = b1;
   }
   c->alloc_out = false;
   vs_bank1_only(c, "vertex output", reg, -1, -1, -1);
   return reg;
}

static void vs_output(struct gctx *c)
{
   uint8_t raw[G5P_USC_MAX];
   if (!c->nvout)
      refuse(c, "no position output");
   unsigned order[ARRAY_SIZE(c->vout)], reg[ARRAY_SIZE(c->vout)], n = 0;
   for (unsigned i = 0; i < c->nvout; i++) {
      if (c->vout[i].at >= 0)
         continue; /* written eagerly */
      order[n] = i;
      reg[i] = c->vs_bank0 ? vs_out_reg(c, c->vout[i].r) : reg_of(c, c->vout[i].r, "vertex output");
      n++;
   }
   if (!n)
      refuse(c, "no final vertex output");
   util_qsort_r(order, n, sizeof(order[0]), vout_order, c);
   /* Native ordinary vertex samples export bank0 with the non-temp VOUT
    * form; other values retain the bank1 form. */
   for (unsigned i = 0; i + 1 < n; i++)
      put(c, raw, g5p_usc_vout(raw, H(c, reg[order[i]]), c->vout[order[i]].slot, vbank(c, reg[order[i]]), false), "vertex output");
   /* Characterized convention: branch to the 64-byte block holding the
    * final output/end-task, padding in between. */
   unsigned target = align(c->size + 6, 64);
   put(c, raw, g5p_usc_branch(raw, target - c->size, false), "branch");
   while (c->size < target)
      put(c, raw, g5p_usc_special(raw, false), "padding");
   unsigned last = order[n - 1];
   put(c, raw, g5p_usc_vout(raw, H(c, reg[last]), c->vout[last].slot, vbank(c, reg[last]), true), "last vertex output");
}

/* Eager vertex outputs whose value instruction idx just produced. */
static void eager_outputs(struct gctx *c, int idx)
{
   uint8_t raw[G5P_USC_MAX];
   for (unsigned i = 0; i < c->nvout; i++)
      if (c->vout[i].at == idx) {
         unsigned r = c->vs_bank0 ? vs_out_reg(c, c->vout[i].r) : reg_of(c, c->vout[i].r, "vertex output");
         put(c, raw, g5p_usc_vout(raw, H(c, r), c->vout[i].slot, vbank(c, r), false), "vertex output");
      }
}

/* fsat folds into a single-use fadd/fsub/fmul/ffma producer. Not in the
 * vertex stage: no vertex form of a saturating fadd/fmul/ffma is witnessed,
 * while the standalone saturate is (v_fsat, vsops-check), so a vertex
 * saturate(x * y) (glmark2 jellyfish's smoothstep) is the product, then
 * v_fsat. */
static void plan_saturate(struct gctx *c)
{
   if (c->vs)
      return;
   foreach_instr_all(ins, c) {
      if (ins->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *a = nir_instr_as_alu(ins);
      if (a->op != nir_op_fsat)
         continue;
      nir_alu_instr *p = alu_of(a->src[0].src.ssa);
      if (!p || (p->op != nir_op_fadd && p->op != nir_op_fsub && p->op != nir_op_fmul && p->op != nir_op_ffma) ||
          !list_is_singular(&p->def.uses))
         continue;
      c->sat_folded[a->def.index] = true;
      c->saturate[p->def.index] = true;
   }
}

/* Structured control flow, phis and local arrays become straight-line
 * selects, the bounded NIR lowering isa-g5p.md (ISA round 2 §2, §4)
 * describes for Apple's own nested if/phi and array probes. Loops are
 * unrolled when their trip count is known. Shaders that are straight-line
 * already skip this, so round-16..18 programs see the unchanged pass list. */
static bool needs_flatten(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   return !exec_list_is_singular(&impl->body) || !exec_list_is_empty(&impl->locals);
}

static bool flatten(nir_shader *nir)
{
   if (!needs_flatten(nir))
      return false;
   bool progress;
   const nir_opt_peephole_select_options sel = {.limit = UINT_MAX, .indirect_load_ok = true,
                                                .expensive_alu_ok = true, .discard_ok = true};
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_function_temp, UINT32_MAX);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      NIR_PASS(progress, nir, nir_opt_peephole_select, &sel);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
   } while (progress);
   return true;
}

/* Vertex tasks commit only the 4-byte global store: the 16-byte form is
 * silently dropped (xfb.md, device run 2), and compute has only ever run
 * the 4-byte form too. Every vertex global access is a 32-bit scalar. */
static nir_mem_access_size_align
mem_access_size_align(nir_intrinsic_op intrin, uint8_t bytes, uint8_t bit_size, uint32_t align_mul,
                      uint32_t align_offset, bool offset_is_const, enum gl_access_qualifier access,
                      const void *cb_data)
{
   return (nir_mem_access_size_align){
      .num_components = 1,
      .bit_size = 32,
      .align = 4,
      .shift = nir_mem_access_shift_method_scalar,
   };
}

/* Vertex stores go last, as Apple's compiler places them (v_st_*: vout ...
 * st, br, vend), so no instruction after a store rewrites its registers.
 * The ALU that only feeds stores (the record offsets) moves with them and
 * isn't held across the shader. */
static bool sink_vs_stores(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   if (!exec_list_is_singular(&impl->body))
      return false;
   nir_block *block = nir_start_block(impl);
   bool any = false;
   nir_foreach_instr_reverse(ins, block) {
      bool sink = false;
      if (ins->type == nir_instr_type_intrinsic) {
         sink = nir_instr_as_intrinsic(ins)->intrinsic == nir_intrinsic_store_global;
      } else if (ins->type == nir_instr_type_alu || ins->type == nir_instr_type_load_const) {
         nir_def *d = nir_instr_def(ins);
         sink = !nir_def_is_unused(d);
         nir_foreach_use_including_if(u, d)
            sink &= !nir_src_is_if(u) && nir_src_use_instr(u)->pass_flags;
      }
      ins->pass_flags = sink;
      any |= sink;
   }
   /* Every record's address arithmetic first, then the stores, so only stores and their address adds follow
    * the first store. A walk reaches the moved instructions again at the end: their flag is clear by then. */
   for (unsigned stores = 0; stores < 2; stores++) {
      nir_foreach_instr_safe(ins, block) {
         if (ins->pass_flags && (ins->type == nir_instr_type_intrinsic) == stores) {
            ins->pass_flags = 0;
            nir_instr_move(nir_after_block(block), ins);
         }
      }
   }
   return nir_progress(any, impl, nir_metadata_control_flow);
}

/* iabs as imax(x, -x): lower_iabs's rule is in nir_opt_algebraic_late,
 * which this backend does not run. */
static bool lower_iabs_instr(nir_builder *b, nir_alu_instr *a, void *data)
{
   if (a->op != nir_op_iabs)
      return false;
   b->cursor = nir_before_instr(&a->instr);
   nir_def *x = nir_ssa_for_alu_src(b, a, 0);
   nir_def_replace(&a->def, nir_imax(b, x, nir_ineg(b, x)));
   return true;
}

/* Vertex sqrt (bench-gaps.md): Apple's v_sqrt keeps its rsqrt in a bank-0
 * scratch, a form the vertex allocator does not use (every vertex temp is
 * bank 1). sqrt(x) = rcp(rsqrt(x)) is two vertex recipes the stage already
 * emits in Apple's forms (v_rcp, v_rsq), and is exact where x * rsqrt(x) is
 * not: 0 -> rcp(inf) = 0, inf -> rcp(0) = inf, x < 0 -> NaN (NIR's own
 * lower_fsqrt rule). */
static bool lower_vs_fsqrt_instr(nir_builder *b, nir_alu_instr *a, void *data)
{
   if (a->op != nir_op_fsqrt || a->def.bit_size != 32)
      return false;
   b->cursor = nir_before_instr(&a->instr);
   nir_def_replace(&a->def, nir_frcp(b, nir_frsq(b, nir_ssa_for_alu_src(b, a, 0))));
   return true;
}

static bool lower(nir_shader *nir)
{
   bool progress;
   /* Projective sampling (GLSL textureProj/texture2DProj, Mesa's fixed-function
    * fragment program; Zink passes 2D projectors on as OpImageSampleProj*):
    * Apple's GPU has no projected sample, Metal divides in the program, so
    * the coordinate (and a compare reference) is divided by the projector
    * before the sample, as AGX does (nir_lower_tex lower_txp). */
   /* texelFetchOffset is integer coordinate addition before the qualified
    * fetch. NIR preserves an array layer instead of adding the offset to it.
    * No packed texture-offset field or new USC recipe is selected. */
   const nir_lower_tex_options txp = {
      .lower_txp = ~0u, .lower_txp_array = true,
      .lower_txf_offset = nir->info.stage == MESA_SHADER_FRAGMENT ||
                          (nir->info.stage == MESA_SHADER_VERTEX &&
                           g5p_vtex_qualified() && g5p_vtex_fetch_offset_probe()),
      /* Size queries use CPU-uploaded view dimensions. Minify them with
       * the qualified integer ALU; NIR preserves an array's layer count.
       * This adds no TPU instruction or shadow-sampling recipe. */
      .lower_txs_lod = nir->info.stage == MESA_SHADER_VERTEX && g5p_vtex_size_lod_probe(),
   };
   NIR_PASS(_, nir, nir_lower_tex, &txp);
   pvr_g5p_tbo_lower(nir);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const, nir_address_format_32bit_offset);
   NIR_PASS(_, nir, nir_lower_flrp, 32, false);
   bool flattened = flatten(nir);
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      const nir_lower_mem_access_bit_sizes_options mem = {
         .callback = mem_access_size_align,
         .modes = nir_var_mem_global,
      };
      NIR_PASS(_, nir, nir_lower_mem_access_bit_sizes, &mem);
   }
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   /* Integer division and modulo as Imagination's PCO lowers them
    * (pco_nir.c): constant divisors to a multiply-high, the rest to the
    * reciprocal estimate and two corrections; the multiply-high to 16-bit
    * partial products (lower_mul_high). Apple's A9 lowering is a loop for a
    * register divisor and a magic multiply for a constant
    * (tools/mesa-a9-hypr-re f_udiv_reg, f_udiv_const), with forms we have
    * not decoded; these use ops we encode. */
   NIR_PASS(_, nir, nir_opt_idiv_const, 32);
   NIR_PASS(_, nir, nir_lower_idiv, &(nir_lower_idiv_options){.allow_fp16 = false});
   NIR_PASS(_, nir, nir_lower_alu);
   NIR_PASS(_, nir, nir_shader_alu_pass, lower_iabs_instr, nir_metadata_control_flow, NULL);
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
   } while (progress);
   /* after the last nir_opt_algebraic, which folds rcp(rsqrt(a)) back into sqrt(a) */
   if (nir->info.stage == MESA_SHADER_VERTEX)
      NIR_PASS(_, nir, nir_shader_alu_pass, lower_vs_fsqrt_instr, nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_opt_shrink_vectors, true);
   NIR_PASS(_, nir, nir_opt_dce);
   if (nir->info.stage == MESA_SHADER_VERTEX)
      NIR_PASS(_, nir, sink_vs_stores);
   return flattened;
}

/* ---- round 19: fragment program key ----
 * ds-isa-state.md §3: blending is compiled into the program. Supported:
 *  - blending off, or One/Zero (source clamped, no tile read);
 *  - premultiplied One/OneMinusSrcAlpha on colour and alpha: Apple's
 *    124-byte epilogue after the clamped source colour;
 *  - straight-over SrcAlpha/OneMinusSrcAlpha colour with One/OneMinusSrcAlpha
 *    alpha: the source is pre-multiplied in f32, then the same epilogue.
 * Everything else is refused with the factors. VkBlendFactor numbering. */
enum { BF_ZERO = 0, BF_ONE = 1, BF_SRC_ALPHA = 6, BF_ONE_MINUS_SRC_ALPHA = 7 };

static unsigned blend_mode(struct gctx *c, const struct pco_g5p_fs_key *k)
{
   if (k->write_mask && k->write_mask != 0xf) {
      uint8_t buf[512];
      const char *err;
      struct g5p_blend_key bk = general_key(k);
      g5p_blend_masked(buf, sizeof(buf), &bk, G5P_BLEND_CONST_BASE, k->write_mask,
                       &c->binfo, &err);
      if (err)
         refuse(c, "partial colour mask epilogue: %s", err);
      return BLEND_GENERAL;
   }
   if (!k->blend)
      return BLEND_NONE;
   if (!k->op_rgb && !k->op_a) {
      if (k->src_rgb == BF_ONE && k->dst_rgb == BF_ZERO && k->src_a == BF_ONE && k->dst_a == BF_ZERO)
         return BLEND_REPLACE;
      if (k->dst_rgb == BF_ONE_MINUS_SRC_ALPHA && k->dst_a == BF_ONE_MINUS_SRC_ALPHA && k->src_a == BF_ONE) {
         if (k->src_rgb == BF_ONE)
            return BLEND_PREMULT;
         if (k->src_rgb == BF_SRC_ALPHA)
            return BLEND_STRAIGHT_OVER;
      }
   }
   if (pco_g5p_blend_general() || (!blend_general_off() && blend_is_proven(k))) {
      uint8_t buf[512];
      const char *err;
      struct g5p_blend_key bk = general_key(k);
      g5p_blend_general(buf, sizeof(buf), &bk, G5P_BLEND_CONST_BASE, &c->binfo, &err);
      if (err)
         refuse(c, "blend factors src %u/%u dst %u/%u op %u/%u: %s", k->src_rgb, k->src_a, k->dst_rgb, k->dst_a,
                k->op_rgb, k->op_a, err);
      return BLEND_GENERAL;
   }
   if (k->op_rgb || k->op_a)
      refuse(c, "blend op %u/%u (only Add has a measured epilogue)", k->op_rgb, k->op_a);
   if (k->src_rgb == BF_SRC_ALPHA && k->dst_rgb == BF_ONE_MINUS_SRC_ALPHA && k->src_a == BF_SRC_ALPHA &&
       k->dst_a == BF_ONE_MINUS_SRC_ALPHA)
      refuse(c, "straight alpha with a SrcAlpha alpha factor: Apple's 110-byte epilogue uses "
                "uncharacterized packed f16 fused-product controls (isa-g5p.md ISA round 2 §6)");
   refuse(c, "blend factors src %u/%u dst %u/%u (rgb/alpha): no measured epilogue", k->src_rgb, k->src_a,
          k->dst_rgb, k->dst_a);
}

struct blend_rw {
   unsigned mode;
   unsigned zero; /* BLEND_GENERAL: channels that blend to constant 0 */
   bool drop;
   bool partial;
   bool clamp_source;
};

static bool rewrite_output(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   struct blend_rw *w = data;
   if (in->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(in).location != FRAG_RESULT_DATA0)
      return false;
   if (w->drop) {
      nir_instr_remove(&in->instr);
      return true;
   }
   if (nir_intrinsic_component(in) || nir_intrinsic_write_mask(in) != 0xf || in->src[0].ssa->num_components != 4) {
      w->partial = true;
      return false;
   }
   b->cursor = nir_before_instr(&in->instr);
   /* A mask alone must not clamp enabled float-render-target channels. */
   nir_def *v = w->clamp_source ? nir_fsat(b, in->src[0].ssa) : in->src[0].ssa;
   if (w->zero) {
      nir_def *ch[4];
      for (unsigned i = 0; i < 4; i++)
         ch[i] = w->zero & (1u << i) ? nir_imm_float(b, 0.0f) : nir_channel(b, v, i);
      v = nir_vec(b, ch, 4);
   }
   if (w->mode == BLEND_STRAIGHT_OVER) {
      nir_def *alpha = nir_channel(b, v, 3);
      v = nir_vec4(b, nir_fmul(b, nir_channel(b, v, 0), alpha), nir_fmul(b, nir_channel(b, v, 1), alpha),
                   nir_fmul(b, nir_channel(b, v, 2), alpha), alpha);
   }
   nir_src_rewrite(&in->src[0], v);
   return true;
}

static void apply_fs_key(struct gctx *c, nir_shader *nir, const struct pco_g5p_fs_key *k)
{
   /* k->srgb: Apple's program is the linear one (ds-isa-state.md §2.2); the
    * gamma is PBE q1 bit 45 (gles3-tex.md §1.2), not a program change. */
   c->blend = blend_mode(c, k);
   if (c->blend == BLEND_NONE && k->write_mask)
      return;
   struct blend_rw w = {.mode = c->blend, .zero = c->blend == BLEND_GENERAL ? c->binfo.zero_lanes : 0,
                        .drop = !k->write_mask, .clamp_source = k->blend};
   nir_shader_intrinsics_pass(nir, rewrite_output, nir_metadata_control_flow, &w);
   if (w.partial)
      refuse(c, "blending with a partially written colour output");
   if (w.drop)
      NIR_PASS(_, nir, nir_opt_dce);
}

/* Discards of a blending program (premultiplied or straight-over, both
 * through Apple's premultiplied epilogue) become a select: every colour
 * store moves to the end and stores transparent black where any discard
 * condition holds, and the source-over epilogue turns that into the tile's
 * own colour. No hardware discard is left in a program that reads the
 * tile, so none runs in the TRANSLUCENT pass that destination-reading
 * draws get. On A10 a hardware discard in that pass lost the colour of
 * other objects in random 32x32 tiles (Hyprland's border shader).
 *
 * From the G9P compiler (OrchardLinux/mesa,
 * src/imagination/g9p/g9p_compile.c discards_to_select(), commits
 * 7013b551398 and 7a64e1d1bf4; MIT), its native source-over form. It runs
 * after flattening, so every discard and colour store is outside control
 * flow; a discard or store left inside it, a memory store, or no discard
 * leaves the program unchanged. */
static bool has_side_effects(nir_intrinsic_instr *in)
{
   switch (in->intrinsic) {
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_store_global:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
   case nir_intrinsic_image_store:
   case nir_intrinsic_image_atomic:
   case nir_intrinsic_image_atomic_swap:
   case nir_intrinsic_bindless_image_store:
      return true;
   default:
      return false;
   }
}

static bool impl_has_side_effects(nir_function_impl *impl)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(ins, block) {
         if (ins->type == nir_instr_type_intrinsic && has_side_effects(nir_instr_as_intrinsic(ins)))
            return true;
      }
   }
   return false;
}

/* Explicit depth export plus storage needs the complete A9 deferred-depth
 * protocol, not just DEPTHF. Keep it refused when storage support is merged:
 * Apple's 2908/2371/211d and early-test modes remain unqualified together. */
static bool storage_with_depth(nir_shader *nir)
{
   bool storage = false, depth = false;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(ins, block) {
            if (ins->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(ins);
            storage |= nir_intrinsic_writes_external_memory(in);
            depth |= in->intrinsic == nir_intrinsic_store_output &&
                     nir_intrinsic_io_semantics(in).location == FRAG_RESULT_DEPTH;
         }
      }
   }
   return storage && depth;
}

static bool discards_to_select(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_intrinsic_instr *discards[16], *stores[8];
   unsigned num_discards = 0, num_stores = 0;
   nir_def *cond = NULL;

   nir_foreach_block(block, impl) {
      const bool top = block->cf_node.parent->type == nir_cf_node_function;
      nir_foreach_instr(ins, block) {
         if (ins->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(ins);
         if (has_side_effects(in))
            return nir_no_progress(impl);
         switch (in->intrinsic) {
         case nir_intrinsic_demote:
         case nir_intrinsic_terminate:
         case nir_intrinsic_demote_if:
         case nir_intrinsic_terminate_if:
            if (!top || num_discards == ARRAY_SIZE(discards))
               return nir_no_progress(impl);
            discards[num_discards++] = in;
            break;
         case nir_intrinsic_store_output:
            if (!top || num_stores == ARRAY_SIZE(stores) ||
                nir_intrinsic_io_semantics(in).location != FRAG_RESULT_DATA0 || in->src[0].ssa->bit_size != 32)
               return nir_no_progress(impl);
            stores[num_stores++] = in;
            break;
         default:
            break;
         }
      }
   }
   if (!num_discards || !num_stores)
      return nir_no_progress(impl);

   nir_builder b = nir_builder_at(nir_after_impl(impl));
   for (unsigned i = 0; i < num_discards; i++) {
      nir_intrinsic_instr *in = discards[i];
      nir_def *k = in->intrinsic == nir_intrinsic_demote_if || in->intrinsic == nir_intrinsic_terminate_if
                      ? in->src[0].ssa
                      : nir_imm_true(&b);
      cond = cond ? nir_ior(&b, cond, k) : k;
      nir_instr_remove(&in->instr);
   }
   for (unsigned i = 0; i < num_stores; i++) {
      nir_intrinsic_instr *st = stores[i];
      nir_def *value = st->src[0].ssa;
      nir_store_output(&b, nir_bcsel(&b, cond, nir_imm_zero(&b, value->num_components, 32), value),
                       nir_imm_int(&b, 0), .base = nir_intrinsic_base(st),
                       .write_mask = nir_intrinsic_write_mask(st), .component = nir_intrinsic_component(st),
                       .src_type = nir_intrinsic_src_type(st), .io_semantics = nir_intrinsic_io_semantics(st));
      nir_instr_remove(&st->instr);
   }
   return nir_progress(true, impl, nir_metadata_none);
}

/* Fragment programs fuse an fmul and the fadd it feeds into one ffma
 * (emit_ffma, Apple's FMA, flown in labwc's blended draws). Our NIR
 * options never fused: the rule is one of nir_opt_algebraic_late's
 * (fadd(contract) of an fmul only fadds read -> ffma), and that pass
 * doesn't run here. This runs just that rule, on the scalar program,
 * for an fmul whose every reader is a contractable 32-bit fadd, so the
 * fmul goes away. Only with the key's fuse_ffma (PVR_G5P_FUSE=ffma), so
 * the programs proven on hardware keep their bytes until fusion is flown,
 * and only with last-read groups: then an operand's group bit is set on
 * its last read alone (vgrp()), so a fused addend that is read again
 * stays live. Without them every read gets group 1.
 * Vertex programs keep their fmul/fadd: the vertex backend emits only
 * Apple's measured FMA forms, and a vertex program above 12 temps wedges
 * the TA.
 *
 * From the G9P compiler (OrchardLinux/mesa 28cdfceab8e, which
 * enables NIR's fusion there: on A10 a shader of 512 multiply-adds went
 * from 15.4 to 12.4 KB and Hyprland's Kawase blur passes got faster). His
 * follow-up 906f3eab2a6 (the addend's last-use bit was always set) is
 * what the last-read requirement above covers. */
static bool only_fadds(nir_alu_instr *mul)
{
   nir_foreach_use_including_if(u, &mul->def) {
      if (nir_src_is_if(u) || nir_src_use_instr(u)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *add = nir_instr_as_alu(nir_src_use_instr(u));
      if (add->op != nir_op_fadd || add->def.num_components != 1 || nir_alu_instr_no_contract(add))
         return false;
   }
   return true;
}

static bool fuse_ffma_instr(nir_builder *b, nir_alu_instr *add, void *data)
{
   if (add->op != nir_op_fadd || add->def.bit_size != 32 || add->def.num_components != 1 ||
       nir_alu_instr_no_contract(add))
      return false;
   for (unsigned i = 0; i < 2; i++) {
      nir_alu_instr *mul = alu_of(add->src[i].src.ssa);
      if (!mul || mul->op != nir_op_fmul || mul->def.num_components != 1 || !only_fadds(mul))
         continue;
      b->cursor = nir_before_instr(&add->instr);
      nir_def *f = nir_ffma(b, nir_mov_alu(b, mul->src[0], 1), nir_mov_alu(b, mul->src[1], 1),
                            nir_mov_alu(b, add->src[1 - i], 1));
      nir_def_replace(&add->def, f);
      return true;
   }
   return false;
}

static void fuse_ffma(struct gctx *c, nir_shader *nir)
{
   bool progress = false;
   if (c->vs || !c->key->fuse_ffma || !c->key->last_read_groups)
      return;
   NIR_PASS(progress, nir, nir_shader_alu_pass, fuse_ffma_instr, nir_metadata_control_flow, NULL);
   if (!progress)
      return;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);
}

/* After lower(): the select, scalar like the rest of the program. */
static void select_discards(struct gctx *c, nir_shader *nir)
{
   bool progress = false;
   if (c->vs || (c->blend != BLEND_PREMULT && c->blend != BLEND_STRAIGHT_OVER))
      return;
   NIR_PASS(progress, nir, discards_to_select);
   c->p->discard_select = progress;
   if (!progress)
      return;
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_cse);
   } while (progress);
}

/* ---- control flow left after flattening (fragment) ----
 * discards_to_if() and discards_to_flag() are G9P's
 * (OrchardLinux/mesa src/imagination/g9p/g9p_compile.c, d1ab45daed3; MIT):
 * `if (c) discard;` becomes demote_if(c); any other discard inside control
 * flow sets a flag, and one demote_if on the flag ends the program (the
 * deferred discards of fs_discards()). The pixel runs on, which only
 * matters for side effects (memory stores are refused in fragment
 * programs). */
static bool cf_left(nir_shader *nir)
{
   return !exec_list_is_singular(&nir_shader_get_entrypoint(nir)->body);
}

static bool discards_to_if(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool progress = false;
   nir_foreach_block_safe(block, impl) {
      nir_if *nif = nir_block_get_following_if(block);
      if (!nif || !exec_list_is_singular(&nif->then_list) || !nir_cf_list_is_empty_block(&nif->else_list))
         continue;
      nir_block *then_block = nir_if_first_then_block(nif);
      if (exec_list_length(&then_block->instr_list) != 1 ||
          nir_block_first_instr(then_block)->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(nir_block_first_instr(then_block));
      if (in->intrinsic != nir_intrinsic_demote && in->intrinsic != nir_intrinsic_terminate)
         continue;
      nir_builder b = nir_builder_at(nir_before_cf_node(&nif->cf_node));
      if (in->intrinsic == nir_intrinsic_demote)
         nir_demote_if(&b, nif->condition.ssa);
      else
         nir_terminate_if(&b, nif->condition.ssa);
      nir_cf_node_remove(&nif->cf_node);
      progress = true;
   }
   return nir_progress(progress, impl, nir_metadata_none);
}

static bool discards_to_flag(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_variable *flag = NULL;
   nir_builder b;
   nir_foreach_block_safe(block, impl) {
      if (block->cf_node.parent->type == nir_cf_node_function)
         continue;
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic != nir_intrinsic_demote && in->intrinsic != nir_intrinsic_terminate &&
             in->intrinsic != nir_intrinsic_demote_if && in->intrinsic != nir_intrinsic_terminate_if)
            continue;
         if (!flag) {
            flag = nir_local_variable_create(impl, glsl_bool_type(), "discarded");
            b = nir_builder_at(nir_before_impl(impl));
            nir_store_var(&b, flag, nir_imm_false(&b), 1);
         }
         b = nir_builder_at(nir_before_instr(instr));
         nir_def *cond = in->intrinsic == nir_intrinsic_demote || in->intrinsic == nir_intrinsic_terminate
                            ? nir_imm_true(&b)
                            : nir_ior(&b, nir_load_var(&b, flag), in->src[0].ssa);
         nir_store_var(&b, flag, cond, 1);
         nir_instr_remove(instr);
      }
   }
   if (!flag)
      return nir_no_progress(impl);
   b = nir_builder_at(nir_after_impl(impl));
   nir_demote_if(&b, nir_load_var(&b, flag));
   return nir_progress(true, impl, nir_metadata_none);
}

/* Key punch: every discard left at the top level (all of them after
 * flattening and cf_discards()) becomes one demote_if on the OR of their
 * conditions, or one demote if any is unconditional, at the end of the
 * program; fs_discards() then emits one token. Apple's compiler gives a
 * program with two discard conditions one `04 60` (f_discard2,
 * tools/mesa-a9-punchds), and the OR of materialized conditions is the
 * flag form discards_to_flag() already runs on hardware (cf7). */
static bool merge_discards(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_intrinsic_instr *list[32];
   unsigned n = 0;
   bool always = false;
   nir_foreach_block(block, impl) {
      if (block->cf_node.parent->type != nir_cf_node_function)
         continue;
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
         if (in->intrinsic != nir_intrinsic_demote && in->intrinsic != nir_intrinsic_terminate &&
             in->intrinsic != nir_intrinsic_demote_if && in->intrinsic != nir_intrinsic_terminate_if)
            continue;
         if (n == ARRAY_SIZE(list))
            return nir_no_progress(impl); /* fs_discards() refuses this many anyway */
         always |= in->intrinsic == nir_intrinsic_demote || in->intrinsic == nir_intrinsic_terminate;
         list[n++] = in;
      }
   }
   if (n < 2)
      return nir_no_progress(impl);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   if (always) {
      nir_demote(&b);
   } else {
      nir_def *k = list[0]->src[0].ssa;
      for (unsigned i = 1; i < n; i++)
         k = nir_ior(&b, k, list[i]->src[0].ssa);
      nir_demote_if(&b, k);
   }
   for (unsigned i = 0; i < n; i++)
      nir_instr_remove(&list[i]->instr);
   return nir_progress(true, impl, nir_metadata_none);
}

static void cf_discards(nir_shader *nir)
{
   bool progress = false;
   NIR_PASS(progress, nir, discards_to_if);
   NIR_PASS(progress, nir, discards_to_flag);
   if (!progress)
      return;
   flatten(nir); /* the flag's variable to SSA, and selects where they fit */
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_opt_dce);
}

/* An arm of an if in a loop that does work and then breaks: the work runs
 * before the if for every lane, its register writes select the new value
 * where the arm is taken, and the arm keeps only the break. Apple's A9
 * compiler does the same (f_loop_break2 computes `last` for every lane, then
 * breaks), so every break stays at its loop's level: a break under a pushed
 * if is not witnessed. Only ALU work, uniform and register reads. */
static bool speculable(nir_instr *i)
{
   switch (i->type) {
   case nir_instr_type_alu:
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_intrinsic:
      switch (nir_instr_as_intrinsic(i)->intrinsic) {
      case nir_intrinsic_load_reg:
      case nir_intrinsic_store_reg:
      case nir_intrinsic_load_push_constant:
      case nir_intrinsic_load_ubo:
      case nir_intrinsic_load_frag_coord:
         return true;
      default:
         return false;
      }
   default:
      return false;
   }
}

static bool hoist_break_work(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool progress = false;
   nir_foreach_block_safe(block, impl) {
      nir_if *nif = nir_block_get_following_if(block);
      unsigned side;
      if (!nif || break_shape(nif, &side) != 2)
         continue;
      nir_block *arm = single_block(side ? &nif->else_list : &nif->then_list);
      nir_instr *brk = nir_block_last_instr(arm);
      bool ok = true;
      nir_foreach_instr(i, arm)
         ok &= i == brk || speculable(i);
      if (!ok)
         continue;
      nir_builder b = nir_builder_at(nir_before_cf_node(&nif->cf_node));
      nir_foreach_instr_safe(i, arm) {
         if (i == brk)
            break;
         nir_intrinsic_instr *in = i->type == nir_instr_type_intrinsic ? nir_instr_as_intrinsic(i) : NULL;
         if (!in || in->intrinsic != nir_intrinsic_store_reg) {
            nir_instr_move(b.cursor, i);
            b.cursor = nir_after_instr(i);
            continue;
         }
         nir_def *v = in->src[0].ssa, *reg = in->src[1].ssa;
         nir_def *old = nir_load_reg(&b, reg);
         nir_def *sel = side ? nir_bcsel(&b, nif->condition.ssa, old, v) : nir_bcsel(&b, nif->condition.ssa, v, old);
         nir_store_reg(&b, sel, reg);
         nir_instr_remove(i);
      }
      progress = true;
   }
   return nir_progress(progress, impl, nir_metadata_none);
}

/* The hardware break operation removes the loop's two mask-stack levels. It
 * cannot also cross an if level. Mesa's common loop simplifier turns every
 * break in such a loop into a flag and one conditional break at the loop's
 * own level, predicating the intervening work as required. Keep the ordinary
 * Apple-shaped loops byte-identical by applying it only when a break is below
 * more than one if. */
static bool nested_break_in_list(struct exec_list *list, unsigned if_depth)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_instr *last = nir_block_last_instr(nir_cf_node_as_block(node));
         if (if_depth > 1 && last && last->type == nir_instr_type_jump &&
             nir_instr_as_jump(last)->type == nir_jump_break)
            return true;
         break;
      }
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         if (nested_break_in_list(&nif->then_list, if_depth + 1) ||
             nested_break_in_list(&nif->else_list, if_depth + 1))
            return true;
         break;
      }
      case nir_cf_node_loop:
         /* A break in a nested loop belongs to that loop. */
         break;
      default:
         UNREACHABLE("invalid control-flow node");
      }
   }
   return false;
}

static bool simplify_nested_breaks_list(struct exec_list *list)
{
   bool progress = false;
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         break;
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= simplify_nested_breaks_list(&nif->then_list);
         progress |= simplify_nested_breaks_list(&nif->else_list);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= simplify_nested_breaks_list(&loop->body);
         if (nested_break_in_list(&loop->body, 0)) {
            nir_simplify_loop(loop, nir_jump_break);
            progress = true;
         }
         break;
      }
      default:
         UNREACHABLE("invalid control-flow node");
      }
   }
   return progress;
}

static bool simplify_nested_breaks(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool progress = simplify_nested_breaks_list(&impl->body);
   return nir_progress(progress, impl, nir_metadata_none);
}

/* Out of SSA for the mask-stack code: phis become NIR registers
 * (G9P's order: convert_from_ssa, then trivialize_registers). */
static void cf_out_of_ssa(nir_shader *nir)
{
   NIR_PASS(_, nir, nir_lower_phis_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_convert_from_ssa, true, false);
   NIR_PASS(_, nir, nir_trivialize_registers);
   NIR_PASS(_, nir, simplify_nested_breaks);
   NIR_PASS(_, nir, hoist_break_work);
   NIR_PASS(_, nir, nir_opt_dce);
}

bool pco_g5p_compile_general(nir_shader *nir, bool half_output, const uint8_t *attrib_components,
                             struct pco_g5p_program *out)
{
   return pco_g5p_compile_general_key(nir, half_output, attrib_components, NULL, out);
}

/* ---- control flow (from the G9P compiler) ----
 * Structured on the execution-mask stack, as G9P's g9p_compile.c
 * (OrchardLinux/mesa d91427625ff: emit_if(), emit_if_else(), emit_loop(),
 * emit_break(), emit_branch_cmp(); MIT) and Apple's A9 compiler: an if is a
 * compare, a push, a jump past the arm when no lane takes it, the arm, else
 * and a jump past the other arm (only when there is one), and a pop. A loop
 * pushes two levels, a break takes the lanes out until the loop's pop and
 * jumps to it when no lane is left, and the back jump repeats the body
 * while any lane runs. Register allocation is the linear one: values read
 * in a loop live to its end (mark()), NIR registers hold one register from
 * their first access (the start of a loop using them) to their last. */
static void emit_list(struct gctx *c, struct exec_list *list);

/* NIR registers whose first access is at or before ip. */
static void decl_alloc(struct gctx *c, int ip)
{
   for (unsigned i = 0; i < c->ndecls; i++) {
      struct val *v = V(c, c->decls[i], 0);
      if (c->decl_held[i] || c->decl_first[i] > ip)
         continue;
      c->decl_held[i] = true;
      v->state = VAL_REG;
      const bool out = c->alloc_out;
      c->alloc_out = c->vs && c->vs_bank0 && c->decl_out[i];
      hold(c, v, alloc_reg(c, -1));
      c->alloc_out = out;
   }
}

static void stack_op(struct gctx *c, enum g5p_stack_op op, bool invert, unsigned levels)
{
   uint8_t raw[G5P_USC_MAX];
   /* the counter starts at 0: Apple's first push or loop reads the immediate 0 */
   bool imm = (op == G5P_ST_PUSH || op == G5P_ST_LOOP) && !c->stack_used;
   bool last = !c->count_pass && c->st_last == (int)c->nput;
   if (c->count_pass)
      c->st_last = c->nput;
   put(c, raw, g5p_usc_stack(raw, op, H(c, c->st) / 2, imm, invert, levels, last && !imm), "mask stack");
   c->stack_used = true;
}

static unsigned jump(struct gctx *c, bool any, int offset)
{
   uint8_t raw[G5P_USC_MAX];
   unsigned at = c->size;
   put(c, raw, g5p_usc_jump(raw, any, offset), "jump");
   return at;
}

static void jump_here(struct gctx *c, unsigned at)
{
   if (!g5p_usc_jump_patch(c->code + at, (int)(c->size - at)))
      refuse(c, "jump: %s", g5p_usc_error ? g5p_usc_error : "?");
}

/* A control-flow compare operand: a register of the compare's bank. Apple's
 * vertex compares (v_loop, v_brk, v_ifelse) read only bank-0 registers, so a
 * vertex value held in bank 1 is first moved into a bank-0 scratch register
 * (the class-98 move, destination bank 0, header 0x21). *group is the
 * compare's read group. */
static unsigned cf_operand(struct gctx *c, unsigned reg, unsigned *group)
{
   if (!c->vs || !vbank(c, reg))
      return reg;
   int b0 = alloc_bank0(c);
   if (b0 < 0)
      refuse(c, "register pressure: no bank-0 register for a vertex control-flow compare");
   uint8_t raw[G5P_USC_MAX];
   put(c, raw, g5p_usc_move(raw, H(c, b0), H(c, reg), 0, *group, vhdr(c, 0x21, reg, -1, -1)),
       "compare operand into bank 0");
   c->scratch |= BITFIELD64_BIT(b0);
   *group = 1;
   return b0;
}

/* The compare an if (or a break) reads; *sense: "passed" means the
 * condition holds. A fused compare of two registers, else the 0.0/1.0
 * boolean as unsigned > 0. */
static void emit_cond(struct gctx *c, nir_if *nif, bool *sense)
{
   uint8_t raw[G5P_USC_MAX];
   nir_alu_instr *cmp = fused_cmp(c, nif->condition.ssa, 0);
   if (!cmp) {
      struct ref r = resolve_value(c, nif->condition.ssa, 0);
      unsigned reg = reg_of(c, r, "if condition");
      unsigned g = vgrp(c, r);
      reg = cf_operand(c, reg, &g);
      put(c, raw, g5p_usc_cf_compare_imm(raw, G5P_UGT, H(c, reg) / 2, g, 0), "if compare");
      *sense = true;
      free_scratch(c);
      return;
   }
   enum g5p_cmp op;
   switch (cmp->op) {
   case nir_op_flt: op = G5P_FLT; *sense = true; break;
   case nir_op_fge: op = G5P_FLT; *sense = false; break;
   case nir_op_feq: op = G5P_FEQ; *sense = true; break;
   case nir_op_fneu: op = G5P_FEQ; *sense = false; break;
   case nir_op_ilt: op = G5P_ILT; *sense = true; break;
   case nir_op_ige: op = G5P_ILT; *sense = false; break;
   case nir_op_ult: op = G5P_ULT; *sense = true; break;
   case nir_op_uge: op = G5P_ULT; *sense = false; break;
   case nir_op_ieq: op = G5P_IEQ; *sense = true; break;
   default: op = G5P_IEQ; *sense = false; break; /* ine */
   }
   struct ref a = resolve_value(c, cmp->src[0].src.ssa, cmp->src[0].swizzle[0]);
   struct ref b = resolve_value(c, cmp->src[1].src.ssa, cmp->src[1].swizzle[0]);
   unsigned ar = reg_of(c, a, "if compare"), br = reg_of(c, b, "if compare");
   unsigned ga = vgrp(c, a), gb = vgrp(c, b);
   ar = cf_operand(c, ar, &ga);
   br = cf_operand(c, br, &gb);
   put(c, raw, g5p_usc_cf_compare(raw, op, H(c, ar) / 2, ga, H(c, br) / 2, gb), "if compare");
   free_scratch(c);
}

static void emit_break(struct gctx *c, bool when_true)
{
   stack_op(c, G5P_ST_BREAK, when_true, 2);
   if (c->nexits == G5P_MAX_EXITS)
      refuse(c, "more than %u breaks", G5P_MAX_EXITS);
   c->exits[c->nexits++] = jump(c, false, 0);
}

static void emit_if_else(struct gctx *c, nir_if *nif, bool sense)
{
   stack_op(c, G5P_ST_PUSH, !sense, 1);
   c->cf_depth++;
   c->pushes++;
   unsigned skip = jump(c, false, 0);
   emit_list(c, &nif->then_list);
   if (!nir_cf_list_is_empty_block(&nif->else_list)) {
      jump_here(c, skip);
      stack_op(c, G5P_ST_ELSE, false, 1);
      skip = jump(c, false, 0);
      emit_list(c, &nif->else_list);
   }
   jump_here(c, skip);
   c->cf_depth--;
   c->pushes--;
   stack_op(c, G5P_ST_POP, false, 1);
}

static void emit_if(struct gctx *c, nir_if *nif)
{
   unsigned side, shape = break_shape(nif, &side);
   bool sense;
   /* Apple breaks only at the loop's own level; an arm emitted inline after
    * a break (below) is at that level too. */
   if (shape && c->pushes)
      refuse(c, "break inside a branch of its loop (under a pushed if)");
   c->now = cf_ip_before(&nif->cf_node);
   decl_alloc(c, c->now);
   emit_cond(c, nif, &sense);
   release(c, c->now);
   if (shape == 1) {
      /* nothing but the break: those lanes leave now, the others take the
       * other arm, inline */
      emit_break(c, side ? !sense : sense);
      emit_list(c, side ? &nif->then_list : &nif->else_list);
      return;
   }
   emit_if_else(c, nif, sense);
   if (shape == 2) {
      /* work, then the break: after the arms, the lanes that took it leave */
      c->now = cf_ip_after(&nif->cf_node);
      decl_alloc(c, c->now);
      emit_cond(c, nif, &sense);
      release(c, c->now);
      emit_break(c, side ? !sense : sense);
   }
}

static void emit_loop(struct gctx *c, nir_loop *l)
{
   c->now = nir_loop_first_block(l)->start_ip;
   decl_alloc(c, c->now);
   stack_op(c, G5P_ST_LOOP, false, 2);
   unsigned first_exit = c->nexits, body = c->size, pushes = c->pushes;
   c->cf_depth++;
   c->pushes = 0;
   emit_list(c, &l->body);
   c->pushes = pushes;
   jump(c, true, (int)body - (int)c->size);
   if (c->nexits == first_exit)
      refuse(c, "loop without a break");
   for (unsigned i = first_exit; i < c->nexits; i++)
      jump_here(c, c->exits[i]);
   c->nexits = first_exit;
   c->cf_depth--;
   stack_op(c, G5P_ST_POP, false, 2);
}

static void emit_block(struct gctx *c, nir_block *block)
{
   nir_foreach_instr(ins, block) {
      c->now = ins->index;
      if (c->ndecls)
         decl_alloc(c, c->now);
      switch (ins->type) {
      case nir_instr_type_alu:
         emit_alu(c, nir_instr_as_alu(ins));
         break;
      case nir_instr_type_intrinsic:
         emit_intrinsic(c, nir_instr_as_intrinsic(ins));
         break;
      case nir_instr_type_tex:
         emit_tex(c, nir_instr_as_tex(ins));
         break;
      case nir_instr_type_jump: {
         /* a break is emitted by the if whose arm it ends (emit_if()) */
         nir_cf_node *p = block->cf_node.parent;
         unsigned side;
         if (p->type != nir_cf_node_if || !break_shape(nir_cf_node_as_if(p), &side) ||
             single_block(side ? &nir_cf_node_as_if(p)->else_list : &nir_cf_node_as_if(p)->then_list) != block)
            refuse(c, "a break that does not end an if arm");
         break;
      }
      default:
         break;
      }
      /* g5p-varyings' interleave: outputs written as soon as final, straight-line programs only (an output
       * written inside an if arm or a loop would be masked) */
      if (c->vs && c->eager_vout && !c->cf)
         eager_outputs(c, ins->index);
      release(c, c->now);
      park(c, ins);
   }
}

static void emit_list(struct gctx *c, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         emit_block(c, nir_cf_node_as_block(node));
         break;
      case nir_cf_node_if:
         emit_if(c, nir_cf_node_as_if(node));
         break;
      case nir_cf_node_loop:
         emit_loop(c, nir_cf_node_as_loop(node));
         break;
      default:
         refuse(c, "control-flow node type %u", node->type);
      }
   }
}

/* Allocation and emission over the lowered, scanned NIR. */
static bool codegen(struct gctx *c)
{
   struct pco_g5p_program *out = c->p;
   if (setjmp(c->fail))
      return false;
   c->nvals = c->impl->ssa_alloc * 4;
   c->vals = calloc(c->nvals, sizeof(*c->vals));
   c->sat_folded = calloc(c->impl->ssa_alloc, sizeof(bool));
   c->saturate = calloc(c->impl->ssa_alloc, sizeof(bool));
   c->fused = calloc(c->impl->ssa_alloc, sizeof(bool));
   c->rh_half = malloc(c->impl->ssa_alloc * sizeof(int));
   if (!c->vals || !c->sat_folded || !c->saturate || !c->fused || !c->rh_half)
      refuse(c, "out of memory");
   for (unsigned i = 0; i < c->nvals; i++)
      c->vals[i].last = -1;
   for (unsigned i = 0; i < c->impl->ssa_alloc; i++)
      c->rh_half[i] = -1;
   memset(c->decl_held, 0, sizeof(c->decl_held));
   c->stack_used = false;
   c->cf_depth = c->nexits = c->pushes = 0;
   plan_saturate(c);
   plan_bools(c);
   if (c->vs)
      plan_vs_layout(c);
   else
      plan_varyings(c);
   liveness(c);
   foreach_instr_all(ins, c) {
      nir_def *d = nir_instr_def(ins);
      if (!d || (ins->type != nir_instr_type_load_const && ins->type != nir_instr_type_undef))
         continue;
      if (c->vs && address_const(d))
         continue; /* a store's immediate dword offset */
      if (d->bit_size != 32 && d->bit_size != 1)
         refuse(c, "%u-bit constant", d->bit_size);
      for (unsigned k = 0; k < d->num_components; k++) {
         struct val *v = V(c, d, k);
         v->state = VAL_CONST;
         v->imm = 0;
         if (ins->type == nir_instr_type_load_const) {
            nir_const_value cv = nir_instr_as_load_const(ins)->value[k];
            v->imm = d->bit_size == 1 ? (cv.b ? F32_ONE : 0) : cv.u32; /* booleans are 0.0 / 1.0 */
         }
      }
   }
   if (c->vs) {
      plan_common(c);
      collect_vs_outputs(c);
      setup_inputs(c);
   } else {
      plan_common(c);
      plan_fs_output(c);
      c->reserved = BITFIELD64_RANGE(0, c->mrt.words); /* packed colour halves, o0..o(2N-1) */
      if (c->cf && c->reserve_st31)
         c->reserved |= BITFIELD64_BIT(NUM_DWORDS - 1);
      c->cube_gradient = false;
      foreach_instr_all(ins, c) {
         if (ins->type == nir_instr_type_tex) {
            nir_tex_instr *t = nir_instr_as_tex(ins);
            c->cube_gradient |= t->op == nir_texop_txd && t->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
         }
      }
      if (c->cube_gradient)
         c->reserved |= BITFIELD64_RANGE(0, 9);
      c->max_dword = MAX2(c->mrt.words, c->cube_gradient ? 9 : 0);
      c->max_written = MAX2(c->max_written, c->cube_gradient ? 9 : 0);
      if (c->cf && !c->count_pass)
         c->reserved |= BITFIELD64_BIT(c->st); /* the mask stack, above every register of the count pass */
      if (!c->cube_gradient) fs_prologue(c);
      hoist_iterators(c);
      release(c, -1);
   }
   emit_list(c, &c->impl->body);
   c->now = LAST_END;
   if (c->vs)
      vs_output(c);
   else {
      /* The fixed tuple overwrites packed-output registers. Emit constant
       * colour halves only after its last use, before dynamic output packs. */
      if (c->cube_gradient) fs_prologue(c);
      fs_output(c);
   }
   out->code = malloc(c->size);
   if (!out->code)
      refuse(c, "out of memory");
   memcpy(out->code, c->code, c->size);
   out->size = c->size;
   if (c->cf && c->vs && c->st >= VS_BANK0)
      c->max_bank0 = MAX2(c->max_bank0, c->st - VS_BANK0 + 1);
   else if (c->cf)
      c->max_dword = MAX2(c->max_dword, c->st + 1);
   out->temps = c->vs && c->vs_bank0 ? MAX2(8, align(c->max_dword, 4)) :
                MAX3(8, align(c->max_dword, 4), align(c->max_bank0, 4));
   if (c->nhomes) {
      out->temps = MAX2(out->temps, align(32 + c->nhomes, 4));
      out->evidence |= PCO_G5P_EV_HIGH_REGS;
   }
   out->written_temps = c->vs ? (c->vs_bank0 ? MAX2(8, align(c->max_written, 4)) :
                                  MAX3(8, align(c->max_written, 4), align(c->max_bank0, 4))) : out->temps;
   /* Mixed-bank programs declare bank0 separately; legacy texture launches retain their proven accounting. */
   out->bank0_temps = c->vs && c->vs_bank0 && c->max_bank0 ? align(c->max_bank0, 8) : 0;
   out->common_dwords = c->common_next;
   if (!c->vs)
      out->varyings = out->num_varyings;
   if (out->varyings > 1)
      out->evidence |= PCO_G5P_EV_VARYINGS;
   out->side_effects = impl_has_side_effects(c->impl);
   out->refusal[0] = 0;
   return true;
}

static void codegen_reset(struct gctx *c)
{
   free(c->vals);
   free(c->sat_folded);
   free(c->saturate);
   free(c->fused);
   free(c->rh_half);
   struct gctx keep = *c;
   memset(c, 0, offsetof(struct gctx, fail));
   c->code = keep.code;
   c->code_cap = keep.code_cap;
   c->lastput = keep.lastput;
   memcpy(c->klast, keep.klast, sizeof(c->klast));
   c->remat = keep.remat;
   c->nir = keep.nir;
   c->impl = keep.impl;
   c->vs = keep.vs;
   c->half_output = keep.half_output;
   c->attrib_components = keep.attrib_components;
   c->key = keep.key;
   c->blend = keep.blend;
   c->binfo = keep.binfo;
   c->p = keep.p;
   c->one_reg = c->zero_reg = c->ione_reg = c->fragcoord = -1;
   c->size = 0;
   c->busy = c->reserved = 0;
   c->max_dword = c->max_written = c->max_bank0 = 0;
   c->tex_bank0 = false;
   c->ntex = c->nranges = c->nssbo = c->common_next = 0;
   c->nvout = c->ndiscards = 0;
   c->discard_always = false;
   c->narrow_tex = NULL;
   c->narrow_iter = NULL;
   memset(c->narrow_sat, 0, sizeof(c->narrow_sat));
   memset(c->out, 0, sizeof(c->out));
   memset(c->out_written, 0, sizeof(c->out_written));
   c->has_depth = false;
   memset(&c->depth, 0, sizeof(c->depth));
   memset(c->slot_written, 0, sizeof(c->slot_written));
   uint32_t ev = c->p->evidence & (PCO_G5P_EV_FLATTEN | PCO_G5P_EV_CF);
   unsigned blend = c->p->blend_mode;
   bool vertex = c->p->vertex;
   bool discard_select = c->p->discard_select;
   memset(c->p, 0, sizeof(*c->p));
   c->p->discard_select = discard_select;
   c->p->evidence = ev;
   c->p->blend_mode = blend;
   c->p->vertex = vertex;
}

/* Round 19: a counting pass records each register's last reading
 * instruction (vgrp/kgrp); the emitting pass sets the group bits from it.
 * Group bits and hints never change allocation, so both passes emit the
 * same instruction sequence. */
static bool two_pass(struct gctx *c)
{
   for (unsigned i = 0; i < c->impl->ssa_alloc * 4; i++)
      c->lastput[i] = -1;
   c->klast[0] = c->klast[1] = c->klast[2] = -1;
   /* Control flow always takes Apple's last-read groups (a read inside a
    * loop is never the last), and the count pass places the mask stack. */
   if (!c->vs && !(c->key->last_read_groups || c->cf))
      return codegen(c);
   c->count_pass = true;
   c->st = c->reserve_st31 ? NUM_DWORDS - 1 : 0;
   c->st_last = -1;
   if (!codegen(c))
      return false;
   if (c->cf && c->vs) {
      /* bank 0, after the program's own bank-0 registers (Apple's v_loop: the
       * stack in bank-0 r0, the loop counter r1) */
      c->st = VS_BANK0 + c->max_bank0;
      if (c->max_bank0 >= NUM_DWORDS) {
         free(c->p->code);
         c->p->code = NULL;
         snprintf(c->p->refusal, sizeof(c->p->refusal),
                  "vertex: register pressure: no bank-0 register left for the mask stack (%u of %u in use)",
                  c->max_bank0, NUM_DWORDS);
         return false;
      }
   } else if (c->cf) {
      /* The ordinary path keeps Apple's allocation: the stack is the first
       * register after the program. If that first count used all 32 low
       * registers, reserve r31 and count again. The second count then fails
       * at the real pressure point, allowing the existing high-register-home
       * retry to park a live value above r31 instead of discovering the full
       * bank only after every value has been released at program end. */
      if (!c->reserve_st31 && c->max_dword >= NUM_DWORDS) {
         free(c->p->code);
         c->p->code = NULL;
         codegen_reset(c);
         c->reserve_st31 = true;
         c->count_pass = true;
         c->st = NUM_DWORDS - 1;
         c->st_last = -1;
         if (!codegen(c))
            return false;
      }
      c->st = c->reserve_st31 ? NUM_DWORDS - 1 : c->max_dword;
   }
   free(c->p->code);
   codegen_reset(c);
   c->count_pass = false;
   return codegen(c);
}

bool pco_g5p_compile_general_key(nir_shader *nir, bool half_output, const uint8_t *attrib_components,
                                 const struct pco_g5p_fs_key *key, struct pco_g5p_program *out)
{
   return pco_g5p_compile_general_link(nir, half_output, attrib_components, key, NULL, out);
}

bool pco_g5p_compile_general_link(nir_shader *nir, bool half_output, const uint8_t *attrib_components,
                                  const struct pco_g5p_fs_key *key, const struct pco_g5p_vs_link *link,
                                  struct pco_g5p_program *out)
{
   static const struct pco_g5p_fs_key default_key = {.write_mask = 0xf};
   memset(out, 0, sizeof(*out));
   struct gctx *c = calloc(1, sizeof(*c));
   if (!c) {
      snprintf(out->refusal, sizeof(out->refusal), "out of memory");
      return false;
   }
   c->nir = nir;
   c->p = out;
   c->vs = nir->info.stage == MESA_SHADER_VERTEX;
   /* Offline test only: the bank-0 placement from the first attempt, so the
    * mixed-bank forms of every emitter appear in ordinary programs
    * (bank-check.py and tuple-check.py over the compiled corpus). */
   if (c->vs && getenv("PVR_G5P_VS_BANK0_FIRST") && strcmp(getenv("PVR_G5P_VS_BANK0_FIRST"), "0"))
      c->vs_bank0 = true;
   c->half_output = half_output;
   c->attrib_components = attrib_components;
   c->key = key ? key : &default_key;
   c->link = link;
   c->one_reg = c->zero_reg = c->ione_reg = c->fragcoord = -1;
   out->vertex = c->vs;
   bool ok = false;
   if (setjmp(c->fail))
      goto done;
   if (!c->vs && nir->info.stage != MESA_SHADER_FRAGMENT)
      refuse(c, "stage %u", nir->info.stage);
   if (!c->vs && !half_output && c->key->color_output == PCO_G5P_COLOR_F16)
      refuse(c, "only packed-f16 colour output (8-bit UNORM targets) is implemented");
   if (!c->vs && c->key->depth_feedback && storage_with_depth(nir))
      refuse(c, "fragment depth export with memory side effects is not qualified");
   if (!c->vs) {
      /* Colour targets: 0 means one. Several targets take no blend and only the full write mask. */
      enum g5p_mrt_refusal mr = g5p_mrt_out_plan_make(c->key->color_targets ? c->key->color_targets : 1, &c->mrt);
      if (mr == G5P_MRT_OK)
         mr = g5p_mrt_check_key(&c->mrt, c->key->blend, c->key->write_mask);
      if (mr != G5P_MRT_OK)
         refuse(c, "%s", g5p_mrt_refusal_str(mr));
      if (c->key->color_output != PCO_G5P_COLOR_F16) {
         if (c->key->color_output != PCO_G5P_COLOR_UINT32 &&
             c->key->color_output != PCO_G5P_COLOR_SINT32)
            refuse(c, "unknown colour output format");
         /* two targets: eight output words, the most one emitpix carries */
         if (c->mrt.targets > 2 || c->key->blend)
            refuse(c, "raw32 output needs one or two unblended targets (four words each, at most eight)");
      } else {
         apply_fs_key(c, nir, c->key);
      }
   }
   out->blend_mode = c->blend == BLEND_REPLACE ? 0 : c->blend;
   if (!c->vs && c->key->color_output != PCO_G5P_COLOR_F16 && c->key->write_mask != 0xf)
      out->blend_mode = BLEND_GENERAL; /* destination read requires TRANSLUCENT ISP */
   if (lower(nir))
      out->evidence |= PCO_G5P_EV_FLATTEN;
   fuse_ffma(c, nir);
   if (!c->vs && cf_left(nir))
      cf_discards(nir);
   if (c->vs || !c->key->punch)
      select_discards(c, nir);
   if (!c->vs && c->key->punch) {
      bool progress = false;
      NIR_PASS(progress, nir, merge_discards);
      if (progress) {
         NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
         do {
            progress = false;
            NIR_PASS(progress, nir, nir_opt_copy_prop);
            NIR_PASS(progress, nir, nir_opt_dce);
            NIR_PASS(progress, nir, nir_opt_constant_folding);
            NIR_PASS(progress, nir, nir_opt_algebraic);
            NIR_PASS(progress, nir, nir_opt_cse);
         } while (progress);
      }
   }
   if (cf_left(nir)) {
      c->cf = true;
      out->evidence |= PCO_G5P_EV_CF;
      cf_out_of_ssa(nir);
      /* Vertex control flow (Apple's v_loop, v_brk, v_ifelse, v_nested): the
       * loop state, the compares and the mask stack in bank 0, the values of
       * outputs in bank 1 (vs-two-bank-allocator.md, stage C). */
      if (c->vs)
         c->vs_bank0 = true;
   }
   if (getenv("G5P_PRINT_NIR"))
      nir_print_shader(nir, stderr);
   c->impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(c->impl);
   nir_index_instrs(c->impl);
   if (c->cf) {
      plan_loops(c, &c->impl->body);
      plan_decls(c);
   }
   struct unsupported u = {{0}};
   scan_cf(c, &c->impl->body, &u);
   if (u.n)
      refuse(c, "%s", u.text);
   c->lastput = malloc(c->impl->ssa_alloc * 4 * sizeof(int));
   c->home = malloc(c->impl->ssa_alloc * 4);
   if (!c->lastput || !c->home)
      refuse(c, "out of memory");
   memset(c->home, -1, c->impl->ssa_alloc * 4);
   ok = two_pass(c);
   if (!ok && !c->vs && strstr(out->refusal, "register pressure")) {
      /* Retry: constants and uniforms re-read per use (round 19). */
      codegen_reset(c);
      c->remat = true;
      ok = two_pass(c);
      if (!ok && strstr(out->refusal, "register pressure")) {
         /* Then each varying read where it is loaded, not all ahead. */
         codegen_reset(c);
         c->lazy_iter = true;
         ok = two_pass(c);
      }
   }
   if (c->vs && (ok ? out->temps > PCO_G5P_VS_TEMPS_FLOWN : !!strstr(out->refusal, "register pressure"))) {
      /* More than 12 temps wedge the TA (xfb.md): write each output as
       * soon as its value is final (Apple's v_s* interleave), not all at
       * the end. Shapes within 12 temps keep their round-17..19 bytes.
       * g5p-vpds sizes the unified store past 12 temps (up to 32), so a
       * program that compiled is replaced only when the interleave needs
       * fewer temps: the shapes it does not shrink keep the bytes g5p-vpds
       * flew (t16/t20/t24, attributes, bump). */
      struct pco_g5p_program first = *out;
      bool first_ok = ok;
      codegen_reset(c);
      c->eager_vout = true;
      ok = two_pass(c);
      if (first_ok && (!ok || out->temps >= first.temps)) {
         if (ok)
            free(out->code);
         *out = first;
         ok = true;
      } else if (first_ok) {
         free(first.code);
      }
   }
   /* Vertex, still out of registers: bank 0 holds what bank 1 cannot, as in
    * Apple's vertex programs (v_press_vec4: bank 1 to r15, bank 0 to r22).
    * Every program that fits bank 1 keeps its bytes. Plain, then with the
    * eager outputs (fewer values live in bank 1 at the end). */
   for (unsigned eager = 0; c->vs && !ok && eager < 2 && strstr(out->refusal, "register pressure"); eager++) {
      codegen_reset(c);
      c->vs_bank0 = true;
      c->eager_vout = eager;
      ok = two_pass(c);
   }
   /* Still out of registers: one more home above r31 per attempt. */
   while (!ok && !c->vs && strstr(out->refusal, "register pressure") && c->nhomes < G5P_MAX_HOMES) {
      int victim = pick_home(c);
      if (victim < 0)
         break;
      c->home[victim] = 32 + c->nhomes++;
      codegen_reset(c);
      ok = two_pass(c);
   }
done:
   free(c->vals);
   free(c->sat_folded);
   free(c->saturate);
   free(c->fused);
   free(c->rh_half);
   free(c->lastput);
   free(c->home);
   free(c->code);
   free(c);
   return ok;
}
