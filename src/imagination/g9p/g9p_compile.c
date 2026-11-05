/* SPDX-License-Identifier: MIT */
/*
 * NIR -> G9P compiler for compute, vertex and fragment shaders (see
 * g9p_compile.h).
 *
 * Code layout follows Apple's compiler: prologue (mask lanes past the thread
 * count), execution-mask stack in the register after the program's own,
 * body, epilogue, jump to the end word. r0 holds the thread's index in its
 * workgroup at entry.
 *
 * Control flow is structured on the mask stack: an if pushes a level, else
 * flips it, the end pops it; a loop pushes two levels (loop, break), breaks
 * disable lanes, and the back jump repeats while any lane is active. NIR is
 * taken out of SSA; its registers get fixed hardware registers, SSA values
 * are allocated by liveness, extended over the loops they are used in.
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/spirv/nir_spirv.h"
#include "util/ralloc.h"
#include "util/u_math.h"

#include <vulkan/vulkan_core.h>

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_schedule.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "util/format/u_format.h"
#include "g9p_compile.h"
#include "g9p_isa.h"

/* r0-r29 for values, the mask stack after them (G9P_REGS: debug override) */
static unsigned
g9p_num_regs(void)
{
   const char *e = getenv("G9P_REGS");

   return e ? MIN2(atoi(e), 63) : 30;
}
#define G9P_NUM_REGS g9p_num_regs()
#define G9P_MAX_LOOPS 64

static const struct spirv_to_nir_options spirv_options = {
   .environment = NIR_SPIRV_VULKAN,
   .ubo_addr_format = nir_address_format_32bit_index_offset,
   .ssbo_addr_format = nir_address_format_32bit_index_offset,
   .phys_ssbo_addr_format = nir_address_format_64bit_global,
   .push_const_addr_format = nir_address_format_32bit_offset,
   .shared_addr_format = nir_address_format_32bit_offset,
   .min_ubo_alignment = 16,
   .min_ssbo_alignment = 4,
};

static const nir_shader_compiler_options nir_options = {
   /* multiply-adds as one instruction (g9p_ffma()), not a mul and an add */
   .float_mul_add32 = nir_float_muladd_support_has_ffma | nir_float_muladd_support_fuse,
   .lower_int64_options = ~0U,
   .lower_fdiv = true,
   .lower_iabs = true,
   .lower_mul_high = true,
   .lower_isign = true,
   .lower_fmod = true,
   .lower_fpow = true,
   .lower_fsqrt = true,
   .lower_ffract = true,
   .lower_fceil = true,
   .lower_ftrunc = true,
   .lower_fround_even = true,
   .lower_flrp32 = true,
   .lower_fsat = true,
   .lower_find_lsb = true,
   .lower_ifind_msb = true,
   .lower_iadd_sat = true,
   .lower_mul_2x32_64 = true,
   .lower_device_index_to_zero = true,
   .lower_extract_byte = true,
   .lower_extract_word = true,
   .lower_insert_byte = true,
   .lower_insert_word = true,
   .max_unroll_iterations = 0,
};

const struct spirv_to_nir_options *
g9p_spirv_options(void)
{
   return &spirv_options;
}

const nir_shader_compiler_options *
g9p_nir_options(void)
{
   return &nir_options;
}

struct g9p_ctx {
   unsigned num_keys;
   struct g9p_code *c;
   struct g9p_shader *out;
   mesa_shader_stage stage;
   /* Fragment: gl_FragCoord's varying */
   unsigned frag_coord_slot;

   /* Vertex: the attribute layout */
   const struct g9p_vertex_attrib *attribs;
   /*
    * Outputs, written at the end (nir_lower_io_to_temporaries keeps their
    * stores there): vertex slots, fragment colour components; their
    * registers stay pinned from the store on
    */
   struct {
      int reg;
      unsigned slot;
   } outputs[4 + 4 * G9P_MAX_VARYINGS];
   unsigned num_outputs;
   int colour[4];

   /* Per SSA scalar (def index * 4 + component) */
   int8_t *reg;
   unsigned *last_use;
   /* Per SSA def: the instruction defining it */
   unsigned *def_ip;
   /* Per decl_reg def: its first use (live range start) */
   unsigned *reg_first;
   /* The decl_reg defs */
   nir_def **decls;
   unsigned num_decls;

   /*
    * Fragment NIR registers parked above r31 (per decl_reg key, -1: none),
    * chosen across generate() attempts when registers run out
    */
   int8_t *home;
   unsigned num_homes;
   bool oom;          /* the last generate() ran out of registers */
   unsigned oom_ip;

   uint64_t busy;   /* registers in use */
   uint64_t pinned; /* NIR registers: never released */
   uint64_t temps;  /* released after the current instruction */
   unsigned max_reg;
   unsigned st;     /* mask stack register */
   unsigned tex_total;  /* textures the shader samples (count_textures()) */
   bool flush_outputs;  /* vertex outputs written at once, for shaders that ran out */
   bool point_size;     /* a point pipeline's vertex shader: the size is the last output */
   int psize_reg;
   uint64_t flushed_outputs;
   int vertex_reg, instance_reg; /* vertex shaders: r0-r1, or copies of them */
   bool reserve_st_low; /* the mask stack is r15, reserved (compile_back()) */
   bool reserve_low;    /* r11-r14 for address work, after running out (compile_back()) */
   unsigned ip;

   struct {
      unsigned start, end;
   } loops[G9P_MAX_LOOPS];
   unsigned num_loops, loop_index;

   /* Jumps to the innermost loop's exit (after a break), patched at its end */
   unsigned *exit_jumps;
   unsigned num_exit_jumps, max_exit_jumps;
};

static unsigned
key(const nir_def *def, unsigned comp)
{
   return def->index * 4 + comp;
}

static void
dump_busy(struct g9p_ctx *ctx)
{
   ctx->oom = true;
   ctx->oom_ip = ctx->ip;
   if (getenv("G9P_DEBUG"))
      fprintf(stderr, "g9p: out of registers at instruction %u\n", ctx->ip);
   if (!getenv("G9P_DEBUG"))
      return;
   /* NIR registers are pinned: the busy ones not listed below */
   fprintf(stderr, "  busy %#" PRIx64 " pinned %#" PRIx64 " temps %#" PRIx64 "\n", ctx->busy,
           ctx->pinned, ctx->temps);
   for (unsigned k = 0; k < ctx->num_keys; k++) {
      if (ctx->reg[k] >= 0 && (ctx->busy & BITFIELD64_BIT(ctx->reg[k])) &&
          !(ctx->pinned & BITFIELD64_BIT(ctx->reg[k])) && ctx->last_use[k] >= ctx->ip)
         fprintf(stderr, "  r%d: value %u.%u, last use %u\n", ctx->reg[k], k / 4, k % 4,
                 ctx->last_use[k]);
   }
}

static int
alloc_reg(struct g9p_ctx *ctx)
{
   /*
    * A vertex shader that ran out: values above r15 first, and r11-r14 stay
    * free, for what reaches r15 only (one_src(), alloc_pair()).
    */
   if (ctx->reserve_low) {
      for (unsigned r = 16; r < G9P_NUM_REGS; r++) {
         if (!(ctx->busy & BITFIELD64_BIT(r))) {
            ctx->busy |= BITFIELD64_BIT(r);
            ctx->max_reg = MAX2(ctx->max_reg, r);
            return r;
         }
      }
   }
   for (unsigned r = 0; r < G9P_NUM_REGS; r++) {
      if (ctx->reserve_low && r >= 11 && r <= 14)
         continue;
      if (!(ctx->busy & BITFIELD64_BIT(r))) {
         ctx->busy |= BITFIELD64_BIT(r);
         ctx->max_reg = MAX2(ctx->max_reg, r);
         return r;
      }
   }
   dump_busy(ctx);
   return -1;
}

/* A scratch register for an address index: its operand reaches r15 only too */
static int
alloc_low_temp(struct g9p_ctx *ctx)
{
   for (unsigned r = 0; r < MIN2(16, G9P_NUM_REGS); r++) {
      if (!(ctx->busy & BITFIELD64_BIT(r))) {
         ctx->busy |= BITFIELD64_BIT(r);
         ctx->temps |= BITFIELD64_BIT(r);
         ctx->max_reg = MAX2(ctx->max_reg, r);
         return r;
      }
   }
   dump_busy(ctx);
   return -1;
}

/* @n consecutive registers below r16, for a vertex shader's sample */
static int
alloc_low_range(struct g9p_ctx *ctx, unsigned n)
{
   for (unsigned r = 0; r + n <= MIN2(16, G9P_NUM_REGS); r++) {
      if (!(ctx->busy & BITFIELD64_RANGE(r, n))) {
         ctx->busy |= BITFIELD64_RANGE(r, n);
         ctx->max_reg = MAX2(ctx->max_reg, r + n - 1);
         return r;
      }
   }
   dump_busy(ctx);
   return -1;
}

/* @r, or a copy of it below r16 for an operand that reaches r15 only */
static int
low_reg(struct g9p_ctx *ctx, int r)
{
   int low;

   if (r < 16)
      return r;
   low = alloc_low_temp(ctx);
   if (low >= 0)
      g9p_alu2(ctx->c, G9P_OR, low, r, r, false);
   return low;
}

/*
 * Forms with one source and an immediate (1189, 1199, 1181, 199e) reach r15
 * only outside fragment code: read r16 they gave wrong values (dEQP vertex
 * shaders of 17 registers and more). Two-source forms reach all registers.
 */
static int
one_src(struct g9p_ctx *ctx, int r)
{
   return ctx->c->fragment ? r : low_reg(ctx, r);
}

/*
 * A pair for a memory address: its operand reaches r15 only (a pair across r16
 * read the wrong memory: dEQP vertex shaders of 17 registers and more)
 */
static int
alloc_pair(struct g9p_ctx *ctx)
{
   for (unsigned r = 0; r + 1 < MIN2(16, G9P_NUM_REGS); r++) {
      if (!(ctx->busy & BITFIELD64_RANGE(r, 2))) {
         ctx->busy |= BITFIELD64_RANGE(r, 2);
         ctx->max_reg = MAX2(ctx->max_reg, r + 1);
         return r;
      }
   }
   dump_busy(ctx);
   return -1;
}

/* @n consecutive registers */
static int
alloc_range(struct g9p_ctx *ctx, unsigned n)
{
   for (unsigned r = 0; r + n <= G9P_NUM_REGS; r++) {
      if (!(ctx->busy & BITFIELD64_RANGE(r, n))) {
         ctx->busy |= BITFIELD64_RANGE(r, n);
         ctx->max_reg = MAX2(ctx->max_reg, r + n - 1);
         return r;
      }
   }
   dump_busy(ctx);
   return -1;
}

static int
alloc_temp(struct g9p_ctx *ctx)
{
   int r = alloc_reg(ctx);

   if (r >= 0)
      ctx->temps |= BITFIELD64_BIT(r);
   return r;
}

static bool
is_const(nir_scalar s, uint32_t *value)
{
   if (nir_scalar_is_const(s)) {
      *value = nir_scalar_as_uint(s);
      return true;
   }
   /*
    * A storage buffer address is vec2(binding index, offset); the offset of
    * a plain binding is 0 (the descriptor address already includes the
    * buffer offset).
    */
   if (nir_scalar_is_intrinsic(s) && s.comp == 1) {
      nir_intrinsic_op op = nir_def_as_intrinsic(s.def)->intrinsic;

      if (op == nir_intrinsic_load_vulkan_descriptor ||
          op == nir_intrinsic_vulkan_resource_index) {
         *value = 0;
         return true;
      }
   }
   return false;
}

/* A temporary holding @value */
static int
const_reg(struct g9p_ctx *ctx, uint32_t value)
{
   int r = alloc_temp(ctx);

   if (r >= 0)
      g9p_mov_imm(ctx->c, r, value);
   return r;
}

/* The register holding @s, materializing a constant into a temporary */
static int
src_reg(struct g9p_ctx *ctx, nir_scalar s)
{
   uint32_t value;
   int r;

   if (is_const(s, &value))
      return const_reg(ctx, value);
   r = ctx->reg[key(s.def, s.comp)];
   if (r < 0 && ctx->home && ctx->home[key(s.def, s.comp)] >= 0) {
      /* parked above r31: back into a temporary for this instruction */
      r = alloc_temp(ctx);
      if (r >= 0)
         g9p_mov_from_high(ctx->c, r, ctx->home[key(s.def, s.comp)]);
      return r;
   }
   if (r < 0)
      fprintf(stderr, "g9p: value %u has no register\n", s.def->index);
   return r;
}

static void
release_scalar(struct g9p_ctx *ctx, nir_def *def, unsigned comp)
{
   unsigned k = key(def, comp);

   if (ctx->last_use[k] > ctx->ip || ctx->reg[k] < 0)
      return;
   /* a NIR register's last use: unpinned and free */
   if (nir_def_is_intrinsic(def) &&
       nir_def_as_intrinsic(def)->intrinsic == nir_intrinsic_decl_reg) {
      ctx->pinned &= ~BITFIELD64_BIT(ctx->reg[k]);
      ctx->busy &= ~BITFIELD64_BIT(ctx->reg[k]);
      return;
   }
   if (!(ctx->pinned & BITFIELD64_BIT(ctx->reg[k])))
      ctx->busy &= ~BITFIELD64_BIT(ctx->reg[k]);
}

/* A result with a home above r31: copied there, its register freed */
static bool
park_def(nir_def *def, void *data)
{
   struct g9p_ctx *ctx = data;

   if (!ctx->home)
      return true;
   for (unsigned c = 0; c < def->num_components; c++) {
      const unsigned k = key(def, c);
      const int r = ctx->reg[k];

      if (ctx->home[k] < 0 || r < 0 || (ctx->pinned & BITFIELD64_BIT(r)))
         continue;
      g9p_mov_to_high(ctx->c, ctx->home[k], r);
      ctx->busy &= ~BITFIELD64_BIT(r);
      ctx->reg[k] = -1;
   }
   return true;
}

/* A NIR register's hardware register, taken at its first use */
static int
nir_reg(struct g9p_ctx *ctx, nir_def *decl)
{
   int r = ctx->reg[key(decl, 0)];

   if (ctx->home && ctx->home[key(decl, 0)] >= 0)
      return ctx->home[key(decl, 0)];

   if (r < 0) {
      r = alloc_reg(ctx);
      if (r < 0)
         return -1;
      ctx->pinned |= BITFIELD64_BIT(r);
      ctx->reg[key(decl, 0)] = r;
   }
   return r;
}

static bool
release_src(nir_src *src, void *data)
{
   struct g9p_ctx *ctx = data;

   for (unsigned i = 0; i < nir_src_num_components(*src); i++)
      release_scalar(ctx, src->ssa, i);
   return true;
}

/* Release the sources whose last use is this instruction, and the temporaries */
static void
release(struct g9p_ctx *ctx, nir_instr *instr)
{
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);

      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++)
         release_scalar(ctx, alu->src[i].src.ssa, alu->src[i].swizzle[0]);
   } else {
      nir_foreach_src(instr, release_src, ctx);
   }
   ctx->busy &= ~ctx->temps;
   ctx->temps = 0;
}


/*
 * A comparison whose only uses are if conditions: the if compares directly
 * (emit_branch_cmp()) and the 0/~0 value is never built. Building it takes
 * a select/move pair, and two of those in a loop after lanes have left it
 * hang the GPU.
 */
static nir_alu_instr *
branch_compare(nir_def *def)
{
   nir_alu_instr *alu;

   if (!nir_def_is_alu(def))
      return NULL;
   alu = nir_def_as_alu(def);
   switch (alu->op) {
   case nir_op_flt32: case nir_op_fge32: case nir_op_feq32: case nir_op_fneu32:
   case nir_op_ilt32: case nir_op_ige32: case nir_op_ult32: case nir_op_uge32:
   case nir_op_ieq32: case nir_op_ine32:
      break;
   default:
      return NULL;
   }
   nir_foreach_use_including_if(src, def) {
      if (!nir_src_is_if(src))
         return NULL;
   }
   return alu;
}

/*
 * branch_compare() runs the compare at the if, but a load_reg names the
 * register, not its value: a store between the compare and the if (a loop's
 * "i++ < n") or in a branch (compared again for a break, emit_if()) would
 * change what is compared. Such operands get a copy at the compare.
 */
static bool
copy_branch_compare_operands(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);

      nir_foreach_block(block, impl) {
         nir_if *nif = nir_block_get_following_if(block);
         nir_alu_instr *cmp = nif ? branch_compare(nif->condition.ssa) : NULL;

         if (!cmp)
            continue;
         for (unsigned i = 0; i < 2; i++) {
            nir_def *def = cmp->src[i].src.ssa;

            if (!nir_def_is_intrinsic(def) ||
                nir_def_as_intrinsic(def)->intrinsic != nir_intrinsic_load_reg)
               continue;
            b.cursor = nir_before_instr(&cmp->instr);
            nir_src_rewrite(&cmp->src[i].src, nir_mov(&b, def));
            cmp->src[i].swizzle[0] = 0;
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }
   return true;
}

/* --- Liveness ------------------------------------------------------------ */

static void
use_scalar(struct g9p_ctx *ctx, nir_def *def, unsigned comp)
{
   unsigned end = ctx->ip;

   /* Used in a loop it is defined outside of: live until the loop's end */
   for (unsigned l = 0; l < ctx->num_loops; l++) {
      if (ctx->loops[l].start <= ctx->ip && ctx->ip <= ctx->loops[l].end &&
          ctx->def_ip[def->index] < ctx->loops[l].start)
         end = MAX2(end, ctx->loops[l].end);
   }
   ctx->last_use[key(def, comp)] = MAX2(ctx->last_use[key(def, comp)], end);

   /*
    * NIR registers (decl_reg, all at the start) take a hardware register
    * only from their first use (from the start of a loop using them) to
    * their last; a load_reg's value keeps its register alive too.
    */
   if (nir_def_is_intrinsic(def)) {
      nir_intrinsic_instr *intr = nir_def_as_intrinsic(def);

      if (intr->intrinsic == nir_intrinsic_decl_reg) {
         unsigned start = ctx->ip;

         for (unsigned l = 0; l < ctx->num_loops; l++) {
            if (ctx->loops[l].start <= ctx->ip && ctx->ip <= ctx->loops[l].end)
               start = MIN2(start, ctx->loops[l].start);
         }
         ctx->reg_first[def->index] = MIN2(ctx->reg_first[def->index], start);
      } else if (intr->intrinsic == nir_intrinsic_load_reg) {
         nir_def *decl = intr->src[0].ssa;

         ctx->last_use[key(decl, 0)] = MAX2(ctx->last_use[key(decl, 0)], end);
      }
   }
}

static bool
use_src(nir_src *src, void *data)
{
   struct g9p_ctx *ctx = data;

   for (unsigned i = 0; i < nir_src_num_components(*src); i++)
      use_scalar(ctx, src->ssa, i);
   return true;
}

/* An if's condition: its value, or the operands of its compare */
static void
use_condition(struct g9p_ctx *ctx, nir_if *nif)
{
   nir_alu_instr *cmp = branch_compare(nif->condition.ssa);

   if (!cmp) {
      use_src(&nif->condition, ctx);
      return;
   }
   for (unsigned i = 0; i < 2; i++)
      use_scalar(ctx, cmp->src[i].src.ssa, cmp->src[i].swizzle[0]);
}

static bool
record_def(nir_def *def, void *data)
{
   struct g9p_ctx *ctx = data;

   ctx->def_ip[def->index] = ctx->ip;
   if (nir_def_is_intrinsic(def) &&
       nir_def_as_intrinsic(def)->intrinsic == nir_intrinsic_decl_reg)
      ctx->decls[ctx->num_decls++] = def;
   return true;
}

/* Numbering (@uses false) and uses (@uses true) in code order */
static void
walk_liveness(struct g9p_ctx *ctx, struct exec_list *list, bool uses)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
            ctx->ip++;
            if (!uses) {
               nir_foreach_def(instr, record_def, ctx);
            } else if (instr->type == nir_instr_type_alu) {
               /* ALU sources read only their swizzled components */
               nir_alu_instr *alu = nir_instr_as_alu(instr);

               for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
                  for (unsigned c = 0; c < alu->def.num_components; c++)
                     use_scalar(ctx, alu->src[i].src.ssa, alu->src[i].swizzle[c]);
               }
            } else {
               nir_foreach_src(instr, use_src, ctx);
            }
         }
         break;
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);

         ctx->ip++;
         if (uses)
            use_condition(ctx, nif);
         walk_liveness(ctx, &nif->then_list, uses);
         /* compared again before a break that ends a branch (emit_if()) */
         if (uses)
            use_condition(ctx, nif);
         walk_liveness(ctx, &nif->else_list, uses);
         if (uses)
            use_condition(ctx, nif);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         unsigned l = ctx->loop_index++;

         if (!uses && l < G9P_MAX_LOOPS) {
            ctx->loops[l].start = ctx->ip + 1;
            ctx->num_loops = MAX2(ctx->num_loops, l + 1);
         }
         walk_liveness(ctx, &loop->body, uses);
         if (!uses && l < G9P_MAX_LOOPS)
            ctx->loops[l].end = ctx->ip;
         break;
      }
      default:
         break;
      }
   }
}

/* --- Code ---------------------------------------------------------------- */

/* The common-store slot of the storage buffer @index refers to */
static int
buffer_slot(struct g9p_ctx *ctx, nir_src index)
{
   nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(index.ssa, 0));
   nir_intrinsic_instr *intr;

   if (!nir_scalar_is_intrinsic(s))
      return -1;
   intr = nir_def_as_intrinsic(s.def);
   if (intr->intrinsic == nir_intrinsic_load_vulkan_descriptor) {
      s = nir_scalar_chase_movs(nir_get_scalar(intr->src[0].ssa, 0));
      if (!nir_scalar_is_intrinsic(s))
         return -1;
      intr = nir_def_as_intrinsic(s.def);
   }
   if (intr->intrinsic != nir_intrinsic_vulkan_resource_index ||
       !nir_src_is_const(intr->src[0]) || nir_src_as_uint(intr->src[0]) != 0)
      return -1;

   return g9p_buffer_slot(ctx->out, nir_intrinsic_desc_set(intr),
                          nir_intrinsic_binding(intr));
}

/*
 * dst = (a COND b) ? x : y as a predicated copy: y, then x under a mask
 * level for the lanes whose compare passed. Apple's select/move pair
 * (g9p_sel()/g9p_movf()) misbehaves while lanes are masked off: two of them
 * in a branch or after a loop break hang the GPU.
 */
static void
emit_select(struct g9p_ctx *ctx, unsigned cond, bool is_float, int a, int b, int x, int y,
            int dst)
{
   g9p_add_imm(ctx->c, dst, y, 0);
   g9p_cmp_reg(ctx->c, cond, a, b, is_float);
   g9p_push_if(ctx->c, ctx->st, false);
   g9p_add_imm(ctx->c, dst, x, 0);
   g9p_pop_levels(ctx->c, ctx->st, 1);
}

/* dst = (a COND b) ? x : y, with constant x and y */
static bool
emit_select_const(struct g9p_ctx *ctx, unsigned cond, bool is_float, int a, int b,
                  uint32_t x, uint32_t y, int dst)
{
   int rx = const_reg(ctx, x), ry = const_reg(ctx, y);

   if (rx < 0 || ry < 0)
      return false;
   emit_select(ctx, cond, is_float, a, b, rx, ry, dst);
   return true;
}

static bool emit_alu_op(struct g9p_ctx *ctx, nir_alu_instr *alu, const int *r,
                        int dst);

static bool
only_tex_uses(nir_def *def)
{
   nir_foreach_use_including_if(src, def) {
      if (nir_src_is_if(src) || nir_src_use_instr(src)->type != nir_instr_type_tex)
         return false;

      /* arrays and cubes take their coordinates in registers (emit_tex_layered()) */
      const nir_tex_instr *tex = nir_instr_as_tex(nir_src_use_instr(src));

      if (tex->is_array || tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE ||
          tex->op == nir_texop_txl || tex->op == nir_texop_txb)
         return false;
   }
   return true;
}

static int varying_component(nir_scalar s);

/*
 * vecN: no code when only texture coordinates read straight from two
 * consecutive varying components use it (emit_tex() iterates them itself),
 * else each component copied into its own register
 */
static bool
emit_vec(struct g9p_ctx *ctx, nir_alu_instr *alu)
{
   nir_scalar s0 = nir_scalar_chase_movs(nir_get_scalar(alu->src[0].src.ssa,
                                                         alu->src[0].swizzle[0]));

   /* a buffer descriptor with its offset zeroed: resolved where it is used */
   if (nir_scalar_is_intrinsic(s0) &&
       (nir_def_as_intrinsic(s0.def)->intrinsic == nir_intrinsic_load_vulkan_descriptor ||
        nir_def_as_intrinsic(s0.def)->intrinsic == nir_intrinsic_vulkan_resource_index))
      return true;
   if (only_tex_uses(&alu->def) && alu->def.num_components >= 2 &&
       varying_component(nir_get_scalar(&alu->def, 0)) >= 0 &&
       varying_component(nir_get_scalar(&alu->def, 1)) ==
          varying_component(nir_get_scalar(&alu->def, 0)) + 1) {
      release(ctx, &alu->instr);
      return true;
   }
   for (unsigned c = 0; c < alu->def.num_components; c++) {
      nir_scalar v = nir_get_scalar(alu->src[c].src.ssa, alu->src[c].swizzle[0]);
      uint32_t imm;
      int src, dst;

      if (!ctx->last_use[key(&alu->def, c)])
         continue;
      if (is_const(v, &imm)) {
         dst = alloc_reg(ctx);
         if (dst < 0)
            return false;
         g9p_mov_imm(ctx->c, dst, imm);
      } else {
         src = src_reg(ctx, v);
         dst = alloc_reg(ctx);
         if (src < 0 || dst < 0)
            return false;
         g9p_alu2(ctx->c, G9P_OR, dst, src, src, false);
      }
      ctx->reg[key(&alu->def, c)] = dst;
   }
   release(ctx, &alu->instr);
   return true;
}

static bool
emit_alu(struct g9p_ctx *ctx, nir_alu_instr *alu)
{
   const unsigned num_inputs = nir_op_infos[alu->op].num_inputs;
   nir_scalar s[3] = { 0 };
   int r[3] = { -1, -1, -1 };
   uint32_t imm;
   int dst;

   if (nir_op_is_vec(alu->op) && alu->def.bit_size == 32)
      return emit_vec(ctx, alu);
   if (alu->def.num_components != 1 || alu->def.bit_size != 32) {
      fprintf(stderr, "g9p: %s: only 32-bit scalar ALU\n", nir_op_infos[alu->op].name);
      return false;
   }
   for (unsigned i = 0; i < num_inputs; i++)
      s[i] = nir_get_scalar(alu->src[i].src.ssa, alu->src[i].swizzle[0]);

   /* compared where it is used (emit_branch_cmp()) */
   if (branch_compare(&alu->def))
      return true;

   if (alu->op == nir_op_mov && nir_scalar_is_intrinsic(s[0]) && s[0].comp == 0) {
      nir_intrinsic_op iop = nir_def_as_intrinsic(s[0].def)->intrinsic;

      /* a binding index: resolved where it is used (buffer_slot()) */
      if (iop == nir_intrinsic_load_vulkan_descriptor ||
          iop == nir_intrinsic_vulkan_resource_index)
         return true;
   }

   /* dst = src + small immediate */
   if (alu->op == nir_op_iadd) {
      if (is_const(s[0], &imm) && imm < 0x40) {
         nir_scalar t = s[0];

         s[0] = s[1];
         s[1] = t;
      }
      if (is_const(s[1], &imm) && imm < 0x40) {
         r[0] = src_reg(ctx, s[0]);
         if (r[0] >= 0)
            r[0] = one_src(ctx, r[0]);
         if (r[0] < 0)
            return false;
         release(ctx, &alu->instr);
         dst = alloc_reg(ctx);
         if (dst < 0)
            return false;
         g9p_add_imm(ctx->c, dst, r[0], imm);
         ctx->reg[key(&alu->def, 0)] = dst;
         return true;
      }
   }
   if (alu->op == nir_op_mov && is_const(s[0], &imm)) {
      release(ctx, &alu->instr);
      dst = alloc_reg(ctx);
      if (dst < 0)
         return false;
      g9p_mov_imm(ctx->c, dst, imm);
      ctx->reg[key(&alu->def, 0)] = dst;
      return true;
   }

   for (unsigned i = 0; i < num_inputs; i++) {
      r[i] = src_reg(ctx, s[i]);
      if (r[i] < 0)
         return false;
   }

   /*
    * The destination may reuse a dying source (the hardware reads sources
    * first), except in multi-instruction sequences: they read their sources
    * (and temporaries) after the first wrote the destination, so they
    * release only afterwards.
    */
   switch (alu->op) {
   case nir_op_flt32: case nir_op_fge32: case nir_op_feq32: case nir_op_fneu32:
   case nir_op_ilt32: case nir_op_ige32: case nir_op_ult32: case nir_op_uge32:
   case nir_op_ieq32: case nir_op_ine32: case nir_op_b32csel:
   case nir_op_fmin: case nir_op_fmax: case nir_op_imin: case nir_op_imax:
   case nir_op_umin: case nir_op_umax:
   case nir_op_fneg: case nir_op_fabs: case nir_op_inot: case nir_op_ineg:
   case nir_op_fsin: case nir_op_fcos: case nir_op_fsign: case nir_op_b2f32:
   case nir_op_b2i32: case nir_op_slt: case nir_op_sge: case nir_op_seq: case nir_op_sne:
   case nir_op_flrp: {
      bool ok;

      dst = alloc_reg(ctx);
      if (dst < 0)
         return false;
      ctx->reg[key(&alu->def, 0)] = dst;
      ok = emit_alu_op(ctx, alu, r, dst);
      release(ctx, &alu->instr);
      return ok;
   }
   default:
      /*
       * Before the release: a copy must not land on the other source, whose
       * register the release may free
       */
      if (alu->op == nir_op_ishl || alu->op == nir_op_ushr || alu->op == nir_op_u2f32 ||
          alu->op == nir_op_i2f32 || alu->op == nir_op_f2u32 || alu->op == nir_op_f2i32) {
         r[0] = one_src(ctx, r[0]);
         if (r[0] < 0)
            return false;
      }
      release(ctx, &alu->instr);
      dst = alloc_reg(ctx);
      if (dst < 0)
         return false;
      ctx->reg[key(&alu->def, 0)] = dst;
      return emit_alu_op(ctx, alu, r, dst);
   }
}

/* The op itself, sources in @r, destination @dst */
static bool
emit_alu_op(struct g9p_ctx *ctx, nir_alu_instr *alu, const int *r, int dst)
{
   switch (alu->op) {
   case nir_op_mov:
      g9p_add_imm(ctx->c, dst, r[0], 0);
      return true;
   case nir_op_iadd:
      g9p_alu2(ctx->c, G9P_IADD, dst, r[0], r[1], false);
      return true;
   case nir_op_isub:
      g9p_alu2(ctx->c, G9P_ISUB, dst, r[0], r[1], false);
      return true;
   case nir_op_imul:
      g9p_alu2(ctx->c, G9P_IMUL, dst, r[0], r[1], false);
      return true;
   case nir_op_iand:
      g9p_alu2(ctx->c, G9P_AND, dst, r[0], r[1], false);
      return true;
   case nir_op_ior:
      g9p_alu2(ctx->c, G9P_OR, dst, r[0], r[1], false);
      return true;
   case nir_op_ixor:
      g9p_alu2(ctx->c, G9P_XOR, dst, r[0], r[1], false);
      return true;
   case nir_op_ishl: {
      const int a = one_src(ctx, r[0]);

      if (a < 0)
         return false;
      g9p_shift(ctx->c, dst, a, r[1], false);
      return true;
   }
   case nir_op_ushr: {
      const int a = one_src(ctx, r[0]);

      if (a < 0)
         return false;
      g9p_shift(ctx->c, dst, a, r[1], true);
      return true;
   }
   case nir_op_ishr:
      g9p_alu2(ctx->c, G9P_ISHR, dst, r[0], r[1], false);
      return true;
   case nir_op_fadd:
      g9p_alu2(ctx->c, G9P_FADD, dst, r[0], r[1], false);
      return true;
   case nir_op_fmul:
      g9p_alu2(ctx->c, G9P_FMUL, dst, r[0], r[1], false);
      return true;
   case nir_op_ffma:
      g9p_ffma(ctx->c, dst, r[0], r[1], r[2]);
      return true;
   case nir_op_u2f32: {
      const int a = one_src(ctx, r[0]);

      if (a < 0)
         return false;
      g9p_convert(ctx->c, dst, a, G9P_U2F);
      return true;
   }
   case nir_op_i2f32: {
      const int a = one_src(ctx, r[0]);

      if (a < 0)
         return false;
      g9p_convert(ctx->c, dst, a, G9P_I2F);
      return true;
   }
   case nir_op_f2u32: {
      const int a = one_src(ctx, r[0]);

      if (a < 0)
         return false;
      g9p_convert(ctx->c, dst, a, G9P_F2U);
      return true;
   }
   case nir_op_f2i32: {
      const int a = one_src(ctx, r[0]);

      if (a < 0)
         return false;
      g9p_convert(ctx->c, dst, a, G9P_F2I);
      return true;
   }

   case nir_op_ffloor:
      g9p_funary(ctx->c, G9P_FFLOOR, dst, r[0]);
      return true;
   case nir_op_frcp:
      g9p_funary(ctx->c, G9P_FRCP, dst, r[0]);
      return true;
   case nir_op_frsq:
      g9p_funary(ctx->c, G9P_FRSQ, dst, r[0]);
      return true;
   case nir_op_flog2:
      g9p_funary(ctx->c, G9P_FLOG2, dst, r[0]);
      return true;
   case nir_op_fexp2:
      g9p_funary(ctx->c, G9P_FEXP2, dst, r[0]);
      return true;

   case nir_op_fsin:
   case nir_op_fcos: {
      /*
       * As Apple's compiler: turns = x / 2pi (+ 1/4 for cos), its fraction
       * times 4, then the two hardware steps, multiplied.
       */
      int k = const_reg(ctx, fui(0.15915494309189535f));
      int q = const_reg(ctx, fui(alu->op == nir_op_fcos ? 0.25f : 0.0f));
      int t = alloc_temp(ctx);

      if (k < 0 || q < 0 || t < 0)
         return false;
      g9p_ffma(ctx->c, dst, r[0], k, q);
      g9p_funary(ctx->c, G9P_FFLOOR, t, dst);
      g9p_alu2(ctx->c, G9P_FADD | G9P_NEG_B, dst, dst, t, false);
      g9p_mov_imm(ctx->c, k, fui(4.0f));
      g9p_alu2(ctx->c, G9P_FMUL, dst, dst, k, false);
      g9p_funary(ctx->c, G9P_FSIN1, t, dst);
      g9p_funary(ctx->c, G9P_FSIN2, k, t);
      g9p_alu2(ctx->c, G9P_FMUL, dst, t, k, false);
      return true;
   }
   case nir_op_fsign: {
      /* 0, then 1 where x > 0, -1 where x < 0 */
      int zero = const_reg(ctx, 0), one = const_reg(ctx, fui(1.0f)),
          minus = const_reg(ctx, fui(-1.0f));

      if (zero < 0 || one < 0 || minus < 0)
         return false;
      emit_select(ctx, G9P_FLT, true, zero, r[0], one, zero, dst);
      g9p_cmp_reg(ctx->c, G9P_FLT, r[0], zero, true);
      g9p_push_if(ctx->c, ctx->st, false);
      g9p_add_imm(ctx->c, dst, minus, 0);
      g9p_pop_levels(ctx->c, ctx->st, 1);
      return true;
   }
   case nir_op_b2f32:
   case nir_op_b2i32: {
      /* booleans are 0 / ~0 */
      int one = const_reg(ctx, alu->op == nir_op_b2f32 ? fui(1.0f) : 1);

      if (one < 0)
         return false;
      g9p_alu2(ctx->c, G9P_AND, dst, r[0], one, false);
      return true;
   }

   case nir_op_flrp: {
      /* a + (b - a) * t */
      int t = alloc_temp(ctx);

      if (t < 0)
         return false;
      g9p_alu2(ctx->c, G9P_FADD | G9P_NEG_B, t, r[1], r[0], false);
      g9p_ffma(ctx->c, dst, t, r[2], r[0]);
      return true;
   }

   /* float results 1.0 / 0.0 (step()) */
   case nir_op_slt:
      return emit_select_const(ctx, G9P_FLT, true, r[0], r[1], fui(1.0f), 0, dst);
   case nir_op_sge:
      return emit_select_const(ctx, G9P_FLT, true, r[0], r[1], 0, fui(1.0f), dst);
   case nir_op_seq:
      return emit_select_const(ctx, G9P_FEQ, true, r[0], r[1], fui(1.0f), 0, dst);
   case nir_op_sne:
      return emit_select_const(ctx, G9P_FEQ, true, r[0], r[1], 0, fui(1.0f), dst);

   case nir_op_flt32:
      return emit_select_const(ctx, G9P_FLT, true, r[0], r[1], ~0u, 0, dst);
   case nir_op_fge32:
      return emit_select_const(ctx, G9P_FLT, true, r[0], r[1], 0, ~0u, dst);
   case nir_op_feq32:
      return emit_select_const(ctx, G9P_FEQ, true, r[0], r[1], ~0u, 0, dst);
   case nir_op_fneu32:
      return emit_select_const(ctx, G9P_FEQ, true, r[0], r[1], 0, ~0u, dst);
   case nir_op_ilt32:
      return emit_select_const(ctx, G9P_SLT, false, r[0], r[1], ~0u, 0, dst);
   case nir_op_ige32:
      return emit_select_const(ctx, G9P_SLT, false, r[0], r[1], 0, ~0u, dst);
   case nir_op_ult32:
      return emit_select_const(ctx, G9P_ULT, false, r[0], r[1], ~0u, 0, dst);
   case nir_op_uge32:
      return emit_select_const(ctx, G9P_ULT, false, r[0], r[1], 0, ~0u, dst);
   case nir_op_ieq32:
      return emit_select_const(ctx, G9P_EQ, false, r[0], r[1], ~0u, 0, dst);
   case nir_op_ine32:
      return emit_select_const(ctx, G9P_EQ, false, r[0], r[1], 0, ~0u, dst);

   case nir_op_b32csel: {
      int zero = const_reg(ctx, 0);

      if (zero < 0)
         return false;
      /* cond == 0 ? y : x */
      emit_select(ctx, G9P_EQ, false, r[0], zero, r[2], r[1], dst);
      return true;
   }
   case nir_op_fmin:
   case nir_op_fmax:
   case nir_op_imin:
   case nir_op_imax:
   case nir_op_umin:
   case nir_op_umax: {
      const unsigned cond =
         (alu->op == nir_op_fmin || alu->op == nir_op_fmax) ? G9P_FLT :
         (alu->op == nir_op_imin || alu->op == nir_op_imax) ? G9P_SLT : G9P_ULT;
      const bool is_min =
         alu->op == nir_op_fmin || alu->op == nir_op_imin || alu->op == nir_op_umin;

      emit_select(ctx, cond, cond == G9P_FLT, r[0], r[1], is_min ? r[0] : r[1],
                  is_min ? r[1] : r[0], dst);
      return true;
   }

   case nir_op_fneg:
   case nir_op_fabs:
   case nir_op_inot:
   case nir_op_ineg: {
      const uint32_t k = alu->op == nir_op_fneg ? 0x80000000u :
                         alu->op == nir_op_fabs ? 0x7fffffffu :
                         alu->op == nir_op_inot ? ~0u : 0;
      int rk = const_reg(ctx, k);

      if (rk < 0)
         return false;
      if (alu->op == nir_op_fabs)
         g9p_alu2(ctx->c, G9P_AND, dst, r[0], rk, false);
      else if (alu->op == nir_op_ineg)
         g9p_alu2(ctx->c, G9P_ISUB, dst, rk, r[0], false);
      else
         g9p_alu2(ctx->c, G9P_XOR, dst, r[0], rk, false);
      return true;
   }
   default:
      fprintf(stderr, "g9p: ALU op %s not supported yet\n", nir_op_infos[alu->op].name);
      return false;
   }
}

/*
 * Loads/stores through a slot address in the common store: storage buffers
 * and push constants. @base is added to the register offset.
 */
static bool
emit_mem(struct g9p_ctx *ctx, nir_intrinsic_instr *intr, int slot, nir_scalar offset,
         unsigned base, nir_src *value)
{
   unsigned bits = value ? nir_src_bit_size(*value) : intr->def.bit_size;
   unsigned comps = value ? nir_src_num_components(*value) : intr->def.num_components;
   int off, pair, data;

   if (slot < 0 || bits != 32 || comps != 1) {
      fprintf(stderr, "g9p: %s: only one 32-bit value through a plain binding\n",
              nir_intrinsic_infos[intr->intrinsic].name);
      return false;
   }

   /* fragment shaders: the buffer's contents in the common store */
   if (ctx->stage == MESA_SHADER_FRAGMENT) {
      const unsigned byte = base + (nir_scalar_is_const(offset) ? nir_scalar_as_uint(offset) : 0);
      int zero;

      if (value || !nir_scalar_is_const(offset) || byte % 4 ||
          byte / 4 >= G9P_FS_BUF_DWORDS) {
         fprintf(stderr, "g9p: %s in a fragment shader: only constant offsets below %u\n",
                 nir_intrinsic_infos[intr->intrinsic].name, 4 * G9P_FS_BUF_DWORDS);
         return false;
      }
      ctx->out->buffers[slot].dwords = MAX2(ctx->out->buffers[slot].dwords, byte / 4 + 1);
      zero = const_reg(ctx, 0);
      release(ctx, &intr->instr);
      data = alloc_reg(ctx);
      if (zero < 0 || data < 0)
         return false;
      g9p_mov_uniform(ctx->c, data,
                      g9p_fs_buf_base(ctx->tex_total) + G9P_FS_BUF_DWORDS * slot + byte / 4, zero);
      ctx->reg[key(&intr->def, 0)] = data;
      return true;
   }

   off = src_reg(ctx, offset);
   if (off < 0)
      return false;
   if (base) {
      int rb = const_reg(ctx, base);
      int t = alloc_temp(ctx);

      if (t < 0 || rb < 0)
         return false;
      g9p_alu2(ctx->c, G9P_IADD, t, off, rb, false);
      off = t;
   }
   off = low_reg(ctx, off);
   if (off < 0)
      return false;
   pair = alloc_pair(ctx);
   if (pair < 0)
      return false;
   ctx->temps |= BITFIELD64_RANGE(pair, 2);
   /* byte offset: shift 0 */
   g9p_addr(ctx->c, pair, 2 * slot, off, 0, false);

   if (value) {
      data = src_reg(ctx, nir_get_scalar(value->ssa, 0));
      if (data < 0)
         return false;
      g9p_mem_at(ctx->c, true, G9P_SIZE_U32, data, pair);
      release(ctx, &intr->instr);
   } else {
      /* the address pair stays busy until the load has read it */
      uint64_t pair_bits = BITFIELD64_RANGE(pair, 2);

      ctx->temps &= ~pair_bits;
      release(ctx, &intr->instr);
      data = alloc_reg(ctx);
      if (data < 0)
         return false;
      g9p_mem_at(ctx->c, false, G9P_SIZE_U32, data, pair);
      g9p_wait(ctx->c);
      ctx->busy &= ~pair_bits;
      ctx->busy |= BITFIELD64_BIT(data);
      ctx->reg[key(&intr->def, 0)] = data;
   }
   return true;
}

/* Keep r[@r] (a store's value) until the end of the program */
static void
pin(struct g9p_ctx *ctx, int r)
{
   ctx->pinned |= BITFIELD64_BIT(r);
   ctx->temps &= ~BITFIELD64_BIT(r);
   ctx->busy |= BITFIELD64_BIT(r);
}

/*
 * Vertex attribute component: base address of its vertex buffer (common
 * store) + vertex index (r0, or the instance index r1 for per-instance
 * bindings) x stride + offset + 4 x component.
 */
static bool
emit_attrib_load(struct g9p_ctx *ctx, nir_intrinsic_instr *intr)
{
   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   unsigned loc = sem.location - VERT_ATTRIB_GENERIC0;
   const struct g9p_vertex_attrib *a;
   int slot, stride_slot, zero, rs, t, pair, data;
   uint32_t offset;

   if (loc >= G9P_MAX_ATTRIBS || !ctx->attribs[loc].present ||
       intr->def.num_components != 1 || intr->def.bit_size != 32 ||
       !nir_src_is_const(intr->src[0])) {
      fprintf(stderr, "g9p: vertex input %u: not a 32-bit attribute component\n", loc);
      return false;
   }
   a = &ctx->attribs[loc];
   slot = g9p_buffer_slot(ctx->out, G9P_VERTEX_BUFFERS, a->binding);
   stride_slot = g9p_buffer_slot(ctx->out, G9P_VERTEX_STRIDES, a->binding);
   if (slot < 0 || stride_slot < 0)
      return false;
   offset = (a->offset & ~3u) + 4 * nir_intrinsic_component(intr) +
            16 * nir_src_as_uint(intr->src[0]);

   /* the stride from its slot: the address form with 0 added (rs:rs+1) */
   zero = const_reg(ctx, 0);
   zero = zero < 0 ? zero : low_reg(ctx, zero);
   rs = alloc_pair(ctx);
   t = alloc_low_temp(ctx);
   if (zero < 0 || rs < 0 || t < 0)
      return false;
   g9p_addr(ctx->c, rs, 2 * stride_slot, zero, 0, false);
   g9p_alu2(ctx->c, G9P_IMUL, t, a->per_instance ? ctx->instance_reg : ctx->vertex_reg, rs, false);
   ctx->busy &= ~BITFIELD64_RANGE(rs, 2);
   if (offset) {
      if (offset < 64) {
         g9p_add_imm(ctx->c, t, t, offset);
      } else {
         int ro = const_reg(ctx, offset);

         ro = ro < 0 ? ro : low_reg(ctx, ro);
         if (ro < 0)
            return false;
         g9p_alu2(ctx->c, G9P_IADD, t, t, ro, false);
      }
   }
   pair = alloc_pair(ctx);
   if (pair < 0)
      return false;
   g9p_addr(ctx->c, pair, 2 * slot, t, 0, false);
   ctx->busy &= ~ctx->temps;
   ctx->temps = 0;
   data = alloc_reg(ctx);
   if (data < 0)
      return false;
   g9p_mem_at(ctx->c, false, G9P_SIZE_U32, data, pair);
   g9p_wait(ctx->c);
   ctx->busy &= ~BITFIELD64_RANGE(pair, 2);
   ctx->busy |= BITFIELD64_BIT(data);
   ctx->reg[key(&intr->def, 0)] = data;
   return true;
}

static bool
perspective(nir_intrinsic_instr *intr)
{
   nir_intrinsic_instr *bary = nir_src_as_intrinsic(intr->src[0]);

   return !bary || nir_intrinsic_interp_mode(bary) != INTERP_MODE_NOPERSPECTIVE;
}

/*
 * Fragment input component: varying VARn component c is component 4n + c
 * of the varyings (each padded to a vec4; the pipeline's DOUTI program
 * places them so), one f32 iterate per component.
 */
static bool
emit_varying_load(struct g9p_ctx *ctx, nir_intrinsic_instr *intr, bool persp)
{
   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_src *offset = nir_get_io_offset_src(intr);
   unsigned n = sem.location - VARYING_SLOT_VAR0;
   int r;

   if (sem.location < VARYING_SLOT_VAR0 || n >= G9P_MAX_VARYINGS ||
       intr->def.num_components != 1 || intr->def.bit_size != 32 ||
       !nir_src_is_const(*offset) || nir_src_as_uint(*offset)) {
      fprintf(stderr, "g9p: fragment input %s not supported yet\n",
              gl_varying_slot_name_for_stage(sem.location, MESA_SHADER_FRAGMENT));
      return false;
   }
   r = alloc_reg(ctx);
   if (r < 0)
      return false;
   g9p_iterate(ctx->c, r, 4 * n + nir_intrinsic_component(intr), 1, persp);
   g9p_wait(ctx->c);
   ctx->reg[key(&intr->def, 0)] = r;
   ctx->out->num_varyings = MAX2(ctx->out->num_varyings, n + 1);
   ctx->out->varying_perspective[n] = persp;
   return true;
}

/* gl_FragCoord: the frag_coord_slot varying, iterated without perspective */
static bool
emit_frag_coord(struct g9p_ctx *ctx, nir_intrinsic_instr *intr)
{
   const unsigned n = ctx->frag_coord_slot;

   for (unsigned c = 0; c < intr->def.num_components; c++) {
      int r;

      if (!ctx->last_use[key(&intr->def, c)])
         continue;
      r = alloc_reg(ctx);
      if (r < 0)
         return false;
      g9p_iterate(ctx->c, r, 4 * n + c, 1, false);
      ctx->reg[key(&intr->def, c)] = r;
   }
   g9p_wait(ctx->c);
   ctx->out->num_varyings = MAX2(ctx->out->num_varyings, n + 1);
   ctx->out->varying_perspective[n] = false;
   ctx->out->uses_frag_coord = true;
   return true;
}

/* An output component: kept in its register until the end */
static bool
emit_output(struct g9p_ctx *ctx, nir_intrinsic_instr *intr)
{
   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_src *offset = nir_get_io_offset_src(intr);
   unsigned comp = nir_intrinsic_component(intr);
   nir_scalar v = nir_get_scalar(intr->src[0].ssa, 0);
   int slot = -1, r;

   if (nir_src_num_components(intr->src[0]) != 1 || nir_src_bit_size(intr->src[0]) != 32 ||
       !nir_src_is_const(*offset) || nir_src_as_uint(*offset)) {
      fprintf(stderr, "g9p: output: one 32-bit component at a time\n");
      return false;
   }
   if (ctx->stage == MESA_SHADER_VERTEX) {
      if (sem.location == VARYING_SLOT_POS) {
         slot = comp;
      } else if (sem.location >= VARYING_SLOT_VAR0 &&
                 sem.location < VARYING_SLOT_VAR0 + G9P_MAX_VARYINGS) {
         unsigned n = sem.location - VARYING_SLOT_VAR0;

         slot = 4 + 4 * n + comp;
         ctx->out->num_varyings = MAX2(ctx->out->num_varyings, n + 1);
      } else if (sem.location == VARYING_SLOT_PSIZ && ctx->point_size) {
         r = src_reg(ctx, v);
         if (r < 0)
            return false;
         if (ctx->temps & BITFIELD64_BIT(r)) {
            int own = alloc_reg(ctx);

            if (own < 0)
               return false;
            g9p_alu2(ctx->c, G9P_OR, own, r, r, false);
            r = own;
         }
         pin(ctx, r);
         ctx->psize_reg = r;
         release(ctx, &intr->instr);
         return true;
      } else {
         /* clip distances: not passed on yet */
         release(ctx, &intr->instr);
         return true;
      }
   } else if (sem.location > FRAG_RESULT_DATA0 && sem.location < FRAG_RESULT_DATA0 + 8) {
      /* one colour attachment so far: the others' outputs are dropped */
      release(ctx, &intr->instr);
      return true;
   } else if (sem.location != FRAG_RESULT_DATA0 && sem.location != FRAG_RESULT_COLOR) {
      fprintf(stderr, "g9p: fragment output %u not supported yet\n", sem.location);
      return false;
   }

   r = src_reg(ctx, v);
   if (r < 0)
      return false;
   if (ctx->temps & BITFIELD64_BIT(r)) {
      /* reloaded from above r31 into a temporary: an output keeps its own */
      int own = alloc_reg(ctx);

      if (own < 0)
         return false;
      g9p_alu2(ctx->c, G9P_OR, own, r, r, false);
      r = own;
   }
   pin(ctx, r);
   if (ctx->stage == MESA_SHADER_VERTEX) {
      /*
       * The previous output now, which frees its register: the last one has to
       * stay for the end of the program.
       */
      if (ctx->flush_outputs && ctx->num_outputs &&
          intr->instr.block->cf_node.parent->type == nir_cf_node_function) {
         const int prev = ctx->outputs[ctx->num_outputs - 1].reg;
         bool live = prev == r;

         g9p_vs_output(ctx->c, prev, ctx->outputs[ctx->num_outputs - 1].slot);
         ctx->flushed_outputs |= BITFIELD64_BIT(ctx->outputs[ctx->num_outputs - 1].slot);
         ctx->num_outputs--;
         for (unsigned k = 0; k < ctx->num_keys && !live; k++)
            live = ctx->reg[k] == prev && ctx->last_use[k] > ctx->ip;
         if (!live) {
            ctx->pinned &= ~BITFIELD64_BIT(prev);
            ctx->busy &= ~BITFIELD64_BIT(prev);
         }
      }
      ctx->outputs[ctx->num_outputs].reg = r;
      ctx->outputs[ctx->num_outputs].slot = slot;
      ctx->num_outputs++;
   } else {
      ctx->colour[comp] = r;
   }
   release(ctx, &intr->instr);
   return true;
}

/*
 * One 32-bit word of the colour attachment's current value (two halves;
 * lower_fb_fetch() makes the f32 components from it): the tile's four
 * halves into an even register pair, the word's register kept. Apple's
 * half-to-f32 conversion (0189 81dd 8hh1 01a0 0304 0070) only writes r0-r2,
 * so the conversion is integer code instead.
 */
static bool
emit_fb_fetch(struct g9p_ctx *ctx, nir_intrinsic_instr *intr)
{
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   unsigned word = nir_intrinsic_component(intr);
   int pair = -1;

   if (ctx->stage != MESA_SHADER_FRAGMENT || word > 1 ||
       (sem.location != FRAG_RESULT_DATA0 && sem.location != FRAG_RESULT_COLOR)) {
      fprintf(stderr, "g9p: only the first colour attachment can be read so far\n");
      return false;
   }
   release(ctx, &intr->instr);
   for (unsigned r = 0; r + 1 < G9P_NUM_REGS; r += 2) {
      if (!(ctx->busy & BITFIELD64_RANGE(r, 2))) {
         pair = r;
         break;
      }
   }
   if (pair < 0) {
      dump_busy(ctx);
      return false;
   }
   ctx->max_reg = MAX2(ctx->max_reg, pair + 1);
   g9p_tile_read(ctx->c, pair);
   g9p_wait(ctx->c);
   if (ctx->last_use[key(&intr->def, 0)]) {
      ctx->reg[key(&intr->def, 0)] = pair + word;
      ctx->busy |= BITFIELD64_BIT(pair + word);
   }
   return true;
}

/* The texture slot of a combined image sampler deref, added if new */
static bool
texture_key(nir_tex_instr *tex, unsigned *set, unsigned *binding, unsigned *index)
{
   int i = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   nir_deref_instr *deref;
   nir_variable *var;

   if (i < 0)
      return false;
   deref = nir_src_as_deref(tex->src[i].src);
   var = nir_deref_instr_get_variable(deref);
   if (!var)
      return false;
   *set = var->data.descriptor_set;
   *binding = var->data.binding;
   *index = 0;
   if (deref->deref_type == nir_deref_type_array) {
      if (!nir_src_is_const(deref->arr.index))
         return false;
      *index = nir_src_as_uint(deref->arr.index);
   }
   return true;
}

static int
texture_slot(struct g9p_ctx *ctx, nir_tex_instr *tex)
{
   unsigned set, binding, index, k;

   if (!texture_key(tex, &set, &binding, &index))
      return -1;
   for (k = 0; k < ctx->out->num_textures; k++) {
      if (ctx->out->textures[k].set == set && ctx->out->textures[k].binding == binding &&
          ctx->out->textures[k].index == index)
         return k;
   }
   if (k == G9P_MAX_TEXTURES) {
      fprintf(stderr, "g9p: more than %u textures in a shader\n", G9P_MAX_TEXTURES);
      return -1;
   }
   ctx->out->textures[k].set = set;
   ctx->out->textures[k].binding = binding;
   ctx->out->textures[k].index = index;
   ctx->out->num_textures++;
   return k;
}

/* How many textures the shader samples: the layout of their common store */
static unsigned
count_textures(nir_function_impl *impl)
{
   struct { unsigned set, binding, index; } seen[G9P_MAX_TEXTURES];
   unsigned n = 0;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         unsigned set, binding, index, k;

         if (instr->type != nir_instr_type_tex ||
             !texture_key(nir_instr_as_tex(instr), &set, &binding, &index))
            continue;
         for (k = 0; k < n; k++) {
            if (seen[k].set == set && seen[k].binding == binding && seen[k].index == index)
               break;
         }
         if (k == n && n < G9P_MAX_TEXTURES) {
            seen[n].set = set, seen[n].binding = binding, seen[n].index = index;
            n++;
         }
      }
   }
   return n;
}

static unsigned
tex_state_dword(const struct g9p_ctx *ctx, unsigned slot)
{
   return g9p_tex_state(ctx->stage == MESA_SHADER_VERTEX ? G9P_VS_TEX_BASE : 0, slot);
}

static unsigned
tex_sampler_dword(const struct g9p_ctx *ctx, unsigned slot)
{
   return g9p_tex_sampler(ctx->stage == MESA_SHADER_VERTEX ? G9P_VS_TEX_BASE : 0,
                          ctx->tex_total, slot);
}

/* The varying component an SSA scalar is a plain load of, or -1 */
static int
varying_component(nir_scalar s)
{
   nir_intrinsic_instr *intr;
   nir_io_semantics sem;

   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_intrinsic(s))
      return -1;
   intr = nir_def_as_intrinsic(s.def);
   if (intr->intrinsic != nir_intrinsic_load_interpolated_input &&
       intr->intrinsic != nir_intrinsic_load_input)
      return -1;
   sem = nir_intrinsic_io_semantics(intr);
   if (sem.location < VARYING_SLOT_VAR0 || sem.location >= VARYING_SLOT_VAR0 + G9P_MAX_VARYINGS)
      return -1;
   return 4 * (sem.location - VARYING_SLOT_VAR0) + nir_intrinsic_component(intr) + s.comp;
}

/*
 * The size of a 2D texture's base level (txs), from its texture state in
 * the common store (dwords 0-1 of its slot): width - 1 at bit 27, height - 1
 * at bit 41
 */
static bool
emit_txs(struct g9p_ctx *ctx, nir_tex_instr *tex)
{
   const int slot = texture_slot(ctx, tex);
   const unsigned u = tex_state_dword(ctx, MAX2(slot, 0));
   int w, h, d0, d1, t, zero;

   if (ctx->stage != MESA_SHADER_FRAGMENT || tex->sampler_dim != GLSL_SAMPLER_DIM_2D ||
       tex->is_array || slot < 0) {
      fprintf(stderr, "g9p: only sizes of 2D combined image samplers so far\n");
      return false;
   }
   release(ctx, &tex->instr);
   w = alloc_reg(ctx);
   h = alloc_reg(ctx);
   zero = const_reg(ctx, 0);
   d0 = alloc_temp(ctx);
   d1 = alloc_temp(ctx);
   t = alloc_temp(ctx);
   if (w < 0 || h < 0 || zero < 0 || d0 < 0 || d1 < 0 || t < 0)
      return false;
   g9p_mov_uniform(ctx->c, d0, u, zero);
   g9p_mov_uniform(ctx->c, d1, u + 1, zero);
   /* width: (d1 & 0x1ff) << 5 | d0 >> 27, plus one */
   g9p_mov_imm(ctx->c, t, 0x1ff);
   g9p_alu2(ctx->c, G9P_AND, w, d1, t, false);
   g9p_mov_imm(ctx->c, t, 5);
   g9p_shift(ctx->c, w, w, t, false);
   g9p_mov_imm(ctx->c, t, 27);
   g9p_shift(ctx->c, d0, d0, t, true);
   g9p_alu2(ctx->c, G9P_OR, w, w, d0, false);
   /* height: (d1 >> 9) & 0x3fff, plus one */
   g9p_mov_imm(ctx->c, t, 9);
   g9p_shift(ctx->c, h, d1, t, true);
   g9p_mov_imm(ctx->c, t, 0x3fff);
   g9p_alu2(ctx->c, G9P_AND, h, h, t, false);
   g9p_mov_imm(ctx->c, t, 1);
   g9p_alu2(ctx->c, G9P_IADD, w, w, t, false);
   g9p_alu2(ctx->c, G9P_IADD, h, h, t, false);
   ctx->busy &= ~ctx->temps;
   ctx->temps = 0;
   ctx->reg[key(&tex->def, 0)] = w;
   ctx->reg[key(&tex->def, 1)] = h;
   return true;
}

/*
 * A sample's explicit level (txl) or bias (txb): an f16 in the low half of a
 * register (Apple's code: 0b0a into a half, or 1189 converting), which
 * lod_source() makes. -1: the sample has none; -2: failed.
 */
static int
lod_source(struct g9p_ctx *ctx, nir_tex_instr *tex)
{
   const nir_tex_src_type type = tex->op == nir_texop_txl ? nir_tex_src_lod : nir_tex_src_bias;
   const int i = nir_tex_instr_src_index(tex, type);
   int r, t;

   if (i < 0)
      return tex->op != nir_texop_txl && tex->op != nir_texop_txb ? -1 : -2;
   r = src_reg(ctx, nir_get_scalar(tex->src[i].src.ssa, 0));
   r = r < 0 ? r : one_src(ctx, r);
   t = r < 0 ? r : alloc_low_temp(ctx);
   if (t < 0)
      return -2;
   g9p_f16_of(ctx->c, t, r);
   return t;
}

/*
 * The source in the words of a sample at @at (PR #4's g5p_usc_sample_ext():
 * byte 6 = 0x80 temp | 0x40 level or 0x20 bias | register, byte 8 = 0x80 LOD
 * source present)
 */
static void
sample_lod(struct g9p_ctx *ctx, nir_tex_instr *tex, unsigned at, int lod)
{
   if (lod < 0)
      return;
   ctx->c->w[at + 3] = (ctx->c->w[at + 3] & 0xff00) | 0x80 |
                       (tex->op == nir_texop_txl ? 0x40 : 0x20) | lod;
   ctx->c->w[at + 4] = (ctx->c->w[at + 4] & 0xff00) | 0x80;
}

/*
 * The fragment shader samples a 2D array or a cube, whose coordinates take
 * r0-r3 (emit_tex_layered())
 */
static bool
uses_layered_tex(nir_function_impl *impl)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex &&
             (nir_instr_as_tex(instr)->is_array ||
              nir_instr_as_tex(instr)->sampler_dim == GLSL_SAMPLER_DIM_CUBE))
            return true;
      }
   }
   return false;
}

/*
 * A 2D array or cube sample, ported from the A9 (G5P) backend (OrchardLinux/mesa
 * PR #5, Berkay Kösebay; Apple's operand layout): the coordinates in r0-r2, where
 * the sample reads them (as g9p_sample_regs() reads r0-r1); an array's layer
 * as an integer in r2's low half, uint(z + 0.5); a cube's direction turned
 * into face coordinates in place by the cube op; then the sample with the
 * dimension in W0. r0-r3 are reserved in such shaders (generate()).
 */
static bool
emit_tex_layered(struct g9p_ctx *ctx, nir_tex_instr *tex)
{
   const bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   int ci = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   int slot = texture_slot(ctx, tex);
   unsigned at;
   int dst, lod;

   if (ctx->stage != MESA_SHADER_FRAGMENT || ci < 0 || slot < 0 || tex->is_shadow ||
       (cube && tex->is_array) ||
       (!cube && tex->sampler_dim != GLSL_SAMPLER_DIM_2D) ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txl && tex->op != nir_texop_txb) ||
       nir_tex_instr_src_index(tex, nir_tex_src_offset) >= 0) {
      fprintf(stderr, "g9p: only plain samples of 2D arrays and cubes\n");
      return false;
   }
   lod = lod_source(ctx, tex);
   if (lod == -2)
      return false;
   for (unsigned c = 0; c < 3; c++) {
      const int r = src_reg(ctx, nir_get_scalar(tex->src[ci].src.ssa, c));

      if (r < 0)
         return false;
      if (c == 2 && !cube) {
         const int half = const_reg(ctx, 0x3f000000);

         if (half < 0)
            return false;
         g9p_alu2(ctx->c, G9P_FADD, 2, r, half, false);
         g9p_convert(ctx->c, 2, 2, G9P_F2U);
      } else {
         g9p_alu2(ctx->c, G9P_OR, c, r, r, false);
      }
   }
   if (cube)
      g9p_cube_coords(ctx->c, 0, 0);
   release(ctx, &tex->instr);
   dst = alloc_range(ctx, 4);
   if (dst < 0)
      return false;
   at = ctx->c->n;
   g9p_sample_regs(ctx->c, dst, tex_state_dword(ctx, slot), tex_sampler_dword(ctx, slot));
   ctx->c->w[at] = (ctx->c->w[at] & ~0x0f00) |
                   (cube ? G9P_TEX_CUBE : G9P_TEX_2D_ARRAY) << 8;
   sample_lod(ctx, tex, at, lod);
   g9p_wait(ctx->c);
   for (unsigned c = 0; c < 4; c++) {
      ctx->reg[key(&tex->def, c)] = dst + c;
      if (c >= tex->def.num_components || !ctx->last_use[key(&tex->def, c)])
         ctx->busy &= ~BITFIELD64_BIT(dst + c);
   }
   return true;
}

/*
 * A vertex shader's sample: explicit level (level 0 without one), the
 * coordinates in registers of their own, as r0-r1 hold the vertex and instance
 * index here.
 */
static bool
emit_tex_vertex(struct g9p_ctx *ctx, nir_tex_instr *tex)
{
   const bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   const int ci = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const int slot = texture_slot(ctx, tex);
   int lod = -1, coord, dst;
   unsigned at;

   if (ci < 0 || slot < 0 || tex->is_shadow || tex->is_array ||
       (!cube && tex->sampler_dim != GLSL_SAMPLER_DIM_2D) ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txl) ||
       nir_tex_instr_src_index(tex, nir_tex_src_offset) >= 0) {
      fprintf(stderr, "g9p: only 2D and cube samples with a level in vertex shaders\n");
      return false;
   }
   if (tex->op == nir_texop_txl) {
      lod = lod_source(ctx, tex);
      if (lod < 0)
         return false;
   } else {
      lod = const_reg(ctx, 0);
      lod = lod < 0 ? lod : low_reg(ctx, lod);
      if (lod < 0)
         return false;
   }
   coord = alloc_low_range(ctx, 4);
   if (coord < 0)
      return false;
   ctx->temps |= BITFIELD64_RANGE(coord, 4);
   for (unsigned c = 0; c < (cube ? 3 : 2); c++) {
      const int r = src_reg(ctx, nir_get_scalar(tex->src[ci].src.ssa, c));

      if (r < 0)
         return false;
      g9p_alu2(ctx->c, G9P_OR, coord + c, r, r, false);
   }
   /* The cube op reads r0-r2 (no other source in Apple's code), reserved in such a shader */
   if (cube) {
      for (unsigned c = 0; c < 3; c++)
         G9P_EMIT(ctx->c, 0x1199, 0x0100 | 2 * c, g9p_srcs(coord + c, coord + c, false) | 0x80,
                  G9P_OR);
      g9p_cube_coords(ctx->c, 2 * coord, 0);
   }
   dst = alloc_range(ctx, 4);
   if (dst < 0)
      return false;
   at = ctx->c->n;
   g9p_sample_vertex(ctx->c, dst, tex_state_dword(ctx, slot), tex_sampler_dword(ctx, slot), coord,
                     lod);
   if (cube) {
      ctx->c->w[at] = (ctx->c->w[at] & ~0x0f00) | G9P_TEX_CUBE << 8;
      ctx->c->w[at + 2] |= 0x8000;
   }
   g9p_wait(ctx->c);
   release(ctx, &tex->instr);
   for (unsigned c = 0; c < 4; c++) {
      ctx->reg[key(&tex->def, c)] = dst + c;
      if (c >= tex->def.num_components || !ctx->last_use[key(&tex->def, c)])
         ctx->busy &= ~BITFIELD64_BIT(dst + c);
   }
   return true;
}

/*
 * A 2D sample, the f32 result in four consecutive registers. Coordinates
 * that are two consecutive varying components (texture(t, v_uv)) are
 * iterated straight to the sampler; computed ones are copied to r0-r1,
 * where the sample reads them (as Apple's code for t.sample(s, uv * 2.0 +
 * 0.25) does; r0-r1 are free until the colour output).
 */
static bool
emit_tex(struct g9p_ctx *ctx, nir_tex_instr *tex)
{
   int ci = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   int slot = texture_slot(ctx, tex);
   int first, second, dst;

   if (tex->op == nir_texop_txs)
      return emit_txs(ctx, tex);
   if (ctx->stage == MESA_SHADER_VERTEX)
      return emit_tex_vertex(ctx, tex);
   if (tex->is_array || tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE)
      return emit_tex_layered(ctx, tex);
   if (ctx->stage != MESA_SHADER_FRAGMENT || tex->sampler_dim != GLSL_SAMPLER_DIM_2D ||
       tex->is_array || tex->is_shadow || ci < 0 || slot < 0 ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txl && tex->op != nir_texop_txb) ||
       nir_tex_instr_src_index(tex, nir_tex_src_offset) >= 0) {
      fprintf(stderr, "g9p: only plain 2D samples of combined image samplers so far\n");
      return false;
   }
   first = varying_component(nir_get_scalar(tex->src[ci].src.ssa, 0));
   second = varying_component(nir_get_scalar(tex->src[ci].src.ssa, 1));
   if (tex->op == nir_texop_tex && first >= 0 && second == first + 1) {
      release(ctx, &tex->instr);
      dst = alloc_range(ctx, 4);
      if (dst < 0)
         return false;
      g9p_sample_varying(ctx->c, dst, first, 2, tex_state_dword(ctx, slot),
                         tex_sampler_dword(ctx, slot));
      ctx->out->num_varyings = MAX2(ctx->out->num_varyings, first / 4 + 1);
   } else {
      const int lod = lod_source(ctx, tex);

      if (lod == -2)
         return false;
      for (unsigned c = 0; c < 2; c++) {
         const int r = src_reg(ctx, nir_get_scalar(tex->src[ci].src.ssa, c));

         if (r < 0)
            return false;
         g9p_alu2(ctx->c, G9P_OR, c, r, r, false);
      }
      release(ctx, &tex->instr);
      dst = alloc_range(ctx, 4);
      if (dst < 0)
         return false;
      {
         const unsigned at = ctx->c->n;

         g9p_sample_regs(ctx->c, dst, tex_state_dword(ctx, slot), tex_sampler_dword(ctx, slot));
         sample_lod(ctx, tex, at, lod);
      }
   }
   g9p_wait(ctx->c);
   for (unsigned c = 0; c < 4; c++) {
      ctx->reg[key(&tex->def, c)] = dst + c;
      if (c >= tex->def.num_components || !ctx->last_use[key(&tex->def, c)])
         ctx->busy &= ~BITFIELD64_BIT(dst + c);
   }
   return true;
}

static bool
emit_intrinsic(struct g9p_ctx *ctx, nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_local_invocation_index:
      /* r0 at entry; reserved from the start when used */
      ctx->reg[key(&intr->def, 0)] = 0;
      return true;

   case nir_intrinsic_load_workgroup_id:
      for (unsigned c = 0; c < intr->def.num_components; c++) {
         int r;

         if (!ctx->last_use[key(&intr->def, c)])
            continue;
         if (c > 1) {
            /* the PDS program writes x and y only so far */
            fprintf(stderr, "g9p: workgroup ID z not supported yet\n");
            return false;
         }
         r = alloc_reg(ctx);
         if (r < 0)
            return false;
         g9p_tg_load(ctx->c, r, c);
         ctx->reg[key(&intr->def, c)] = r;
      }
      g9p_wait(ctx->c);
      ctx->out->uses_workgroup_id = true;
      return true;

   case nir_intrinsic_vulkan_resource_index:
   case nir_intrinsic_load_vulkan_descriptor:
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_centroid:
   case nir_intrinsic_load_barycentric_sample:
      /* resolved where they are used */
      return true;

   case nir_intrinsic_load_front_face: {
      /* Apple's read of the face register, then 0/~0 */
      int r, zero, dst;

      if (ctx->stage != MESA_SHADER_FRAGMENT)
         return false;
      r = alloc_temp(ctx);
      zero = const_reg(ctx, 0);
      dst = alloc_reg(ctx);
      if (r < 0 || zero < 0 || dst < 0)
         return false;
      G9P_EMIT(ctx->c, 0x0198, g9p_dst(ctx->c, r), 0x0131, 0x0110, 0x0154, 0x0038);
      if (!emit_select_const(ctx, G9P_EQ, false, r, zero, 0, ~0u, dst))
         return false;
      ctx->reg[key(&intr->def, 0)] = dst;
      release(ctx, &intr->instr);
      return true;
   }
   case nir_intrinsic_load_vertex_id:
      /* r0 at entry (the vertex kick's DOUTW), pinned in vertex shaders */
      ctx->reg[key(&intr->def, 0)] = 0;
      return true;
   case nir_intrinsic_load_instance_id:
      /* r1 at entry (the vertex kick's second DOUTW), pinned likewise */
      ctx->reg[key(&intr->def, 0)] = 1;
      return true;
   case nir_intrinsic_load_base_instance: {
      /* firstInstance is not passed in: draws with one are refused */
      int r = alloc_reg(ctx);

      if (r < 0)
         return false;
      g9p_mov_imm(ctx->c, r, 0);
      ctx->reg[key(&intr->def, 0)] = r;
      return true;
   }
   case nir_intrinsic_load_input:
      if (ctx->stage == MESA_SHADER_VERTEX)
         return emit_attrib_load(ctx, intr);
      return emit_varying_load(ctx, intr, true);
   case nir_intrinsic_load_frag_coord:
      return emit_frag_coord(ctx, intr);
   case nir_intrinsic_demote:
   case nir_intrinsic_terminate:
      /*
       * `6804` discards every lane, the mask stack notwithstanding: only
       * outside control flow (conditional ones are discard_if, see
       * discards_to_if())
       */
      if (intr->instr.block->cf_node.parent->type != nir_cf_node_function) {
         fprintf(stderr, "g9p: discard inside control flow not supported yet\n");
         return false;
      }
      g9p_discard(ctx->c);
      ctx->out->uses_discard = true;
      return true;
   case nir_intrinsic_demote_if:
   case nir_intrinsic_terminate_if: {
      /* discard where the condition (0 or ~0) is not 0 */
      int r = src_reg(ctx, nir_get_scalar(intr->src[0].ssa, 0)), zero = const_reg(ctx, 0);

      if (r < 0 || zero < 0)
         return false;
      g9p_discard_cmp(ctx->c, G9P_DISCARD_INE, r, zero);
      release(ctx, &intr->instr);
      ctx->out->uses_discard = true;
      return true;
   }
   case nir_intrinsic_load_interpolated_input:
      return emit_varying_load(ctx, intr, perspective(intr));
   case nir_intrinsic_store_output:
      return emit_output(ctx, intr);
   case nir_intrinsic_load_output:
      return emit_fb_fetch(ctx, intr);

   case nir_intrinsic_store_ssbo:
      return emit_mem(ctx, intr, buffer_slot(ctx, intr->src[1]),
                      nir_get_scalar(intr->src[2].ssa, 0), 0, &intr->src[0]);
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_load_ubo:
      /* uniform buffers: the same descriptor (address first) and loads */
      return emit_mem(ctx, intr, buffer_slot(ctx, intr->src[0]),
                      nir_get_scalar(intr->src[1].ssa, 0), 0, NULL);
   case nir_intrinsic_load_push_constant:
      return emit_mem(ctx, intr, g9p_buffer_slot(ctx->out, G9P_PUSH_CONSTANTS, 0),
                      nir_get_scalar(intr->src[0].ssa, 0), nir_intrinsic_base(intr),
                      NULL);

   case nir_intrinsic_decl_reg: {
      int r;

      if (nir_intrinsic_num_components(intr) != 1 || nir_intrinsic_bit_size(intr) != 32 ||
          nir_intrinsic_num_array_elems(intr)) {
         fprintf(stderr, "g9p: only scalar 32-bit NIR registers\n");
         return false;
      }
      /* allocated at its first use (nir_reg()) */
      (void)r;
      return true;
   }
   case nir_intrinsic_load_reg: {
      /* nir_trivialize_registers: read right before its uses */
      int r = nir_reg(ctx, intr->src[0].ssa);

      if (r < 0)
         return false;
      if (r >= 32) {
         /* parked above r31: into a register of the load's own */
         int low = alloc_reg(ctx);

         if (low < 0)
            return false;
         g9p_mov_from_high(ctx->c, low, r);
         r = low;
      }
      ctx->reg[key(&intr->def, 0)] = r;
      release(ctx, &intr->instr);
      return true;
   }
   case nir_intrinsic_store_reg: {
      int dst = nir_reg(ctx, intr->src[1].ssa);
      nir_scalar v = nir_get_scalar(intr->src[0].ssa, 0);
      uint32_t imm;
      int src;

      if (is_const(v, &imm)) {
         g9p_mov_imm(ctx->c, dst, imm);
      } else {
         src = src_reg(ctx, v);
         if (src < 0)
            return false;
         if (dst >= 32)
            g9p_mov_to_high(ctx->c, dst, src);
         else if (src != dst)
            g9p_add_imm(ctx->c, dst, src, 0);
      }
      release(ctx, &intr->instr);
      return true;
   }

   default:
      fprintf(stderr, "g9p: intrinsic %s not supported yet\n",
              nir_intrinsic_infos[intr->intrinsic].name);
      return false;
   }
}

static bool
emit_instr(struct g9p_ctx *ctx, nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_alu:
      return emit_alu(ctx, nir_instr_as_alu(instr));
   case nir_instr_type_intrinsic:
      return emit_intrinsic(ctx, nir_instr_as_intrinsic(instr));
   case nir_instr_type_tex:
      return emit_tex(ctx, nir_instr_as_tex(instr));
   case nir_instr_type_deref:
      /* texture derefs: resolved by emit_tex() */
      return true;
   case nir_instr_type_jump:
      /* a break is emitted by the if whose branch it ends (emit_if()) */
      return nir_instr_as_jump(instr)->type == nir_jump_return ||
             (nir_instr_as_jump(instr)->type == nir_jump_break &&
              instr == nir_block_last_instr(instr->block) &&
              instr->block->cf_node.parent->type == nir_cf_node_if);
   default:
      return false;
   }
}

static nir_block *
only_block(struct exec_list *list)
{
   if (exec_list_length(list) != 1)
      return NULL;
   return nir_cf_node_as_block(exec_node_data(nir_cf_node, exec_list_get_head(list), node));
}

/* A branch that ends by breaking out of the innermost loop */
static bool
ends_with_break(struct exec_list *list)
{
   nir_block *block = only_block(list);
   nir_instr *instr = block ? nir_block_last_instr(block) : NULL;

   return instr && instr->type == nir_instr_type_jump &&
          nir_instr_as_jump(instr)->type == nir_jump_break;
}

static bool
is_empty(struct exec_list *list)
{
   nir_block *block = only_block(list);

   return block && exec_list_is_empty(&block->instr_list);
}

static bool emit_cf_list(struct g9p_ctx *ctx, struct exec_list *list);

/*
 * Compare for an if; *@sense is true when "compare passed" means the
 * condition holds.
 */
static bool
emit_branch_cmp(struct g9p_ctx *ctx, nir_if *nif, bool *sense)
{
   nir_alu_instr *cmp = branch_compare(nif->condition.ssa);
   unsigned cond;
   bool is_float = false;
   int a, b;

   if (!cmp) {
      int r = src_reg(ctx, nir_get_scalar(nif->condition.ssa, 0));

      if (r >= 0)
         r = one_src(ctx, r);
      if (r < 0)
         return false;
      /* condition true: != 0, as unsigned > 0 */
      g9p_cmp_imm(ctx->c, G9P_UGT, r, 0);
      *sense = true;
      return true;
   }

   switch (cmp->op) {
   case nir_op_flt32:  cond = G9P_FLT; *sense = true;  is_float = true; break;
   case nir_op_fge32:  cond = G9P_FLT; *sense = false; is_float = true; break;
   case nir_op_feq32:  cond = G9P_FEQ; *sense = true;  is_float = true; break;
   case nir_op_fneu32: cond = G9P_FEQ; *sense = false; is_float = true; break;
   case nir_op_ilt32:  cond = G9P_SLT; *sense = true;  break;
   case nir_op_ige32:  cond = G9P_SLT; *sense = false; break;
   case nir_op_ult32:  cond = G9P_ULT; *sense = true;  break;
   case nir_op_uge32:  cond = G9P_ULT; *sense = false; break;
   case nir_op_ieq32:  cond = G9P_EQ;  *sense = true;  break;
   default:            cond = G9P_EQ;  *sense = false; break; /* ine32 */
   }
   a = src_reg(ctx, nir_get_scalar(cmp->src[0].src.ssa, cmp->src[0].swizzle[0]));
   b = src_reg(ctx, nir_get_scalar(cmp->src[1].src.ssa, cmp->src[1].swizzle[0]));
   if (a < 0 || b < 0)
      return false;
   g9p_cmp_reg(ctx->c, cond, a, b, is_float);
   ctx->busy &= ~ctx->temps;
   ctx->temps = 0;
   return true;
}

/* After the if: its condition's registers whose last use it was */
static void
release_condition(struct g9p_ctx *ctx, nir_if *nif)
{
   nir_alu_instr *cmp = branch_compare(nif->condition.ssa);

   if (!cmp) {
      release_src(&nif->condition, ctx);
      return;
   }
   for (unsigned i = 0; i < 2; i++)
      release_scalar(ctx, cmp->src[i].src.ssa, cmp->src[i].swizzle[0]);
}

/*
 * Break, then skip to the loop's exit when no lane is left, as Apple's
 * compiler does: the instructions after a break must not run with every
 * lane inactive (a second break then hangs the GPU).
 */
static bool
emit_break(struct g9p_ctx *ctx, bool when_true)
{
   g9p_loop_break(ctx->c, ctx->st, when_true);
   if (ctx->num_exit_jumps == ctx->max_exit_jumps) {
      fprintf(stderr, "g9p: too many breaks\n");
      return false;
   }
   ctx->exit_jumps[ctx->num_exit_jumps++] = g9p_jump(ctx->c, G9P_JMP_NONE, 0);
   return true;
}

/*
 * Push, then branch, else, else branch, pop; each branch is skipped when no
 * lane takes it (jumps to the else and to the pop, as Apple's compiler
 * emits them).
 */
static bool
emit_if_else(struct g9p_ctx *ctx, nir_if *nif, bool sense)
{
   unsigned to_else, to_pop;

   /* keep the lanes for which the condition holds */
   g9p_push_if(ctx->c, ctx->st, !sense);
   to_else = g9p_jump(ctx->c, G9P_JMP_NONE, 0);
   if (!emit_cf_list(ctx, &nif->then_list))
      return false;
   g9p_jump_patch(ctx->c, to_else, 2 * (int)(ctx->c->n - to_else));
   g9p_else(ctx->c, ctx->st);
   to_pop = g9p_jump(ctx->c, G9P_JMP_NONE, 0);
   if (!emit_cf_list(ctx, &nif->else_list))
      return false;
   g9p_jump_patch(ctx->c, to_pop, 2 * (int)(ctx->c->n - to_pop));
   g9p_pop_levels(ctx->c, ctx->st, 1);
   return true;
}

static bool
emit_if(struct g9p_ctx *ctx, nir_if *nif)
{
   bool sense;

   if (!emit_branch_cmp(ctx, nif, &sense))
      return false;

   /*
    * A branch ending with a break (side 0: then, 1: else):
    * - nothing before the break: those lanes leave the loop at once, and
    *   every lane still active took the other branch, which runs inline;
    * - otherwise both branches run under their mask level, and after the
    *   pop the lanes that took the breaking branch leave (compared again;
    *   the operands stay live, see walk_liveness()).
    */
   for (unsigned side = 0; side < 2; side++) {
      struct exec_list *taken = side ? &nif->else_list : &nif->then_list;
      struct exec_list *other = side ? &nif->then_list : &nif->else_list;
      nir_block *block = only_block(taken);
      /* break the lanes whose compare passed (or failed) */
      bool when_passed = side ? !sense : sense;

      if (!ends_with_break(taken))
         continue;
      if (nir_block_first_instr(block) == nir_block_last_instr(block)) {
         if (!emit_break(ctx, when_passed))
            return false;
         ctx->ip++;
         if (!emit_cf_list(ctx, other))
            return false;
         release_condition(ctx, nif);
         return true;
      }

      if (!emit_if_else(ctx, nif, sense))
         return false;
      if (!emit_branch_cmp(ctx, nif, &sense) || !emit_break(ctx, when_passed))
         return false;
      release_condition(ctx, nif);
      return true;
   }

   if (!emit_if_else(ctx, nif, sense))
      return false;
   release_condition(ctx, nif);
   return true;
}

static bool
emit_loop(struct g9p_ctx *ctx, nir_loop *loop)
{
   unsigned body;

   unsigned first_exit = ctx->num_exit_jumps;

   g9p_loop_begin(ctx->c, ctx->st);
   body = ctx->c->n;
   if (!emit_cf_list(ctx, &loop->body))
      return false;
   g9p_jump(ctx->c, G9P_JMP_ANY, -2 * (int)(ctx->c->n - body));
   /* this loop's breaks skip to here when no lane is left */
   for (unsigned j = first_exit; j < ctx->num_exit_jumps; j++)
      g9p_jump_patch(ctx->c, ctx->exit_jumps[j],
                     2 * (int)(ctx->c->n - ctx->exit_jumps[j]));
   ctx->num_exit_jumps = first_exit;
   g9p_pop_levels(ctx->c, ctx->st, 2);
   return true;
}

static bool
emit_cf_list(struct g9p_ctx *ctx, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
            ctx->ip++;
            /* NIR registers whose live range starts here (a loop's start) */
            for (unsigned d = 0; d < ctx->num_decls; d++) {
               if (ctx->reg_first[ctx->decls[d]->index] == ctx->ip &&
                   nir_reg(ctx, ctx->decls[d]) < 0)
                  return false;
            }
            if (!emit_instr(ctx, instr)) {
               if (getenv("G9P_DEBUG") || !ctx->oom) {
                  fprintf(stderr, "g9p: cannot compile: ");
                  nir_print_instr(instr, stderr);
                  fprintf(stderr, "\n");
               }
               return false;
            }
            nir_foreach_def(instr, park_def, ctx);
         }
         break;
      case nir_cf_node_if:
         ctx->ip++;
         if (!emit_if(ctx, nir_cf_node_as_if(node)))
            return false;
         break;
      case nir_cf_node_loop:
         if (!emit_loop(ctx, nir_cf_node_as_loop(node)))
            return false;
         break;
      default:
         return false;
      }
   }
   return true;
}

/* One code generation with the mask stack in @st */
static bool
generate(struct g9p_ctx *ctx, nir_function_impl *impl, unsigned st)
{
   uint16_t *body;
   unsigned body_n;

   bool stack;

   memset(ctx->reg, -1, impl->ssa_alloc * 4);
   ctx->c->n = 0;
   ctx->c->uses_stack = false;
   ctx->busy = ctx->pinned = ctx->temps = 0;
   ctx->max_reg = 0;
   ctx->st = st;
   ctx->out->num_buffers = 0;
   ctx->out->uses_workgroup_id = false;
   ctx->out->num_varyings = 0;
   /* varyings the shader does not read are iterated with perspective */
   for (unsigned i = 0; i < G9P_MAX_VARYINGS; i++)
      ctx->out->varying_perspective[i] = true;
   ctx->out->num_textures = 0;
   ctx->out->uses_discard = false;
   ctx->num_exit_jumps = 0;
   ctx->num_outputs = 0;
   ctx->psize_reg = -1;
   for (unsigned i = 0; i < 4; i++)
      ctx->colour[i] = -1;

   /*
    * Vertex shaders: r0 is the vertex index and r1 the instance index
    * throughout (attribute loads). Fragment shaders: r0-r1 take the colour
    * output's halves at the end.
    */
   if (ctx->stage == MESA_SHADER_VERTEX || ctx->stage == MESA_SHADER_FRAGMENT)
      ctx->busy = ctx->pinned = BITFIELD64_RANGE(0, 2);
   if (ctx->reserve_st_low)
      ctx->busy |= BITFIELD64_BIT(15), ctx->pinned |= BITFIELD64_BIT(15);
   /* 2D array and cube samples: their coordinates in r0-r3 (emit_tex_layered()) */
   if (ctx->stage == MESA_SHADER_FRAGMENT && uses_layered_tex(impl))
      ctx->busy |= BITFIELD64_RANGE(0, 4), ctx->pinned |= BITFIELD64_RANGE(0, 4);
   /* a vertex shader's cube samples use r0-r3 too: the indices kept elsewhere */
   ctx->vertex_reg = 0;
   ctx->instance_reg = 1;
   if (ctx->stage == MESA_SHADER_VERTEX && uses_layered_tex(impl)) {
      ctx->busy |= BITFIELD64_RANGE(0, 4), ctx->pinned |= BITFIELD64_RANGE(0, 4);
      ctx->vertex_reg = alloc_pair(ctx);
      if (ctx->vertex_reg < 0)
         return false;
      ctx->instance_reg = ctx->vertex_reg + 1;
      ctx->pinned |= BITFIELD64_RANGE(ctx->vertex_reg, 2);
      g9p_alu2(ctx->c, G9P_OR, ctx->vertex_reg, 0, 0, false);
      g9p_alu2(ctx->c, G9P_OR, ctx->instance_reg, 1, 1, false);
   }

   /* compute: r0 holds the local index from the start when it is used */
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_load_local_invocation_index)
            ctx->busy |= BITFIELD64_BIT(0);
      }
   }

   ctx->ip = 0;
   if (!emit_cf_list(ctx, &impl->body))
      return false;

   /*
    * Frame. Compute: prologue (thread count), mask stack, body, epilogue,
    * end. Vertex and fragment: the mask stack only when the body pushes
    * (control flow, selects), then the outputs; the last vertex output
    * write is the end.
    */
   stack = ctx->stage == MESA_SHADER_COMPUTE || ctx->c->uses_stack;
   body_n = ctx->c->n;
   body = malloc(body_n * sizeof(uint16_t) + 1);
   if (!body)
      return false;
   memcpy(body, ctx->c->w, body_n * sizeof(uint16_t));
   ctx->c->n = 0;
   if (ctx->stage == MESA_SHADER_COMPUTE)
      g9p_prologue(ctx->c, 4 * ctx->out->num_buffers);
   if (stack)
      g9p_push_init(ctx->c, st);
   g9p_emit(ctx->c, body, body_n);
   if (stack)
      g9p_pop(ctx->c, st);
   free(body);

   switch (ctx->stage) {
   case MESA_SHADER_VERTEX:
      if (!ctx->num_outputs && !ctx->point_size) {
         fprintf(stderr, "g9p: vertex shader without outputs\n");
         return false;
      }
      /* points: the size is the last output dword (Apple's vertex code) */
      if (ctx->point_size) {
         if (ctx->psize_reg < 0) {
            ctx->psize_reg = alloc_reg(ctx);
            if (ctx->psize_reg < 0)
               return false;
            g9p_mov_imm(ctx->c, ctx->psize_reg, 0x3f800000);
         }
         ctx->outputs[ctx->num_outputs].reg = ctx->psize_reg;
         ctx->outputs[ctx->num_outputs].slot = 4 + 4 * ctx->out->num_varyings;
         ctx->num_outputs++;
      }
      for (unsigned i = 0; i + 1 < ctx->num_outputs; i++)
         g9p_vs_output(ctx->c, ctx->outputs[i].reg, ctx->outputs[i].slot);
      g9p_end_vertex(ctx->c, ctx->outputs[ctx->num_outputs - 1].reg,
                     ctx->outputs[ctx->num_outputs - 1].slot);
      ctx->out->num_outputs = 4 + 4 * ctx->out->num_varyings + ctx->point_size;
      break;
   case MESA_SHADER_FRAGMENT: {
      bool colour = false;

      for (unsigned i = 0; i < 4; i++)
         colour |= ctx->colour[i] >= 0;
      if (colour) {
         /* unwritten components: 0 */
         for (unsigned i = 0; i < 4; i++) {
            if (ctx->colour[i] < 0) {
               ctx->colour[i] = alloc_reg(ctx);
               if (ctx->colour[i] < 0)
                  return false;
               g9p_mov_imm(ctx->c, ctx->colour[i], 0);
            }
         }
         for (unsigned i = 0; i < 4; i++)
            g9p_to_half(ctx->c, i, ctx->colour[i]);
         if (ctx->out->native_blend)
            g9p_blend_src_over(ctx->c);
      }
      g9p_end_fragment(ctx->c, colour);
      break;
   }
   default:
      g9p_end(ctx->c);
      break;
   }
   return true;
}

int
g9p_buffer_slot(struct g9p_shader *out, unsigned set, unsigned binding)
{
   unsigned k;

   for (k = 0; k < out->num_buffers; k++) {
      if (out->buffers[k].set == set && out->buffers[k].binding == binding)
         return k;
   }
   if (k == G9P_MAX_BUFFERS)
      return -1;
   out->buffers[k].set = set;
   out->buffers[k].binding = binding;
   out->buffers[k].dwords = 0;
   out->num_buffers++;
   return k;
}



static void
optimize(nir_shader *nir)
{
   bool progress;

   do {
      progress = false;
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      NIR_PASS(progress, nir, nir_lower_alu_to_scalar, NULL, NULL);
      NIR_PASS(progress, nir, nir_lower_phis_to_scalar, NULL, NULL);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
   } while (progress);
}

static unsigned
type_size_vec4(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/*
 * A buffer descriptor's second component (the offset of the
 * 32bit_index_offset address format) is always 0 here: the descriptor
 * address already includes the buffer offset. A literal 0 lets the
 * offsets of every buffer load fold to constants.
 */
static bool
zero_descriptor_offset(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_def *v;

   if (intr->intrinsic != nir_intrinsic_load_vulkan_descriptor &&
       intr->intrinsic != nir_intrinsic_vulkan_resource_index)
      return false;
   if (intr->def.num_components != 2)
      return false;
   b->cursor = nir_after_instr(&intr->instr);
   v = nir_vec2(b, nir_channel(b, &intr->def, 0), nir_imm_int(b, 0));
   nir_def_rewrite_uses_after(&intr->def, v);
   return true;
}

/*
 * Framebuffer fetch (nir_lower_blend's load_output): each f32 component
 * from a raw 32-bit word of the tile (emit_fb_fetch(), dest_type uint32),
 * f16 to f32 in integer code: (h & 0x7fff) << 13 + 0x38000000, clamped to
 * [0, 1] as the attachment's unorm format would. The tile holds what the last
 * shader wrote, unclamped until it is written to memory: a blend that went
 * negative read back positive (dEQP fragment_ops).
 */
static bool
lower_fb_fetch_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_def *comps[4];

   if (intr->intrinsic != nir_intrinsic_load_output ||
       nir_intrinsic_dest_type(intr) == nir_type_uint32)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   for (unsigned i = 0; i < intr->def.num_components; i++) {
      const unsigned k = nir_intrinsic_component(intr) + i;
      nir_def *word = nir_load_output(b, 1, 32, nir_imm_int(b, 0),
                                      .base = nir_intrinsic_base(intr), .component = k / 2,
                                      .dest_type = nir_type_uint32,
                                      .io_semantics = nir_intrinsic_io_semantics(intr));
      nir_def *h = k & 1 ? nir_ushr_imm(b, word, 16) : nir_iand_imm(b, word, 0xffff);
      nir_def *bits = nir_iadd_imm(b, nir_ishl_imm(b, nir_iand_imm(b, h, 0x7fff), 13),
                                   0x38000000);

      /* h - 1 >= 0x7fff: zero or negative */
      nir_def *low = nir_uge_imm(b, nir_iand_imm(b, nir_iadd_imm(b, h, -1), 0xffff), 0x7fff);

      comps[i] = nir_bcsel(b, low, nir_imm_int(b, 0),
                           nir_bcsel(b, nir_uge_imm(b, h, 0x3c00),
                                     nir_imm_int(b, 0x3f800000), bits));
   }
   nir_def_replace(&intr->def, nir_vec(b, comps, intr->def.num_components));
   return true;
}

/* One function, globals as locals; buffers and push constants lowered */
static void
lower_front(nir_shader *nir)
{
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   nir_remove_non_entrypoints(nir);
   NIR_PASS(_, nir, nir_opt_deref);
   NIR_PASS(_, nir, nir_lower_variable_initializers, ~0);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_split_struct_vars, nir_var_function_temp);
   /* local arrays indexed dynamically: if-ladders, then SSA */
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
            nir_var_function_temp | nir_var_shader_in | nir_var_shader_out, UINT32_MAX);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo | nir_var_mem_ubo,
            nir_address_format_32bit_index_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, zero_descriptor_offset,
            nir_metadata_control_flow, NULL);
}

/*
 * Inputs and outputs: outputs through temporaries (their stores at the
 * end), lowered to scalar load/store intrinsics with the location as base
 */
static void
lower_io(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);

   NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries, impl, nir_var_shader_out);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_in | nir_var_shader_out)
      var->data.driver_location = var->data.location;
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            type_size_vec4, 0);
}

/*
 * Loads that are cheap to repeat (uniform buffers, push constants,
 * varyings, the fragment position): a copy right before each use in another
 * block, instead of one value live across branches (graphics shaders have
 * 32 registers; Hyprland's rounded-rectangle shader kept dozens of uniforms
 * live from its first block). @every_use: for a vertex shader that ran out
 * of registers, a copy before each use after the first, attributes too.
 */
static bool
rematerialize_loads(nir_shader *nir, bool every_use)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         nir_intrinsic_instr *intr;

         if (instr->type != nir_instr_type_intrinsic)
            continue;
         intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_ubo &&
             intr->intrinsic != nir_intrinsic_load_push_constant &&
             intr->intrinsic != nir_intrinsic_load_interpolated_input &&
             intr->intrinsic != nir_intrinsic_load_frag_coord &&
             !(intr->intrinsic == nir_intrinsic_load_input &&
               (nir->info.stage == MESA_SHADER_FRAGMENT || every_use)))
            continue;
         bool first = true;

         nir_foreach_use_safe(src, &intr->def) {
            nir_instr *user = nir_src_use_instr(src);
            nir_instr *copy;

            if (nir_src_is_if(src) || user->type == nir_instr_type_phi)
               continue;
            if (user->block == instr->block && (!every_use || first)) {
               first = false;
               continue;
            }
            copy = nir_instr_clone(nir, instr);
            nir_instr_insert_before(user, copy);
            nir_src_rewrite(src, &nir_instr_as_intrinsic(copy)->def);
            progress = true;
         }
      }
   }
   return nir_progress(progress, impl, nir_metadata_control_flow);
}

/*
 * A fragment shader reads its buffers from the common store, as uniform
 * operands with a fixed index: a load at a computed offset becomes a chain of
 * selects over the offsets it can have (GL's uniform arrays indexed in a loop)
 */
static bool
lower_dynamic_ubo_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_def *value;
   unsigned n;

   if (intr->intrinsic != nir_intrinsic_load_ubo || nir_src_is_const(intr->src[1]) ||
       intr->def.num_components != 1 || intr->def.bit_size != 32)
      return false;
   n = MIN2(nir_intrinsic_range(intr) / 4, G9P_FS_BUF_DWORDS);
   b->cursor = nir_before_instr(&intr->instr);
   value = nir_imm_int(b, 0);
   for (unsigned k = 0; k < n; k++) {
      const unsigned at = nir_intrinsic_range_base(intr) + 4 * k;
      nir_def *load = nir_load_ubo(b, 1, 32, intr->src[0].ssa, nir_imm_int(b, at),
                                   .align_mul = 4, .align_offset = 0,
                                   .range_base = nir_intrinsic_range_base(intr),
                                   .range = nir_intrinsic_range(intr));

      value = nir_bcsel(b, nir_ieq_imm(b, intr->src[1].ssa, at), load, value);
   }
   nir_def_replace(&intr->def, value);
   return true;
}

/* Optimization, out of SSA, then code for @ctx->stage */
static bool
compile_back(nir_shader *nir, struct g9p_shader *out, mesa_shader_stage stage,
             const struct g9p_vertex_attrib *attribs, unsigned frag_coord_slot,
             bool repeat_loads, bool point_size, void *mem_ctx)
{
   struct g9p_ctx ctx = { .point_size = point_size, .flush_outputs = repeat_loads };
   nir_function_impl *impl;
   unsigned st;

   /* texture2DProj */
   {
      const nir_lower_tex_options tex = { .lower_txp = ~0u };

      NIR_PASS(_, nir, nir_lower_tex, &tex);
   }
   /* integer division through floats: no such instruction (GLSL ES 1.00 int and ivec /) */
   {
      const nir_lower_idiv_options idiv = { .allow_fp16 = false };

      NIR_PASS(_, nir, nir_lower_idiv, &idiv);
      /* its umul_high (nir_shader_compiler_options.lower_mul_high) */
      NIR_PASS(_, nir, nir_lower_alu);
   }
   optimize(nir);
   NIR_PASS(_, nir, nir_lower_io_to_scalar,
            nir_var_mem_ssbo | nir_var_mem_ubo | nir_var_mem_push_const |
               nir_var_shader_in | nir_var_shader_out,
            NULL, NULL);
   NIR_PASS(_, nir, nir_opt_algebraic_late);
   optimize(nir);
   if (stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_dynamic_ubo_instr,
               nir_metadata_control_flow, NULL);
      optimize(nir);
   }
   NIR_PASS(_, nir, nir_lower_bool_to_int32);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, rematerialize_loads, repeat_loads);
   NIR_PASS(_, nir, nir_opt_dce);
   /* values computed early for late uses: compute them right before */
   NIR_PASS(_, nir, nir_opt_move, nir_move_alu | nir_move_const_undef |
            nir_move_comparisons | nir_move_copies | nir_move_load_ubo |
            nir_move_load_input);
   NIR_PASS(_, nir, nir_convert_from_ssa, true, false);
   /* a shader that ran out of registers: order by register pressure */
   if (repeat_loads) {
      const nir_schedule_options sched = { .threshold = 16, .fallback = true };

      nir_schedule(nir, &sched);
   }
   NIR_PASS(_, nir, nir_trivialize_registers);
   copy_branch_compare_operands(nir);

   impl = nir_shader_get_entrypoint(nir);
   if (!impl)
      return false;
   nir_index_ssa_defs(impl);
   ctx.tex_total = count_textures(impl);
   if (getenv("G9P_DEBUG"))
      nir_print_shader(nir, stderr);

   if (stage == MESA_SHADER_COMPUTE) {
      for (unsigned i = 0; i < 3; i++)
         out->local_size[i] = nir->info.workgroup_size[i];
   }

   out->stage = stage;
   ctx.out = out;
   ctx.stage = stage;
   ctx.attribs = attribs;
   ctx.frag_coord_slot = frag_coord_slot;
   ctx.c = rzalloc(mem_ctx, struct g9p_code);
   ctx.c->fragment = stage == MESA_SHADER_FRAGMENT;
   ctx.reg = ralloc_array(mem_ctx, int8_t, impl->ssa_alloc * 4);
   ctx.num_keys = impl->ssa_alloc * 4;
   ctx.last_use = rzalloc_array(mem_ctx, unsigned, impl->ssa_alloc * 4);
   ctx.def_ip = rzalloc_array(mem_ctx, unsigned, impl->ssa_alloc);
   ctx.reg_first = ralloc_array(mem_ctx, unsigned, impl->ssa_alloc);
   memset(ctx.reg_first, 0xff, impl->ssa_alloc * sizeof(unsigned));
   ctx.decls = ralloc_array(mem_ctx, nir_def *, impl->ssa_alloc);
   ctx.max_exit_jumps = 256;
   ctx.exit_jumps = ralloc_array(mem_ctx, unsigned, ctx.max_exit_jumps);

   ctx.ip = 0;
   walk_liveness(&ctx, &impl->body, false);
   if (ctx.loop_index > G9P_MAX_LOOPS) {
      fprintf(stderr, "g9p: too many loops\n");
      return false;
   }
   ctx.ip = 0;
   ctx.loop_index = 0;
   walk_liveness(&ctx, &impl->body, true);

   /*
    * Twice: to find the register count, then with the mask stack after it.
    * Fragment shaders that run out of registers park the NIR register live
    * at that point with the furthest last use above r31 and try again.
    */
   ctx.home = ralloc_array(mem_ctx, int8_t, impl->ssa_alloc * 4);
   memset(ctx.home, -1, impl->ssa_alloc * 4);
   for (;;) {
      unsigned best = 0;
      int victim = -1;

      ctx.oom = false;
      if (generate(&ctx, impl, ctx.reserve_st_low ? 15 : G9P_NUM_REGS))
         break;
      if (ctx.oom && stage == MESA_SHADER_VERTEX && !ctx.reserve_low) {
         ctx.reserve_low = true;
         ctx.reserve_st_low = true;
         continue;
      }
      if (!ctx.oom || stage == MESA_SHADER_COMPUTE || ctx.num_homes >= 96)
         return false;
      for (unsigned d = 0; d < ctx.num_decls; d++) {
         const unsigned k = key(ctx.decls[d], 0);

         if (ctx.home[k] < 0 && ctx.reg_first[ctx.decls[d]->index] <= ctx.oom_ip &&
             ctx.last_use[k] >= ctx.oom_ip && ctx.last_use[k] > best) {
            best = ctx.last_use[k];
            victim = k;
         }
      }
      /* else the live value (not a NIR register's) with the furthest last use */
      for (unsigned k = 0, ssa_best = 0; victim < 0 && k < ctx.num_keys; k++) {
         const int r = ctx.reg[k];

         if (r < 0 || r >= 32 || ctx.home[k] >= 0 || ctx.last_use[k] <= ctx.oom_ip ||
             (ctx.pinned | ctx.temps) & BITFIELD64_BIT(r) || ctx.last_use[k] <= ssa_best)
            continue;
         ssa_best = ctx.last_use[k];
         best = k + 1;
      }
      if (victim < 0 && best && best <= ctx.num_keys && ctx.home[best - 1] < 0 &&
          ctx.reg[best - 1] >= 0)
         victim = best - 1;
      if (victim < 0) {
         fprintf(stderr, "g9p: out of registers at instruction %u\n", ctx.oom_ip);
         return false;
      }
      ctx.home[victim] = 32 + ctx.num_homes++;
   }
   st = ctx.max_reg + 1;
   /*
    * Vertex shaders: the mask stack below r16, selects use it too (dEQP vertex
    * shaders of 17 registers and more drew wrong values or hung the geometry
    * job). Reserved in the second pass.
    */
   if (stage == MESA_SHADER_VERTEX && ctx.reserve_st_low) {
      st = MAX2(ctx.max_reg, 15);
   } else if (stage == MESA_SHADER_VERTEX && st > 15) {
      ctx.reserve_st_low = true;
      if (!generate(&ctx, impl, 15)) {
         /* without r15, address work can run out of low registers */
         if (!ctx.oom || ctx.reserve_low)
            return false;
         ctx.reserve_low = true;
         ctx.oom = false;
         if (!generate(&ctx, impl, 15))
            return false;
      }
      st = MAX2(ctx.max_reg, 15);
   } else if (!generate(&ctx, impl, st))
      return false;

   out->num_regs = MAX2(st + 1, ctx.num_homes ? 32 + ctx.num_homes : 0);
   out->code = ctx.c->w;
   out->code_size = 2 * ctx.c->n;
   if (stage != MESA_SHADER_COMPUTE && st + 1 > 32) {
      fprintf(stderr, "g9p: %u registers, a graphics task has 32\n", out->num_regs);
      return false;
   }
   if (getenv("G9P_DEBUG")) {
      fprintf(stderr, "g9p: %u registers, %u bytes of code\n", out->num_regs,
              out->code_size);
      for (unsigned i = 0; i < ctx.c->n; i++)
         fprintf(stderr, "%04x%c", ctx.c->w[i], (ctx.c->w[i] & 0x100) ? ' ' : '\n');
   }
   return true;
}

bool
g9p_compile_compute(nir_shader *nir, struct g9p_shader *out, void *mem_ctx)
{
   const nir_lower_compute_system_values_options csv = {
      .lower_cs_local_id_to_index = true,
      .global_id_is_32bit = true,
   };

   memset(out, 0, sizeof(*out));
   lower_front(nir);
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv);
   return compile_back(nir, out, MESA_SHADER_COMPUTE, NULL, 0, false, false, mem_ctx);
}

/*
 * The fragment position as a varying: window coordinates from gl_Position
 * and the viewport (G9P_VIEWPORT buffer), written to varying @slot
 */
static void
add_frag_coord_output(nir_shader *nir, unsigned slot)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_variable *pos = nir_find_variable_with_location(nir, nir_var_shader_out,
                                                       VARYING_SLOT_POS);
   nir_variable *out;
   nir_builder b;
   nir_def *p, *vp[6], *inv_w, *index;

   if (!pos)
      return;
   out = nir_variable_create(nir, nir_var_shader_out, glsl_vec4_type(), "g9p_frag_coord");
   out->data.location = VARYING_SLOT_VAR0 + slot;
   out->data.interpolation = INTERP_MODE_NOPERSPECTIVE;

   b = nir_builder_at(nir_after_impl(impl));
   p = nir_load_var(&b, pos);
   index = nir_vulkan_resource_index(&b, 2, 32, nir_imm_int(&b, 0),
                                     .desc_set = G9P_VIEWPORT, .binding = 0,
                                     .desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
   for (unsigned i = 0; i < 6; i++)
      vp[i] = nir_load_ubo(&b, 1, 32, index, nir_imm_int(&b, 4 * i), .align_mul = 4,
                           .range = ~0);
   inv_w = nir_frcp(&b, nir_channel(&b, p, 3));
   nir_store_var(&b, out,
                 nir_vec4(&b,
                          nir_ffma(&b, nir_fmul(&b, nir_channel(&b, p, 0), inv_w), vp[0], vp[1]),
                          nir_ffma(&b, nir_fmul(&b, nir_channel(&b, p, 1), inv_w), vp[2], vp[3]),
                          nir_ffma(&b, nir_fmul(&b, nir_channel(&b, p, 2), inv_w), vp[4], vp[5]),
                          inv_w),
                 0xf);
}

unsigned
g9p_frag_coord_slot(const nir_shader *vs)
{
   unsigned slot = 0;

   nir_foreach_variable_with_modes(var, vs, nir_var_shader_out) {
      if (var->data.location >= VARYING_SLOT_VAR0)
         slot = MAX2(slot, var->data.location - VARYING_SLOT_VAR0 +
                              glsl_count_attribute_slots(var->type, false));
   }
   return MIN2(slot, G9P_MAX_VARYINGS - 1);
}

/*
 * Attributes of formats with channels other than 32 bits: their packed
 * words (a 32-bit load each), unpacked to the components the shader reads
 * (nir_format_unpack_rgba() fills missing ones with 0, 0, 0, 1). Attributes without a buffer read 0, 0, 0, 1,
 * GL's value for a disabled array (Zink binds none for Qt Quick's _qt_order).
 */
static bool
lower_attrib_format(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct g9p_vertex_attrib *attribs = data;
   const struct util_format_description *desc;
   nir_def *packed, *rgba, *comps[4];
   nir_io_semantics sem;
   unsigned loc;

   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;
   sem = nir_intrinsic_io_semantics(intr);
   loc = sem.location - VERT_ATTRIB_GENERIC0;
   if (loc >= G9P_MAX_ATTRIBS)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   if (!attribs[loc].present) {
      for (unsigned i = 0; i < intr->def.num_components; i++) {
         const bool w = nir_intrinsic_component(intr) + i == 3;
         const bool is_int = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) !=
                             nir_type_float;

         comps[i] = w ? (is_int ? nir_imm_int(b, 1) : nir_imm_float(b, 1.0f)) : nir_imm_int(b, 0);
      }
      nir_def_replace(&intr->def, nir_vec(b, comps, intr->def.num_components));
      return true;
   }
   desc = util_format_description(attribs[loc].format);
   if (!desc)
      return false;
   if (desc->channel[0].size == 32) {
      /* fewer channels than the shader reads (a vec3 position): GL's 0, 0, 0, 1 */
      const unsigned first = nir_intrinsic_component(intr);
      const unsigned wanted = intr->def.num_components;
      const bool is_int = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) !=
                          nir_type_float;
      nir_def *load;

      if (first + wanted <= desc->nr_channels)
         return false;
      load = first < desc->nr_channels
                ? nir_load_input(b, desc->nr_channels - first, 32, intr->src[0].ssa,
                                 .base = nir_intrinsic_base(intr),
                                 .range = nir_intrinsic_range(intr), .component = first,
                                 .dest_type = nir_intrinsic_dest_type(intr), .io_semantics = sem)
                : NULL;
      for (unsigned i = 0; i < wanted; i++) {
         const unsigned k = first + i;

         comps[i] = k < desc->nr_channels ? nir_channel(b, load, i)
                    : k == 3 ? (is_int ? nir_imm_int(b, 1) : nir_imm_float(b, 1.0f))
                             : nir_imm_int(b, 0);
      }
      nir_def_replace(&intr->def, nir_vec(b, comps, wanted));
      return true;
   }

   /* an attribute off a word boundary: the words around it (emit_input aligns) */
   const unsigned shift = (attribs[loc].offset % 4) * 8;
   const unsigned words = DIV_ROUND_UP(desc->block.bits, 32);

   packed = nir_load_input(b, DIV_ROUND_UP(desc->block.bits + shift, 32), 32, intr->src[0].ssa,
                           .base = nir_intrinsic_base(intr),
                           .range = nir_intrinsic_range(intr), .component = 0,
                           .dest_type = nir_type_uint32, .io_semantics = sem);
   if (shift) {
      for (unsigned i = 0; i < words; i++) {
         comps[i] = nir_ushr_imm(b, nir_channel(b, packed, i), shift);
         if (i + 1 < packed->num_components)
            comps[i] = nir_ior(b, comps[i],
                               nir_ishl_imm(b, nir_channel(b, packed, i + 1), 32 - shift));
      }
      packed = nir_vec(b, comps, words);
   }
   rgba = nir_format_unpack_rgba(b, packed, attribs[loc].format);
   nir_def_replace(&intr->def, nir_channels(b, rgba, nir_component_mask(intr->def.num_components)
                                                        << nir_intrinsic_component(intr)));
   return true;
}

bool
g9p_compile_vertex(nir_shader *nir, const struct g9p_vertex_attrib *attribs,
                   int frag_coord_slot, struct g9p_shader *out, bool point_size, void *mem_ctx)
{
   memset(out, 0, sizeof(*out));
   lower_front(nir);
   if (frag_coord_slot >= 0)
      add_frag_coord_output(nir, frag_coord_slot);
   lower_io(nir);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_attrib_format,
            nir_metadata_control_flow, (void *)attribs);
   /* out of registers: again with a value's loads repeated, not kept live */
   nir_shader *copy = nir_shader_clone(mem_ctx, nir);

   return compile_back(nir, out, MESA_SHADER_VERTEX, attribs, 0, false, point_size, mem_ctx) ||
          compile_back(copy, out, MESA_SHADER_VERTEX, attribs, 0, true, point_size, mem_ctx);
}

/*
 * `if (c) discard;` (an if with only a demote/terminate in its then-branch
 * and an empty else) to demote_if(c)/terminate_if(c), which compile to a
 * compare and `6004`
 */
static bool
discards_to_if(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool progress = false;

   nir_foreach_block_safe(block, impl) {
      nir_if *nif = nir_block_get_following_if(block);
      nir_block *then_block;
      nir_intrinsic_instr *intr;
      nir_builder b;

      if (!nif || !exec_list_is_singular(&nif->then_list) ||
          !nir_cf_list_is_empty_block(&nif->else_list))
         continue;
      then_block = nir_if_first_then_block(nif);
      if (exec_list_length(&then_block->instr_list) != 1)
         continue;
      if (nir_block_first_instr(then_block)->type != nir_instr_type_intrinsic)
         continue;
      intr = nir_instr_as_intrinsic(nir_block_first_instr(then_block));
      if (intr->intrinsic != nir_intrinsic_demote && intr->intrinsic != nir_intrinsic_terminate)
         continue;
      b = nir_builder_at(nir_before_cf_node(&nif->cf_node));
      if (intr->intrinsic == nir_intrinsic_demote)
         nir_demote_if(&b, nif->condition.ssa);
      else
         nir_terminate_if(&b, nif->condition.ssa);
      nir_cf_node_remove(&nif->cf_node);
      progress = true;
   }
   return nir_progress(progress, impl, nir_metadata_none);
}

/*
 * Discards left inside control flow (nested, or with more around them):
 * each sets a flag instead, and one discard_if on it ends the shader. The
 * shader runs on for such pixels, which only matters for side effects
 * (none in the shaders seen so far).
 */
static bool
discards_to_flag(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_variable *flag = NULL;
   nir_builder b;

   nir_foreach_block_safe(block, impl) {
      if (block->cf_node.parent->type == nir_cf_node_function)
         continue;
      nir_foreach_instr_safe(instr, block) {
         nir_intrinsic_instr *intr;
         nir_def *cond;

         if (instr->type != nir_instr_type_intrinsic)
            continue;
         intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_demote &&
             intr->intrinsic != nir_intrinsic_terminate &&
             intr->intrinsic != nir_intrinsic_demote_if &&
             intr->intrinsic != nir_intrinsic_terminate_if)
            continue;
         if (!flag) {
            flag = nir_local_variable_create(impl, glsl_bool_type(), "discarded");
            b = nir_builder_at(nir_before_impl(impl));
            nir_store_var(&b, flag, nir_imm_false(&b), 1);
         }
         b = nir_builder_at(nir_before_instr(instr));
         cond = intr->intrinsic == nir_intrinsic_demote ||
                      intr->intrinsic == nir_intrinsic_terminate
                   ? nir_imm_true(&b)
                   : nir_ior(&b, nir_load_var(&b, flag), intr->src[0].ssa);
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

/*
 * Discards of a shader that reads the tile (blending): each colour output
 * keeps the tile's value where the pixel is discarded, instead of a
 * hardware discard. Hyprland's border shader (blending, discarding inside
 * the window) lost its border in random 32x32 tiles with the hardware
 * discard in the translucent pass, while the same draw alone (vk-tri) was
 * right. A discarded pixel writes nothing wherever the discard is, so
 * the outputs are stored at the end, selected on all discard conditions.
 * Needs every discard and colour store outside control flow (after
 * discards_to_if and discards_to_flag); otherwise the hardware discard
 * stays.
 */
static bool
discards_to_select(nir_shader *nir, bool zero)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_intrinsic_instr *discards[16], *stores[8];
   unsigned num_discards = 0, num_stores = 0;
   nir_def *cond = NULL;
   nir_builder b;

   nir_foreach_block(block, impl) {
      const bool top = block->cf_node.parent->type == nir_cf_node_function;

      nir_foreach_instr(instr, block) {
         nir_intrinsic_instr *intr;

         if (instr->type != nir_instr_type_intrinsic)
            continue;
         intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_demote:
         case nir_intrinsic_terminate:
         case nir_intrinsic_demote_if:
         case nir_intrinsic_terminate_if:
            if (!top || num_discards == ARRAY_SIZE(discards))
               return nir_no_progress(impl);
            discards[num_discards++] = intr;
            break;
         case nir_intrinsic_store_output:
            if (!top || num_stores == ARRAY_SIZE(stores) ||
                nir_intrinsic_io_semantics(intr).location < FRAG_RESULT_DATA0 ||
                intr->src[0].ssa->bit_size != 32)
               return nir_no_progress(impl);
            stores[num_stores++] = intr;
            break;
         default:
            break;
         }
      }
   }
   if (!num_discards || !num_stores)
      return nir_no_progress(impl);

   b = nir_builder_at(nir_after_impl(impl));
   for (unsigned i = 0; i < num_discards; i++) {
      nir_intrinsic_instr *intr = discards[i];
      nir_def *c = intr->intrinsic == nir_intrinsic_demote_if ||
                         intr->intrinsic == nir_intrinsic_terminate_if
                      ? intr->src[0].ssa
                      : nir_imm_true(&b);

      cond = cond ? nir_ior(&b, cond, c) : c;
      nir_instr_remove(&intr->instr);
   }
   for (unsigned i = 0; i < num_stores; i++) {
      nir_intrinsic_instr *st = stores[i];
      nir_io_semantics sem = nir_intrinsic_io_semantics(st);
      nir_def *value = st->src[0].ssa;
      nir_def *fb;

      sem.fb_fetch_output = 1;
      /* native source-over: transparent black keeps the tile's colour */
      fb = zero ? nir_imm_zero(&b, value->num_components, 32)
                : nir_load_output(&b, value->num_components, 32, nir_imm_int(&b, 0),
                                  .base = nir_intrinsic_base(st),
                                  .component = nir_intrinsic_component(st),
                                  .dest_type = nir_type_float32, .io_semantics = sem);
      nir_store_output(&b, nir_bcsel(&b, cond, fb, value), nir_imm_int(&b, 0),
                       .base = nir_intrinsic_base(st),
                       .write_mask = nir_intrinsic_write_mask(st),
                       .component = nir_intrinsic_component(st),
                       .src_type = nir_intrinsic_src_type(st),
                       .io_semantics = nir_intrinsic_io_semantics(st));
      nir_instr_remove(&st->instr);
   }
   return nir_progress(true, impl, nir_metadata_none);
}

/*
 * Premultiplied source-over on all four channels of a colour attachment
 * (Hyprland's and Zink's usual blend): done natively at the end
 */
/* The dynamic blend constants of nir_lower_blend: a buffer the draw uploads */
static bool
lower_blend_constant(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   unsigned c;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_blend_const_color_r_float: c = 0; break;
   case nir_intrinsic_load_blend_const_color_g_float: c = 1; break;
   case nir_intrinsic_load_blend_const_color_b_float: c = 2; break;
   case nir_intrinsic_load_blend_const_color_a_float: c = 3; break;
   default: return false;
   }
   b->cursor = nir_before_instr(&intr->instr);

   nir_def *index = nir_vulkan_resource_index(b, 2, 32, nir_imm_int(b, 0),
                                              .desc_set = G9P_BLEND_CONSTANTS, .binding = 0,
                                              .desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

   nir_def_replace(&intr->def,
                   nir_load_ubo(b, 1, 32, index, nir_imm_int(b, 4 * c), .align_mul = 4,
                                .range = ~0));
   return true;
}

static bool
is_src_over(const nir_lower_blend_options *blend)
{
   const nir_lower_blend_rt *rt = &blend->rt[0];
   const nir_lower_blend_channel *ch[2] = { &rt->rgb, &rt->alpha };

   if (blend->logicop_enable || rt->colormask != 0xf)
      return false;
   for (unsigned i = 0; i < 2; i++) {
      if (ch[i]->func != PIPE_BLEND_ADD || ch[i]->src_factor != PIPE_BLENDFACTOR_ONE ||
          ch[i]->dst_factor != PIPE_BLENDFACTOR_INV_SRC_ALPHA)
         return false;
   }
   return true;
}

/*
 * texelFetch of a 2D texture (Zink's blits): an explicit-level sample at
 * the texel's centre, (i + 0.5) / size, which reads that texel alone with
 * nearest and linear filtering. The size is the base level's (txs), right
 * for the single-level views blits read.
 */
static bool
lower_txf(nir_builder *b, nir_tex_instr *tex, void *data)
{
   const int ci = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const int li = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   nir_def *size, *coord;

   if (tex->op != nir_texop_txf || tex->sampler_dim != GLSL_SAMPLER_DIM_2D || tex->is_array ||
       ci < 0)
      return false;
   size = nir_i2f32(b, nir_get_texture_size(b, tex));
   coord = nir_fdiv(b, nir_fadd_imm(b, nir_i2f32(b, tex->src[ci].src.ssa), 0.5), size);
   nir_src_rewrite(&tex->src[ci].src, coord);
   if (li >= 0)
      nir_src_rewrite(&tex->src[li].src, nir_i2f32(b, tex->src[li].src.ssa));
   else
      nir_tex_instr_add_src(tex, nir_tex_src_lod, nir_imm_float(b, 0.0f));
   tex->op = nir_texop_txl;
   return true;
}

bool
g9p_compile_fragment(nir_shader *nir, unsigned frag_coord_slot,
                     const nir_lower_blend_options *blend, struct g9p_shader *out,
                     void *mem_ctx)
{
   memset(out, 0, sizeof(*out));
   lower_front(nir);
   NIR_PASS(_, nir, discards_to_if);
   NIR_PASS(_, nir, discards_to_flag);
   NIR_PASS(_, nir, nir_shader_tex_pass, lower_txf, nir_metadata_control_flow, NULL);
   lower_io(nir);
   out->native_blend = blend && is_src_over(blend) && !getenv("G9P_NO_NATIVE_BLEND");
   if (blend && !out->native_blend) {
      NIR_PASS(_, nir, nir_lower_blend, blend);
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_blend_constant,
               nir_metadata_control_flow, NULL);
   }
   out->uses_fbfetch = nir->info.fs.uses_fbfetch_output || out->native_blend;
   if (out->uses_fbfetch)
      NIR_PASS(_, nir, discards_to_select, out->native_blend);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_fb_fetch_instr, nir_metadata_control_flow,
            NULL);
   nir_shader *copy = nir_shader_clone(mem_ctx, nir);

   return compile_back(nir, out, MESA_SHADER_FRAGMENT, NULL, frag_coord_slot, false, false, mem_ctx) ||
          compile_back(copy, out, MESA_SHADER_FRAGMENT, NULL, frag_coord_slot, true, false, mem_ctx);
}
