/* SPDX-License-Identifier: MIT */
/*
 * NIR -> G9P compiler for compute shaders (see g9p_compile.h).
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/spirv/nir_spirv.h"
#include "util/ralloc.h"
#include "util/u_math.h"

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
   .lower_int64_options = ~0U,
   .lower_fdiv = true,
   .lower_fmod = true,
   .lower_fpow = true,
   .lower_fsqrt = true,
   .lower_ffract = true,
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
   struct g9p_compute_shader *out;

   /* Per SSA scalar (def index * 4 + component) */
   int8_t *reg;
   unsigned *last_use;
   /* Per SSA def: the instruction defining it */
   unsigned *def_ip;

   uint64_t busy;   /* registers in use */
   uint64_t pinned; /* NIR registers: never released */
   uint64_t temps;  /* released after the current instruction */
   unsigned max_reg;
   unsigned st;     /* mask stack register */
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
   fprintf(stderr, "g9p: out of registers at instruction %u\n", ctx->ip);
   if (!getenv("G9P_DEBUG"))
      return;
   for (unsigned k = 0; k < ctx->num_keys; k++) {
      if (ctx->reg[k] >= 0 && (ctx->busy & BITFIELD64_BIT(ctx->reg[k])) &&
          !(ctx->pinned & BITFIELD64_BIT(ctx->reg[k])))
         fprintf(stderr, "  r%d: value %u.%u, last use %u\n", ctx->reg[k], k / 4, k % 4,
                 ctx->last_use[k]);
   }
}

static int
alloc_reg(struct g9p_ctx *ctx)
{
   for (unsigned r = 0; r < G9P_NUM_REGS; r++) {
      if (!(ctx->busy & BITFIELD64_BIT(r))) {
         ctx->busy |= BITFIELD64_BIT(r);
         ctx->max_reg = MAX2(ctx->max_reg, r);
         return r;
      }
   }
   dump_busy(ctx);
   return -1;
}

static int
alloc_pair(struct g9p_ctx *ctx)
{
   for (unsigned r = 0; r + 1 < G9P_NUM_REGS; r++) {
      if (!(ctx->busy & BITFIELD64_RANGE(r, 2))) {
         ctx->busy |= BITFIELD64_RANGE(r, 2);
         ctx->max_reg = MAX2(ctx->max_reg, r + 1);
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
   if (r < 0)
      fprintf(stderr, "g9p: value %u has no register\n", s.def->index);
   return r;
}

static void
release_scalar(struct g9p_ctx *ctx, nir_def *def, unsigned comp)
{
   unsigned k = key(def, comp);

   if (ctx->last_use[k] <= ctx->ip && ctx->reg[k] >= 0 &&
       !(ctx->pinned & BITFIELD64_BIT(ctx->reg[k])))
      ctx->busy &= ~BITFIELD64_BIT(ctx->reg[k]);
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
emit_alu(struct g9p_ctx *ctx, nir_alu_instr *alu)
{
   const unsigned num_inputs = nir_op_infos[alu->op].num_inputs;
   nir_scalar s[3] = { 0 };
   int r[3] = { -1, -1, -1 };
   uint32_t imm;
   int dst;

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
   case nir_op_slt: case nir_op_sge: case nir_op_seq: case nir_op_sne:
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
   case nir_op_ishl:
      g9p_shift(ctx->c, dst, r[0], r[1], false);
      return true;
   case nir_op_ushr:
      g9p_shift(ctx->c, dst, r[0], r[1], true);
      return true;
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
   case nir_op_u2f32:
      g9p_convert(ctx->c, dst, r[0], G9P_U2F);
      return true;
   case nir_op_i2f32:
      g9p_convert(ctx->c, dst, r[0], G9P_I2F);
      return true;
   case nir_op_f2u32:
      g9p_convert(ctx->c, dst, r[0], G9P_F2U);
      return true;
   case nir_op_f2i32:
      g9p_convert(ctx->c, dst, r[0], G9P_F2I);
      return true;

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
   case nir_op_b2f32: {
      /* booleans are 0 / ~0 */
      int one = const_reg(ctx, fui(1.0f));

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
      /* resolved where they are used */
      return true;

   case nir_intrinsic_store_ssbo:
      return emit_mem(ctx, intr, buffer_slot(ctx, intr->src[1]),
                      nir_get_scalar(intr->src[2].ssa, 0), 0, &intr->src[0]);
   case nir_intrinsic_load_ssbo:
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
      r = alloc_reg(ctx);
      if (r < 0)
         return false;
      ctx->pinned |= BITFIELD64_BIT(r);
      ctx->reg[key(&intr->def, 0)] = r;
      return true;
   }
   case nir_intrinsic_load_reg:
      /* nir_trivialize_registers: read right before its uses */
      ctx->reg[key(&intr->def, 0)] = ctx->reg[key(intr->src[0].ssa, 0)];
      return true;
   case nir_intrinsic_store_reg: {
      int dst = ctx->reg[key(intr->src[1].ssa, 0)];
      nir_scalar v = nir_get_scalar(intr->src[0].ssa, 0);
      uint32_t imm;
      int src;

      if (is_const(v, &imm)) {
         g9p_mov_imm(ctx->c, dst, imm);
      } else {
         src = src_reg(ctx, v);
         if (src < 0)
            return false;
         if (src != dst)
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
            if (!emit_instr(ctx, instr)) {
               fprintf(stderr, "g9p: cannot compile: ");
               nir_print_instr(instr, stderr);
               fprintf(stderr, "\n");
               return false;
            }
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

   memset(ctx->reg, -1, impl->ssa_alloc * 4);
   ctx->c->n = 0;
   ctx->busy = ctx->pinned = ctx->temps = 0;
   ctx->max_reg = 0;
   ctx->st = st;
   ctx->out->num_buffers = 0;
   ctx->out->uses_workgroup_id = false;
   ctx->num_exit_jumps = 0;

   /* r0 holds the local index from the start when it is used */
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

   /* Frame: prologue, mask stack, body, epilogue, end */
   body_n = ctx->c->n;
   body = malloc(body_n * sizeof(uint16_t) + 1);
   if (!body)
      return false;
   memcpy(body, ctx->c->w, body_n * sizeof(uint16_t));
   ctx->c->n = 0;
   g9p_prologue(ctx->c, 4 * ctx->out->num_buffers);
   g9p_push_init(ctx->c, st);
   g9p_emit(ctx->c, body, body_n);
   g9p_pop(ctx->c, st);
   g9p_end(ctx->c);
   free(body);
   return true;
}

int
g9p_buffer_slot(struct g9p_compute_shader *out, unsigned set, unsigned binding)
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

bool
g9p_compile_compute(nir_shader *nir, struct g9p_compute_shader *out, void *mem_ctx)
{
   const nir_lower_compute_system_values_options csv = {
      .lower_cs_local_id_to_index = true,
      .global_id_is_32bit = true,
   };
   struct g9p_ctx ctx = { 0 };
   nir_function_impl *impl;
   unsigned st;

   memset(out, 0, sizeof(*out));

   /* One function, globals as locals */
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
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            nir_address_format_32bit_index_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);
   optimize(nir);
   NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_mem_ssbo | nir_var_mem_push_const,
            NULL, NULL);
   NIR_PASS(_, nir, nir_opt_algebraic_late);
   optimize(nir);
   NIR_PASS(_, nir, nir_lower_bool_to_int32);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_dce);
   /* values computed early for late uses: compute them right before */
   NIR_PASS(_, nir, nir_opt_move, nir_move_alu | nir_move_const_undef |
            nir_move_comparisons | nir_move_copies);
   NIR_PASS(_, nir, nir_convert_from_ssa, true, false);
   NIR_PASS(_, nir, nir_trivialize_registers);

   impl = nir_shader_get_entrypoint(nir);
   if (!impl)
      return false;
   nir_index_ssa_defs(impl);
   if (getenv("G9P_DEBUG"))
      nir_print_shader(nir, stderr);

   for (unsigned i = 0; i < 3; i++)
      out->local_size[i] = nir->info.workgroup_size[i];

   ctx.out = out;
   ctx.c = rzalloc(mem_ctx, struct g9p_code);
   ctx.reg = ralloc_array(mem_ctx, int8_t, impl->ssa_alloc * 4);
   ctx.num_keys = impl->ssa_alloc * 4;
   ctx.last_use = rzalloc_array(mem_ctx, unsigned, impl->ssa_alloc * 4);
   ctx.def_ip = rzalloc_array(mem_ctx, unsigned, impl->ssa_alloc);
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

   /* Twice: to find the register count, then with the mask stack after it */
   if (!generate(&ctx, impl, G9P_NUM_REGS))
      return false;
   st = ctx.max_reg + 1;
   if (!generate(&ctx, impl, st))
      return false;

   out->num_regs = st + 1;
   out->code = ctx.c->w;
   out->code_size = 2 * ctx.c->n;
   if (getenv("G9P_DEBUG")) {
      fprintf(stderr, "g9p: %u registers, %u bytes of code\n", out->num_regs,
              out->code_size);
      for (unsigned i = 0; i < ctx.c->n; i++)
         fprintf(stderr, "%04x%c", ctx.c->w[i], (ctx.c->w[i] & 0x100) ? ' ' : '\n');
   }
   return true;
}
