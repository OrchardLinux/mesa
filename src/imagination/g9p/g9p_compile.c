/* SPDX-License-Identifier: MIT */
/*
 * NIR -> G9P compiler for compute shaders (see g9p_compile.h).
 *
 * Code layout follows Apple's compiler: prologue (mask lanes past the thread
 * count), mask stack in the register after the program's own, body,
 * epilogue, jump to the end word. Values live in r0-r7; r0 holds the
 * thread's index in its workgroup at entry.
 */

#include <stdio.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/spirv/nir_spirv.h"
#include "util/ralloc.h"

#include "g9p_compile.h"
#include "g9p_isa.h"

#define G9P_NUM_REGS 8

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
   .lower_find_lsb = true,
   .lower_ifind_msb = true,
   .lower_iadd_sat = true,
   .lower_mul_2x32_64 = true,
   .lower_device_index_to_zero = true,
   .max_unroll_iterations = 32,
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
   struct g9p_code *c;
   struct g9p_compute_shader *out;

   /* Per SSA scalar (def index * 4 + component) */
   int8_t *reg;
   unsigned *last_use;

   unsigned busy; /* register bitmask */
   unsigned temps; /* registers to release after the current instruction */
   unsigned max_reg;
   unsigned ip;
};

static unsigned
key(const nir_def *def, unsigned comp)
{
   return def->index * 4 + comp;
}

static int
alloc_reg(struct g9p_ctx *ctx)
{
   for (unsigned r = 0; r < G9P_NUM_REGS; r++) {
      if (!(ctx->busy & BITFIELD_BIT(r))) {
         ctx->busy |= BITFIELD_BIT(r);
         ctx->max_reg = MAX2(ctx->max_reg, r);
         return r;
      }
   }
   return -1;
}

static int
alloc_pair(struct g9p_ctx *ctx)
{
   for (unsigned r = 0; r + 1 < G9P_NUM_REGS; r++) {
      if (!(ctx->busy & BITFIELD_RANGE(r, 2))) {
         ctx->busy |= BITFIELD_RANGE(r, 2);
         ctx->max_reg = MAX2(ctx->max_reg, r + 1);
         return r;
      }
   }
   return -1;
}

static int
alloc_temp(struct g9p_ctx *ctx)
{
   int r = alloc_reg(ctx);

   if (r >= 0)
      ctx->temps |= BITFIELD_BIT(r);
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

/* The register holding @s, materializing a constant into a temporary */
static int
src_reg(struct g9p_ctx *ctx, nir_scalar s)
{
   uint32_t value;

   if (is_const(s, &value)) {
      int r = alloc_temp(ctx);

      if (r >= 0)
         g9p_mov_imm(ctx->c, r, value);
      return r;
   }
   return ctx->reg[key(s.def, s.comp)];
}

static bool
last_use(const struct g9p_ctx *ctx, nir_scalar s)
{
   uint32_t value;

   return !is_const(s, &value) && ctx->last_use[key(s.def, s.comp)] == ctx->ip;
}

/* Release the sources whose last use is this instruction, and the temporaries */
static bool
release_src(nir_src *src, void *data)
{
   struct g9p_ctx *ctx = data;

   for (unsigned i = 0; i < nir_src_num_components(*src); i++) {
      unsigned k = key(src->ssa, i);

      if (ctx->last_use[k] == ctx->ip && ctx->reg[k] >= 0)
         ctx->busy &= ~BITFIELD_BIT(ctx->reg[k]);
   }
   return true;
}

static void
release(struct g9p_ctx *ctx, nir_instr *instr)
{
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);

      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         unsigned k = key(alu->src[i].src.ssa, alu->src[i].swizzle[0]);

         if (ctx->last_use[k] == ctx->ip && ctx->reg[k] >= 0)
            ctx->busy &= ~BITFIELD_BIT(ctx->reg[k]);
      }
   } else {
      nir_foreach_src(instr, release_src, ctx);
   }
   ctx->busy &= ~ctx->temps;
   ctx->temps = 0;
}

static bool
mark_use(nir_src *src, void *data)
{
   struct g9p_ctx *ctx = data;

   for (unsigned i = 0; i < nir_src_num_components(*src); i++)
      ctx->last_use[key(src->ssa, i)] = ctx->ip;
   return true;
}

static bool
def_used(const struct g9p_ctx *ctx, const nir_def *def, unsigned comp)
{
   return ctx->last_use[key(def, comp)] != 0;
}

/* The common-store slot of the storage buffer @index refers to */
static int
buffer_slot(struct g9p_ctx *ctx, nir_src index)
{
   nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(index.ssa, 0));
   nir_intrinsic_instr *intr;
   unsigned set, binding, k;

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

   set = nir_intrinsic_desc_set(intr);
   binding = nir_intrinsic_binding(intr);
   for (k = 0; k < ctx->out->num_buffers; k++) {
      if (ctx->out->buffers[k].set == set && ctx->out->buffers[k].binding == binding)
         return k;
   }
   if (k == G9P_MAX_BUFFERS)
      return -1;
   ctx->out->buffers[k].set = set;
   ctx->out->buffers[k].binding = binding;
   ctx->out->num_buffers++;
   return k;
}

static bool
emit_alu(struct g9p_ctx *ctx, nir_alu_instr *alu)
{
   nir_scalar s0 = nir_get_scalar(alu->src[0].src.ssa, alu->src[0].swizzle[0]);
   nir_scalar s1 = { 0 };
   unsigned op;
   uint32_t imm;
   int a, b, dst;

   if (alu->def.num_components != 1 || alu->def.bit_size != 32) {
      fprintf(stderr, "g9p: %s: only 32-bit scalar ALU\n", nir_op_infos[alu->op].name);
      return false;
   }
   if (nir_op_infos[alu->op].num_inputs > 1)
      s1 = nir_get_scalar(alu->src[1].src.ssa, alu->src[1].swizzle[0]);

   switch (alu->op) {
   case nir_op_mov:
      if (nir_scalar_is_intrinsic(s0) && s0.comp == 0) {
         nir_intrinsic_op iop = nir_def_as_intrinsic(s0.def)->intrinsic;

         /* a binding index: resolved where it is used (buffer_slot()) */
         if (iop == nir_intrinsic_load_vulkan_descriptor ||
             iop == nir_intrinsic_vulkan_resource_index)
            return true;
      }
      if (is_const(s0, &imm)) {
         release(ctx, &alu->instr);
         dst = alloc_reg(ctx);
         if (dst < 0)
            return false;
         g9p_mov_imm(ctx->c, dst, imm);
      } else {
         a = src_reg(ctx, s0);
         release(ctx, &alu->instr);
         dst = alloc_reg(ctx);
         if (a < 0 || dst < 0)
            return false;
         g9p_add_imm(ctx->c, dst, a, 0);
      }
      ctx->reg[key(&alu->def, 0)] = dst;
      return true;

   case nir_op_iadd:
      if (is_const(s0, &imm) && imm < 0x40) {
         nir_scalar t = s0;
         s0 = s1;
         s1 = t;
      }
      if (is_const(s1, &imm) && imm < 0x40) {
         a = src_reg(ctx, s0);
         release(ctx, &alu->instr);
         dst = alloc_reg(ctx);
         if (a < 0 || dst < 0)
            return false;
         g9p_add_imm(ctx->c, dst, a, imm);
         ctx->reg[key(&alu->def, 0)] = dst;
         return true;
      }
      op = G9P_IADD;
      break;
   case nir_op_isub:
      op = G9P_ISUB;
      break;
   case nir_op_imul:
      op = G9P_IMUL;
      break;
   case nir_op_iand:
      op = G9P_AND;
      break;
   case nir_op_ior:
      op = G9P_OR;
      break;
   case nir_op_ixor:
      op = G9P_XOR;
      break;
   case nir_op_ishl:
      /* by a constant: a multiply */
      if (!is_const(s1, &imm)) {
         fprintf(stderr, "g9p: ishl by a register\n");
         return false;
      }
      a = src_reg(ctx, s0);
      b = alloc_temp(ctx);
      if (a < 0 || b < 0)
         return false;
      g9p_mov_imm(ctx->c, b, 1u << (imm & 31));
      {
         bool kill = last_use(ctx, s0);

         release(ctx, &alu->instr);
         dst = alloc_reg(ctx);
         if (dst < 0)
            return false;
         g9p_alu2(ctx->c, G9P_IMUL, dst, a, b, kill);
      }
      ctx->reg[key(&alu->def, 0)] = dst;
      return true;
   default:
      fprintf(stderr, "g9p: ALU op %s not supported yet\n", nir_op_infos[alu->op].name);
      return false;
   }

   a = src_reg(ctx, s0);
   b = src_reg(ctx, s1);
   if (a < 0 || b < 0)
      return false;
   {
      bool kill = (is_const(s0, &imm) || last_use(ctx, s0)) &&
                  (is_const(s1, &imm) || last_use(ctx, s1));

      release(ctx, &alu->instr);
      dst = alloc_reg(ctx);
      if (dst < 0)
         return false;
      g9p_alu2(ctx->c, op, dst, a, b, kill);
   }
   ctx->reg[key(&alu->def, 0)] = dst;
   return true;
}

static bool
emit_ssbo(struct g9p_ctx *ctx, nir_intrinsic_instr *intr, bool store)
{
   nir_src index = intr->src[store ? 1 : 0];
   nir_scalar offset = nir_get_scalar(intr->src[store ? 2 : 1].ssa, 0);
   unsigned bits = store ? nir_src_bit_size(intr->src[0]) : intr->def.bit_size;
   unsigned comps = store ? nir_src_num_components(intr->src[0]) : intr->def.num_components;
   int slot = buffer_slot(ctx, index);
   int off, pair, data;

   if (slot < 0 || bits != 32 || comps != 1) {
      fprintf(stderr, "g9p: %s: only one 32-bit value through a plain binding\n",
              nir_intrinsic_infos[intr->intrinsic].name);
      return false;
   }

   off = src_reg(ctx, offset);
   pair = alloc_pair(ctx);
   if (off < 0 || pair < 0)
      return false;
   ctx->temps |= BITFIELD_RANGE(pair, 2);
   /* byte offset: shift 0 */
   g9p_addr(ctx->c, pair, 2 * slot, off, 0, false);

   if (store) {
      data = src_reg(ctx, nir_get_scalar(intr->src[0].ssa, 0));
      if (data < 0)
         return false;
      g9p_mem_at(ctx->c, true, G9P_SIZE_U32, data, pair);
      release(ctx, &intr->instr);
   } else {
      /* the address pair stays busy until the load has read it */
      release(ctx, &intr->instr);
      ctx->busy |= BITFIELD_RANGE(pair, 2);
      data = alloc_reg(ctx);
      if (data < 0)
         return false;
      g9p_mem_at(ctx->c, false, G9P_SIZE_U32, data, pair);
      g9p_wait(ctx->c);
      ctx->busy &= ~BITFIELD_RANGE(pair, 2);
      ctx->busy |= BITFIELD_BIT(data);
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

         if (!def_used(ctx, &intr->def, c))
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
      return emit_ssbo(ctx, intr, true);
   case nir_intrinsic_load_ssbo:
      return emit_ssbo(ctx, intr, false);

   default:
      fprintf(stderr, "g9p: intrinsic %s not supported yet\n",
              nir_intrinsic_infos[intr->intrinsic].name);
      return false;
   }
}

static void
optimize(nir_shader *nir)
{
   bool progress;

   do {
      progress = false;
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      NIR_PASS(progress, nir, nir_lower_alu_to_scalar, NULL, NULL);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
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
   unsigned count_half, st;
   uint16_t body[G9P_MAX_CODE_WORDS];
   unsigned body_n;

   memset(out, 0, sizeof(*out));

   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            nir_address_format_32bit_index_offset);
   optimize(nir);
   NIR_PASS(_, nir, nir_opt_algebraic_late);
   optimize(nir);

   impl = nir_shader_get_entrypoint(nir);
   if (!impl || exec_list_length(&impl->body) != 1) {
      fprintf(stderr, "g9p: control flow not supported yet\n");
      return false;
   }
   nir_index_ssa_defs(impl);

   for (unsigned i = 0; i < 3; i++)
      out->local_size[i] = nir->info.workgroup_size[i];

   ctx.out = out;
   ctx.c = rzalloc(mem_ctx, struct g9p_code);
   ctx.reg = ralloc_array(mem_ctx, int8_t, impl->ssa_alloc * 4);
   ctx.last_use = rzalloc_array(mem_ctx, unsigned, impl->ssa_alloc * 4);
   memset(ctx.reg, -1, impl->ssa_alloc * 4);

   /* Liveness: the last instruction using each scalar (instructions from 1) */
   ctx.ip = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         ctx.ip++;
         if (instr->type == nir_instr_type_alu) {
            /* ALU sources read only their swizzled components */
            nir_alu_instr *alu = nir_instr_as_alu(instr);

            for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
               for (unsigned c = 0; c < alu->def.num_components; c++)
                  ctx.last_use[key(alu->src[i].src.ssa, alu->src[i].swizzle[c])] = ctx.ip;
            }
         } else {
            nir_foreach_src(instr, mark_use, &ctx);
         }
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_load_local_invocation_index)
            ctx.busy |= BITFIELD_BIT(0);
      }
   }

   ctx.ip = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         bool ok = true;

         ctx.ip++;
         switch (instr->type) {
         case nir_instr_type_load_const:
         case nir_instr_type_undef:
            break;
         case nir_instr_type_alu:
            ok = emit_alu(&ctx, nir_instr_as_alu(instr));
            break;
         case nir_instr_type_intrinsic:
            ok = emit_intrinsic(&ctx, nir_instr_as_intrinsic(instr));
            break;
         case nir_instr_type_jump:
            ok = nir_instr_as_jump(instr)->type == nir_jump_return;
            break;
         default:
            fprintf(stderr, "g9p: instruction type %u not supported yet\n", instr->type);
            ok = false;
         }
         if (!ok) {
            fprintf(stderr, "g9p: cannot compile: ");
            nir_print_instr(instr, stderr);
            fprintf(stderr, "\n");
            return false;
         }
      }
   }

   /* Frame: prologue, mask stack after our registers, body, epilogue, end */
   body_n = ctx.c->n;
   memcpy(body, ctx.c->w, body_n * sizeof(uint16_t));
   ctx.c->n = 0;
   count_half = 4 * out->num_buffers;
   st = ctx.max_reg + 1;
   g9p_prologue(ctx.c, count_half);
   g9p_push_init(ctx.c, st);
   g9p_emit(ctx.c, body, body_n);
   g9p_pop(ctx.c, st);
   g9p_end(ctx.c);

   out->code = ctx.c->w;
   out->code_size = 2 * ctx.c->n;
   return true;
}
