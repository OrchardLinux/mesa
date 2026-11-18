/* SPDX-License-Identifier: MIT */
/*
 * Apple's A9 GPU (G5P, has_apple_g5p): SPIR-V to NIR for the G5P backend
 * (src/imagination/g5p), as the A9 ICD prepares its graphics stages
 * (g5p_nir.c of the route-b tree: g5p_nir(), graphics_nir(), vs_sysvals()),
 * then pco_g5p_compile_general_link(). The backend's recognised forms depend
 * on this pass sequence, so it is the ICD's, not vk_pipeline's. There is no
 * captured-shape matcher and no out-of-process fallback here: the general
 * backend only.
 */

#include <stdio.h>
#include <string.h>

#include "compiler/glsl_types.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_xfb_info.h"
#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv.h"
#include "compiler/spirv/spirv_info.h"
#include "util/u_call_once.h"
#include "util/u_math.h"
#include "vk_nir.h"
#include "vk_shader_module.h"
#include "vk_util.h"

#include "g5p_mrt_out.h"
#include "pco/pco.h"
#include "pco_g5p_general.h"
#include "pvr_g5p.h"
#include "pvr_g5p_shader_dump.h"
#include "pvr_g5p_vertex.h"

/* nir->options must outlive the shader (the general backend runs NIR passes
 * after the preparation returns), so the options are static. */
static nir_shader_compiler_options g5p_options;
static nir_shader_compiler_options g5p_vertex_options;

static void
g5p_init_options(void)
{
   g5p_options = *pco_nir_options();
   /* Apple's A9 compiler fully unrolls up to 96 trips (lane-b-alu-codegen-spec.md §2). */
   g5p_options.max_unroll_iterations = 96;
   /* G5P has a measured native FMA; do not inherit PCO's decomposition. */
   g5p_options.float_mul_add32 = nir_float_muladd_support_has_ffma;
   /* Selects are compare/write + inverse MOV pairs (no fused fcsel_*), and
    * fsqrt/fmin/fmax have measured complete lowerings. */
   g5p_options.has_fused_comp_and_csel = false;
   g5p_options.lower_fsqrt = false;
   g5p_options.lower_fminmax_signed_zero = false;
   /* Float results of compares become b2f32 of the compare; integer abs/sign
    * become imax/imin; the multiply-high of nir_lower_idiv becomes 16-bit
    * partial products; extract_u8/u16 stay shifts and masks. Every op they
    * produce has a measured encoding. */
   g5p_options.lower_scmp = true;
   g5p_options.lower_iabs = true;
   g5p_options.lower_isign = true;
   g5p_options.lower_mul_high = true;
   g5p_options.lower_extract_byte = true;
   g5p_options.lower_extract_word = true;
   /* GLSL's zero/full-width bitfields lower to the existing measured integer
    * shifts, masks and selects. Keep PCO's BFE/BFI capabilities disabled. */
   g5p_options.lower_bitfield_extract = true;
   g5p_options.lower_bitfield_insert = true;
   /* GLSL's norm packings (packUnorm2x16() ...) lower to the measured float clamps, rounds, conversions and
    * shifts (G9P's 26b4dcbc83d ("pvr/g9p: packing built-ins"); the half packings: pco_g5p_general.c lower_half_packing()) */
   g5p_options.lower_pack_snorm_2x16 = true;
   g5p_options.lower_unpack_snorm_2x16 = true;
   g5p_options.lower_pack_unorm_2x16 = true;
   g5p_options.lower_unpack_unorm_2x16 = true;
   g5p_options.lower_pack_snorm_4x8 = true;
   g5p_options.lower_unpack_snorm_4x8 = true;
   g5p_options.lower_pack_unorm_4x8 = true;
   g5p_options.lower_unpack_unorm_4x8 = true;
   /* saturating integer add/subtract (naga's bounds clamps: usub_sat(arrayLength, 1)): compare and select */
   g5p_options.lower_usub_sat = true;
   g5p_options.lower_uadd_sat = true;
   g5p_options.lower_iadd_sat = true;
   g5p_vertex_options = g5p_options;
   g5p_vertex_options.max_unroll_iterations = 16;
   /* The isolated Apple iv_reg_pop/reverse witnesses qualify native VS
    * register forms too. Do not lower them to SWAR multiplies. */
}

/*
 * Zink lowers a GL uniform block to an array of uints, so a vec4 element of
 * a uniform array indexed by a register arrives as four 32-bit loads at
 * (i << 4) + c, c + 4, c + 8 and c + 12. Apple's compiler reads such an
 * element with one 16-byte load (v_cexpr: vec4 at shift 4); the four are
 * merged into that form. Only complete, 16-byte aligned quads in one block
 * with a shift of 4 are merged, so every other load keeps its shape.
 */

/* A term of a UBO offset the backend reads as a constant: an immediate, or
 * the descriptor's offset component, which is 0. */
static bool
ubo_offset_const(nir_scalar s, uint32_t *v)
{
   if (nir_scalar_is_const(s)) {
      *v = nir_scalar_as_uint(s);
      return true;
   }
   if (nir_scalar_is_intrinsic(s) && nir_scalar_intrinsic_op(s) == nir_intrinsic_load_vulkan_descriptor &&
       s.comp == 1) {
      *v = 0;
      return true;
   }
   return false;
}

static bool
dyn_ubo_split(nir_intrinsic_instr *in, nir_scalar *x, uint32_t *c)
{
   if (in->intrinsic != nir_intrinsic_load_ubo || in->def.num_components != 1 || in->def.bit_size != 32)
      return false;
   nir_scalar s = nir_get_scalar(in->src[1].ssa, 0);
   *c = 0;
   for (unsigned guard = 0; guard < 8 && nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd; guard++) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0), b = nir_scalar_chase_alu_src(s, 1);
      uint32_t v;
      if (ubo_offset_const(b, &v)) {
         *c += v;
         s = a;
      } else if (ubo_offset_const(a, &v)) {
         *c += v;
         s = b;
      } else {
         return false;
      }
   }
   if (!nir_scalar_is_alu(s))
      return false;
   nir_op op = nir_scalar_alu_op(s);
   if (op != nir_op_ishl && op != nir_op_imul)
      return false;
   nir_scalar k = nir_scalar_chase_alu_src(s, 1);
   if (!nir_scalar_is_const(k) || nir_scalar_as_uint(k) != (op == nir_op_ishl ? 4u : 16u))
      return false;
   *x = s;
   return true;
}

static bool
vec4_dyn_ubo(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_function_impl (impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block (block, impl) {
         bool again;
         do {
            again = false;
            nir_foreach_instr (instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *lo = nir_instr_as_intrinsic(instr), *q[4] = { lo };
               nir_scalar x, y;
               uint32_t c, d;
               if (!dyn_ubo_split(lo, &x, &c) || c % 16)
                  continue;
               unsigned found = 1;
               for (nir_instr *i = nir_instr_next(instr); i && found < 4; i = nir_instr_next(i)) {
                  if (i->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_instr *in = nir_instr_as_intrinsic(i);
                  if (!dyn_ubo_split(in, &y, &d) || in->src[0].ssa != lo->src[0].ssa || !nir_scalar_equal(x, y) ||
                      d < c + 4 || d > c + 12 || (d - c) % 4 || q[(d - c) / 4])
                     continue;
                  q[(d - c) / 4] = in;
                  found++;
               }
               if (found < 4)
                  continue;
               b.cursor = nir_before_instr(instr);
               /* the first component's offset, (i << 4) + c: vec4 at shift 4 */
               nir_def *v = nir_load_ubo(&b, 4, 32, lo->src[0].ssa, lo->src[1].ssa,
                                         .access = nir_intrinsic_access(lo), .align_mul = 16, .align_offset = 0,
                                         .range_base = 0, .range = ~0u);
               for (unsigned j = 0; j < 4; j++) {
                  nir_def_rewrite_uses(&q[j]->def, nir_channel(&b, v, j));
                  if (j)
                     nir_instr_remove(&q[j]->instr);
               }
               nir_instr_remove(instr);
               progress = again = true;
               break; /* the block changed under the iterator */
            }
         } while (again);
      }
      nir_progress(progress, impl, nir_metadata_control_flow);
   }
   return progress;
}

/*
 * Fragment code (the backend refuses any fragment index this does not
 * bound): each register-indexed UBO index is
 * clamped to the block, umin(i, N) with N the last element whose bytes lie
 * inside the UBO variable (robust-access semantics). A negative index taken
 * by a helper lane would otherwise read 4 GiB past the pointer.
 */
static bool
ubo_block_bytes(nir_shader *nir, nir_def *d, unsigned *bytes)
{
   for (unsigned guard = 0; guard < 8; guard++) {
      nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(d, 0));
      if (!nir_scalar_is_intrinsic(s))
         return false;
      nir_intrinsic_instr *in = nir_def_as_intrinsic(s.def);
      if (in->intrinsic == nir_intrinsic_load_vulkan_descriptor) {
         d = in->src[0].ssa;
         continue;
      }
      if (in->intrinsic != nir_intrinsic_vulkan_resource_index || !nir_src_is_const(in->src[0]) ||
          nir_src_as_uint(in->src[0]))
         return false;
      nir_foreach_variable_with_modes (var, nir, nir_var_mem_ubo) {
         if (var->data.descriptor_set != nir_intrinsic_desc_set(in) ||
             var->data.binding != nir_intrinsic_binding(in))
            continue;
         /* Zink declares each block as an array of one; the index is 0 (checked above) */
         *bytes = glsl_get_explicit_size(glsl_without_array(var->type), false);
         return *bytes != 0;
      }
      return false;
   }
   return false;
}

/* A constant term of a UBO offset as the backend evaluates it
 * (pco_g5p_general.c eval_offset): ubo_offset_const() or a sum of them. */
static bool
ubo_offset_eval(nir_scalar s, uint32_t *v)
{
   s = nir_scalar_chase_movs(s);
   if (ubo_offset_const(s, v))
      return true;
   uint32_t x, y;
   if (nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd &&
       ubo_offset_eval(nir_scalar_chase_alu_src(s, 0), &x) && ubo_offset_eval(nir_scalar_chase_alu_src(s, 1), &y)) {
      *v = x + y;
      return true;
   }
   return false;
}

/*
 * A register-indexed UBO load the backend's direct form does not take:
 * offset = (x << s) + C with s from ishl or a multiply by a power of two
 * (pco_g5p_general.c dyn_ubo_offset), read as a scalar at s != 1 or as a
 * vec4 at s == 4, s <= 4 and C a multiple of the load. Anything else with a
 * non-constant offset (a sum of two dynamic terms, a non-power-of-two
 * stride, a larger or smaller shift) is general and goes through
 * general_dyn_ubo(). A shift by a non-constant amount is neither.
 */
static bool
dyn_ubo_general(nir_intrinsic_instr *in)
{
   if (in->intrinsic != nir_intrinsic_load_ubo || in->def.bit_size != 32)
      return false;
   uint32_t c = 0, v;
   nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(in->src[1].ssa, 0));
   if (ubo_offset_eval(s, &v))
      return false; /* a constant offset: preloaded */
   for (unsigned guard = 0; guard < 8 && nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd; guard++) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0), b = nir_scalar_chase_alu_src(s, 1);
      if (ubo_offset_eval(a, &v))
         s = nir_scalar_chase_movs(b);
      else if (ubo_offset_eval(b, &v))
         s = nir_scalar_chase_movs(a);
      else
         return true; /* two dynamic terms */
      c += v;
   }
   unsigned shift = 0;
   if (nir_scalar_is_alu(s) && (nir_scalar_alu_op(s) == nir_op_ishl || nir_scalar_alu_op(s) == nir_op_imul)) {
      if (!ubo_offset_eval(nir_scalar_chase_alu_src(s, 1), &v))
         return false;
      if (nir_scalar_alu_op(s) == nir_op_imul) {
         if (!util_is_power_of_two_nonzero(v))
            return true; /* a non-power-of-two stride */
         v = util_logbase2(v);
      }
      shift = v;
   }
   /* the backend sees the load after its trailing unread components are trimmed */
   const unsigned n = MAX2(util_last_bit(nir_def_components_read(&in->def)), 1);
   const unsigned loaded = (n == 2 || n == 3) && shift == 4 && !(c % 16) ? 4 : n;
   return !((loaded == 4 && shift == 4) || (n == 1 && shift != 1)) || shift > 4 || c % (4 * loaded);
}

/* s / 2^k for an offset term built from constants, iadd, a multiply by a
 * constant and a shift by a constant, when each part divides exactly (the
 * element index Apple's compiler computes); NULL otherwise. */
static nir_def *
dyn_ubo_div(nir_builder *b, nir_scalar s, unsigned k, unsigned depth)
{
   const uint32_t mask = (1u << k) - 1;
   uint32_t v;
   s = nir_scalar_chase_movs(s);
   if (ubo_offset_eval(s, &v))
      return v & mask ? NULL : nir_imm_int(b, (int32_t)v >> k);
   if (!nir_scalar_is_alu(s) || depth > 8)
      return NULL;
   nir_scalar x = nir_scalar_chase_alu_src(s, 0), y = nir_scalar_chase_alu_src(s, 1);
   switch (nir_scalar_alu_op(s)) {
   case nir_op_iadd: {
      nir_def *p = dyn_ubo_div(b, x, k, depth + 1);
      nir_def *q = p ? dyn_ubo_div(b, y, k, depth + 1) : NULL;
      if (!q)
         return NULL;
      if (nir_scalar_is_const(nir_get_scalar(q, 0)))
         return nir_iadd_imm(b, p, nir_scalar_as_uint(nir_get_scalar(q, 0)));
      if (nir_scalar_is_const(nir_get_scalar(p, 0)))
         return nir_iadd_imm(b, q, nir_scalar_as_uint(nir_get_scalar(p, 0)));
      return nir_iadd(b, p, q);
   }
   case nir_op_imul: {
      if (!nir_scalar_is_const(y)) {
         nir_scalar t = x;
         x = y;
         y = t;
      }
      if (!nir_scalar_is_const(y))
         return NULL;
      const int32_t m = (int32_t)nir_scalar_as_uint(y);
      if (!(m & mask))
         return nir_imul_imm(b, nir_channel(b, x.def, x.comp), (uint32_t)(m >> k));
      nir_def *p = dyn_ubo_div(b, x, k, depth + 1);
      return p ? nir_imul_imm(b, p, (uint32_t)m) : NULL;
   }
   case nir_op_ishl: {
      if (!nir_scalar_is_const(y))
         return NULL;
      const unsigned n = nir_scalar_as_uint(y) & 31;
      if (n >= k)
         return nir_ishl_imm(b, nir_channel(b, x.def, x.comp), n - k);
      return dyn_ubo_div(b, x, k - n, depth + 1); /* (x << n) / 2^k = x / 2^(k - n) */
   }
   default:
      return NULL;
   }
}

/* dyn_ubo_div() would succeed (nothing is built) */
static bool
dyn_ubo_divides(nir_scalar s, unsigned k, unsigned depth)
{
   const uint32_t mask = (1u << k) - 1;
   uint32_t v;
   s = nir_scalar_chase_movs(s);
   if (ubo_offset_eval(s, &v))
      return !(v & mask);
   if (!nir_scalar_is_alu(s) || depth > 8)
      return false;
   nir_scalar x = nir_scalar_chase_alu_src(s, 0), y = nir_scalar_chase_alu_src(s, 1);
   switch (nir_scalar_alu_op(s)) {
   case nir_op_iadd:
      return dyn_ubo_divides(x, k, depth + 1) && dyn_ubo_divides(y, k, depth + 1);
   case nir_op_imul:
      if (!nir_scalar_is_const(y)) {
         nir_scalar t = x;
         x = y;
         y = t;
      }
      return nir_scalar_is_const(y) &&
             (!(nir_scalar_as_uint(y) & mask) || dyn_ubo_divides(x, k, depth + 1));
   case nir_op_ishl: {
      if (!nir_scalar_is_const(y))
         return false;
      const unsigned n = nir_scalar_as_uint(y) & 31;
      return n >= k || dyn_ubo_divides(x, k - n, depth + 1);
   }
   default:
      return false;
   }
}

/* offset = d + c: the dynamic part and the sum of the constant terms */
static nir_scalar
dyn_ubo_peel(nir_intrinsic_instr *in, uint32_t *c)
{
   nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(in->src[1].ssa, 0));
   uint32_t v;
   *c = 0;
   for (unsigned guard = 0; guard < 8 && nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd; guard++) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0), b = nir_scalar_chase_alu_src(s, 1);
      if (ubo_offset_eval(a, &v))
         s = nir_scalar_chase_movs(b);
      else if (ubo_offset_eval(b, &v))
         s = nir_scalar_chase_movs(a);
      else
         break;
      *c += v;
   }
   return s;
}

/*
 * Zink reads a GL uniform block as an array of uints, so a vec4 at a general
 * offset arrives as four scalar loads at d + c, c + 4, c + 8, c + 12. When
 * d + c is a multiple of 16 by construction (k*48 + 32), the four are one
 * vec4 load, which general_dyn_ubo() reads with one 16-byte load at shift 4
 * (as Apple's compiler reads a float4 at p[k*3 + 1]). Only complete quads of general loads are merged, as
 * vec4_dyn_ubo() does for the direct form.
 */
static bool
vec4_general_dyn_ubo(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_function_impl (impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block (block, impl) {
         bool again;
         do {
            again = false;
            nir_foreach_instr (instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *lo = nir_instr_as_intrinsic(instr), *q[4] = { lo };
               uint32_t c, e;
               if (lo->def.num_components != 1 || !dyn_ubo_general(lo))
                  continue;
               nir_scalar d = dyn_ubo_peel(lo, &c);
               if (c % 16 || !dyn_ubo_divides(d, 4, 0))
                  continue;
               unsigned found = 1;
               for (nir_instr *i = nir_instr_next(instr); i && found < 4; i = nir_instr_next(i)) {
                  if (i->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_instr *in = nir_instr_as_intrinsic(i);
                  if (in->def.num_components != 1 || !dyn_ubo_general(in) || in->src[0].ssa != lo->src[0].ssa)
                     continue;
                  nir_scalar y = dyn_ubo_peel(in, &e);
                  if (!nir_scalar_equal(d, y) || e < c + 4 || e > c + 12 || (e - c) % 4 || q[(e - c) / 4])
                     continue;
                  q[(e - c) / 4] = in;
                  found++;
               }
               if (found < 4)
                  continue;
               /* the range that covers all four (the bound when the block size is unknown) */
               uint64_t end = 0;
               for (unsigned j = 0; j < 4; j++)
                  end = MAX2(end, nir_intrinsic_range(q[j]) == ~0u ? UINT64_MAX
                                  : (uint64_t)nir_intrinsic_range_base(q[j]) + nir_intrinsic_range(q[j]));
               b.cursor = nir_before_instr(instr);
               nir_def *v = nir_load_ubo(&b, 4, 32, lo->src[0].ssa, lo->src[1].ssa,
                                         .access = nir_intrinsic_access(lo), .align_mul = 16, .align_offset = 0,
                                         .range_base = 0, .range = end >= UINT32_MAX ? ~0u : (uint32_t)end);
               for (unsigned j = 0; j < 4; j++) {
                  nir_def_rewrite_uses(&q[j]->def, nir_channel(&b, v, j));
                  if (j)
                     nir_instr_remove(&q[j]->instr);
               }
               nir_instr_remove(instr);
               progress = again = true;
               break; /* the block changed under the iterator */
            }
         } while (again);
      }
      nir_progress(progress, impl, nir_metadata_control_flow);
   }
   return progress;
}

/*
 * The general register-indexed UBO load, as Apple's A9 compiler reads a
 * constant buffer at a computed offset (Metal, constant float *p:
 * p[k*3 + j + 2] is imul, iadd, iadd and addr.add_shift at shift 2; a
 * constant float4 *p read at p[k*3 + 1] is the same at shift 4, one 16-byte
 * load): the element index is computed with integer
 * ALU, then the pointer pair = buffer + (index << s) and one 4- or 16-byte
 * load. Here offset' = umin(offset >> s, last) << s, the backend's direct
 * form with a clamped index (robust access; the pair add does not
 * sign-extend), with offset >> s built term by term where it divides
 * exactly (k*12 + j*4 + 8 at s = 2 is k*3 + j + 2) and a ushr otherwise.
 * s is 4 for a vec4 whose alignment is 16, else 2 per 32-bit component; a
 * vec4 below 16-byte alignment becomes four scalar loads. A load whose
 * block size is unknown, or below 4-byte alignment, keeps its shape (and
 * stays refused by name).
 */
static void
dyn_ubo_rewrite(nir_builder *b, nir_intrinsic_instr *in, unsigned shift, unsigned size)
{
   const unsigned bytes = shift == 4 ? 16 : 4;
   b->cursor = nir_before_instr(&in->instr);
   nir_scalar off = nir_get_scalar(in->src[1].ssa, 0);
   nir_def *index = dyn_ubo_div(b, off, shift, 0);
   if (!index)
      index = nir_ushr_imm(b, nir_channel(b, off.def, off.comp), shift);
   index = nir_umin(b, index, nir_imm_int(b, (size - bytes) >> shift));
   nir_src_rewrite(&in->src[1], nir_ishl_imm(b, index, shift));
}

static bool
general_dyn_ubo(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   nir_shader *nir = data;
   if (!dyn_ubo_general(in) || nir_intrinsic_align(in) < 4)
      return false;
   const unsigned n = in->def.num_components;
   const unsigned shift = n == 4 && (nir_intrinsic_align(in) >= 16 ||
                                     dyn_ubo_divides(nir_get_scalar(in->src[1].ssa, 0), 4, 0)) ? 4 : 2;
   /* the block's size; else the bytes the load's deref chain can reach
    * (nir_lower_explicit_io's range, e.g. Zink's ubos[n] with n > 1) */
   unsigned size;
   if (!ubo_block_bytes(nir, in->src[0].ssa, &size)) {
      if (nir_intrinsic_range(in) == ~0u || nir_intrinsic_range_base(in) + (uint64_t)nir_intrinsic_range(in) > UINT32_MAX)
         return false;
      size = nir_intrinsic_range_base(in) + nir_intrinsic_range(in);
   }
   if (size < (shift == 4 ? 16 : 4))
      return false;
   if (shift == 4 || n == 1) {
      dyn_ubo_rewrite(b, in, shift, size);
      return true;
   }
   b->cursor = nir_before_instr(&in->instr);
   nir_def *comp[NIR_MAX_VEC_COMPONENTS];
   for (unsigned k = 0; k < n; k++) {
      comp[k] = nir_load_ubo(b, 1, 32, in->src[0].ssa, nir_iadd_imm(b, in->src[1].ssa, 4 * k),
                             .access = nir_intrinsic_access(in), .align_mul = 4, .align_offset = 0,
                             .range_base = 0, .range = ~0u);
      dyn_ubo_rewrite(b, nir_def_as_intrinsic(comp[k]), 2, size);
      b->cursor = nir_after_instr(nir_def_instr(comp[k]));
   }
   nir_def_replace(&in->def, nir_vec(b, comp, n));
   return true;
}

static bool
clamp_dyn_ubo(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_function_impl (impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block (block, impl) {
         nir_foreach_instr_safe (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if (in->intrinsic != nir_intrinsic_load_ubo || in->def.bit_size != 32 || dyn_ubo_general(in))
               continue;
            /* offset = shift(x) + c, the shape the backend folds */
            nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(in->src[1].ssa, 0));
            uint32_t c = 0, v;
            bool ok = true;
            for (unsigned guard = 0; guard < 8 && nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd;
                 guard++) {
               nir_scalar a0 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
               nir_scalar a1 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
               if (ubo_offset_const(a1, &v)) {
                  c += v;
                  s = a0;
               } else if (ubo_offset_const(a0, &v)) {
                  c += v;
                  s = a1;
               } else {
                  ok = false;
                  break;
               }
            }
            if (!ok || ubo_offset_const(s, &v))
               continue; /* a constant offset (the descriptor term reads 0) */
            unsigned shift = 0;
            nir_scalar x = s;
            if (nir_scalar_is_alu(s) && (nir_scalar_alu_op(s) == nir_op_ishl || nir_scalar_alu_op(s) == nir_op_imul)) {
               nir_scalar k = nir_scalar_chase_alu_src(s, 1);
               if (!nir_scalar_is_const(k))
                  continue;
               uint32_t kv = nir_scalar_as_uint(k);
               if (nir_scalar_alu_op(s) == nir_op_imul) {
                  if (!util_is_power_of_two_nonzero(kv))
                     continue;
                  kv = util_logbase2(kv);
               }
               if (kv > 31)
                  continue;
               shift = kv;
               x = nir_scalar_chase_alu_src(s, 0);
            }
            unsigned size, bytes = in->def.num_components * 4;
            /* A negative intermediate index can still yield a valid final
             * address: (2-i)*8+4 is folded to (-i)*8+20. Clamp the complete
             * 32-bit byte address, preserving its known low offset bits,
             * rather than clamping -i before adding the base. */
            uint32_t low = c & ((1u << shift) - 1);
            if (!ubo_block_bytes(nir, in->src[0].ssa, &size) || low > size || size - low < bytes)
               continue;
            uint32_t last = (size - low - bytes) >> shift;
            b.cursor = nir_before_instr(instr);
            nir_def *address = nir_iadd_imm(&b,
               nir_ishl_imm(&b, nir_channel(&b, x.def, x.comp), shift), c);
            nir_def *i = nir_umin(&b, nir_ushr_imm(&b, address, shift), nir_imm_int(&b, last));
            nir_def *off = nir_iadd_imm(&b, nir_ishl_imm(&b, i, shift), low);
            nir_src_rewrite(&in->src[1], off);
            progress = true;
         }
      }
      nir_progress(progress, impl, nir_metadata_control_flow);
   }
   return progress;
}

static nir_shader *g5p_nir_prepare(nir_shader *nir);

/* Storage-buffer accesses the backend encodes: a 32-bit scalar, or a vec4
 * load at a 16-byte-aligned offset (one 16-byte addressed load). Wider or
 * misaligned vectors become scalars at offset + 4k (WGSL's vec2/vec3 and
 * struct members), and stores become one scalar store per written
 * component. A constant offset keeps a vec4 that the fixed load already
 * encodes, so earlier programs are unchanged. */
static bool
split_ssbo_access(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   if (in->intrinsic == nir_intrinsic_load_ssbo) {
      const unsigned n = in->def.num_components;
      if (in->def.bit_size != 32 || n == 1)
         return false;
      if (n == 4 && (nir_intrinsic_align(in) >= 16 ||
                     (nir_src_is_const(in->src[1]) && !(nir_src_as_uint(in->src[1]) % 16))))
         return false;
      b->cursor = nir_before_instr(&in->instr);
      nir_def *comp[NIR_MAX_VEC_COMPONENTS];
      for (unsigned k = 0; k < n; k++)
         comp[k] = nir_load_ssbo(b, 1, 32, in->src[0].ssa, nir_iadd_imm(b, in->src[1].ssa, 4 * k),
                                 .access = nir_intrinsic_access(in), .align_mul = 4, .align_offset = 0);
      nir_def_replace(&in->def, nir_vec(b, comp, n));
      return true;
   }
   if (in->intrinsic == nir_intrinsic_store_ssbo) {
      const unsigned n = in->src[0].ssa->num_components;
      const unsigned mask = nir_intrinsic_write_mask(in);
      if (in->src[0].ssa->bit_size != 32 || (n == 1 && mask == 1))
         return false;
      b->cursor = nir_before_instr(&in->instr);
      u_foreach_bit(k, mask)
         nir_store_ssbo(b, nir_channel(b, in->src[0].ssa, k), in->src[1].ssa,
                        nir_iadd_imm(b, in->src[2].ssa, 4 * k), .write_mask = 1,
                        .access = nir_intrinsic_access(in), .align_mul = 4, .align_offset = 0);
      nir_instr_remove(&in->instr);
      return true;
   }
   return false;
}

/* After constant folding: a register-indexed vec2/vec3 UBO load becomes
 * scalars at offset + 4k (the backend encodes scalar and vec4 forms;
 * vec4_dyn_ubo() then merges complete quads). Constant-offset loads are
 * preloaded and stay as they are. */
static bool
split_dyn_ubo(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   const unsigned n = in->def.num_components;
   if (in->intrinsic != nir_intrinsic_load_ubo || in->def.bit_size != 32 || (n != 2 && n != 3) ||
       nir_src_is_const(in->src[1]))
      return false;
   b->cursor = nir_before_instr(&in->instr);
   nir_def *comp[NIR_MAX_VEC_COMPONENTS];
   for (unsigned k = 0; k < n; k++)
      comp[k] = nir_load_ubo(b, 1, 32, in->src[0].ssa, nir_iadd_imm(b, in->src[1].ssa, 4 * k),
                             .access = nir_intrinsic_access(in), .align_mul = 4, .align_offset = 0,
                             .range_base = 0, .range = ~0u);
   nir_def_replace(&in->def, nir_vec(b, comp, n));
   return true;
}

/* SPIR-V -> optimised NIR (the ICD's g5p_nir()) */
static nir_shader *
g5p_nir(const uint32_t *spv, size_t words, mesa_shader_stage stage, const char *entry)
{
   static util_once_flag once = UTIL_ONCE_FLAG_INIT;
   util_call_once(&once, g5p_init_options);
   struct spirv_capabilities caps = { .Shader = true, .Float16 = true, .Int16 = true, .Int8 = true,
                                      .Int64 = true, .Float64 = true, .TransformFeedback = true };
   struct spirv_to_nir_options opts = {
      .environment = NIR_SPIRV_VULKAN,
      .capabilities = &caps,
      .ubo_addr_format = nir_address_format_32bit_index_offset,
      .ssbo_addr_format = nir_address_format_32bit_index_offset,
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .push_const_addr_format = nir_address_format_32bit_offset,
      .shared_addr_format = nir_address_format_32bit_offset,
      .temp_addr_format = nir_address_format_32bit_offset,
      .skip_os_break_in_debug_build = true,
   };
   opts.workarounds.force_exact_glsl_fma = true;
   nir_shader *nir = spirv_to_nir(spv, words, NULL, stage, entry, &opts,
      stage == MESA_SHADER_VERTEX ? &g5p_vertex_options : &g5p_options);
   if (!nir)
      return NULL;
   return g5p_nir_prepare(nir);
}

/* The ICD's passes after spirv_to_nir(), for any stage's NIR */
static nir_shader *
g5p_nir_prepare(nir_shader *nir)
{
   const mesa_shader_stage stage = nir->info.stage;
   nir_lower_variable_initializers(nir, nir_var_function_temp);
   nir_lower_returns(nir);
   nir_inline_functions(nir);
   nir_remove_non_entrypoints(nir);
   /* gl_PerVertex: one variable per member, each at its builtin's slot, as
    * the Vulkan runtime does (vk_nir.c); without it a member keeps its index
    * as its location (glslang's gl_PointSize became slot 1, COL0) */
   nir_split_var_copies(nir);
   nir_split_per_member_structs(nir);
   /* gl_ClipDistance/gl_CullDistance sizes, before IO lowering, after the
    * dead outputs are gone (vk_nir.c: glslang declares an unwritten
    * gl_ClipDistance[1] and gl_CullDistance[1] in every gl_PerVertex) */
   if (stage == MESA_SHADER_VERTEX) {
      const nir_remove_dead_variables_options dead = { .can_remove_var = nir_vk_is_not_xfb_output };
      nir_remove_dead_variables(nir, nir_var_shader_out, &dead);
      nir_gather_clip_cull_distance_sizes_from_vars(nir);
   }
   nir_lower_vars_to_ssa(nir);
   nir_opt_deref(nir);
   nir_lower_system_values(nir);
   /* XFB decorations live on the variables: gather them before IO lowering,
    * which then attaches them to the store_output intrinsics. */
   if (nir->info.has_transform_feedback_varyings)
      nir_shader_gather_xfb_info(nir);
   nir_lower_io_passes(nir, false);
   nir_lower_explicit_io(nir, nir_var_mem_ssbo | nir_var_mem_ubo, nir_address_format_32bit_index_offset);
   nir_shader_intrinsics_pass(nir, split_ssbo_access, nir_metadata_control_flow, NULL);
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
   } while (progress);
   progress = false;
   if (nir_shader_intrinsics_pass(nir, split_dyn_ubo, nir_metadata_control_flow, NULL)) {
      NIR_PASS(_, nir, nir_opt_constant_folding);
      NIR_PASS(_, nir, nir_opt_algebraic);
      NIR_PASS(_, nir, nir_opt_cse);
   }
   NIR_PASS(progress, nir, vec4_dyn_ubo);
   if (stage == MESA_SHADER_FRAGMENT && pco_g5p_fs_dyn_ubo())
      NIR_PASS(progress, nir, clamp_dyn_ubo);
   /* after the clamp, which leaves these loads alone: their index is clamped here */
   NIR_PASS(_, nir, vec4_general_dyn_ubo);
   if (nir_shader_intrinsics_pass(nir, general_dyn_ubo, nir_metadata_control_flow, nir)) {
      NIR_PASS(_, nir, nir_opt_constant_folding);
      NIR_PASS(_, nir, nir_opt_cse);
      progress = true;
   }
   if (progress) {
      NIR_PASS(_, nir, nir_opt_copy_prop);
      NIR_PASS(_, nir, nir_opt_dce);
   }
   nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
   return nir;
}

/* Outputs at locations DATA0+N..DATA7 never have an attachment and are
 * discarded (Zink broadcasts gl_FragColor to all 8). */
static bool
drop_unattached_output(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   if (in->intrinsic != nir_intrinsic_store_output)
      return false;
   unsigned loc = nir_intrinsic_io_semantics(in).location;
   if (loc < FRAG_RESULT_DATA0 || loc > FRAG_RESULT_DATA7 ||
       g5p_mrt_slot_is_attached(loc - FRAG_RESULT_DATA0, *(const unsigned *)data))
      return false;
   nir_instr_remove(&in->instr);
   return true;
}

/* SampleMask at one sample: gl_SampleMaskIn is 1, and a written mask without
 * bit 0 removes the fragment's only sample, a demote (the discard path's
 * rules apply). */
static bool
lower_sample_mask_1x(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   bool *used = data;
   b->cursor = nir_before_instr(&in->instr);
   if (in->intrinsic == nir_intrinsic_load_sample_mask_in) {
      *used = true;
      nir_def_replace(&in->def, nir_imm_int(b, 1));
      return true;
   }
   if (in->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(in).location != FRAG_RESULT_SAMPLE_MASK)
      return false;
   *used = true;
   nir_def *mask = nir_channel(b, in->src[0].ssa, 0);
   nir_demote_if(b, nir_ieq_imm(b, nir_iand_imm(b, mask, 1), 0));
   nir_instr_remove(&in->instr);
   return true;
}

/* gl_PointSize has no effect outside point pipelines (polygon mode FILL
 * only): its store is dropped unless transform feedback captures it. A
 * point pipeline keeps it (the TA reads it after the position), and gets
 * 1.0 when the program writes none (maintenance5's default). */
static bool
drop_point_size(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   if (in->intrinsic != nir_intrinsic_store_output || nir_intrinsic_io_semantics(in).location != VARYING_SLOT_PSIZ)
      return false;
   nir_instr_remove(&in->instr);
   return true;
}

/* The program stores gl_PointSize (its intrinsics: outputs_written is not
 * kept up to date by the passes before) */
static bool
find_point_size(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   if (in->intrinsic == nir_intrinsic_store_output && nir_intrinsic_io_semantics(in).location == VARYING_SLOT_PSIZ)
      *(bool *)data = true;
   return false;
}

static bool
xfb_captures(const nir_shader *nir, gl_varying_slot slot)
{
   if (!nir->xfb_info)
      return false;
   for (unsigned i = 0; i < nir->xfb_info->output_count; i++)
      if (nir->xfb_info->outputs[i].location == slot)
         return true;
   return false;
}

/* The colour slots (FRAG_RESULT_DATA0 + s) a fragment shader stores */
static bool
find_color_slots(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   (void)b;
   if (in->intrinsic != nir_intrinsic_store_output)
      return false;
   const unsigned loc = nir_intrinsic_io_semantics(in).location;
   if (loc >= FRAG_RESULT_DATA0 && loc < FRAG_RESULT_DATA0 + 8)
      *(uint8_t *)data |= 1u << (loc - FRAG_RESULT_DATA0);
   return false;
}

/* Several colour targets: one emitpix writes every target, so an attached slot the shader leaves unwritten
 * (Vulkan: undefined) is written zeros, as is the padding fourth target of a 3-target pass. A shader that
 * stores no colour at all is left as it is (it ends without emitpix). */
static void
pad_color_slots(nir_shader *nir, unsigned color_targets)
{
   uint8_t written = 0;
   nir_shader_intrinsics_pass(nir, find_color_slots, nir_metadata_all, &written);
   const uint8_t all = BITFIELD_MASK(color_targets);
   if (!(written & all) || (written & all) == all)
      return;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   for (unsigned s = 0; s < color_targets; s++) {
      if (written & (1u << s))
         continue;
      nir_store_output(&b, nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 0.0f), nir_imm_int(&b, 0), .base = 0,
                       .component = 0, .write_mask = 0xf, .src_type = nir_type_float32,
                       .io_semantics = (nir_io_semantics){ .location = FRAG_RESULT_DATA0 + s, .num_slots = 1 });
      nir->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + s);
   }
   nir_index_ssa_defs(impl);
}

/* The clip distance components a vertex program stores (a bit per
 * distance; compact CLIP_DIST0/1, constant offsets) */
static bool
clip_written(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   if (in->intrinsic != nir_intrinsic_store_output)
      return false;
   const unsigned loc = nir_intrinsic_io_semantics(in).location;
   if (loc != VARYING_SLOT_CLIP_DIST0 && loc != VARYING_SLOT_CLIP_DIST1)
      return false;
   nir_src *off = nir_get_io_offset_src(in);
   const unsigned first = 4 * (loc - VARYING_SLOT_CLIP_DIST0 + (nir_src_is_const(*off) ? nir_src_as_uint(*off) : 0)) +
                          nir_intrinsic_component(in);
   *(unsigned *)data |= nir_intrinsic_write_mask(in) << first;
   return false;
}

/* A declared clip distance the program never stores is 0 (on the plane,
 * not clipped): the TA reads every declared one (plane0..n-1), and an
 * unwritten output dword would clip by a stale register. GL's disabled
 * planes are already stored as 0 (Zink's nir_lower_clip_disable). */
static void
fill_clip_distances(nir_shader *nir)
{
   const unsigned n = nir->info.clip_distance_array_size;
   unsigned written = 0;
   if (!n)
      return;
   nir_shader_intrinsics_pass(nir, clip_written, nir_metadata_all, &written);
   const unsigned missing = BITFIELD_MASK(n) & ~written;
   if (!missing)
      return;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   u_foreach_bit(i, missing) {
      const gl_varying_slot loc = i < 4 ? VARYING_SLOT_CLIP_DIST0 : VARYING_SLOT_CLIP_DIST1;
      nir_store_output(&b, nir_imm_float(&b, 0.0f), nir_imm_int(&b, 0), .base = 0, .component = i % 4,
                       .write_mask = 1, .src_type = nir_type_float32,
                       .io_semantics = (nir_io_semantics){ .location = loc, .num_slots = 1 });
      nir->info.outputs_written |= BITFIELD64_BIT(loc);
   }
   nir_index_ssa_defs(impl);
}

/* G5P renders layer 0 of single-layer views only (no layered rendering is
 * exposed): a vertex shader's gl_Layer is dropped and a fragment shader's
 * gl_Layer reads 0. vk_meta's rectangle and blit shaders use both. */
static bool
lower_layer_0(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   if (in->intrinsic == nir_intrinsic_store_output &&
       nir_intrinsic_io_semantics(in).location == VARYING_SLOT_LAYER) {
      nir_instr_remove(&in->instr);
      return true;
   }
   if (in->intrinsic == nir_intrinsic_load_layer_id) {
      b->cursor = nir_before_instr(&in->instr);
      nir_def_replace(&in->def, nir_imm_int(b, 0));
      return true;
   }
   return false;
}

/* vk_meta's blit samples a 2D-array view of one layer: sample it as 2D
 * (pvr_arch_g5p_tex.c gives that view 2D state; linear images have no array
 * state). The array index is always 0, since the view has one layer. */
static bool
meta_tex_2d(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   const int c = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (!tex->is_array || tex->sampler_dim != GLSL_SAMPLER_DIM_2D || c < 0)
      return false;
   b->cursor = nir_before_instr(instr);
   nir_src_rewrite(&tex->src[c].src, nir_trim_vector(b, tex->src[c].src.ssa, 2));
   tex->is_array = false;
   tex->coord_components = 2;
   return true;
}

static void
meta_blit_2d(nir_shader *nir)
{
   if (!nir->info.name || strncmp(nir->info.name, "vk-meta-blit", 12))
      return;
   nir_foreach_variable_with_modes(var, nir, nir_var_uniform) {
      const struct glsl_type *t = glsl_without_array(var->type);
      if (glsl_type_is_sampler(t) && glsl_sampler_type_is_array(t) &&
          glsl_get_sampler_dim(t) == GLSL_SAMPLER_DIM_2D)
         var->type = glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, glsl_get_sampler_result_type(t));
   }
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_deref && nir_instr_as_deref(instr)->deref_type == nir_deref_type_var)
               nir_instr_as_deref(instr)->type = nir_instr_as_deref(instr)->var->type;
         }
      }
   }
   nir_shader_instructions_pass(nir, meta_tex_2d, nir_metadata_control_flow, NULL);
}

/*
 * The one-layer program variant (pvr_g5p_pipeline::one_layer): every 2D
 * array sample the backend would draw as an array sample (all but the
 * constant layer-0 ones, layer0_as_2d() in pco_g5p_general.c) becomes a 2D
 * sample of the view's base layer. With one layer that is exact: Vulkan's
 * layer is clamp(RNE(z), 0, layers - 1) for filtered samples, and a texel
 * fetch at any other layer is out of bounds (undefined: robustImageAccess is
 * not exposed). The A9 has no array state for a linear image (AGXMetalA9
 * ImageStateEncoderGen2::ImageStateFields: a linear 2D array encodes no
 * texture type), so a one-layer view of a linear image (every single-level
 * 2D colour image on G5P) needs this form when a program samples it at a
 * layer it computes (Dolphin's EFB copy, restore and palette programs). Apple
 * never meets the case: Metal types its textures at creation and makes every
 * array twiddled. The draw selects this variant (g5p_cmd_draw()) only when an
 * array sample's view has no array state, and its uploads refuse a
 * multi-layer view by name (PCO_G5P_UPLOAD_IMAGE_ONE_LAYER).
 */
thread_local bool pvr_g5p_one_layer_compile;

static bool
one_layer_as_2d(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *t = nir_instr_as_tex(instr);
   if (!t->is_array || t->sampler_dim != GLSL_SAMPLER_DIM_2D || t->coord_components != 3 ||
       (t->op != nir_texop_tex && t->op != nir_texop_txb && t->op != nir_texop_txl &&
        t->op != nir_texop_txd && t->op != nir_texop_txf && t->op != nir_texop_tg4))
      return false;
   const int ci = nir_tex_instr_src_index(t, nir_tex_src_coord);
   if (ci < 0)
      return false;
   /* layer0_as_2d()'s own samples keep their form (and their bytes) */
   nir_scalar z = nir_scalar_chase_movs(nir_get_scalar(t->src[ci].src.ssa, 2));
   if (nir_scalar_is_const(z) &&
       (t->op == nir_texop_txf ? nir_scalar_as_int(z) == 0 : nir_scalar_as_float(z) <= 0.5))
      return false;
   b->cursor = nir_before_instr(instr);
   nir_src_rewrite(&t->src[ci].src, nir_trim_vector(b, t->src[ci].src.ssa, 2));
   t->coord_components = 2;
   t->is_array = false;
   t->backend_flags |= PCO_G5P_TEX_ONE_LAYER_AS_2D;
   return true;
}

static void
one_layer_lower(nir_shader *nir)
{
   if (nir_shader_instructions_pass(nir, one_layer_as_2d, nir_metadata_control_flow, NULL)) {
      nir_opt_dce(nir);
      nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
   }
}

/* The graphics lowerings of a prepared stage (the ICD's graphics_nir()) */
static nir_shader *
graphics_lower(nir_shader *nir, unsigned color_targets, bool *sample_mask, bool point_size, bool *psize_dropped)
{
   const mesa_shader_stage stage = nir ? nir->info.stage : MESA_SHADER_NONE;

   if (nir)
      meta_blit_2d(nir);
   if (nir && pvr_g5p_one_layer_compile)
      one_layer_lower(nir);
   if (nir && nir_shader_intrinsics_pass(nir, lower_layer_0, nir_metadata_control_flow, NULL)) {
      nir->info.outputs_written &= ~BITFIELD64_BIT(VARYING_SLOT_LAYER);
      BITSET_CLEAR(nir->info.system_values_read, SYSTEM_VALUE_LAYER_ID);
      nir_opt_constant_folding(nir);
      nir_opt_dce(nir);
      nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
   }
   if (nir && stage == MESA_SHADER_VERTEX && !point_size && !xfb_captures(nir, VARYING_SLOT_PSIZ) &&
       nir_shader_intrinsics_pass(nir, drop_point_size, nir_metadata_control_flow, NULL)) {
      /* polygon mode POINT needs to know (pco_g5p_program::psize_dropped) */
      if (psize_dropped)
         *psize_dropped = true;
      nir->info.outputs_written &= ~BITFIELD64_BIT(VARYING_SLOT_PSIZ);
      nir_opt_dce(nir);
      nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
   }
   if (nir && stage == MESA_SHADER_VERTEX)
      fill_clip_distances(nir);
   bool writes_psize = false;
   if (nir && stage == MESA_SHADER_VERTEX && point_size)
      nir_shader_intrinsics_pass(nir, find_point_size, nir_metadata_all, &writes_psize);
   if (nir && stage == MESA_SHADER_VERTEX && point_size && !writes_psize) {
      nir_function_impl *impl = nir_shader_get_entrypoint(nir);
      nir_builder b = nir_builder_at(nir_after_impl(impl));
      nir_store_output(&b, nir_imm_float(&b, 1.0f), nir_imm_int(&b, 0), .base = 0, .component = 0,
                       .write_mask = 1, .src_type = nir_type_float32,
                       .io_semantics = (nir_io_semantics){ .location = VARYING_SLOT_PSIZ, .num_slots = 1 });
      nir->info.outputs_written |= BITFIELD64_BIT(VARYING_SLOT_PSIZ);
      nir_index_ssa_defs(impl);
   }
   if (nir && stage == MESA_SHADER_FRAGMENT &&
       nir_shader_intrinsics_pass(nir, drop_unattached_output, nir_metadata_control_flow, &color_targets)) {
      nir_opt_dce(nir);
      nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
   }
   if (nir && stage == MESA_SHADER_FRAGMENT && color_targets > 1)
      pad_color_slots(nir, color_targets);
   if (nir && stage == MESA_SHADER_FRAGMENT &&
       nir_shader_intrinsics_pass(nir, lower_sample_mask_1x, nir_metadata_control_flow, sample_mask)) {
      nir->info.outputs_written &= ~BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK);
      BITSET_CLEAR(nir->info.system_values_read, SYSTEM_VALUE_SAMPLE_MASK_IN);
      nir_opt_constant_folding(nir);
      nir_opt_algebraic(nir);
      nir_opt_dce(nir);
      nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
   }
   return nir;
}

static nir_shader *
graphics_nir(const uint32_t *spv, size_t words, mesa_shader_stage stage, const char *entry, unsigned color_targets,
             bool *sample_mask, bool point_size, bool *psize_dropped)
{
   return graphics_lower(g5p_nir(spv, words, stage, entry), color_targets, sample_mask, point_size, psize_dropped);
}

/* Vertex system values: the hardware index the PDS writes is the draw's
 * first vertex + i, which is gl_VertexIndex; the base instance is 0, as the
 * instance index already holds it (the fetch program adds the first instance
 * to t1, Apple's g3b-basevertex form). */
static bool
lower_vs_sysvals(nir_builder *b, nir_intrinsic_instr *in, void *data)
{
   b->cursor = nir_before_instr(&in->instr);
   switch (in->intrinsic) {
   case nir_intrinsic_load_raw_vertex_id:
      nir_def_replace(&in->def, nir_load_vertex_id(b));
      return true;
   case nir_intrinsic_load_raw_vertex_offset:
   case nir_intrinsic_load_base_instance:
      nir_def_replace(&in->def, nir_imm_int(b, 0));
      return true;
   default:
      return false;
   }
}

static void
vs_sysvals(nir_shader *nir)
{
   if (nir_shader_intrinsics_pass(nir, lower_vs_sysvals, nir_metadata_control_flow, NULL)) {
      nir_opt_constant_folding(nir);
      nir_opt_algebraic(nir);
      nir_opt_dce(nir);
   }
   nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
}

/* The SPIR-V module declares ExecutionMode Xfb (execution modes precede
 * every function) */
static bool
spirv_xfb(const uint32_t *spv, size_t words)
{
   for (size_t i = 5; i < words;) {
      const unsigned op = spv[i] & 0xffff, n = spv[i] >> 16;
      if (!n)
         return false;
      if (op == SpvOpExecutionMode && n >= 3 && i + 2 < words && spv[i + 2] == SpvExecutionModeXfb)
         return true;
      if (op == SpvOpFunction)
         return false;
      i += n;
   }
   return false;
}

/* The transform feedback variant (xfb != NULL): the capture layout from the
 * shader's xfb_info, then nir_lower_xfb_to_stores (64-bit global addresses,
 * outputs kept, so the same draw rasterizes). The backend turns those stores
 * into 4-byte vertex stores at the per-draw block's addresses
 * (PCO_G5P_UPLOAD_XFB). The A9 ICD's pvr_g5p_general_xfb(). */
static bool
xfb_lower(nir_shader *nir, struct g5p_xfb_layout *layout, char *refusal, size_t refusal_size)
{
   const nir_xfb_info *x = nir->xfb_info;
   memset(layout, 0, sizeof(*layout));
   if (!x || !x->buffers_written)
      return true; /* nothing captured: no variant */
   if (x->output_count > G5P_XFB_MAX_RANGES || x->streams_written > 1) {
      snprintf(refusal, refusal_size, "transform feedback: %u outputs (at most %u), streams %#x (stream 0 only)",
               x->output_count, G5P_XFB_MAX_RANGES, x->streams_written);
      return false;
   }
   layout->buffers = x->buffers_written;
   for (unsigned b = 0; b < G5P_XFB_BUFFERS; b++)
      layout->stride[b] = x->buffers[b].stride;
   for (unsigned i = 0; i < x->output_count; i++) {
      const nir_xfb_output_info *o = &x->outputs[i];
      if (o->data_is_16bit) {
         snprintf(refusal, refusal_size, "transform feedback of 16-bit data");
         return false;
      }
      if (o->location == VARYING_SLOT_CLIP_DIST0 || o->location == VARYING_SLOT_CLIP_DIST1) {
         snprintf(refusal, refusal_size, "transform feedback of gl_ClipDistance (capture not qualified)");
         return false;
      }
      const unsigned r = layout->nranges++;
      layout->ranges[r].buffer = o->buffer;
      layout->ranges[r].offset = o->offset;
      layout->ranges[r].size = 4 * util_bitcount(o->component_mask);
   }
   const nir_lower_xfb_to_stores_options opts = { .address_format = nir_address_format_64bit_global,
                                                  .keep_outputs = true };
   NIR_PASS(_, nir, nir_lower_xfb_to_stores, &opts);
   return true;
}

static bool
compile_stage(const VkPipelineShaderStageCreateInfo *stage_info, bool half_output, uint32_t samples,
              struct pvr_g5p_vertex_layout *vertex, const struct pco_g5p_fs_key *key,
              const struct pco_g5p_vs_link *link, struct g5p_xfb_layout *xfb, struct pco_g5p_program *out)
{
   memset(out, 0, sizeof(*out));

   const mesa_shader_stage stage = stage_info->stage == VK_SHADER_STAGE_VERTEX_BIT     ? MESA_SHADER_VERTEX
                                   : stage_info->stage == VK_SHADER_STAGE_FRAGMENT_BIT ? MESA_SHADER_FRAGMENT
                                                                                       : MESA_SHADER_NONE;
   if (stage == MESA_SHADER_NONE) {
      snprintf(out->refusal, sizeof(out->refusal), "a %#x stage (vertex and fragment only)", stage_info->stage);
      return false;
   }
   if (stage_info->pSpecializationInfo && stage_info->pSpecializationInfo->mapEntryCount) {
      snprintf(out->refusal, sizeof(out->refusal), "specialization constants");
      return false;
   }

   /* the SPIR-V words: the module's, or an inline module (maintenance5); or the driver's own NIR (an internal
    * module, vk_shader_module_handle_from_nir(): the clear quad's stages) */
   VK_FROM_HANDLE(vk_shader_module, module, stage_info->module);
   const VkShaderModuleCreateInfo *inline_ci = vk_find_struct_const(stage_info->pNext, SHADER_MODULE_CREATE_INFO);
   /* vk_meta's stages are NIR without a module (pvr_arch_g5p_meta.c) */
   const VkPipelineShaderStageNirCreateInfoMESA *nir_ci =
      vk_find_struct_const(stage_info->pNext, PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA);
   const nir_shader *internal = module ? module->nir : nir_ci ? nir_ci->nir : NULL;
   const uint32_t *spv = NULL;
   size_t bytes = 0;
   if (module && !module->nir) {
      spv = (const uint32_t *)module->data;
      bytes = module->size;
   } else if (!internal && inline_ci) {
      spv = inline_ci->pCode;
      bytes = inline_ci->codeSize;
   } else if (!internal) {
      snprintf(out->refusal, sizeof(out->refusal), "a shader stage without SPIR-V");
      return false;
   }
   if (!internal && (bytes < 20 || bytes % 4)) {
      snprintf(out->refusal, sizeof(out->refusal), "SPIR-V of %zu bytes", bytes);
      return false;
   }
   if (internal && internal->info.stage != stage) {
      snprintf(out->refusal, sizeof(out->refusal), "an internal module of another stage");
      return false;
   }
   if (xfb && (stage != MESA_SHADER_VERTEX || internal || !spirv_xfb(spv, bytes / 4))) {
      memset(xfb, 0, sizeof(*xfb)); /* no capture: no variant */
      return true;
   }

   glsl_type_singleton_init_or_ref();
   const char *entry = stage_info->pName ? stage_info->pName : "main";
   const unsigned color_targets = key && key->color_targets ? key->color_targets : 1;
   bool sample_mask = false;
   const bool point_size = stage == MESA_SHADER_VERTEX && link && link->point_size;
   bool psize_dropped = false;
   nir_shader *clone = internal ? nir_shader_clone(NULL, internal) : NULL;
   if (clone && !clone->options) {
      /* vk_meta's NIR comes without compiler options: G5P's, as spirv_to_nir() gets */
      static util_once_flag once = UTIL_ONCE_FLAG_INIT;
      util_call_once(&once, g5p_init_options);
      clone->options = stage == MESA_SHADER_VERTEX ? &g5p_vertex_options : &g5p_options;
   }
   nir_shader *nir = internal ? graphics_lower(g5p_nir_prepare(clone), color_targets,
                                               &sample_mask, point_size, &psize_dropped)
                              : graphics_nir(spv, bytes / 4, stage, entry, color_targets, &sample_mask, point_size,
                                             &psize_dropped);
   bool ok = false;
   if (!nir) {
      snprintf(out->refusal, sizeof(out->refusal), "SPIR-V -> NIR failed");
   } else if (sample_mask && samples > 1) {
      /* the lowering above is the one-sample rule; at 4x the mask needs the program's coverage feedback */
      snprintf(out->refusal, sizeof(out->refusal), "gl_SampleMask or gl_SampleMaskIn at %u samples "
               "(the coverage feedback is not built)", samples);
   } else if (stage == MESA_SHADER_VERTEX && vertex && !pvr_g5p_vertex_lower(nir, vertex)) {
      /* the vertex input layout (pvr_g5p_vertex_nir.c) */
      snprintf(out->refusal, sizeof(out->refusal), "%s", vertex->refusal);
   } else if (xfb && !xfb_lower(nir, xfb, out->refusal, sizeof(out->refusal))) {
      /* refused by name */
   } else if (xfb && !xfb->buffers) {
      ok = true; /* Xfb mode, nothing captured: no variant */
   } else {
      if (stage == MESA_SHADER_VERTEX)
         vs_sysvals(nir);
      ok = pco_g5p_compile_general_link(nir, half_output, vertex ? vertex->components : NULL, key, link, out);
      out->psize_dropped = psize_dropped;
   }
   ralloc_free(nir);
   glsl_type_singleton_decref();
   return ok;
}

bool
pvr_g5p_compile_stage(const VkPipelineShaderStageCreateInfo *stage_info, bool half_output, uint32_t samples,
                      struct pvr_g5p_vertex_layout *vertex, const struct pco_g5p_fs_key *key,
                      const struct pco_g5p_vs_link *link, struct pco_g5p_program *out)
{
   return compile_stage(stage_info, half_output, samples, vertex, key, link, NULL, out);
}

bool
pvr_g5p_stage_has_xfb(const VkPipelineShaderStageCreateInfo *stage_info)
{
   VK_FROM_HANDLE(vk_shader_module, module, stage_info->module);
   const VkShaderModuleCreateInfo *inline_ci = vk_find_struct_const(stage_info->pNext, SHADER_MODULE_CREATE_INFO);
   if (stage_info->stage != VK_SHADER_STAGE_VERTEX_BIT)
      return false;
   if (module && !module->nir)
      return spirv_xfb((const uint32_t *)module->data, module->size / 4);
   return inline_ci && spirv_xfb(inline_ci->pCode, inline_ci->codeSize / 4);
}

bool
pvr_g5p_compile_vs_xfb(const VkPipelineShaderStageCreateInfo *stage_info, struct pvr_g5p_vertex_layout *vertex,
                       const struct pco_g5p_vs_link *link, struct g5p_xfb_layout *layout,
                       struct pco_g5p_program *out)
{
   return compile_stage(stage_info, false, 1, vertex, NULL, link, layout, out);
}

/*
 * The background program of a render whose colour attachment is loaded
 * (loadOp LOAD): the attachment, bound as a combined image/sampler at set 0
 * binding 0, read at the fragment's own pixel,
 *   color = texelFetch(prev, ivec2(gl_FragCoord.xy), 0),
 * built with nir_builder (Mesa's way for a driver's own shaders), then the
 * same preparation and backend as every graphics stage, with the
 * last-read source groups of every fragment program. Its uploads are the
 * image state and the sampler at that binding (out->uploads).
 */
static bool
g5p_compile_load_type(struct pco_g5p_program *out, unsigned raw)
{
   static util_once_flag once = UTIL_ONCE_FLAG_INIT;
   util_call_once(&once, g5p_init_options);
   memset(out, 0, sizeof(*out));
   glsl_type_singleton_init_or_ref();

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &g5p_options, "g5p-load");
   nir_variable *prev = nir_variable_create(b.shader, nir_var_uniform,
                                            glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, (raw == 1 || raw == 5 || raw == 6 || raw == 7 || raw == 10) ? GLSL_TYPE_UINT : (raw == 2 || raw == 8 || raw == 9 || raw == 11) ? GLSL_TYPE_INT : GLSL_TYPE_FLOAT),
                                            "prev");
   prev->data.descriptor_set = 0;
   prev->data.binding = 0;
   nir_variable *color = nir_variable_create(b.shader, nir_var_shader_out, (raw == 1 || raw == 5 || raw == 6 || raw == 7 || raw == 10) ? glsl_uvec4_type() : (raw == 2 || raw == 8 || raw == 9 || raw == 11) ? glsl_ivec4_type() : glsl_vec4_type(), "color");
   color->data.location = FRAG_RESULT_DATA0;
   /* texelFetch (Mesa's form of a load): with the last-read group bits it reads every channel (the LOAD probe's
    * one-channel result was the default group rule, not the read op) */
   nir_def *xy = nir_f2i32(&b, nir_trim_vector(&b, nir_load_frag_coord(&b), 2));
   nir_def *texel = nir_txf(&b, xy, .texture_deref = nir_build_deref_var(&b, prev), .lod = nir_imm_int(&b, 0),
                            .dim = GLSL_SAMPLER_DIM_2D, .dest_type = (raw == 1 || raw == 5 || raw == 6 || raw == 7 || raw == 10) ? nir_type_uint32 : (raw == 2 || raw == 8 || raw == 9 || raw == 11) ? nir_type_int32 : nir_type_float32);
   nir_store_var(&b, color, texel, 0xf);

   bool sample_mask = false; /* the load program reads no sample mask */
   nir_shader *nir = graphics_lower(g5p_nir_prepare(b.shader), 1, &sample_mask, false, NULL);
   /* Apple's source groups, as every pipeline's fragment program (g5p_check_state()): 1 only on a register's last
    * read. The default, 1 on every read, returns a wrong value on a re-read: the LOAD probe's texelFetch wrote only
    * output component 0 (B G R A = 0 0 R 0). */
   struct pco_g5p_fs_key key = PCO_G5P_FS_KEY_DEFAULT;
   key.last_read_groups = true;
   key.packed_uint_bits = (raw == 5 || raw == 9) ? 16 : (raw == 6 || raw == 8 || raw == 10 || raw == 11) ? 8 : raw == 7 ? 10 : 0;
   key.packed_sint = raw == 8 || raw == 9 || raw == 11;
   key.packed_scalar8 = raw == 10 || raw == 11;
   key.color_output = (raw == 1 || raw == 5 || raw == 6 || raw == 7 || raw == 10) ? PCO_G5P_COLOR_UINT32 : (raw == 2 || raw == 8 || raw == 9 || raw == 11) ? PCO_G5P_COLOR_SINT32 : PCO_G5P_COLOR_F16;
   bool ok = nir && pco_g5p_compile_general_link(nir, true, NULL, &key, NULL, out);
   if (!nir)
      snprintf(out->refusal, sizeof(out->refusal), "the load program's NIR");
   ralloc_free(nir);
   glsl_type_singleton_decref();
   return ok;
}

/* The clear quad's stages (pvr_g5p.h) */
static nir_shader *
g5p_clear_nir_type(bool vertex, unsigned raw)
{
   static util_once_flag once = UTIL_ONCE_FLAG_INIT;
   util_call_once(&once, g5p_init_options);
   glsl_type_singleton_init_or_ref();

   nir_builder b = nir_builder_init_simple_shader(vertex ? MESA_SHADER_VERTEX : MESA_SHADER_FRAGMENT, &g5p_options,
                                                  vertex ? "g5p-clear-vs" : "g5p-clear-fs");
   if (vertex) {
      nir_variable *pos = nir_variable_create(b.shader, nir_var_shader_in, glsl_vec4_type(), "pos");
      pos->data.location = VERT_ATTRIB_GENERIC0;
      nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
      out->data.location = VARYING_SLOT_POS;
      nir_store_var(&b, out, nir_load_var(&b, pos), 0xf);
   } else {
      nir_variable *color = nir_variable_create(b.shader, nir_var_shader_out, raw == 1 ? glsl_uvec4_type() : raw == 2 ? glsl_ivec4_type() : glsl_vec4_type(), "color");
      color->data.location = FRAG_RESULT_DATA0;
      nir_store_var(&b, color, nir_load_push_constant(&b, 4, 32, nir_imm_int(&b, 0), .base = 0, .range = 16), 0xf);
      b.shader->num_uniforms = 16;
   }
   return b.shader;
}

void
pvr_g5p_clear_nir_free(nir_shader *nir)
{
   ralloc_free(nir);
   glsl_type_singleton_decref();
}

bool
pvr_g5p_compile_load(struct pco_g5p_program *out)
{
   return g5p_compile_load_type(out, 0);
}

bool
pvr_g5p_compile_load_raw(struct pco_g5p_program *out, bool is_signed)
{
   return g5p_compile_load_type(out, is_signed ? 2 : 1);
}

/*
 * The 4x LOAD background (a loaded 4-sample colour attachment), Apple's loader (an iOS capture
 * of a 4x colour load: the 1x loader plus the own-sample operand, dimension f9, kicked at
 * sample rate): a sample of the target's 2D-multisample state at the pixel centre in normalized
 * coordinates, color = texture(prev, gl_FragCoord.xy / textureSize(prev, 0)), compiled with the key's
 * ms_own_sample, so the sample reads the sample the task shades (pco_g5p_general.c
 * emit_tex_ms_own_sample()). Its uploads: the image state, the sampler, the size block.
 */
bool
pvr_g5p_compile_load_ms(struct pco_g5p_program *out)
{
   static util_once_flag once = UTIL_ONCE_FLAG_INIT;
   util_call_once(&once, g5p_init_options);
   memset(out, 0, sizeof(*out));
   glsl_type_singleton_init_or_ref();

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &g5p_options, "g5p-load-ms4");
   nir_variable *prev = nir_variable_create(b.shader, nir_var_uniform,
                                            glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
                                            "prev");
   prev->data.descriptor_set = 0;
   prev->data.binding = 0;
   nir_variable *color = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "color");
   color->data.location = FRAG_RESULT_DATA0;
   nir_deref_instr *deref = nir_build_deref_var(&b, prev);
   nir_def *size = nir_i2f32(&b, nir_trim_vector(&b, nir_txs(&b, .texture_deref = deref, .lod = nir_imm_int(&b, 0),
                                                              .dim = GLSL_SAMPLER_DIM_2D,
                                                              .dest_type = nir_type_int32), 2));
   nir_def *uv = nir_fdiv(&b, nir_trim_vector(&b, nir_load_frag_coord(&b), 2), size);
   nir_def *texel = nir_tex(&b, uv, .texture_deref = deref, .sampler_deref = deref, .dim = GLSL_SAMPLER_DIM_2D,
                            .dest_type = nir_type_float32);
   nir_store_var(&b, color, texel, 0xf);

   bool sample_mask = false;
   nir_shader *nir = graphics_lower(g5p_nir_prepare(b.shader), 1, &sample_mask, false, NULL);
   struct pco_g5p_fs_key key = PCO_G5P_FS_KEY_DEFAULT;
   key.last_read_groups = true;
   key.ms_own_sample = true;
   bool ok = nir && pco_g5p_compile_general_link(nir, true, NULL, &key, NULL, out);
   if (!nir)
      snprintf(out->refusal, sizeof(out->refusal), "the 4x load program's NIR");
   ralloc_free(nir);
   glsl_type_singleton_decref();
   return ok;
}

bool
pvr_g5p_compile_load_mrt(struct pco_g5p_program *out, unsigned targets, bool bgra)
{
   static util_once_flag once = UTIL_ONCE_FLAG_INIT;
   util_call_once(&once, g5p_init_options);
   memset(out, 0, sizeof(*out));
   glsl_type_singleton_init_or_ref();

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &g5p_options, "g5p-load-mrt");
   nir_def *xy = nir_f2i32(&b, nir_trim_vector(&b, nir_load_frag_coord(&b), 2));
   for (unsigned t = 0; t < targets; t++) {
      nir_variable *prev = nir_variable_create(b.shader, nir_var_uniform,
                                               glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
                                               "prev");
      prev->data.descriptor_set = 0;
      prev->data.binding = t;
      nir_variable *color = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "color");
      color->data.location = FRAG_RESULT_DATA0 + t;
      nir_def *texel = nir_txf(&b, xy, .texture_deref = nir_build_deref_var(&b, prev), .lod = nir_imm_int(&b, 0),
                               .dim = GLSL_SAMPLER_DIM_2D, .dest_type = nir_type_float32);
      nir_store_var(&b, color, texel, 0xf);
   }
   bool sample_mask = false;
   nir_shader *nir = graphics_lower(g5p_nir_prepare(b.shader), targets, &sample_mask, false, NULL);
   struct pco_g5p_fs_key key = PCO_G5P_FS_KEY_DEFAULT;
   key.last_read_groups = true;
   key.color_targets = targets;
   key.color_output = bgra ? PCO_G5P_COLOR_PACKED_BGRA8 : PCO_G5P_COLOR_PACKED_RGBA8;
   bool ok = nir && pco_g5p_compile_general_link(nir, true, NULL, &key, NULL, out);
   if (!nir)
      snprintf(out->refusal, sizeof(out->refusal), "the compact load program's NIR");
   ralloc_free(nir);
   glsl_type_singleton_decref();
   return ok;
}

nir_shader *
pvr_g5p_clear_nir(bool vertex)
{
   return g5p_clear_nir_type(vertex, 0);
}

nir_shader *
pvr_g5p_clear_nir_raw(bool vertex, bool is_signed)
{
   return g5p_clear_nir_type(vertex, is_signed ? 2 : 1);
}

bool pvr_g5p_compile_load_rgba16ui(struct pco_g5p_program *out)
{
   return g5p_compile_load_type(out, 5);
}

bool pvr_g5p_compile_load_rgba8ui(struct pco_g5p_program *out)
{
   return g5p_compile_load_type(out, 6);
}

bool pvr_g5p_compile_load_rgb10a2ui(struct pco_g5p_program *out)
{
   return g5p_compile_load_type(out, 7);
}

bool pvr_g5p_compile_load_packed_sint(struct pco_g5p_program *out, unsigned bits)
{
   return g5p_compile_load_type(out, bits == 8 ? 8 : 9);
}

bool pvr_g5p_compile_load_scalar8(struct pco_g5p_program *out, bool is_signed)
{
   return g5p_compile_load_type(out, is_signed ? 11 : 10);
}
