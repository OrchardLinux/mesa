/* SPDX-License-Identifier: MIT
 * Generated-by: LLM
 */
#ifndef G5P_NIR_INTEGER_H
#define G5P_NIR_INTEGER_H
#include "compiler/nir/nir_builder.h"

/* Fill every bit below the top set bit, count, then subtract one. This is
 * exact over all uint32 values, including zero (0 - 1 = UINT32_MAX). The
 * inherited NIR lowering maps signed MSB through x ^ (x >> 31) / select
 * and LSB through x & -x. No float conversion, zero-undefined CLZ, or
 * unwitnessed native integer opcode is needed. */
static bool
g5p_lower_msb_instr(nir_builder *b, nir_alu_instr *a, void *data)
{
   if (a->op != nir_op_ufind_msb || a->def.bit_size != 32)
      return false;
   b->cursor = nir_before_instr(&a->instr);
   nir_def *x = nir_ssa_for_alu_src(b, a, 0);
   for (unsigned shift = 1; shift < 32; shift <<= 1)
      x = nir_ior(b, x, nir_ushr_imm(b, x, shift));
   nir_def_replace(&a->def, nir_iadd_imm(b, nir_bit_count(b, x), -1));
   return true;
}
#endif
