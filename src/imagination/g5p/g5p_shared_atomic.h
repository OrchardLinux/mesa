/* SPDX-License-Identifier: MIT */
#ifndef G5P_SHARED_ATOMIC_H
#define G5P_SHARED_ATOMIC_H
#include "compiler/nir/nir.h"
#include "../g9p/g9p_isa.h"

/* The exclusive (load-linked, store-conditional) form of g9p_shared_access(): Apple's A9 LL/SC loop sets the size
 * field to 16, W0 bit 15 alone (observed role). The retry jump is g5p_barrier.h's LL/SC retry: G9P_JMP_ANY with
 * the condition bits 5..6 set. */
#define G5P_SHARED_EXCLUSIVE 16u
#define G5P_JMP_SC_RETRY (G9P_JMP_ANY | 3u << 5)

static inline void
g5p_shared_exclusive(struct g9p_code *c, bool store, unsigned r,
                     unsigned k, int offset_half)
{
   g9p_shared_access(c, G5P_SHARED_EXCLUSIVE, store, r, k, offset_half);
}

/* The device-proven g5p_shared_barrier_fenced() LL/SC retry structure.
 * SC always runs, including compare failure, so cmpxchg is strong. */
static inline void
g5p_shared_atomic(struct g9p_code *c, nir_atomic_op op, unsigned old,
                  unsigned val, unsigned cmp, unsigned next, unsigned k,
                  int offset_half, unsigned st)
{
   assert(!c->fragment && k < 4096 && old < 32 && next < 32);
   c->uses_stack = true;
   g9p_wait(c);
   /* push two levels */
   G9P_EMIT(c, G9P_W0_STACK, g9p_st_dst(c, st), g9p_st_src(c, st),
            G9P_STACK_CTL | G9P_STACK_PUSH | 2 << G9P_STACK_LEVELS_SHIFT);
   unsigned ll = c->n;
   g5p_shared_exclusive(c, false, old, k, offset_half);
   g9p_wait(c);
   switch (op) {
   case nir_atomic_op_iadd:
      g9p_alu2(c, G9P_IADD, next, old, val, false); break;
   case nir_atomic_op_iand:
      g9p_alu2(c, G9P_AND, next, old, val, false); break;
   case nir_atomic_op_ior:
      g9p_alu2(c, G9P_OR, next, old, val, false); break;
   case nir_atomic_op_ixor:
      g9p_alu2(c, G9P_XOR, next, old, val, false); break;
   case nir_atomic_op_xchg:
      g9p_alu2(c, G9P_OR, next, val, val, false); break;
   case nir_atomic_op_imin:
   case nir_atomic_op_umin:
   case nir_atomic_op_imax:
   case nir_atomic_op_umax:
   case nir_atomic_op_cmpxchg: {
      bool cas = op == nir_atomic_op_cmpxchg;
      bool is_min = op == nir_atomic_op_imin || op == nir_atomic_op_umin;
      unsigned cond = cas ? G9P_EQ :
                      op == nir_atomic_op_imin || op == nir_atomic_op_imax ? G9P_SLT : G9P_ULT;
      unsigned x = cas ? val : is_min ? old : val;
      unsigned y = cas ? old : is_min ? val : old;
      /* Same compare/select sequence as g9p_compile.c emit_select(). */
      g9p_add_imm(c, next, y, 0);
      g9p_cmp_reg(c, cond, old, cas ? cmp : val, false);
      g9p_push_if(c, st, false);
      g9p_add_imm(c, next, x, 0);
      g9p_pop_levels(c, st, 1);
      break;
   }
   default: UNREACHABLE("unsupported shared atomic");
   }
   g5p_shared_exclusive(c, true, next, k, offset_half);
   unsigned retry = g9p_jump(c, G5P_JMP_SC_RETRY, 0);
   g9p_jump_patch(c, retry, 2 * ((int)ll - (int)retry));
   /* pop both levels, the stack register still live (no last-read bit) */
   G9P_EMIT(c, G9P_W0_STACK, g9p_st_dst(c, st), g9p_st_src(c, st),
            G9P_STACK_CTL | G9P_STACK_POP | 2 << G9P_STACK_LEVELS_SHIFT);
   g9p_wait(c);
}
#endif
