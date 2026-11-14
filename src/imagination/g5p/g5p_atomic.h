/* SPDX-License-Identifier: MIT */
#ifndef G5P_ATOMIC_H
#define G5P_ATOMIC_H
#include "../g9p/g9p_isa.h"
/* Native register-address device atomics, RE §3.1 (HC and A9 CAP).
 * Result is the old32-bit value; slot0 completion makes it available. */
static inline void
g5p_device_atomic(struct g9p_code *c, unsigned group, unsigned op,
                  unsigned pair, unsigned dst, unsigned zero, unsigned val)
{
   assert(!c->fragment && group <= 2 && op <= 3);
   G9P_EMIT(c, 0x8110 | group << 12,
               (1 | pair << 2) << 8 | op << 6 | 2 * dst,
               0x8900 | 2 * zero, 0x1102, 0x8000 | 2 * val);
   g9p_wait(c);
}
/* RE §3.3: serialize the task's32 lanes around the exclusive sequence.
 * release executes on compare failure too; no unlocked CAS emulation. */
static inline void
g5p_device_cas(struct g9p_code *c, unsigned pair, unsigned old,
               unsigned cmp, unsigned val, unsigned i, unsigned st)
{
   assert(!c->fragment && pair < 32 && old < 32 && i < 32);
   /* Observed in every Apple device CAS, absent from native ALU atomics.
    * Exact broader mode semantics remain unknown. */
   G9P_EMIT(c,0x000e);g9p_wait(c);
   G9P_EMIT(c, 0x0198, 0x0300 | 2*i, 0x0101, 0x0120, 0x01b9, 0x0038);
   G9P_EMIT(c, 0x21c8, 0x0300 | 2*st, 0x0100 | st<<2, 0x86d0);
   unsigned loop=c->n;
   G9P_EMIT(c, 0x1181, 0x0121 | i<<10, 0x0110, 0xa1bd, 0x0070);
   G9P_EMIT(c, 0x21c8, 0x0300 | 2*st, 0x0100 | st<<2, 0x46d1);
   unsigned skip=g9p_jump(c,G9P_JMP_NONE,0);
   G9P_EMIT(c, 0x2310, 0x0100 | pair<<10 | 2*old, 0x8900, 0x1000);
   g9p_wait(c);
   g9p_cmp_reg(c,G9P_EQ,old,cmp,false);
   G9P_EMIT(c, 0x2330, 0x0180 | pair<<10 | 2*val, 0x8900, 0x1000);
   g9p_wait(c);
   G9P_EMIT(c, 0xc330, 0x0100 | pair<<10, 0x0900, 0x1000);
   g9p_jump_patch(c,skip,2*(c->n-skip));
   assert(2*(c->n-skip)==0x28);
   G9P_EMIT(c, 0x21c8, 0x0300 | 2*st, 0x0100 | st<<2, 0x46d2);
   G9P_EMIT(c, 0x1199, 0x0300 | 2*i, 0x0103 | i<<10, 0x0120, 0x01a1, 0x0070);
   g9p_cmp_imm(c,G9P_EQ,i,32);
   G9P_EMIT(c, 0x21c8, 0x0300 | 2*st, 0x0100 | st<<2, 0xa6d7);
   g9p_wait(c);
   unsigned again=g9p_jump(c,G9P_JMP_ANY,0);
   g9p_jump_patch(c,again,2*((int)loop-(int)again));
   assert(2*((int)loop-(int)again)==-0x62);
   G9P_EMIT(c, 0x21c8, 0x0300 | 2*st, 0x0100 | st<<2, 0x86d2);
   G9P_EMIT(c,0x040e);
}
#endif
