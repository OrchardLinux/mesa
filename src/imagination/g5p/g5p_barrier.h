/* SPDX-License-Identifier: MIT */
#ifndef G5P_BARRIER_H
#define G5P_BARRIER_H
#include "../g9p/g9p_isa.h"
/* compute-isa-shared-atomic.md §2.2: Apple's software workgroup barrier.
 * T already contains task_count-1 in its low half. Target = ceil(local/32)-1
 * is inferred beyond the CAP-executed local64 target1. N counts down to1.
 * Only shared memory ordering is covered; device fences require more work. */
static inline void
g5p_shared_barrier_fenced(struct g9p_code *c, unsigned r, unsigned T,
                   unsigned k, unsigned st, unsigned N, int fence_pair)
{
   assert(!c->fragment && k < 128 && N && N <= 31);
   g9p_wait(c);
   G9P_EMIT(c, 0x1181, 0x0101 | T << 9, 0x0120, 0xa5bd, 0x0070);
   unsigned skip = g9p_jump(c, 0x0120, 0);
   G9P_EMIT(c, 0x0182, 0x0101, 0x0120, 0x0121, 0x0110, 0xa1bd, 0x00b0);
   G9P_EMIT(c, 0x21c8, 0x0300 | 2 * st, 0x0100 | st << 2, 0x86d1);
   unsigned ll = c->n;
   G9P_EMIT(c, 0x8314, 0x0100 | 2 * r, 0x8100 | k, 0x0200);
   g9p_wait(c);
   G9P_EMIT(c, 0x398a, 0x0300 | 2 * r, 0x017f | T << 9,
               0x71fe, 0x0100 | r << 2, 0x3fbd, 0x0058);
   G9P_EMIT(c, 0x1189, 0x0300 | 2 * r, 0x0103 | r << 10, 0x0120, 0x0ea1);
   G9P_EMIT(c, 0x21a8, 0x0300 | (2 * r + 1), 0x0102 | r << 2, 0x06c5);
   G9P_EMIT(c, 0x8314, 0x0180 | 2 * r, 0x8100 | k, 0x0200);
   unsigned retry = g9p_jump(c, 0x0161, 0);
   g9p_jump_patch(c, retry, 2 * ((int)ll - (int)retry));
   assert(2 * ((int)ll - (int)retry) == -0x32);
   if (fence_pair >= 0) {
      G9P_EMIT(c, 0xc330, 0x0100 | (unsigned)fence_pair << 10, 0x0900, 0x1000);
      g9p_wait(c);
   }
   G9P_EMIT(c, 0x1181, 0x0101 | r << 10, 0x0120, 0xa5bd, 0x0070);
   G9P_EMIT(c, (N << 11) | 0x042f);
   unsigned last = g9p_jump(c, 0x0120, 0);
   unsigned spin = c->n;
   G9P_EMIT(c, (N << 11) | 0x000f);
   G9P_EMIT(c, 0x1314, 0x0100 | T, 0x8100 | (2 * k + 1), 0x0200);
   g9p_wait(c);
   G9P_EMIT(c, 0x3181, 0x0100 | T << 9 | (2 * r + 1) << 1, 0xacbd);
   unsigned again = g9p_jump(c, 0x0121, 0);
   g9p_jump_patch(c, again, 2 * ((int)spin - (int)again));
   assert(2 * ((int)spin - (int)again) == -0x12);
   unsigned pop = c->n;
   g9p_jump_patch(c, last, 2 * (pop - last));
   assert(2 * (pop - last) == 0x1e);
   G9P_EMIT(c, 0x21c8, 0x0300 | 2 * st, 0x0100 | st << 2, 0x86d2);
   g9p_jump_patch(c, skip, 2 * (c->n - skip));
   assert(2 * (c->n - skip) == 0x86 + (fence_pair >= 0 ? 10 : 0));
   g9p_wait(c);
}
static inline void
g5p_shared_barrier(struct g9p_code *c, unsigned r, unsigned T,
                   unsigned k, unsigned st, unsigned N)
{ g5p_shared_barrier_fenced(c,r,T,k,st,N,-1); }
#endif
