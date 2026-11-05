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
#endif
