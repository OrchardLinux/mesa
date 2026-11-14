/* SPDX-License-Identifier: MIT */
#ifndef G5P_END_BARRIER_H
#define G5P_END_BARRIER_H
#include <stdint.h>
#include "g5p_usc.h"
/* The compute flush program: the USC program of the compute end-of-pass
 * barrier task (pvr_arch_g5p_compute.c upload_end_barrier()), the one iOS
 * runs from that task. iOS builds it as a driver program
 * (AGCStreamTokenComputeFlushProgram, or its firmware-interrupt variant, in
 * AGXCompilerCore); here it is built from the USC encoders.
 *
 * The task's DMA puts the fence operand in common dwords 0 and 1. The
 * program moves them into r0:r1, fences device memory twice around a wait,
 * emits code 0xb with no words, waits, and ends at a 64-byte boundary.
 * @usc holds at least 128 bytes; returns the byte count, 0 on an encoder
 * refusal. */
#define G5P_END_BARRIER_FENCE_PAIR 0 /* r0:r1 */
static inline unsigned
g5p_end_barrier_helper(uint8_t *usc)
{
   unsigned at = 0, n;
#define G5P_EB_STEP(call) do { if (!(n = (call))) return 0; at += n; } while (0)
   G5P_EB_STEP(g5p_usc_mov_shared(usc + at, 2, 1, 0)); /* r1 = common dword 1 */
   G5P_EB_STEP(g5p_usc_mov_shared(usc + at, 0, 0, 0)); /* r0 = common dword 0 */
   G5P_EB_STEP(g5p_usc_mem_fence(usc + at, G5P_END_BARRIER_FENCE_PAIR, G5P_FENCE_FORM_FLUSH_PROGRAM));
   G5P_EB_STEP(g5p_usc_wait(usc + at, 0));
   G5P_EB_STEP(g5p_usc_mem_fence(usc + at, G5P_END_BARRIER_FENCE_PAIR, G5P_FENCE_FORM_FLUSH_PROGRAM));
   G5P_EB_STEP(g5p_usc_emit_code(usc + at, G5P_EMIT_CODE_FLUSH_PROGRAM));
   G5P_EB_STEP(g5p_usc_wait(usc + at, 0));
   G5P_EB_STEP(g5p_usc_branch(usc + at, 64 - at, false));
   while (at < 64)
      G5P_EB_STEP(g5p_usc_special(usc + at, false));
   G5P_EB_STEP(g5p_usc_special(usc + at, true));
#undef G5P_EB_STEP
   return at;
}
#endif
