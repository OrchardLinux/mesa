/* SPDX-License-Identifier: MIT */
/*
 * -Dimagination-g9p-es3: the encoders the OpenGL ES 3.0 enablers add to
 * g9p_isa.h's. Each one is the instruction our Apple A9 (G5P) backend emits
 * for the same job, ported because the two ISAs match on every form
 * compared (A9 notes, the 11-02 reply to the A10 notes, section 1.1). The A9
 * evidence is named at each encoder; none of them has run on an A10.
 */
#ifndef G9P_ES3_ISA_H
#define G9P_ES3_ISA_H

#include "g9p_isa.h"

/* Colour targets a fragment shader can write (output halves 4t..4t+3) */
#define G9P_ES3_MAX_TARGETS 4

/*
 * Output the colour of @words 32-bit output registers (two per 8-bit RGBA
 * target, r0 upwards) and end the fragment shader. The emit's word count is
 * a 3-bit field, so 8 is 0: Apple's A9 compiler emits 1c21 0008 for one
 * target (g9p_end_fragment()'s 211c 0800), 1c41 0008 for two and 1c01 0008
 * for four (A9 notes, gles3-continued107.md section 0.3; hardware-proven on
 * the A9 for two and four targets, mrt.md).
 */
static inline void
g9p_es3_end_fragment(struct g9p_code *c, unsigned words)
{
   G9P_EMIT(c, (uint16_t)(0x001c | ((words & 7) << 4 | 1) << 8), 0x0800);
   G9P_EMIT(c, 0x8000);
   while (c->n % 32)
      G9P_EMIT(c, 0x0000);
}

/*
 * End of tile, one PBE write: the image state at common-store dword 8 x
 * @state8 (units of eight dwords, the emit's operand is 2 x @state8), the
 * target's output-half offset in r[@offset_reg]. Writes before the last one
 * set W1 bit 13 and are followed by 044c. Our A9 generator's form
 * (a9/g5p_mrt_programs.h, byte-equal to Apple's two- and four-target end of
 * tile); for one target it is g9p_gfx_device_init()'s 051c 5100 0800.
 */
static inline void
g9p_es3_pbe_emit(struct g9p_code *c, unsigned offset_reg, unsigned state8, bool last)
{
   G9P_EMIT(c, 0x051c, (last ? 0x5100 : 0x7100) | 2 * offset_reg, 0x0800 | 2 * state8);
   if (!last)
      G9P_EMIT(c, 0x044c);
}

/*
 * Several colour targets' device programs (pvr_arch_g9p_es3_gfx.h has the
 * layout of their common store). The end of tile writing the targets in
 * @mask: our A9 generator's program (a9/g5p_mrt_programs.h, byte-equal to
 * Apple's two- and four-target end of tile), with the offsets as
 * immediates where Apple's code loads them from the common store (the same
 * length, so the end word lands where it does on the A9)
 */
static inline void
g9p_es3_eot_program(struct g9p_code *c, unsigned mask)
{
   unsigned list[G9P_ES3_MAX_TARGETS], m = 0;

   for (unsigned t = 0; t < G9P_ES3_MAX_TARGETS; t++) {
      if (mask & (1u << t))
         list[m++] = t;
   }
   c->n = 0;
   c->fragment = true;
   /* the k-th store's offset register: m - 1 - k, as Apple's A9 code */
   for (unsigned k = 0; k < m; k++)
      g9p_mov_imm(c, m - 1 - k, 4 * list[k]);
   for (unsigned k = 0; k < m; k++)
      g9p_es3_pbe_emit(c, m - 1 - k, k, k + 1 == m);
   g9p_end(c);
}

/* The background clear of @n targets: common dword h to output half h */
static inline void
g9p_es3_clear_program(struct g9p_code *c, unsigned n)
{
   c->n = 0;
   c->fragment = true;
   for (unsigned h = 0; h < 4 * n; h++)
      G9P_EMIT(c, 0x018a, 0x0100 | h, 0x0101, 0x01a0, 0x0101 | 4 * h, 0x4140, 0x0104,
               0x00b0);
   g9p_es3_end_fragment(c, 2 * n);
}

/*
 * The background load of @n targets: target t's image state at dword 16t,
 * its sampler at 16t + 8 and its coordinate scale at 16t + 12, as the
 * one-target load program has them at 0, 8 and 12. The pixel position is
 * converted once into r(2n)-r(2n+1), above the output registers; each
 * target's coordinates go to r0-r1, where the sample reads them, and the
 * targets are sampled from the last to the first, so that target 0's
 * halves (r0-r1) are written last.
 */
static inline void
g9p_es3_load_program(struct g9p_code *c, unsigned n)
{
   const unsigned x = 2 * n, y = 2 * n + 1;

   c->n = 0;
   c->fragment = true;
   G9P_EMIT(c, 0x211c, 0x6800);
   g9p_wait(c);
   g9p_convert(c, x, 0, G9P_U2F);
   g9p_convert(c, y, 1, G9P_U2F);
   for (int t = n - 1; t >= 0; t--) {
      const unsigned sx = 16 * t + 12, sy = 16 * t + 13;

      G9P_EMIT(c, 0x0199, g9p_dst(c, 0), 0x0181 | (sx & 31) << 2 | x << 10,
               0x4100 | (sx >> 5) << 1, G9P_FMUL);
      G9P_EMIT(c, 0x0199, g9p_dst(c, 1), 0x0181 | (sy & 31) << 2 | y << 10,
               0x4100 | (sy >> 5) << 1, G9P_FMUL);
      G9P_EMIT(c, 0xf518, 0x0100 | 4 * t, 0x8100 | (16 * t) << 8 | (16 * t + 8), 0x0700,
               0x0200);
      g9p_wait(c);
   }
   g9p_es3_end_fragment(c, 2 * n);
}

#endif /* G9P_ES3_ISA_H */
