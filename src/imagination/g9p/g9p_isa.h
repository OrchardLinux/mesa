/* SPDX-License-Identifier: MIT */
/*
 * USC instruction encoders for the Apple A10 GPU's shader ISA
 * (applegpu_g9p). Each instruction is a run of 16-bit words; bit 8 of a word
 * means another word follows. The encodings are the ones decoded and tested
 * on hardware in the orchard gpu-001 ISA.md (its tools/isa/g9p.py is the
 * reference assembler these mirror).
 *
 * Registers are 32-bit, r0-r7 verified (ISA.md "Register-indexed memory
 * access": the encoding above r7 is open). At entry r0 holds the thread's
 * index in its workgroup.
 */
#ifndef G9P_ISA_H
#define G9P_ISA_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#define G9P_MAX_CODE_WORDS 8192

struct g9p_code {
   uint16_t w[G9P_MAX_CODE_WORDS];
   unsigned n;
};

static inline void
g9p_emit(struct g9p_code *c, const uint16_t *w, unsigned count)
{
   assert(c->n + count <= G9P_MAX_CODE_WORDS);
   for (unsigned i = 0; i < count; i++)
      c->w[c->n++] = w[i];
}

#define G9P_EMIT(c, ...)                                     \
   do {                                                      \
      const uint16_t _w[] = { __VA_ARGS__ };                 \
      g9p_emit((c), _w, sizeof(_w) / sizeof(_w[0]));         \
   } while (0)

/* Two-source ALU opcodes (ISA.md "Loads and two-source ALU") */
#define G9P_IADD 0xa1
#define G9P_IMUL 0xa2
#define G9P_ISUB 0xa5
#define G9P_AND  0xc8
#define G9P_OR   0xce
#define G9P_XOR  0xc6

/* Memory access sizes: W0 bits 11-15 of loads and stores */
#define G9P_SIZE_U8  1
#define G9P_SIZE_U16 2
#define G9P_SIZE_U32 4

/*
 * Program entry: compares the thread count in the common store (uniform
 * half @count_half, i.e. dword @count_half / 2, both halves = threads per
 * group) with r0 and disables the lanes past it.
 */
static inline void
g9p_prologue(struct g9p_code *c, unsigned count_half)
{
   G9P_EMIT(c, 0x1181, 0x8181 | 2 * count_half, 0x4100, 0xacb7);
}

/* Start the execution-mask stack in register @st (first one after ours) */
static inline void
g9p_push_init(struct g9p_code *c, unsigned st)
{
   G9P_EMIT(c, 0x01c8, 0x0300 | 2 * st, 0x0101, 0x0120, 0x46d1);
}

/* Pop the mask stack before the end */
static inline void
g9p_pop(struct g9p_code *c, unsigned st)
{
   G9P_EMIT(c, 0x21c8, 0x0300 | 2 * st, 0x0180 | st << 2, 0x46d2);
}

/*
 * End of program: a jump to the end word, which must start the next 64-byte
 * block after `c000` padding (an `8000` right after the jump hangs the GPU
 * intermittently).
 */
static inline void
g9p_end(struct g9p_code *c)
{
   unsigned jump = 2 * c->n;
   unsigned end = (jump + 6 + 63) / 64 * 64;

   G9P_EMIT(c, 0x0100, 0x0100 | (end - jump), 0x0000);
   while (2 * c->n < end)
      G9P_EMIT(c, 0xc000);
   G9P_EMIT(c, 0x8000);
   while (c->n % 32)
      G9P_EMIT(c, 0xc000);
}

/* dst = imm (32-bit) */
static inline void
g9p_mov_imm(struct g9p_code *c, unsigned dst, uint32_t imm)
{
   uint16_t w0 = 0x0198 | ((imm >> 9) & 1) << 13;
   uint16_t w2 = 0x0101 | (imm & 0x3f) << 1;
   uint16_t w3 = 0x0120 | ((imm >> 6) & 7) << 1 | ((imm >> 10) & 1) << 4 |
                 ((imm >> 14) & 1) << 6 | ((imm >> 15) & 1) << 7 |
                 ((imm >> 11) & 7) << 12;

   if (imm >> 16) {
      w3 |= 1 | ((imm >> 24) & 1) << 15;
      G9P_EMIT(c, w0, 0x0300 | 2 * dst, w2, w3,
               0x0100 | ((imm >> 16) & 0xff) | ((imm >> 25) & 0x7f) << 9,
               0x0154, 0x0038);
   } else {
      G9P_EMIT(c, w0, 0x0300 | 2 * dst, w2, w3, 0x0154, 0x0038);
   }
}

/* Two register sources; @kill marks both as last use */
static inline uint16_t
g9p_srcs(unsigned a, unsigned b, bool kill)
{
   return 0x0100 | a << 2 | b << 10 | (kill ? 0x8080 : 0);
}

/* dst = a OP b */
static inline void
g9p_alu2(struct g9p_code *c, unsigned op, unsigned dst, unsigned a, unsigned b,
         bool kill)
{
   G9P_EMIT(c, 0x3199, 0x0300 | 2 * dst, g9p_srcs(a, b, kill), op);
}

/* dst = src + imm, 0 <= imm < 64 */
static inline void
g9p_add_imm(struct g9p_code *c, unsigned dst, unsigned src, unsigned imm)
{
   assert(imm < 0x40);
   G9P_EMIT(c, 0x1199, 0x0300 | 2 * dst, 0x0101 | imm << 1 | src << 10, 0x0120,
            0x09a1, 0x0070);
}

/*
 * dst = threadgroup memory word @word (the workgroup ID is in words 0-2,
 * written by the kernel task's PDS program); follow with g9p_wait().
 */
static inline void
g9p_tg_load(struct g9p_code *c, unsigned dst, unsigned word)
{
   G9P_EMIT(c, 0x2314, 0x0100 | 2 * dst, 0x8100 | (word & 0xff),
            0x0200 | (word >> 8));
}

/* Wait for outstanding loads */
static inline void
g9p_wait(struct g9p_code *c)
{
   G9P_EMIT(c, 0x004d);
}

/*
 * r[dst]:r[dst+1] = buffer base + (r[idx] << shift); the base is the
 * uniform at common-store dword @base_dword. @shift 0 takes @idx as a byte
 * offset.
 */
static inline void
g9p_addr(struct g9p_code *c, unsigned dst, unsigned base_dword, unsigned idx,
         unsigned shift, bool kill)
{
   G9P_EMIT(c, 0x1189, 0x0300 | 2 * dst,
            (kill ? 0x8000 : 0) | (1 | idx << 2) << 8 | 0x81 | 4 * base_dword,
            0x4100, 0x01a1, 0x0600 | shift << 13);
}

/* Load (store) @size bytes at address pair r[pair] into (from) r[reg] */
static inline void
g9p_mem_at(struct g9p_code *c, bool store, unsigned size, unsigned reg,
           unsigned pair)
{
   G9P_EMIT(c, size << 11 | 0x0310,
            (1 | pair << 2) << 8 | (store ? 0x80 : 0) | reg << 1, 0x8800);
}

#endif /* G9P_ISA_H */
