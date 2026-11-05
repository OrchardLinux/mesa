/* SPDX-License-Identifier: MIT */
/*
 * USC instruction encoders for the Apple A10 GPU's shader ISA
 * (applegpu_g9p). Each instruction is a run of 16-bit words; bit 8 of a word
 * means another word follows. The encodings are the ones decoded and tested
 * on hardware in the A10 notes' ISA.md (g9p/README.md; its tools/isa/g9p.py is the
 * reference assembler these mirror).
 *
 * Registers are 32-bit; how many a thread has is set by the CDM kernel
 * record (G9P_CDM_KERNEL_REGS). At entry r0 holds the thread's index in its
 * workgroup.
 */
#ifndef G9P_ISA_H
#define G9P_ISA_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#define G9P_MAX_CODE_WORDS 65536

struct g9p_code {
   uint16_t w[G9P_MAX_CODE_WORDS];
   unsigned n;
   /* a fragment shader: registers in the other bank, see g9p_w0() */
   bool fragment;
   /* the code pushes onto the mask stack (needs g9p_push_init()) */
   bool uses_stack;
};

/*
 * Register bank. Compute and vertex code (Apple's too) names its registers
 * with W0 bits 11-13 and W1 bit 9 set (3199 03dd, 1199 03dd, 399e 03dd);
 * fragment code, whose iterates, immediates and colour output live in the
 * other bank, with them clear (0199 01dd, 019e 01dd). Mixing the two in a
 * fragment shader reads and writes registers nothing else sees (ISA.md
 * "Fragment register bank"). The mask stack follows (g9p_st_dst()).
 */
static inline uint16_t
g9p_w0(const struct g9p_code *c, uint16_t w0)
{
   return c->fragment ? w0 & ~0x3800 : w0;
}

static inline uint16_t
g9p_dst(const struct g9p_code *c, unsigned r)
{
   return (c->fragment ? 0x0100 : 0x0300) | 2 * r;
}

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
#define G9P_SIZE_2X32 5

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

/*
 * The execution-mask stack register @st as destination (W1) and source
 * (W2): in compute and vertex code r@st; in fragment code, like Apple's
 * (01c8 010d 011a 86d0), the upper half of r@st in the fragment bank
 */
static inline uint16_t
g9p_st_dst(const struct g9p_code *c, unsigned st)
{
   return c->fragment ? 0x0100 | (2 * st + 1) : 0x0300 | 2 * st;
}

static inline uint16_t
g9p_st_src(const struct g9p_code *c, unsigned st)
{
   return c->fragment ? 0x0100 | (2 * st + 1) << 1 : 0x0100 | st << 2;
}

/* Start the execution-mask stack in register @st (first one after ours) */
static inline void
g9p_push_init(struct g9p_code *c, unsigned st)
{
   G9P_EMIT(c, 0x01c8, g9p_st_dst(c, st), 0x0101, 0x0120, 0x46d1);
}

/* Pop the mask stack before the end */
static inline void
g9p_pop(struct g9p_code *c, unsigned st)
{
   G9P_EMIT(c, g9p_w0(c, 0x21c8), g9p_st_dst(c, st), 0x0080 | g9p_st_src(c, st), 0x46d2);
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
      G9P_EMIT(c, w0, g9p_dst(c, dst), w2, w3,
               0x0100 | ((imm >> 16) & 0xff) | ((imm >> 25) & 0x7f) << 9,
               0x0154, 0x0038);
   } else {
      G9P_EMIT(c, w0, g9p_dst(c, dst), w2, w3, 0x0154,
               0x0038);
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
   G9P_EMIT(c, g9p_w0(c, 0x3199), g9p_dst(c, dst), g9p_srcs(a, b, kill), op);
}

/*
 * Fragment registers above r31 (Apple's compiler uses up to r101 in fragment
 * code): as a destination the usual 0x0100 | 2r; as a source the operand
 * word's 5-bit fields hold r & 31, bit 0 is set, and an extension word after
 * it holds r >> 5 (first operand at bit 1, second at bit 9), as for uniform
 * operands. Tested with pvr-tri (r40, r77, r100: ISA.md "Registers above
 * r31"). The compiler parks NIR registers there ("homes") and moves them
 * with OR, which copies any bit pattern.
 */
static inline void
g9p_mov_to_high(struct g9p_code *c, unsigned hi, unsigned src)
{
   assert(c->fragment && src < 32 && hi < 128);
   G9P_EMIT(c, 0x0199, 0x0100 | 2 * hi, 0x0100 | src << 2 | src << 10, 0x00ce);
}

static inline void
g9p_mov_from_high(struct g9p_code *c, unsigned dst, unsigned hi)
{
   assert(c->fragment && dst < 32 && hi < 128);
   G9P_EMIT(c, 0x0199, 0x0100 | 2 * dst, 0x0101 | (hi & 31) << 2 | (hi & 31) << 10,
            0x0100 | (hi >> 5) << 1 | (hi >> 5) << 9, 0x00ce);
}

/* dst = src + imm, 0 <= imm < 64 */
static inline void
g9p_add_imm(struct g9p_code *c, unsigned dst, unsigned src, unsigned imm)
{
   assert(imm < 0x40);
   G9P_EMIT(c, g9p_w0(c, 0x1199), g9p_dst(c, dst), 0x0101 | imm << 1 | src << 10, 0x0120,
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
   G9P_EMIT(c, g9p_w0(c, 0x1189), g9p_dst(c, dst),
            (kill ? 0x8000 : 0) | (1 | idx << 2) << 8 | 0x81 | 4 * base_dword,
            0x4100, 0x01a1, 0x0600 | shift << 13);
}

/*
 * Load (store) @size bytes at address pair r[pair] into (from) r[reg]. W0
 * bits 8-9 are the load's bank bits: fragment code clears them (Apple's
 * fragment loads are 2012/3812 where compute has 2312, probes/fsload.metal)
 */
static inline void
g9p_mem_at(struct g9p_code *c, bool store, unsigned size, unsigned reg,
           unsigned pair)
{
   G9P_EMIT(c, size << 11 | (c->fragment ? 0x0010 : 0x0310),
            (1 | pair << 2) << 8 | (store ? 0x80 : 0) | reg << 1, 0x8800);
}

/* Float ops of the two-source form; bit 14 negates the second source */
#define G9P_FADD 0x44
#define G9P_FMUL 0x48
#define G9P_NEG_B 0x4000
/* Arithmetic shift right, two-source form */
#define G9P_ISHR 0xfc

/* Compare conditions (integer: ISA.md "Control flow"; float: "fops") */
#define G9P_ULT 0xb6
#define G9P_UGT 0xb7
#define G9P_EQ  0xbd
#define G9P_SLT 0xbe
#define G9P_SGT 0xbf
#define G9P_FLT 0x59
#define G9P_FGT 0x5d
#define G9P_FEQ 0x55

/*
 * dst = common-store dword @u (a uniform operand, Apple's 0x81 | u << 2 with
 * the high bits in the extension word 0x4100 | (u >> 5) << 1) + r[zero]: a
 * move from the common store (probes/fsuni*.metal)
 */
static inline void
g9p_mov_uniform(struct g9p_code *c, unsigned dst, unsigned u, unsigned zero)
{
   G9P_EMIT(c, g9p_w0(c, 0x3199), g9p_dst(c, dst), 0x0181 | (u & 31) << 2 | zero << 10,
            0x4100 | (u >> 5) << 1, G9P_IADD);
}

/*
 * dst = a * b + c. The third operand word is 0x0100 | c << 2; bit 7 is its
 * last-use (kill) flag, as in g9p_srcs(), and must stay clear while c is
 * read again later
 */
static inline void
g9p_ffma(struct g9p_code *c, unsigned dst, unsigned a, unsigned b, unsigned cc)
{
   G9P_EMIT(c, g9p_w0(c, 0x399e), g9p_dst(c, dst), g9p_srcs(a, b, false), 0x0100 | cc << 2,
            0x004c);
}

/*
 * Float unary ops (2198 03dd 01ss 00op, W0 219c without last-use bit):
 * floor 0x40, rcp 0x80, rsqrt 0x81, log2 0x84, exp2 0x85; 0x82 and 0x86
 * are the two steps of sin/cos (see g9p_compile.c). Checked on hardware
 * against the host's math (ISA.md "Transcendentals").
 */
#define G9P_FFLOOR 0x40
#define G9P_FRCP   0x80
#define G9P_FRSQ   0x81
#define G9P_FSIN1  0x82
#define G9P_FLOG2  0x84
#define G9P_FEXP2  0x85
#define G9P_FSIN2  0x86

static inline void
g9p_funary(struct g9p_code *c, unsigned op, unsigned dst, unsigned src)
{
   G9P_EMIT(c, g9p_w0(c, 0x219c), g9p_dst(c, dst), 0x0100 | src << 2, op);
}

/*
 * Unary conversions: u2f 0x14, i2f 0x16 (to float, W4 0x0468), f2u 0x10,
 * f2i 0x12 (to integer, W4 0x0460).
 */
#define G9P_U2F 0x14
#define G9P_I2F 0x16
#define G9P_F2U 0x10
#define G9P_F2I 0x12

static inline void
g9p_convert(struct g9p_code *c, unsigned dst, unsigned src, unsigned op)
{
   G9P_EMIT(c, g9p_w0(c, 0x1199), g9p_dst(c, dst), 0x0101 | op | src << 10, 0x0120,
            (op & 4) ? 0x0468 : 0x0460);
}

/* dst = a << b (left) or a >> b (logical), b a register */
static inline void
g9p_shift(struct g9p_code *c, unsigned dst, unsigned a, unsigned b, bool right)
{
   G9P_EMIT(c, g9p_w0(c, 0x199e), g9p_dst(c, dst), 0x0101 | a << 10, 0x0120, 0x0100 | b << 2,
            right ? 0x04f4 : 0x04f0);
}

/*
 * Select: dst = cc where (a COND b); g9p_movf() then sets the other lanes.
 * Without last-use bits W0 is 399e.
 */
static inline void
g9p_sel(struct g9p_code *c, unsigned cond, unsigned dst, unsigned a, unsigned b,
        unsigned cc)
{
   G9P_EMIT(c, g9p_w0(c, 0x399e), g9p_dst(c, dst), g9p_srcs(a, b, false), 0x0180 | cc << 2,
            0x6000 | cond);
}

static inline void
g9p_movf(struct g9p_code *c, unsigned dst, unsigned d)
{
   G9P_EMIT(c, g9p_w0(c, 0x21f8), g9p_dst(c, dst), 0x0180 | d << 2, 0x0154, 0x0038);
}

/* Compare a register with an immediate (< 1024), for a following push */
static inline void
g9p_cmp_imm(struct g9p_code *c, unsigned cond, unsigned reg, unsigned imm)
{
   G9P_EMIT(c, g9p_w0(c, 0x1181) | ((imm >> 9) & 1) << 13, 0x0101 | (imm & 0x3f) << 1 | reg << 10,
            0x0120 | ((imm >> 6) & 7) << 1, 0xa100 | cond, 0x0070);
}

/*
 * Compare two registers, for a following push or break: integer conditions
 * (W2 a0cc) or float ones (60cc; Apple's compiler adds a word 0x1000 that
 * negates, for NaN handling).
 */
static inline void
g9p_cmp_reg(struct g9p_code *c, unsigned cond, unsigned a, unsigned b, bool is_float)
{
   G9P_EMIT(c, g9p_w0(c, 0x3181), g9p_srcs(a, b, false), (is_float ? 0x6000 : 0xa000) | cond);
}

/* Mask stack: push (disable the lanes whose compare failed, or passed) */
static inline void
g9p_push_if(struct g9p_code *c, unsigned st, bool invert)
{
   c->uses_stack = true;
   G9P_EMIT(c, g9p_w0(c, 0x21c8), g9p_st_dst(c, st), g9p_st_src(c, st), 0x46d1 | invert << 13);
}

static inline void
g9p_else(struct g9p_code *c, unsigned st)
{
   G9P_EMIT(c, g9p_w0(c, 0x21c8), g9p_st_dst(c, st), g9p_st_src(c, st), 0x46d4);
}

static inline void
g9p_pop_levels(struct g9p_code *c, unsigned st, unsigned levels)
{
   G9P_EMIT(c, g9p_w0(c, 0x21c8), g9p_st_dst(c, st), 0x0080 | g9p_st_src(c, st), 0x06d2 | levels << 14);
}

/* Loops: start (two levels), break the lanes whose compare passed/failed */
static inline void
g9p_loop_begin(struct g9p_code *c, unsigned st)
{
   c->uses_stack = true;
   G9P_EMIT(c, g9p_w0(c, 0x21c8), g9p_st_dst(c, st), g9p_st_src(c, st), 0x86d0);
}

static inline void
g9p_loop_break(struct g9p_code *c, unsigned st, bool when_true)
{
   G9P_EMIT(c, g9p_w0(c, 0x21c8), g9p_st_dst(c, st), g9p_st_src(c, st), 0x86d7 | when_true << 13);
}

/*
 * Jumps: byte offset from the jump instruction, two words of 16 offset bits
 * each. Bit 8 of a word is its continuation flag, so offset bit 8 (16 + 8 in
 * the second word) sits in bit 0, which even offsets leave free (ISA.md
 * "Jump offsets": e.g. 968 -> 03c9, -72 -> ffb9 feff).
 */
#define G9P_JMP_ANY  0x0101 /* back to a loop body while any lane is active */
#define G9P_JMP_NONE 0x0900 /* forward when no lane is active */

static inline void
g9p_jump_patch(struct g9p_code *c, unsigned at, int offset)
{
   uint32_t off = (uint32_t)offset;

   c->w[at + 1] = (off & 0xfefe) | 0x0100 | ((off >> 8) & 1);
   c->w[at + 2] = ((off >> 16) & 0xfefe) | ((off >> 24) & 1);
}

static inline unsigned
g9p_jump(struct g9p_code *c, uint16_t op, int offset)
{
   unsigned at = c->n;

   G9P_EMIT(c, op, 0, 0);
   g9p_jump_patch(c, at, offset);
   return at;
}

/* --- Graphics (RENDER.md "Own programs", "Varyings ...") */

/*
 * Vertex output: slot @slot (0-3 the position, then the varyings'
 * components) = r[@reg]. The last output write of a vertex shader takes the
 * end word's place, at the start of a 64-byte block: g9p_end_vertex().
 */
static inline void
g9p_vs_output(struct g9p_code *c, unsigned reg, unsigned slot)
{
   G9P_EMIT(c, 0x151e, 0x0100 | 2 * reg, slot);
}

static inline void
g9p_end_vertex(struct g9p_code *c, unsigned reg, unsigned slot)
{
   unsigned jump = 2 * c->n;
   unsigned end = (jump + 6 + 63) / 64 * 64;

   G9P_EMIT(c, 0x0100, 0x0100 | (end - jump), 0x0000);
   while (2 * c->n < end)
      G9P_EMIT(c, 0xc000);
   G9P_EMIT(c, 0x155e, 0xc100 | 2 * reg, slot);
   while (c->n % 32)
      G9P_EMIT(c, 0x0000);
}

/*
 * Fragment iterate: @comps (1-4) components of the varyings, from component
 * index @first (each varying's components in order; the DOUTI program
 * places them), as f32 into r[@dst]...; perspective-correct or not (W0 bit
 * 9). W0 bits 11-12 are the count modulo 4 (Apple: 0b08 one, 1308 two,
 * 0308 four). A perspective iterate's index counts the W the DOUTI program
 * places first (1 + @first); a non-perspective one's does not (@first).
 * Follow with g9p_wait().
 */
static inline void
g9p_iterate(struct g9p_code *c, unsigned dst, unsigned first, unsigned comps,
            bool perspective)
{
   G9P_EMIT(c, (comps & 3) << 11 | (perspective ? 0x0308 : 0x0108),
            (perspective + first) << 9 | 0x0100 | 2 * dst, 0x0080);
}

/*
 * Texture sample, its coordinates straight from the varyings: an iterate of
 * @comps components from component @first whose result goes to the sampler
 * (W2 0x8000), then the sample with the texture state at common-store dword
 * @state and the sampler at @sampler into f32 r[@dst]..r[@dst + 3] (W0
 * f51a; f518 writes the colour halves). Follow with g9p_wait(). Computed
 * coordinates: g9p_sample_regs().
 */
static inline void
g9p_sample_varying(struct g9p_code *c, unsigned dst, unsigned first, unsigned comps,
                   unsigned state, unsigned sampler)
{
   G9P_EMIT(c, (comps & 3) << 11 | 0x0308, (1 + first) << 9 | 0x0100, 0x8000);
   G9P_EMIT(c, 0xf51a, 0x0100 | 2 * dst, 0x8100 | state << 8 | sampler, 0x0700, 0x0200);
}

/*
 * Texture sample with the coordinates in f32 r0-r1 (Apple's code for
 * computed coordinates: the same sample word after an iterate into r0-r1
 * and arithmetic on them); otherwise as g9p_sample_varying()
 */
static inline void
g9p_sample_regs(struct g9p_code *c, unsigned dst, unsigned state, unsigned sampler)
{
   G9P_EMIT(c, 0xf51a, 0x0100 | 2 * dst, 0x8100 | state << 8 | sampler, 0x0700, 0x0200);
}

/*
 * Half @h of r0-r1 (the colour output) = f16 of f32 r[@src]: a multiply by
 * an f16 1.0 immediate writing a half register
 */
static inline void
g9p_to_half(struct g9p_code *c, unsigned h, unsigned src)
{
   G9P_EMIT(c, 0x0189, 0x0100 | h, 0x0101 | src << 10, 0x7130, 0x0b48, 0x0070);
}

/*
 * Read the tile's colour (framebuffer fetch) as four halves into r[dst] and
 * r[dst + 1]; follow with g9p_wait(). The operand is the first half's index
 * (`2804`: halves 4-7, r2-r3, in Apple's source-over blend; offline probes
 * with Metal's [[color(0)]] input).
 */
static inline void
g9p_tile_read(struct g9p_code *c, unsigned dst)
{
   G9P_EMIT(c, 0x211c, 0x2800 | 2 * dst);
}

/*
 * Discard the active lanes (`6804`; Apple's compiler emits it alone for an
 * unconditional discard and `6004` after a compare for a conditional one)
 */
static inline void
g9p_discard(struct g9p_code *c)
{
   G9P_EMIT(c, 0x6804);
}

/*
 * Conditional discard: Apple's compare form `0189 0100 SSSS W3 [1000]`, then
 * `6004`. W3: integer a2b9 a != b, e2b9 a == b, e2b2 unsigned <, e2b3
 * unsigned >, e2ba signed <; float 435a a < b, 435e a > b, 4356 a == b, and
 * with bit 8 clear (and no fifth word) their negations, e.g. 425a a >= b
 * (probes/discard2.metal, discard3.metal)
 */
#define G9P_DISCARD_INE 0xa2b9

static inline void
g9p_discard_cmp(struct g9p_code *c, unsigned w3, unsigned a, unsigned b)
{
   if (w3 & 0x0100)
      G9P_EMIT(c, 0x0189, 0x0100, g9p_srcs(a, b, false), w3, 0x1000);
   else
      G9P_EMIT(c, 0x0189, 0x0100, g9p_srcs(a, b, false), w3);
   G9P_EMIT(c, 0x6004);
}

/* Output the colour in r0-r1 (4 halves) and end the fragment shader */
/*
 * Premultiplied source-over on the colour in halves 0-3 (r0-r1), as Apple's
 * compiler appends it (`blend.metal` with One / OneMinusSourceAlpha, ISA.md
 * "Tile read and blending"): the tile into halves 4-7 (r2-r3), half 10 (r5)
 * = 1 - half 3, then half i = half (4 + i) x half 10 + half i
 */
static inline void
g9p_blend_src_over(struct g9p_code *c)
{
   G9P_EMIT(c, 0x211c, 0x2804);
   G9P_EMIT(c, 0x004d);
   G9P_EMIT(c, 0x418d, 0x810a, 0x0701, 0x7130, 0x8104, 0x0070);
   G9P_EMIT(c, 0x418e, 0x0100, 0x1588, 0x0180, 0x010c, 0x8000);
   G9P_EMIT(c, 0x418e, 0x0101, 0x158a, 0x0182, 0x010c, 0x8000);
   G9P_EMIT(c, 0x418e, 0x0102, 0x158c, 0x0184, 0x010c, 0x8000);
   G9P_EMIT(c, 0x018a, 0x0103, 0x958e, 0x0186, 0x010c, 0x8000);
}

static inline void
g9p_end_fragment(struct g9p_code *c, bool colour)
{
   if (colour)
      G9P_EMIT(c, 0x211c, 0x0800);
   G9P_EMIT(c, 0x8000);
   while (c->n % 32)
      G9P_EMIT(c, 0x0000);
}

/*
 * Texture dimensions of the sample instruction: W0 bits 8-11 (2D 5, as
 * g9p_sample_regs()'s f51a; 2D array 7, cube d). From the A9 (PR #5).
 */
#define G9P_TEX_2D_ARRAY 0x7
#define G9P_TEX_CUBE     0xd

/*
 * The cube op: the direction in halves @src_half.. (f32 x, y, z) becomes the
 * cube sample's coordinates in halves @dst_half.. (Apple's A9 fragment code
 * 087d dst (src << 1 | 1) 0080 before a cube sample of a computed direction)
 */
static inline void
g9p_cube_coords(struct g9p_code *c, unsigned dst_half, unsigned src_half)
{
   G9P_EMIT(c, 0x7d08, (uint16_t)(((src_half << 1) | 1) << 8 | dst_half), 0x8000);
}

#endif /* G9P_ISA_H */
