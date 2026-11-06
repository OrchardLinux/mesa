/* SPDX-License-Identifier: MIT
 * G5P USC field encoders: a C port of tools/gpu-userspace/g5p_usc.py and
 * g5p_usc_resources.py (ISA round 18, linux-a9 isa-g5p.md).
 *
 * Every function writes one complete instruction into out[] (at most
 * G5P_USC_MAX bytes) and returns its length, or 0 when a field is outside
 * the measured/representable domain of the reference encoder; the reason is
 * then in g5p_usc_error. Ranges are software bounds of the encoded fields,
 * not measured GPU limits. Source selectors count 16-bit halves; their
 * group bit and the header byte are explicit because no physical bank
 * meaning has been established for them (see isa-g5p.md). The Python
 * reference decoder checks every byte these functions produce
 * (tools/gpu-userspace/check-usc-r18.py).
 */
#ifndef G5P_USC_H
#define G5P_USC_H

#include <stdbool.h>
#include <stdint.h>

#define G5P_USC_MAX 24 /* round 2: literal A + literal C + routing = 20 bytes */

extern _Thread_local const char *g5p_usc_error;

enum g5p_binop {
   G5P_BIN_FADD, G5P_BIN_FMUL, G5P_BIN_IADD, G5P_BIN_IMUL,
   G5P_BIN_AND, G5P_BIN_OR, G5P_BIN_XOR, G5P_BIN_ISHR,
   G5P_BIN_ISUB, /* two registers only (Apple's 00a5; G9P_ISUB) */
   G5P_BIN_COUNT,
};
enum g5p_minmax { G5P_UMIN, G5P_UMAX, G5P_IMIN, G5P_IMAX, G5P_MINMAX_COUNT };
enum g5p_convert { G5P_F2I32, G5P_F2U32, G5P_I2F32, G5P_U2F32, G5P_F2I32_RTE, G5P_CONVERT_COUNT };
/* The 8- and 16-bit sources of the convert to float (Apple's vertex format
 * conversions): bit 0 signed, bit 1 normalized, bit 2 16-bit. */
enum g5p_small_format {
   G5P_SMALL_U8, G5P_SMALL_S8, G5P_SMALL_UN8, G5P_SMALL_SN8,
   G5P_SMALL_U16, G5P_SMALL_S16, G5P_SMALL_UN16, G5P_SMALL_SN16,
   G5P_SMALL_FORMAT_COUNT
};
enum g5p_unary { G5P_FLOOR, G5P_LOG2, G5P_EXP2, G5P_NOT, G5P_POPCOUNT, G5P_BITREVERSE, G5P_UNARY_COUNT };
enum g5p_tex_dim { G5P_TEX_2D, G5P_TEX_2D_ARRAY, G5P_TEX_3D, G5P_TEX_CUBE, G5P_TEX_DIM_COUNT };
/* ISA round 2 (isa-g5p.md "ISA round 2"): compare/write relations, SFU
 * cores and the literal kinds of encode_binary_literal. */
enum g5p_cmp { G5P_FLT, G5P_FGT, G5P_FEQ, G5P_ILT, G5P_IGT, G5P_ULT, G5P_UGT, G5P_IEQ, G5P_CMP_COUNT };
enum g5p_sfu { G5P_RCP_APPROX, G5P_RSQRT_APPROX, G5P_SIN_CORE, G5P_COS_CORE, G5P_SFU_COUNT };
enum g5p_literal { G5P_LIT_U32, G5P_LIT_F16, G5P_LIT_F32 };

extern const char *const g5p_binop_name[G5P_BIN_COUNT];
extern const char *const g5p_minmax_name[G5P_MINMAX_COUNT];
extern const char *const g5p_convert_name[G5P_CONVERT_COUNT];
extern const char *const g5p_unary_name[G5P_UNARY_COUNT];
extern const char *const g5p_tex_dim_name[G5P_TEX_DIM_COUNT];
extern const char *const g5p_cmp_name[G5P_CMP_COUNT];
extern const char *const g5p_sfu_name[G5P_SFU_COUNT];

struct g5p_src {
   unsigned half;  /* 16-bit half index, 0..63 */
   unsigned group; /* selector bit 7 */
   bool neg, abs;
};

/* Compact-source ALU (binary, FMA, integer min/max). Defaults in the
 * reference: bank 0, header 1, every source group 1. */
struct g5p_alu {
   unsigned dst_half, bank, header;
   struct g5p_src a, b, c;
   bool saturate, result_half, source_half;
   unsigned source_extension; /* 0 or 4 (class 9d/9e) */
   unsigned dst_control;      /* 0 or 0x80 (round 2) */
};
#define G5P_ALU(dst) \
   (struct g5p_alu){ .dst_half = (dst), .header = 1, .a.group = 1, .b.group = 1, .c.group = 1 }

unsigned g5p_usc_binary(uint8_t *out, enum g5p_binop op, const struct g5p_alu *f);
unsigned g5p_usc_fma(uint8_t *out, const struct g5p_alu *f);
unsigned g5p_usc_minmax(uint8_t *out, enum g5p_minmax op, const struct g5p_alu *f);
/* A = literal (raw bits), B = compact source. f16: promoted binary16
 * literal for fadd/fmul; otherwise u32 literal for integer operations.
 * long_form: -1 picks the reference default (long iff value > 0xffff). */
unsigned g5p_usc_binary_literal(uint8_t *out, enum g5p_binop op, unsigned dst_half,
                                uint32_t value, unsigned b_half, unsigned bank,
                                unsigned b_group, unsigned header, bool f16, int long_form);
unsigned g5p_usc_fma_literal(uint8_t *out, unsigned dst_half, uint32_t value, unsigned b_half,
                             unsigned c_half, unsigned bank, unsigned b_group, unsigned c_group,
                             unsigned header);
/* Vertex integer forms with a literal A (Apple's vertex probes m_imin_lit,
 * m_umax_lit, m_imax_neg, m_umin_big, m_isub_lit, m_isub_big, m_ineg):
 * min/max(literal, B) and literal - B; 0 - B is ineg. */
unsigned g5p_usc_minmax_literal(uint8_t *out, enum g5p_minmax op, unsigned dst_half, uint32_t value,
                                unsigned b_half, unsigned bank, unsigned b_group, unsigned header);
unsigned g5p_usc_isub_literal(uint8_t *out, unsigned dst_half, uint32_t value, unsigned b_half,
                              unsigned bank, unsigned b_group, unsigned header);
unsigned g5p_usc_shift(uint8_t *out, bool left, unsigned dst_half, unsigned value_half,
                       unsigned amount, bool immediate, unsigned bank, unsigned value_group,
                       unsigned amount_group, unsigned header);
unsigned g5p_usc_convert(uint8_t *out, enum g5p_convert op, unsigned dst_half, unsigned src_half,
                         unsigned bank, unsigned src_group, unsigned header);
unsigned g5p_usc_unary(uint8_t *out, enum g5p_unary op, unsigned dst_half, unsigned src_half,
                       unsigned bank, unsigned src_group, unsigned header, bool negate);
unsigned g5p_usc_move(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned bank,
                      unsigned src_group, unsigned header);
unsigned g5p_usc_immediate(uint8_t *out, unsigned dst_half, uint32_t value, unsigned bank,
                           int long_form);
unsigned g5p_usc_special(uint8_t *out, bool end);
unsigned g5p_usc_task_end(uint8_t *out);
unsigned g5p_usc_branch(uint8_t *out, int delta, bool conditional);
unsigned g5p_usc_iterator(uint8_t *out, unsigned dst_half, unsigned slot, unsigned count,
                          bool narrow, bool saturate);
/* Iterator modes (Apple's applegpu_g5p compiler, varyings.md): b1 low bits
 * 3 = perspective-correct, 1 = no perspective; b1 bit 6 = centroid; b0 bit 0
 * = the second dependent-read counter (Apple's 9th..16th coefficient read
 * of a batch, waited with `4d02`). */
enum g5p_iter_flags {
   G5P_ITER_LINEAR = 1u << 0,
   G5P_ITER_CENTROID = 1u << 1,
   G5P_ITER_DRC1 = 1u << 2,
};
unsigned g5p_usc_iterator_flags(uint8_t *out, unsigned dst_half, unsigned slot, unsigned count,
                                bool narrow, bool saturate, unsigned flags);
/* Flat varying component: load the C coefficient of a coefficient set,
 * `14 23 DD 01 AA 01 00 02` (AA = coefficient dword 4 * set + 2; b0 bit 0 =
 * the second dependent-read counter). */
unsigned g5p_usc_coeff_load(uint8_t *out, unsigned dst_half, unsigned coeff_dword, bool drc1);
unsigned g5p_usc_coord_iterator(uint8_t *out, unsigned dst_half, unsigned slot, unsigned count);
unsigned g5p_usc_emit(uint8_t *out, unsigned words, unsigned source_half);
/* Total word count of a 1/2/4-target fragment (2/4/8; 8 encodes as field 0). */
unsigned g5p_usc_emit_mrt(uint8_t *out, unsigned words, unsigned source_half);
unsigned g5p_usc_vout(uint8_t *out, unsigned src_half, unsigned slot, bool temp, bool last);
unsigned g5p_usc_wait(uint8_t *out, unsigned control);

/* Resources (g5p_usc_resources.py). bank 1 = the VS temp destination
 * (ISA round 2). */
unsigned g5p_usc_mov_shared(uint8_t *out, unsigned dst_half, unsigned src_dword, unsigned bank);

struct g5p_tex {
   unsigned dst_half, image_qword, sampler_qword, coord_dword;
   bool coord_control;
   unsigned class_variant;
   enum g5p_tex_dim dim;
   bool read;    /* integer-coordinate read (mode 'read') */
   bool narrow;  /* class 0x18 packed-f16 result (implicit sample only) */
   /* explicit LOD / gradient */
   unsigned lod_bank, lod_dword;
   bool gradient, lod_control;
   unsigned length; /* 0 = minimum */
};
unsigned g5p_usc_sample(uint8_t *out, const struct g5p_tex *t);
unsigned g5p_usc_texture_lod(uint8_t *out, const struct g5p_tex *t);
unsigned g5p_usc_gather(uint8_t *out, const struct g5p_tex *t);

/* textypes (docs/gpu/experiments/textypes.md §3, Apple's probes.metal):
 * the general sample/read form. b1 = component mask << 4 | dimension
 * (5 2D, 7 2D array, b 3D, d cube); b5 = coordinate dword << 1 | 0x80 | 0x40
 * offset | 1 (continuation); b6 = LOD source: 0x80 temp | mode (0x40 level,
 * 0x20 bias, 0x60 gradients) | dword; b7 = 1 | 0x80 compare; b8 = 0x80 LOD
 * source present | 0x40 extras (compare reference f32, then the offset
 * half) | first extras half. The six-byte form is the plain sample. */
enum g5p_tex_lod { G5P_TEX_LOD_NONE, G5P_TEX_LOD_LEVEL, G5P_TEX_LOD_BIAS, G5P_TEX_LOD_GRAD };
struct g5p_tex_ext {
   unsigned dst_half, image_qword, sampler_qword, coord_dword;
   enum g5p_tex_dim dim;
   unsigned mask;          /* result components: 0xf, or 0x1 for a compare */
   bool read;              /* integer coordinates, u32 level */
   enum g5p_tex_lod lod;   /* source in temp dword lod_dword */
   unsigned lod_dword;
   bool compare, offset;   /* extras at half extras_half: [ref f32][offset u16] */
   unsigned extras_half;
};
unsigned g5p_usc_sample_ext(uint8_t *out, const struct g5p_tex_ext *t);
/* 16-bit immediate into one half (class 0x88, the 32-bit move's split map;
 * Apple's texel-offset probes witness bits 0-9). */
unsigned g5p_usc_half_immediate(uint8_t *out, unsigned dst_half, uint16_t value);
/* Cube direction (x, y, z) in temps -> the cube sample's coordinates, by the
 * iterator unit: `08 7d DH SH 00 80`, SH = source half << 1 | 1 (Apple's
 * f_cube_alu, f_cube_hi; a cube sample of raw directions reads them as 2D on
 * face 0, device 2026-11-02). */
unsigned g5p_usc_vertex_texture_level(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned group);
unsigned g5p_usc_cube_coords(uint8_t *out, unsigned dst_half, unsigned src_half);
/* f32 -> u16 (truncating) into one half: Apple's array-layer conversion
 * `89 01 DH 01 09 SS 20 01 60 06`. */
unsigned g5p_usc_f2u16_half(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned group);
/* length 0 = minimum */
unsigned g5p_usc_buffer_load(uint8_t *out, unsigned dst_half, unsigned byte_count,
                             unsigned base_common_dword, unsigned index, unsigned length);
unsigned g5p_usc_address_load(uint8_t *out, unsigned dst_half, unsigned byte_count,
                              unsigned address_half, unsigned length);
/* Register-indexed buffer access (gles3-tex.md §3): pair = 64-bit pointer at
 * common dword base + (index << shift), then a class-10 load from the pair. */
unsigned g5p_usc_address_add_shift(uint8_t *out, unsigned pair_half, unsigned base_common_dword,
                                   unsigned index_half, unsigned shift, bool kill, bool vs);
unsigned g5p_usc_vertex_address_load(uint8_t *out, unsigned dst_half, unsigned byte_count,
                                     unsigned address_half);

/* ---- ISA round 2 (g5p_usc.py / g5p_usc_resources.py, 2026-10-23) ----
 * Control fields stay explicit (header, class extension, destination
 * control, groups, routing) exactly as in the reference; callers choose
 * the combinations the compiler corpus witnessed. route < 0 = no routing
 * parcel; long_form < 0 = the reference default. */
unsigned g5p_usc_fadd_literal_mods(uint8_t *out, unsigned dst_half, uint16_t value, unsigned b_half,
                                   unsigned bank, unsigned b_group, unsigned header, bool negate_b,
                                   bool abs_b, bool saturate);
unsigned g5p_usc_binary_literal2(uint8_t *out, enum g5p_binop op, unsigned dst_half, uint32_t value,
                                 unsigned b_half, unsigned bank, unsigned b_group, unsigned header,
                                 enum g5p_literal type, int long_form, unsigned source_extension,
                                 unsigned dst_control, bool negate_b);
/* One 8- or 16-bit component of the half register src_half (2 * r + 1: the
 * high 16 bits) to f32; high_byte: bits 15:8 of that half (8-bit formats). */
unsigned g5p_usc_convert_small(uint8_t *out, enum g5p_small_format fmt, unsigned dst_half, unsigned src_half,
                               bool high_byte, unsigned bank, unsigned src_group, unsigned header,
                               unsigned source_extension);
/* The f16 in half register src_half to f32: -0.0 + x with an f16 B source. */
unsigned g5p_usc_half_to_float(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned bank,
                               unsigned src_group, unsigned header);
unsigned g5p_usc_convert2(uint8_t *out, enum g5p_convert op, unsigned dst_half, unsigned src_half,
                          unsigned bank, unsigned src_group, unsigned header,
                          unsigned source_extension, unsigned dst_control);
unsigned g5p_usc_unary2(uint8_t *out, enum g5p_unary op, unsigned dst_half, unsigned src_half,
                        unsigned bank, unsigned src_group, unsigned header, bool negate,
                        unsigned source_extension, unsigned dst_control, bool absolute);
/* Inline shared (common-store) sources S(s,g), s = dword 0..255. */
unsigned g5p_usc_alu_shared(uint8_t *out, enum g5p_binop op, unsigned dst_half, unsigned a_dword,
                            unsigned b_half, unsigned bank, unsigned b_group, unsigned header,
                            unsigned source_extension, unsigned dst_control, bool negate_b,
                            int route, unsigned a_shared_group);
unsigned g5p_usc_fma_shared_ac(uint8_t *out, unsigned dst_half, unsigned a_dword, unsigned b_half,
                               unsigned c_dword, unsigned bank, unsigned b_group, unsigned header,
                               unsigned source_extension, unsigned dst_control, int route,
                               unsigned a_shared_group, unsigned c_shared_group);
unsigned g5p_usc_fma_shared_a(uint8_t *out, unsigned dst_half, unsigned a_dword, unsigned b_half,
                              unsigned c_half, unsigned bank, unsigned b_group, unsigned c_group,
                              unsigned header, unsigned source_extension, unsigned dst_control,
                              int route, unsigned a_shared_group);
unsigned g5p_usc_alu_shared_pair(uint8_t *out, enum g5p_binop op, unsigned dst_half, unsigned a_dword,
                                 unsigned b_dword, unsigned bank, unsigned a_shared_group,
                                 unsigned b_shared_group);
/* min/max(shared A, B) (Apple's m_imin_u, m_umax_u). */
unsigned g5p_usc_minmax_shared(uint8_t *out, enum g5p_minmax op, unsigned dst_half, unsigned a_dword,
                               unsigned b_half, unsigned bank, unsigned b_group, unsigned header,
                               unsigned a_shared_group);
/* Relations, selections, discard. */
struct g5p_ctl {
   unsigned bank, header, source_extension, dst_control;
};
#define G5P_CTL ((struct g5p_ctl){.header = 1})
unsigned g5p_usc_compare_write(uint8_t *out, enum g5p_cmp op, unsigned dst_half, struct g5p_src a,
                               struct g5p_src b, struct g5p_src c, struct g5p_ctl k,
                               unsigned relation_control);
unsigned g5p_usc_predicated_move(uint8_t *out, unsigned dst_half, struct g5p_src s, struct g5p_ctl k);
unsigned g5p_usc_float_minmax_test(uint8_t *out, bool max, unsigned dst_half, struct g5p_src a,
                                   struct g5p_src b, unsigned bank, unsigned header);
unsigned g5p_usc_minmax_merge(uint8_t *out, unsigned dst_half, struct g5p_src s, unsigned bank,
                              unsigned header);
unsigned g5p_usc_sfu_core(uint8_t *out, enum g5p_sfu op, unsigned dst_half, struct g5p_src s,
                          struct g5p_ctl k, bool absolute);
unsigned g5p_usc_square(uint8_t *out, unsigned dst_half, struct g5p_src s, struct g5p_ctl k);
unsigned g5p_usc_discard_test(uint8_t *out, enum g5p_cmp op, unsigned dst_half, struct g5p_src a,
                              struct g5p_src b, unsigned relation_control);
unsigned g5p_usc_discard(uint8_t *out, int src_half);
/* Fragment depth replacement (A9 depth(any) compiler witnesses). */
unsigned g5p_usc_depth_feedback(uint8_t *out, unsigned src_half);
struct g5p_lit3 {
   unsigned opcode, dst_half;
   uint32_t value;
   unsigned b_half, b_group;
   bool c_literal;        /* C is c_value instead of c_half/c_group */
   unsigned c_half, c_group;
   uint32_t c_value;
   struct g5p_ctl k;
   unsigned operation_control;
   int routing;           /* < 0: none */
   int long_form, c_long_form;
   bool result_half;
};
unsigned g5p_usc_literal_ternary(uint8_t *out, const struct g5p_lit3 *t);
/* FP32 FMA: full-width literal multiplicand and addend, no rounding between. */
unsigned g5p_usc_fma_f32_literals(uint8_t *out, unsigned dst_half, unsigned b_half,
                                   uint32_t multiplier, uint32_t addend, unsigned bank);
unsigned g5p_usc_repeat_fma(uint8_t *out, unsigned dst_half, struct g5p_src a, struct g5p_src b,
                            struct g5p_ctl k);
unsigned g5p_usc_predicated_sfu(uint8_t *out, unsigned dst_half, struct g5p_src s, unsigned bank,
                                unsigned header);
unsigned g5p_usc_bool_to_half(uint8_t *out, unsigned dst_half, unsigned src_half);
unsigned g5p_usc_cond_select(uint8_t *out, unsigned dst_half, struct g5p_src cond, struct g5p_src t,
                             struct g5p_src f, unsigned bank, unsigned source_extension, unsigned header);
/* Raster position, tile access and native-f16 blend arithmetic. */
unsigned g5p_usc_position_xy(uint8_t *out, unsigned dst_half, unsigned words, unsigned source_half);
unsigned g5p_usc_position_zw(uint8_t *out, unsigned dst_half, bool z, bool both, int long_form);
unsigned g5p_usc_tile_read(uint8_t *out, unsigned dst_half, unsigned words);
unsigned g5p_usc_half_binary(uint8_t *out, bool mul, unsigned dst_half, struct g5p_src a, struct g5p_src b,
                             struct g5p_ctl k, int source_control);
unsigned g5p_usc_half_fma(uint8_t *out, unsigned dst_half, struct g5p_src a, struct g5p_src b,
                          struct g5p_src c, struct g5p_ctl k, unsigned source_control);
unsigned g5p_usc_half_one_minus(uint8_t *out, unsigned dst_half, unsigned src_half);
unsigned g5p_usc_tile_component(uint8_t *out, unsigned dst_half, unsigned src_half);

/* Native-f16 blend-epilogue forms of Apple's 45 blend probes (one Metal
 * blend setting each; g5p_usc.py, docs/gpu/experiments/blend-forms.md).
 * b0 = 0x88 | (sources - 1) | class extension 4; a literal or common-store
 * source's own parcel follows the parcel holding its selector; an operation
 * control with bit 0 set continues to a [routing, source control] parcel.
 * Source control bit 7/6/5 (slot A/B/C) is set exactly where that slot reads
 * a common-store dword or a tile component not preceded by the -0.0 step.
 * Source modifiers (neg/abs in struct g5p_src) are refused: these forms
 * carry their negations as named arguments. Return bytes, 0 on refusal. */
/* `1c 11 DD 29 CC 00`: one word from component CC (only 2: alpha in DD+1). */
unsigned g5p_usc_tile_read_component(uint8_t *out, unsigned dst_half, unsigned words,
                                     unsigned first_component);
/* tile_component with destination control 0 or 0x80. */
unsigned g5p_usc_tile_component2(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned dst_control);
/* half_binary plus source control 0xc0 and negate_b (control bit 5:
 * Subtract / ReverseSubtract). source_control < 0: no control parcel. */
unsigned g5p_usc_half_binary2(uint8_t *out, bool mul, unsigned dst_half, struct g5p_src a,
                              struct g5p_src b, struct g5p_ctl k, int source_control, bool negate_b);
/* half_fma with source control 0x80 or none (< 0): `... 0c 00`. */
unsigned g5p_usc_half_fma2(uint8_t *out, unsigned dst_half, struct g5p_src a, struct g5p_src b,
                           struct g5p_src c, struct g5p_ctl k, int source_control);
/* x * x: `88 HH DD DB SS 01 08 01 30 CC`, CC 0 or 0xc0. */
unsigned g5p_usc_half_square(uint8_t *out, unsigned dst_half, struct g5p_src s, struct g5p_ctl k,
                             unsigned source_control);
/* f16 literal A + B: `89 HH DD DB` split literal, `04 CC 70 00`; negate_b =
 * CC bit 7 (0x3c00 - B: OneMinus factors), control 0 or 2 (0x8000 + B:
 * the tile step). g5p_usc_half_one_minus is the 8d/0x41/0x80/group-0 case. */
unsigned g5p_usc_half_binary_literal(uint8_t *out, unsigned dst_half, uint16_t value, struct g5p_src b,
                                     struct g5p_ctl k, bool negate_b, unsigned control);
/* Blend-colour component c[a_dword] (common store) op B:
 * `89 HH DD DB SA BB (40|hi) 41 OP CC RR SC`; routing 0/0x70, source
 * control 0x80/0xc0. */
unsigned g5p_usc_half_binary_shared(uint8_t *out, bool mul, unsigned dst_half, unsigned a_dword,
                                    unsigned a_shared_group, struct g5p_src b, struct g5p_ctl k,
                                    bool negate_b, unsigned routing, unsigned source_control);
/* c[a_dword] into a half: `88 HH DD DB SA 01 (40|hi) 41 28 01 39 00`. */
unsigned g5p_usc_half_move_shared(uint8_t *out, unsigned dst_half, unsigned a_dword,
                                  unsigned a_shared_group, struct g5p_ctl k);
/* a_value * B + c_value, both short f16 literals (Apple: 0xbc00 * x + 0x3c00);
 * `... 0c 01 RR SC` with routing 0x70 and source control 0x80. */
unsigned g5p_usc_half_fma_literal(uint8_t *out, unsigned dst_half, uint16_t a_value, struct g5p_src b,
                                  uint16_t c_value, struct g5p_ctl k, unsigned routing,
                                  unsigned source_control);
/* min/max `... OP (RR|1) 0a 00`: min = 0x30/0x20, max = 0x38/0x20 or, with
 * opcode 0x30, 0x30/0x80 (the alpha-lane pair). opcode 0: 0x30 min, 0x38 max. */
unsigned g5p_usc_half_minmax(uint8_t *out, bool max, unsigned opcode, unsigned dst_half, struct g5p_src a,
                             struct g5p_src b, struct g5p_ctl k);
/* The alpha-lane pair's second instruction `... 28 21 3a 00`: A = the
 * min/max result, B = that min/max's A. */
unsigned g5p_usc_half_minmax_fixup(uint8_t *out, unsigned dst_half, struct g5p_src a, struct g5p_src b,
                                   struct g5p_ctl k);
/* A + B * (C - A) `... 14 (08|1) 00 SC`; negate_c: A + B * (-C - A). */
unsigned g5p_usc_half_lerp(uint8_t *out, unsigned dst_half, struct g5p_src a, struct g5p_src b,
                           struct g5p_src c, struct g5p_ctl k, bool negate_c, unsigned source_control);
/* A*B + C*D (0x10) or A*B - C*D (0x12, subtract); class 0x8b, source
 * control 0x20/0x80 or none (< 0). */
unsigned g5p_usc_half_dot2(uint8_t *out, unsigned dst_half, struct g5p_src a, struct g5p_src b,
                           struct g5p_src c, struct g5p_src d, struct g5p_ctl k, bool subtract,
                           int source_control);

/* Round 19 (g5p_recipes.c): Apple's complete reference lowerings with
 * explicit half indices (d = result; x = input, d == x except rsqrt;
 * y = second input of min/max; t = scratch) and the premultiplied blend
 * epilogue from the tile read to the last FMA. Return bytes, 0 on failure. */
enum g5p_recipe {
   G5P_RC_RCP, G5P_RC_RSQ, G5P_RC_SQRT, G5P_RC_SIGN, G5P_RC_MIN, G5P_RC_MAX, G5P_RC_CEIL, G5P_RC_TRUNC,
   G5P_RC_RINT, G5P_RC_SIN, G5P_RC_COS, G5P_RC_COUNT
};
extern const char *const g5p_recipe_name[G5P_RC_COUNT];
/* flags: an operand the recipe reads is read again afterwards (the group
 * bit of its last read in the recipe is then 0, as in Apple's programs). */
#define G5P_RC_X_LIVE 1u /* rsqrt: the input */
#define G5P_RC_Y_LIVE 2u /* min/max: the second input */
#define G5P_RC_VERTEX 4u /* vertex program: temps in bank 1 (rcp, rsqrt, min, max only) */
unsigned g5p_recipe(enum g5p_recipe r, uint8_t *out, unsigned max, unsigned d, unsigned x, unsigned y,
                    unsigned t, unsigned flags, const char **err);
unsigned g5p_blend_epilogue(uint8_t *out, unsigned max);

/* General blend epilogue (g5p_blend.c, hyprland-gpu.md §11.2 stage 2): the
 * colour in halves 0..3 (saturated f16) -> op(src * Fs, dst * Fd) per
 * channel, VkBlendFactor/VkBlendOp numbering, blend constants as four f32
 * common dwords from const_base. Channels in zero_lanes blend to constant 0:
 * the caller zeroes them in the source first (the epilogue passes them
 * through). Returns bytes (0 with *err NULL: no epilogue needed), 0 with *err
 * set on refusal. */
struct g5p_blend_key {
   uint8_t src_rgb, dst_rgb, op_rgb, src_a, dst_a, op_a;
};
struct g5p_blend_info {
   unsigned max_half;   /* highest half written or read */
   bool reads_tile;     /* the program reads the destination */
   bool uses_constants; /* reads common const_base..+3 */
   unsigned zero_lanes; /* channels the caller sets to 0 */
};
unsigned g5p_blend_general(uint8_t *out, unsigned max, const struct g5p_blend_key *k, unsigned const_base,
                           struct g5p_blend_info *info, const char **err);
/* Disabled channels retain the destination after the blend equation. */
unsigned g5p_blend_masked(uint8_t *out, unsigned max, const struct g5p_blend_key *k, unsigned const_base,
                        unsigned write_mask, struct g5p_blend_info *info, const char **err);

/* ---- Vertex-stage memory stores and integer forms (xfb.md, XFB probes) ----
 * Byte templates from Apple's applegpu_g5p vertex functions (tools/
 * mesa-a9-xfb-re/probes.metal); only witnessed field values are accepted.
 *
 * 64-bit address add: dst:dst+1 = shared pointer pair + (u32 offset << shift),
 * `89 11 DD 03 SS OO XX 41 a1 01 00 (shift<<5|6)` (v_st_vid, v_st_xfb, ...). */
unsigned g5p_usc_address_add(uint8_t *out, unsigned dst_half, unsigned ptr_dword, unsigned ptr_group,
                             unsigned off_half, unsigned off_group, unsigned shift);
/* Store of 4/8/12/16 bytes from consecutive registers at the 64-bit address
 * in addr_half plus an immediate dword offset (4-byte stores only: 1 in
 * v_st_bytes, 2 in v_st_stride12, 3 in v_st_off12):
 * `10 (0x23+8(n-1)) (0x80|data_half) (addr_half<<1|1) OO 88`.
 * Vertex tasks commit only the 4-byte form (xfb.md, device run 2). */
/* Host fragment store predicate and class70/71 memory side effect;
 * helper-lane interpretation is inferred, pending graphics-storage rungs. */
unsigned g5p_usc_fs_store_predicate(uint8_t *out);
unsigned g5p_usc_fs_store(uint8_t *out,unsigned bytes,unsigned data_half,unsigned addr_half,unsigned slot);
unsigned g5p_usc_global_store(uint8_t *out, unsigned bytes, unsigned data_half, unsigned addr_half,
                              unsigned dword_offset);
/* Vertex integer add of a literal: short prefix + `a1 09 70 00` up to 0xffff
 * (literal bit 9 in header bit 5), long prefix + `a1 01 70 00` above
 * (v_iadd_lit, v_iadd_lit_mid, v_iadd_lit_big). */
unsigned g5p_usc_iadd_literal(uint8_t *out, unsigned dst_half, uint32_t value, unsigned b_half,
                              unsigned bank, unsigned b_group);
/* dst = src << shift, Apple's vertex idiom for a power-of-two multiply:
 * `99 11 DD 03 01 SS 20 01 a1 05 00 (shift<<5)` (v_shl4, v_imul_lit*). */
unsigned g5p_usc_shl_literal(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned bank,
                             unsigned src_group, unsigned shift);
/* Register move with destination control (v_st_gather: `98 21 DD 83 SS 01 54 01 38 00`). */
unsigned g5p_usc_move2(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned bank,
                       unsigned src_group, unsigned header, unsigned dst_control);
/* Control flow (from the G9P encoders, d91427625ff; g5p_usc.c):
 * the execution-mask stack in the upper half of register st, the compares
 * that set the relation it reads, and jumps (byte offset from the jump's
 * start; any: back while some lane runs, else forward when none does).
 * imm_src: the program's first push or loop reads 0 instead of the register
 * (st < 128; above r31 the source takes an extension word); last:
 * the program's last stack operation (Apple's last-read bit). invert: push
 * keeps the lanes whose compare failed; break takes the lanes whose
 * compare passed. */
enum g5p_stack_op { G5P_ST_PUSH, G5P_ST_ELSE, G5P_ST_POP, G5P_ST_LOOP, G5P_ST_BREAK };
unsigned g5p_usc_stack(uint8_t *out, enum g5p_stack_op op, unsigned st, bool imm_src, bool invert,
                       unsigned levels, bool last);
unsigned g5p_usc_jump(uint8_t *out, bool any, int offset);
unsigned g5p_usc_jump_patch(uint8_t *jump, int offset);
unsigned g5p_usc_cf_compare(uint8_t *out, enum g5p_cmp op, unsigned a, unsigned a_group, unsigned b,
                            unsigned b_group);
unsigned g5p_usc_cf_compare_imm(uint8_t *out, enum g5p_cmp op, unsigned reg, unsigned group, unsigned imm);

/* Registers above r31 (from the G9P encoders, d1ab45daed3; g5p_usc.c):
 * the two-source ALU form with any of r0..r127, fragment bank. A source above
 * r31 takes the extension word. op: the binop code. */
unsigned g5p_usc_alu2_wide(uint8_t *out, unsigned op, unsigned dst, unsigned a, unsigned a_group, unsigned b,
                           unsigned b_group);

/* Derivatives (Apple's A9 dfdx/dfdy, tools/mesa-a9-hypr-re f_ddx_*, f_ddy_*):
 * the class-98 unary form with op 44 and a modifier word, 0200 across x and
 * 0400 down y; class 9c (source_extension 4) when the source is read again.
 * Metal has one derivative per axis: fine and coarse requests take it. */
unsigned g5p_usc_derivative(uint8_t *out, bool y, unsigned dst_half, unsigned src_half, unsigned src_group,
                            unsigned header, unsigned source_extension, unsigned dst_control);

/* Exactly one instruction under the parcel continuation rule. */
bool g5p_usc_is_one_instruction(const uint8_t *raw, unsigned n);

#endif
