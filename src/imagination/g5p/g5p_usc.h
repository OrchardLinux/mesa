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

/*
 * Encode for the A8 (G4P) instead of the A9: set once by the driver for an A8
 * device (one process drives one Apple GPU). The forms that differ, from
 * Apple's A8 compiler output (iPod touch 6, iOS 12.4, a8-shadercap): emitpix
 * takes an extension halfword, varyings iterate with opcode 0x18, and texture
 * samples have another form (refused until decoded). Everything else Apple's
 * A8 compiler emits the A9's way (the vertex programs byte for byte).
 */
extern bool g5p_usc_g4p;
/*
 * On the A8, the A9's instruction where the A8 form captured from Apple cannot express the operands (today: a
 * texture sample whose image is not at common qword 0, or whose sampler is beyond common dword 7). Set by the
 * driver for an A8 (PVR_A8_A9_FORMS, default on). These are A9 instructions no A8 capture has witnessed.
 */
extern bool g5p_usc_g4p_a9_forms;
/*
 * texelFetch on the A8: a fragment 2D fetch is drawn as the A8's sample with a point, non-normalized, clamped copy of
 * the texture's sampler, as Apple's read (an iOS capture of a Metal read()): the integer coordinate as a float,
 * byte 6 bit 2 set (g5p_usc_g4p_texfetch = 2; 1, the texel's centre without the bit, reads the same on the iPod
 * touch 6). A constant level above 0 (the sampler's LOD range clamped to it) only with PVR_A8_TEXFETCH_LEVEL:
 *   1 ("level"): the coordinate in the level's texels; 2 ("scale"): in level 0's texels (times 2^level);
 *   3 ("apple"): every fetch in Apple's read form (AGXCompilerCore's read(): an explicit level, 0 when the source
 *   names none): byte 6 0xac (mode 2 + bit 2), the level as an f32 in the register after the coordinate (byte 7
 *   0x09, as in an iOS capture of a Metal sample at a varying level), through Apple's constant read sampler
 *   (normalized, clamped to the edge, point, LOD 0..max; PCO_G5P_UPLOAD_SAMPLER_READ), the coordinate unscaled.
 *   Not yet run on the hardware.
 */
extern int g5p_usc_g4p_texfetch;
extern int g5p_usc_g4p_texfetch_level;
/*
 * Vertex-stage textures on the A8, refused by default. PVR_A8_VTEX=1 takes the A9's qualified vertex recipe over the
 * A8's four-dword image and sampler words. It does not work yet: on the iPod touch 6 a vertex texelFetch read zeros
 * and a vertex textureLod locked the TA. No capture shows an A8 vertex texture; the A8's vertex programs are
 * otherwise the A9's.
 */
extern bool g5p_usc_g4p_vtex;
/*
 * PVR_A8_VTEX_APPLE=1 (not yet run on the hardware): vertex textures in Apple's form instead (it enables them on its
 * own), as an iOS capture shows it for AGXMetalA8's vertex programs (a sample, a read of RGBA32Float and of
 * RGBA32Sint): the A8's 12-byte sample with the coordinate and the result in bank 1 and Apple's constant level 0,
 * 1a 7d 00 41 00 01 a8|ac|a4 05 00 01 1c 80 (g5p_tex.vtx), the image and sampler at common dwords 0 and 4; a read's
 * sampler is the fetch sampler's words, as Apple's (0x0202240000000fff). The A9's recipe (PVR_A8_VTEX=1) uses the A9's
 * own sample instruction (1a f5 .. 01 08 85 c6 01 80 81, a coupled f16 level), which no A8 program contains.
 */
extern bool g5p_usc_g4p_vtex_apple;
/* A fragment texel-buffer read on the A8 (st/mesa's pixel-buffer uploads) as a 2D texelFetch of the buffer's linear
 * state (PVR_A8_TBO, on by default; 0 refuses it) */
extern bool g5p_usc_g4p_tbo;

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
/* G5P_TEX_2D_MS (code 0xf9): the 2D-multisample fetch, its sample operand in the register after the
 * coordinates (Apple's 4x LOAD background and Metal texture2d_ms probes); only the narrow
 * sample (g5p_usc_sample()) takes it. */
enum g5p_tex_dim { G5P_TEX_2D, G5P_TEX_2D_ARRAY, G5P_TEX_3D, G5P_TEX_CUBE, G5P_TEX_2D_MS, G5P_TEX_DIM_COUNT };
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
 * the second dependent-read counter). The A8 form is Apple's `18 09 DD 01 AA
 * 01 60 00` (f_iter), on the first counter only. */
unsigned g5p_usc_coeff_load(uint8_t *out, unsigned dst_half, unsigned coeff_dword, bool drc1);
unsigned g5p_usc_coord_iterator(uint8_t *out, unsigned dst_half, unsigned slot, unsigned count);
unsigned g5p_usc_emit(uint8_t *out, unsigned words, unsigned source_half);
/* Total word count of a 1/2/4-target fragment (2/4/8; 8 encodes as field 0). */
unsigned g5p_usc_emit_mrt(uint8_t *out, unsigned words, unsigned source_half);
unsigned g5p_usc_vout(uint8_t *out, unsigned src_half, unsigned slot, bool temp, bool last);
unsigned g5p_usc_wait(uint8_t *out, unsigned control);
/* Device memory fence on the 64-bit operand in r[2 pair]:r[2 pair + 1], in
 * the form Apple uses for each kind of program (g5p_usc.c). */
enum g5p_fence_form {
   G5P_FENCE_FORM_KERNEL,        /* compiler-built kernels */
   G5P_FENCE_FORM_FLUSH_PROGRAM, /* the compute flush program */
};
unsigned g5p_usc_mem_fence(uint8_t *out, unsigned pair, enum g5p_fence_form form);
/* Zero-word emit; the code is control bits 4-7. */
enum g5p_emit_code {
   G5P_EMIT_CODE_FLUSH_PROGRAM = 0xb,
};
unsigned g5p_usc_emit_code(uint8_t *out, enum g5p_emit_code code);

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
   /* The A8's multi-image form (Apple's, from an iOS capture): byte 4 = the image's common dword * 2, the
    * sampler's four-dword slot in byte 1 bits 7:6 and byte 5 bits 2:1. image_qword and sampler_qword then hold
    * dwords. */
   bool a8_slots;
   /* g5p_usc_g4p_texfetch = 2: Apple's read flag, byte 6 bit 2 */
   bool read_flag;
   /* Apple's explicit level in a register (an iOS capture of a Metal sample at a varying level:
    * 1a 59 00 41 00 01 a8 09 00 01 00 80): mode 2, byte 7 0x09, the level an f32 in the register after the
    * coordinate's two */
   bool lod_reg;
   /* Apple's vertex-stage sample (an iOS capture, AGXMetalA8's v_sample / v_read_f32 / v_read_i32:
    * 1a 7d 00 41 00 01 a8|ac|a4 05 00 01 1c 80): the coordinate and the result in bank 1 (byte 1 bit 2, byte 7
    * bit 2), mode 2 with Apple's constant level 0 (byte 10 0x1c); byte 6 0xa8 for a sample, 0xac for a read of a float
    * or normalized texture, 0xa4 (int_result) for a read of an integer one */
   bool vtx;
   bool int_result;
   unsigned lod_common; /* vtx: the common dword holding the f32 level (byte 10 = its half index, 2 * dword) */
};
unsigned g5p_usc_sample(uint8_t *out, const struct g5p_tex *t);
/* The 2D-multisample fetch's sample operand "the task's own sample" (a sample-rate program reads the
 * sample it shades): a 12-byte move of the special value with that index into dst_half, bank 0 (the fields
 * are listed at the encoder). It is not g5p_usc_immediate()'s move: its source kind is "special value" where
 * the immediate's is "literal". Witnesses: the 4x LOAD background (an iOS capture of a 4x colour load, half
 * 4) and texture2d_ms::read(c, sample_id) (a Metal probe, half 12, which the fetch reads unchanged). */
unsigned g5p_usc_ms_own_sample(uint8_t *out, unsigned dst_half);
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
/* Complete native recipes only. Independent continuation-field choices
 * are unqualified and are deliberately not exposed by this interface. */
enum g5p_tex_tuple {
   G5P_TEX_TUPLE_ORDINARY,
   G5P_TEX_TUPLE_COUPLED_SHARED_LEVEL,
   G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL,
   G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL_OFFSET,
   G5P_TEX_TUPLE_COUPLED_GRAD,
   G5P_TEX_TUPLE_COUPLED_GRAD_OFFSET,
   G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL_OFFSET_3D,
   G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL_COMPARE,
   G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL_COMPARE_OFFSET,
};
struct g5p_tex_ext {
   unsigned dst_half, image_qword, sampler_qword, coord_dword;
   enum g5p_tex_dim dim;
   unsigned mask;          /* result components: 0xf, or 0x1 for a compare */
   bool read;              /* integer coordinates, u32 level */
   enum g5p_tex_lod lod;   /* source in temp dword lod_dword */
   unsigned lod_dword;
   bool compare, offset;   /* extras at half extras_half: [ref f32][offset u16] */
   unsigned extras_half;
   enum g5p_tex_tuple tuple;
   bool lod_shared; /* prefix level source: common store instead of register */
   bool vertex;     /* a vertex-stage sample (the A8 takes the A9's recipe there, g5p_usc_g4p_vtex) */
};
/* A9 compute's register-level array or cube sample and fetch: the coupled temporary-level tuple, the fetch's
 * coordinate kept live (fields at the encoder). */
unsigned g5p_usc_sample_register_lod(uint8_t *out, const struct g5p_tex_ext *t);
unsigned g5p_usc_sample_ext(uint8_t *out, const struct g5p_tex_ext *t);
/* 16-bit immediate into one half (class 0x88, the 32-bit move's split map;
 * Apple's texel-offset probes witness bits 0-9). */
unsigned g5p_usc_half_immediate(uint8_t *out, unsigned dst_half, uint16_t value);
/* Native v_2d_attribute_{pos,neg}: packed offset into a bank1 half. */
unsigned g5p_usc_vertex_half_immediate(uint8_t *out, unsigned dst_half, uint16_t value);
/* Packed XYZ nibbles, all12 value bits and both destination banks witnessed. */
unsigned g5p_usc_half_immediate_xyz(uint8_t *out, unsigned dst_half, uint16_t value, unsigned bank);
/* Cube direction (x, y, z) in temps -> the cube sample's coordinates, by the
 * iterator unit: `08 7d DH SH 00 80`, SH = source half << 1 | 1 (Apple's
 * f_cube_alu, f_cube_hi; a cube sample of raw directions reads them as 2D on
 * face 0, device 2026-11-02). */
unsigned g5p_usc_vertex_texture_level(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned group);
unsigned g5p_usc_vertex_array_layer(uint8_t *out, unsigned dst_half, unsigned src_half);
/* Native cube direction is fixed to bank0; this form writes bank1.
 * bit2 is the output-bank selector, not a direction-bank selector. */
unsigned g5p_usc_cube_coords_bank1(uint8_t *out, unsigned dst_half, unsigned src_half);
unsigned g5p_usc_cube_coords(uint8_t *out, unsigned dst_half, unsigned src_half);
/* Fixed bank0 r0..5 gradients + r6..8 direction -> r0..2 coordinates,
 * r3..8 projected gradients. No independent relocation is qualified. */
unsigned g5p_usc_cube_gradient_tuple(uint8_t *out);
/* f32 -> u16 (truncating) into one half: Apple's array-layer conversion
 * `89 01 DH 01 09 SS 20 01 60 06`. */
unsigned g5p_usc_f2u16_half(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned group);
/* One byte of a packed UNORM8 colour: the f32 in common-store dword src_dword as an unsigned normalized byte in
 * byte 0 or 1 of dst_half, bank 0; the byte-1 form keeps byte 0 of dst_half (the fields are listed at the
 * encoder). The clear of 5..8 colour targets (g5p_mrt_programs.h). */
/* Common-store dword src_dword (group 1) into the 16-bit dst_half of bank 0 (fields at the encoder): the
 * end-of-tile program's state moves (g5p_mrt_programs.h). */
unsigned g5p_usc_mov_shared_half(uint8_t *out, unsigned dst_half, unsigned src_dword);
unsigned g5p_usc_unorm8_byte(uint8_t *out, unsigned dst_half, unsigned byte, unsigned src_dword);
/* Integer min/max of a short literal and common-store dword c_dword (group 1) into the 32-bit register at
 * dst_half, bank 0 (fields at the encoder): the A8's RGBA8UI clear clamps each channel with umin(255, c). */
unsigned g5p_usc_minmax_literal_shared(uint8_t *out, enum g5p_minmax op, unsigned dst_half, unsigned dst_control,
                                       uint32_t value, unsigned c_dword);
/* The 16-bit dst_half of bank 0 = half b_half | half c_half << 8 (fields at the encoder): the A8's RGBA8UI clear
 * packs two clamped channels into one half with it. */
unsigned g5p_usc_pack_byte_half(uint8_t *out, unsigned dst_half, unsigned b_half, unsigned c_half,
                                unsigned shift);
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
/* The discard test at 4 raster samples: A != 0.0 writes the kill's sample mask 15 (all four samples) into the
 * destination, where the one-sample test writes 1 (our one-sample g5p_usc_discard_test() kills where A != 0.0
 * too). Apple's compiler, the same fragment program under raster_sample_count 4 (if (v != 0.0f)
 * discard_fragment(), v in h0 and, read again, h4): a packed ternary with the 16-bit result, -0.0f as a long
 * literal A, v as B and the mask 15 as a short literal C, the float-equality opcode and the != relation (the
 * fields are listed at the encoder). At one sample the same compiler writes the mask 1 with a two-source form.
 * With the == relation the form killed the samples it should keep on the device. The kill that follows is
 * g5p_usc_discard() of the destination; an unconditional discard is g5p_usc_discard(-1) at 4 samples as at one. */
unsigned g5p_usc_discard_test_ms4(uint8_t *out, unsigned dst_half, struct g5p_src a);
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
unsigned g5p_usc_position_zw(uint8_t *out, unsigned dst_half, bool set1, bool both, int long_form);
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
/* `1c (words<<4|1) DD 29 CC 00`: words from f16 component CC;
 * ios-witness-r1 mrtm-blend1/3bm select components4/6 across MRTs. */
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

unsigned g5p_blend_target(uint8_t *out, unsigned max, const struct g5p_blend_key *k,
                          unsigned const_base, unsigned mask, unsigned target, unsigned targets,
                          struct g5p_blend_info *info, const char **err);

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
 * helper-lane interpretation is inferred, pending graphics-storage cases. */
unsigned g5p_usc_fs_store_predicate(uint8_t *out);
unsigned g5p_usc_fs_store(uint8_t *out,unsigned bytes,unsigned data_half,unsigned addr_half,unsigned slot);
unsigned g5p_usc_global_store(uint8_t *out, unsigned bytes, unsigned data_half, unsigned addr_half,
                              unsigned dword_offset);
/* Vertex integer add of a literal: short prefix + `a1 09 70 00` up to 0xffff
 * (literal bit 9 in header bit 5), long prefix + `a1 01 70 00` above
 * (v_iadd_lit, v_iadd_lit_mid, v_iadd_lit_big). */
unsigned g5p_usc_iadd_literal(uint8_t *out, unsigned dst_half, uint32_t value, unsigned b_half,
                              unsigned bank, unsigned b_group);
unsigned g5p_usc_iadd_literal2(uint8_t *out, unsigned dst_half, uint32_t value, unsigned b_half,
                               unsigned bank, unsigned b_group, unsigned header);
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
/* The vertex form: A | B with both sources in bank 1 (both bank bits in the header), any of r0..r127 (above r31 the extension
 * word), into dst_half of bank dst_bank. Apple's A9 vertex programs read attributes above r31 in place this way
 * (Apple's compiler, 44 attribute dwords: r44 <- r40 | r36 with the extension word). With a == b it copies
 * a register (the G9P home move). */
unsigned g5p_usc_vs_or_wide(uint8_t *out, unsigned dst_half, unsigned dst_bank, unsigned a, unsigned a_group,
                            unsigned b, unsigned b_group);

/* Derivatives (Apple's A9 dfdx/dfdy, tools/mesa-a9-hypr-re f_ddx_*, f_ddy_*):
 * the class-98 unary form with op 44 and a modifier word, 0200 across x and
 * 0400 down y; class 9c (source_extension 4) when the source is read again.
 * Metal has one derivative per axis: fine and coarse requests take it. */
unsigned g5p_usc_derivative(uint8_t *out, bool y, unsigned dst_half, unsigned src_half, unsigned src_group,
                            unsigned header, unsigned source_extension, unsigned dst_control);

/* Exactly one instruction under the parcel continuation rule. */
bool g5p_usc_is_one_instruction(const uint8_t *raw, unsigned n);

#endif
