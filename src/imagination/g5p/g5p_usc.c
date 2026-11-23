/* SPDX-License-Identifier: MIT
 * G5P USC field encoders, ported from tools/gpu-userspace/g5p_usc.py and
 * g5p_usc_resources.py. Field checks follow the reference one for one: a
 * value the reference rejects with ValueError returns 0 here.
 */
#include "g5p_usc.h"
#include "g5p_mrt_out.h"

#include <string.h>

_Thread_local const char *g5p_usc_error;
bool g5p_usc_g4p;
bool g5p_usc_g4p_a9_forms;
int g5p_usc_g4p_texfetch;
int g5p_usc_g4p_texfetch_level;
bool g5p_usc_g4p_vtex;
bool g5p_usc_g4p_vtex_apple;
bool g5p_usc_g4p_tbo;
bool g5p_usc_g4p_fge_swap;
bool g5p_usc_g4p_pos_iter;
bool g5p_usc_g4p_r8_load;

const char *const g5p_binop_name[G5P_BIN_COUNT] = {"fadd", "fmul", "iadd", "imul", "and", "or", "xor", "ishr",
                                                       "isub"};
const char *const g5p_minmax_name[G5P_MINMAX_COUNT] = {"umin", "umax", "imin", "imax"};
const char *const g5p_convert_name[G5P_CONVERT_COUNT] = {"f2i32", "f2u32", "i2f32", "u2f32", "f2i32_rte"};
const char *const g5p_unary_name[G5P_UNARY_COUNT] = {"floor", "log2", "exp2", "not", "popcount", "bitreverse"};
const char *const g5p_tex_dim_name[G5P_TEX_DIM_COUNT] = {"2d", "2d_array", "3d", "cube", "2d_ms"};
const char *const g5p_cmp_name[G5P_CMP_COUNT] = {"flt", "fgt", "feq", "ilt", "igt", "ult", "ugt", "ieq"};
const char *const g5p_sfu_name[G5P_SFU_COUNT] = {"rcp_approx", "rsqrt_approx", "sin_core", "cos_core"};

static const uint8_t binop_code[G5P_BIN_COUNT] = {0x44, 0x48, 0xa1, 0xa2, 0xc8, 0xce, 0xc6, 0xfc, 0xa5};
static const uint8_t minmax_code[G5P_MINMAX_COUNT] = {0xb2, 0xb3, 0xba, 0xbb};
static const uint8_t convert_tag[G5P_CONVERT_COUNT] = {0x13, 0x11, 0x17, 0x15, 0x17};
static const uint8_t convert_code[G5P_CONVERT_COUNT] = {0x60, 0x60, 0x68, 0x68, 0x60};
static const uint8_t cmp_code[G5P_CMP_COUNT] = {0x59, 0x5d, 0x55, 0xbe, 0xbf, 0xb6, 0xb7, 0xbd};
static const uint8_t sfu_code[G5P_SFU_COUNT] = {0x80, 0x81, 0x82, 0x86};
static const uint8_t unary_code[G5P_UNARY_COUNT] = {0x40, 0x84, 0x85, 0xc5, 0xe8, 0xe4};
static const uint8_t dim_code[G5P_TEX_DIM_COUNT] = {0xf5, 0xf7, 0xfb, 0xfd, 0xf9};
static const uint8_t mov_trailer[4] = {0x54, 0x01, 0x38, 0x00};

/* Split immediate map (g5p_isa.IMM_MAP): bit -> (byte, bit). */
static const uint8_t imm_byte[32] = {4, 4, 4, 4, 4, 4, 6, 6, 6, 1, 6, 7, 7, 7, 6, 6,
                                     8, 8, 8, 8, 8, 8, 8, 8, 7, 9, 9, 9, 9, 9, 9, 9};
static const uint8_t imm_bit[32] = {1, 2, 3, 4, 5, 6, 1, 2, 3, 5, 4, 4, 5, 6, 6, 7,
                                    0, 1, 2, 3, 4, 5, 6, 7, 7, 1, 2, 3, 4, 5, 6, 7};

#define REJECT(why) do { g5p_usc_error = (why); return 0; } while (0)
#define CHECK(ok, why) do { if (!(ok)) REJECT(why); } while (0)

bool g5p_usc_is_one_instruction(const uint8_t *raw, unsigned n)
{
   if (!n || n % 2 || n > G5P_USC_MAX)
      return false;
   for (unsigned i = 1; i + 2 < n; i += 2)
      if (!(raw[i] & 1))
         return false;
   return !(raw[n - 1] & 1);
}

static unsigned finish(uint8_t *out, const uint8_t *raw, unsigned n)
{
   CHECK(g5p_usc_is_one_instruction(raw, n), "encoding does not form exactly one instruction");
   memcpy(out, raw, n);
   g5p_usc_error = NULL;
   return n;
}

/* An A8 form: the A9's one-instruction length rule is not established for it */
static unsigned finish_g4p(uint8_t *out, const uint8_t *raw, unsigned n)
{
   memcpy(out, raw, n);
   g5p_usc_error = NULL;
   return n;
}

/*
 * The A8 varying iterator: 0x18, (count % 4) << 3 | 1, the destination half,
 * 0x81 (0xa1 for four components), slot << 2, then 0 (32-bit) or, for the
 * narrow (16-bit) form, 01 04 00. Apple's A8 output (a8-shadercap, iOS 12.4):
 * 18 09 04 81 10 00 (one f32 component at slot 4 into half 4), 18 11 00 81 04
 * 00 (two f32), 18 19 00 81 04 01 04 00 (three f16), 18 01 00 a1 04 01 04 00
 * (four f16), where the A9 writes 08 0b 04 08, 08 13 00 03 00 80, 0a 1b 00 02
 * and 0a 03 00 02. Only those shapes are accepted; the narrow form only at
 * slot 1, the one captured (its 0x04 may be the slot or a constant).
 */
static unsigned iterator_g4p(uint8_t *out, unsigned dst_half, unsigned slot, unsigned count, bool narrow,
                             bool saturate, unsigned flags)
{
   /* by name: the mode a refused shader needs */
   CHECK(!saturate, "A8 saturating iterator not captured");
   CHECK(!(flags & G5P_ITER_LINEAR), "A8 no-perspective iterator not captured");
   CHECK(!(flags & G5P_ITER_CENTROID), "A8 centroid iterator not captured");
   CHECK(!(flags & G5P_ITER_DRC1), "A8 iterator on the second dependent-read counter not captured");
   CHECK(narrow ? (count == 3 || count == 4) : (count == 1 || count == 2),
         "A8 iterator: only 1-2 f32 or 3-4 f16 components are captured");
   CHECK(!narrow || slot == 1, "A8 narrow iterator captured at slot 1 only");
   CHECK(dst_half < 256 && slot < 64, "A8 iterator destination/slot range");
   const uint8_t raw[8] = {0x18, (uint8_t)(((count % 4) << 3) | 1), (uint8_t)dst_half,
                           (uint8_t)(0x81 | (count == 4 ? 0x20 : 0)), (uint8_t)(slot << 2),
                           narrow ? 1 : 0, 0x04, 0x00};
   return finish_g4p(out, raw, narrow ? 8 : 6);
}

/* _destination / _selector / _header of the reference. */
static bool destination(uint8_t *raw, unsigned dst_half, unsigned bank)
{
   if (bank > 1) {
      g5p_usc_error = "only destination bank fields0/1 characterized";
      return false;
   }
   if (dst_half > 255) {
      g5p_usc_error = "destination half must be u8";
      return false;
   }
   raw[0] = dst_half;
   raw[1] = (bank << 1) | 1;
   return true;
}

static int selector(unsigned half, unsigned group, unsigned cont)
{
   if (half > 63) {
      g5p_usc_error = "source half must be u6";
      return -1;
   }
   if (group > 1) {
      g5p_usc_error = "source group must be u1";
      return -1;
   }
   return (half << 1) | (group << 7) | cont;
}

static bool header_ok(unsigned header)
{
   if (header > 255) {
      g5p_usc_error = "header must be u8";
      return false;
   }
   if (!(header & 1)) {
      g5p_usc_error = "header must continue to destination parcel";
      return false;
   }
   return true;
}

/* _controlled_destination: dst_control 0 or 0x80. */
static bool cdestination(uint8_t *raw, unsigned dst_half, unsigned bank, unsigned dst_control)
{
   if (dst_control != 0 && dst_control != 0x80) {
      g5p_usc_error = "unmeasured destination control";
      return false;
   }
   if (!destination(raw, dst_half, bank))
      return false;
   raw[1] |= dst_control;
   return true;
}

#define SEL(var, half, group, cont) \
   int var = selector(half, group, cont); \
   if (var < 0) return 0

unsigned g5p_usc_binary(uint8_t *out, enum g5p_binop op, const struct g5p_alu *f)
{
   CHECK(op < G5P_BIN_COUNT, "unsupported binary operation");
   bool floating = op == G5P_BIN_FADD || op == G5P_BIN_FMUL;
   CHECK(floating || !(f->a.neg || f->b.neg || f->a.abs || f->b.abs || f->saturate ||
                       f->result_half || f->source_half),
         "integer modifiers not characterized");
   CHECK((f->source_extension == 0 || f->source_extension == 4) &&
         !(f->source_extension && f->result_half), "unmeasured source class extension");
   unsigned code = binop_code[op] | (f->saturate ? 2 : 0);
   unsigned flags = (f->a.neg ? 0x80 : 0) | (f->b.neg ? 0x40 : 0);
   flags |= (f->result_half ? 2 : 0) | (f->source_half ? 0x0c : 0);
   unsigned absolute = (f->a.abs ? 0x80 : 0) | (f->b.abs ? 0x40 : 0);
   uint8_t raw[G5P_USC_MAX];
   CHECK(header_ok(f->header), g5p_usc_error);
   raw[0] = (f->result_half ? 0x89 : 0x99) | f->source_extension;
   raw[1] = f->header;
   if (!cdestination(raw + 2, f->dst_half, f->bank, f->dst_control))
      return 0;
   SEL(a, f->a.half, f->a.group, 0);
   SEL(b, f->b.half, f->b.group, 1);
   raw[4] = a;
   raw[5] = b;
   raw[6] = code;
   raw[7] = flags | (absolute ? 1 : 0);
   unsigned n = 8;
   if (absolute) {
      raw[n++] = 0;
      raw[n++] = absolute;
   }
   return finish(out, raw, n);
}

unsigned g5p_usc_fma(uint8_t *out, const struct g5p_alu *f)
{
   CHECK((f->source_extension == 0 || f->source_extension == 4) &&
         !(f->source_extension && f->result_half), "unmeasured source class extension");
   unsigned flags = (f->a.neg ? 0x80 : 0) | (f->b.neg ? 0x40 : 0) | (f->c.neg ? 0x20 : 0);
   flags |= (f->result_half ? 2 : 0) | (f->source_half ? 0x1c : 0);
   unsigned absolute = (f->a.abs ? 0x80 : 0) | (f->b.abs ? 0x40 : 0) | (f->c.abs ? 0x20 : 0);
   uint8_t raw[G5P_USC_MAX];
   CHECK(header_ok(f->header), g5p_usc_error);
   raw[0] = (f->result_half ? 0x8a : 0x9a) | f->source_extension;
   raw[1] = f->header;
   if (!cdestination(raw + 2, f->dst_half, f->bank, f->dst_control))
      return 0;
   SEL(a, f->a.half, f->a.group, 0);
   SEL(b, f->b.half, f->b.group, 1);
   SEL(c, f->c.half, f->c.group, 0);
   raw[4] = a;
   raw[5] = b;
   raw[6] = c;
   raw[7] = 1;
   raw[8] = f->saturate ? 0x4e : 0x4c;
   raw[9] = flags | (absolute ? 1 : 0);
   unsigned n = 10;
   if (absolute) {
      raw[n++] = 0;
      raw[n++] = absolute;
   }
   return finish(out, raw, n);
}

unsigned g5p_usc_minmax(uint8_t *out, enum g5p_minmax op, const struct g5p_alu *f)
{
   CHECK(op < G5P_MINMAX_COUNT, "unsupported integer min/max");
   CHECK(f->source_extension == 0 || f->source_extension == 4, "unmeasured source class extension");
   uint8_t raw[G5P_USC_MAX];
   CHECK(header_ok(f->header), g5p_usc_error);
   raw[0] = 0x99 | f->source_extension;
   raw[1] = f->header;
   if (!destination(raw + 2, f->dst_half, f->bank))
      return 0;
   SEL(a, f->a.half, f->a.group, 0);
   SEL(b, f->b.half, f->b.group, 1);
   raw[4] = a;
   raw[5] = b;
   raw[6] = minmax_code[op];
   raw[7] = 0x21;
   raw[8] = 8;
   raw[9] = 0;
   return finish(out, raw, 10);
}

static unsigned literal_prefix(uint8_t *raw, unsigned cls, unsigned dst_half, unsigned bank,
                               uint32_t value, unsigned b_half, unsigned b_group, unsigned header,
                               int long_form)
{
   CHECK(header_ok(header), g5p_usc_error);
   CHECK(!(header & 0x20), "header bit5 belongs to literal bit9");
   bool wide = long_form < 0 ? value > 0xffff : long_form;
   CHECK(wide || value <= 0xffff, "short literal exceeds16 bits");
   raw[0] = cls;
   raw[1] = header;
   if (!destination(raw + 2, dst_half, bank))
      return 0;
   raw[4] = 1;
   SEL(b, b_half, b_group, 1);
   raw[5] = b;
   raw[6] = 0x20;
   raw[7] = 1;
   unsigned n = 8;
   if (wide) {
      raw[n++] = 0;
      raw[n++] = 1;
      raw[6] |= 1;
   }
   for (unsigned bit = 0; bit < 32; bit++)
      if (imm_byte[bit] < n && (value >> bit) & 1)
         raw[imm_byte[bit]] |= 1u << imm_bit[bit];
   return n;
}

unsigned g5p_usc_binary_literal2(uint8_t *out, enum g5p_binop op, unsigned dst_half, uint32_t value,
                                 unsigned b_half, unsigned bank, unsigned b_group, unsigned header,
                                 enum g5p_literal type, int long_form, unsigned source_extension,
                                 unsigned dst_control, bool negate_b)
{
   CHECK(op < G5P_BIN_COUNT && op != G5P_BIN_ISUB && type <= G5P_LIT_F32, "unsupported literal operation/type");
   bool floating = op == G5P_BIN_FADD || op == G5P_BIN_FMUL;
   CHECK(!(type == G5P_LIT_U32 && floating), "float ALU requires literal_type f16 or f32");
   CHECK(type != G5P_LIT_F16 || (floating && value <= 0xffff && long_form <= 0),
         "f16 literal form requires short float ALU");
   CHECK(type != G5P_LIT_F32 || floating, "f32 literal requires float operation");
   CHECK((source_extension == 0 || source_extension == 4) && !(negate_b && type == G5P_LIT_U32),
         "unmeasured literal source modifier");
   uint8_t raw[G5P_USC_MAX];
   unsigned n = literal_prefix(raw, 0x99 | source_extension, dst_half, bank, value, b_half, b_group,
                               header, long_form);
   if (!n)
      return 0;
   if (!cdestination(raw + 2, dst_half, bank, dst_control))
      return 0;
   raw[n++] = binop_code[op];
   raw[n++] = (type == G5P_LIT_F16 ? 9 : 1) | (negate_b ? 0x80 : 0);
   raw[n++] = 0x70;
   raw[n++] = 0;
   return finish(out, raw, n);
}

/* Vertex standalone modifiers (vsops-probes.py v_fneg/v_fabs/v_fsat): fadd of
 * the f16 literal -0.0 and B with B's negate (byte 1 of the tail, bit 7), B's
 * absolute value (tail byte 3, bit 7) and saturate (opcode 0x46). */
unsigned g5p_usc_fadd_literal_mods(uint8_t *out, unsigned dst_half, uint16_t value, unsigned b_half,
                                   unsigned bank, unsigned b_group, unsigned header, bool negate_b,
                                   bool abs_b, bool saturate)
{
   uint8_t raw[G5P_USC_MAX];
   unsigned n = literal_prefix(raw, 0x99, dst_half, bank, value, b_half, b_group, header, -1);
   if (!n)
      return 0;
   if (!cdestination(raw + 2, dst_half, bank, 0))
      return 0;
   raw[n++] = binop_code[G5P_BIN_FADD] | (saturate ? 2 : 0);
   raw[n++] = 9 | (negate_b ? 0x80 : 0);
   raw[n++] = 0x70;
   raw[n++] = abs_b ? 0x80 : 0;
   return finish(out, raw, n);
}

unsigned g5p_usc_binary_literal(uint8_t *out, enum g5p_binop op, unsigned dst_half,
                                uint32_t value, unsigned b_half, unsigned bank,
                                unsigned b_group, unsigned header, bool f16, int long_form)
{
   return g5p_usc_binary_literal2(out, op, dst_half, value, b_half, bank, b_group, header,
                                  f16 ? G5P_LIT_F16 : G5P_LIT_U32, long_form, 0, 0, false);
}

unsigned g5p_usc_fma_literal(uint8_t *out, unsigned dst_half, uint32_t value, unsigned b_half,
                             unsigned c_half, unsigned bank, unsigned b_group, unsigned c_group,
                             unsigned header)
{
   CHECK(value <= 0xffff, "f16 literal must be u16");
   uint8_t raw[G5P_USC_MAX];
   unsigned n = literal_prefix(raw, 0x9a, dst_half, bank, value, b_half, b_group, header, 0);
   if (!n)
      return 0;
   SEL(c, c_half, c_group, 0);
   raw[n++] = c;
   raw[n++] = 1;
   raw[n++] = 0x4c;
   raw[n++] = 9;
   raw[n++] = 0x70;
   raw[n++] = 0;
   return finish(out, raw, n);
}

/* Apple's vertex m_imin_lit 991100030f812001ba217c00 (min(x, 7)), m_imax_neg
 * (long literal -3), m_umin_big (0x12345): the literal prefix, then the
 * min/max code, 0x21, 0x7c. */
unsigned g5p_usc_minmax_literal(uint8_t *out, enum g5p_minmax op, unsigned dst_half, uint32_t value,
                                unsigned b_half, unsigned bank, unsigned b_group, unsigned header)
{
   CHECK(op < G5P_MINMAX_COUNT, "unsupported integer min/max");
   uint8_t raw[G5P_USC_MAX];
   unsigned n = literal_prefix(raw, 0x99, dst_half, bank, value, b_half, b_group, header, -1);
   if (!n)
      return 0;
   raw[n++] = minmax_code[op];
   raw[n++] = 0x21;
   raw[n++] = 0x7c;
   raw[n++] = 0;
   return finish(out, raw, n);
}

/* literal - B: Apple's vertex m_isub_lit 9911000349812201a504 (100 - x),
 * m_ineg 9911000301812001a504 (0 - x), m_isub_big (0x12345 - x, long form,
 * last byte 0). */
unsigned g5p_usc_isub_literal(uint8_t *out, unsigned dst_half, uint32_t value, unsigned b_half,
                              unsigned bank, unsigned b_group, unsigned header)
{
   uint8_t raw[G5P_USC_MAX];
   unsigned n = literal_prefix(raw, 0x99, dst_half, bank, value, b_half, b_group, header, -1);
   if (!n)
      return 0;
   raw[n++] = binop_code[G5P_BIN_ISUB];
   raw[n++] = value > 0xffff ? 0 : 4;
   return finish(out, raw, n);
}

unsigned g5p_usc_shift(uint8_t *out, bool left, unsigned dst_half, unsigned value_half,
                       unsigned amount, bool immediate, unsigned bank, unsigned value_group,
                       unsigned amount_group, unsigned header)
{
   uint8_t raw[G5P_USC_MAX];
   CHECK(header_ok(header), g5p_usc_error);
   raw[0] = 0x9a;
   raw[1] = header;
   if (!destination(raw + 2, dst_half, bank))
      return 0;
   raw[4] = 1;
   SEL(v, value_half, value_group, 1);
   raw[5] = v;
   raw[6] = 0x20;
   raw[7] = 1;
   unsigned n = 8;
   if (immediate) {
      CHECK(amount < 32, "shift amount must be u5");
      raw[n++] = (amount << 1) | 1;
      raw[n++] = 1;
      raw[n++] = 0x20;
      raw[n++] = 1;
   } else {
      SEL(s, amount, amount_group, 0);
      raw[n++] = s;
      raw[n++] = 1;
   }
   raw[n++] = left ? 0xf0 : 0xf4;
   raw[n++] = immediate ? 0x14 : 4;
   return finish(out, raw, n);
}

unsigned g5p_usc_convert(uint8_t *out, enum g5p_convert op, unsigned dst_half, unsigned src_half,
                         unsigned bank, unsigned src_group, unsigned header)
{
   return g5p_usc_convert2(out, op, dst_half, src_half, bank, src_group, header, 0, 0);
}

unsigned g5p_usc_convert2(uint8_t *out, enum g5p_convert op, unsigned dst_half, unsigned src_half,
                          unsigned bank, unsigned src_group, unsigned header,
                          unsigned source_extension, unsigned dst_control)
{
   CHECK(op < G5P_CONVERT_COUNT, "unsupported conversion");
   CHECK(source_extension == 0 || source_extension == 4, "unmeasured conversion class");
   uint8_t raw[G5P_USC_MAX];
   CHECK(header_ok(header), g5p_usc_error);
   raw[0] = 0x99 | source_extension;
   raw[1] = header;
   if (!cdestination(raw + 2, dst_half, bank, dst_control))
      return 0;
   raw[4] = convert_tag[op];
   SEL(s, src_half, src_group, 1);
   raw[5] = s;
   raw[6] = 0x20;
   raw[7] = 1;
   raw[8] = convert_code[op];
   raw[9] = 4;
   return finish(out, raw, 10);
}

/* Apple's vertex format conversions (our decode of the c_* and v_*norm*
 * vertex probes): the convert's tag is (format << 1) | 1, its source a half
 * register, and its last byte 0x0c, 0x4c for the half's high byte
 * (c_un8_b3 9911060305832001684c: un8, r0's high half, high byte). A source
 * read again takes class 9d, header bit 6 (v_unorm4x8). */
unsigned g5p_usc_convert_small(uint8_t *out, enum g5p_small_format fmt, unsigned dst_half, unsigned src_half,
                               bool high_byte, unsigned bank, unsigned src_group, unsigned header,
                               unsigned source_extension)
{
   CHECK(fmt < G5P_SMALL_FORMAT_COUNT, "unsupported small conversion format");
   CHECK(!high_byte || fmt < G5P_SMALL_U16, "a 16-bit source has no high byte");
   CHECK(source_extension == 0 || source_extension == 4, "unmeasured conversion class");
   uint8_t raw[G5P_USC_MAX];
   CHECK(header_ok(header), g5p_usc_error);
   raw[0] = 0x99 | source_extension;
   raw[1] = header;
   if (!destination(raw + 2, dst_half, bank))
      return 0;
   raw[4] = (fmt << 1) | 1;
   SEL(s, src_half, src_group, 1);
   raw[5] = s;
   raw[6] = 0x20;
   raw[7] = 1;
   raw[8] = 0x68;
   raw[9] = high_byte ? 0x4c : 0x0c;
   return finish(out, raw, 10);
}

/* Apple's c_f16_h0/h1, v_half2: 991106030181a001440d7000, the f16 literal
 * fadd (A = -0.0) with byte 9 bit 2 set, its B read as f16 from the half. */
unsigned g5p_usc_half_to_float(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned bank,
                               unsigned src_group, unsigned header)
{
   uint8_t raw[G5P_USC_MAX];
   unsigned n = literal_prefix(raw, 0x99, dst_half, bank, 0x8000, src_half, src_group, header, 0);
   if (!n)
      return 0;
   raw[n++] = binop_code[G5P_BIN_FADD];
   raw[n++] = 0x0d;
   raw[n++] = 0x70;
   raw[n++] = 0;
   return finish(out, raw, n);
}

unsigned g5p_usc_unary(uint8_t *out, enum g5p_unary op, unsigned dst_half, unsigned src_half,
                       unsigned bank, unsigned src_group, unsigned header, bool negate)
{
   return g5p_usc_unary2(out, op, dst_half, src_half, bank, src_group, header, negate, 0, 0, false);
}

unsigned g5p_usc_unary2(uint8_t *out, enum g5p_unary op, unsigned dst_half, unsigned src_half,
                        unsigned bank, unsigned src_group, unsigned header, bool negate,
                        unsigned source_extension, unsigned dst_control, bool absolute)
{
   CHECK(op < G5P_UNARY_COUNT, "unsupported unary operation");
   CHECK(!negate || op == G5P_FLOOR, "unary source negation measured only for floor");
   CHECK((source_extension == 0 || source_extension == 4) && (!absolute || op == G5P_LOG2),
         "unmeasured unary extension/absolute modifier");
   uint8_t raw[G5P_USC_MAX];
   CHECK(header_ok(header), g5p_usc_error);
   raw[0] = 0x98 | source_extension;
   raw[1] = header;
   if (!cdestination(raw + 2, dst_half, bank, dst_control))
      return 0;
   SEL(s, src_half, src_group, 0);
   raw[4] = s;
   raw[5] = 1;
   raw[6] = unary_code[op];
   raw[7] = (negate ? 0x80 : 0) | (absolute ? 1 : 0);
   unsigned n = 8;
   if (absolute) {
      raw[n++] = 0;
      raw[n++] = 0x80;
   }
   return finish(out, raw, n);
}

unsigned g5p_usc_move(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned bank,
                      unsigned src_group, unsigned header)
{
   uint8_t raw[G5P_USC_MAX];
   CHECK(header_ok(header), g5p_usc_error);
   raw[0] = 0x98;
   raw[1] = header;
   if (!destination(raw + 2, dst_half, bank))
      return 0;
   SEL(s, src_half, src_group, 0);
   raw[4] = s;
   raw[5] = 1;
   memcpy(raw + 6, mov_trailer, 4);
   return finish(out, raw, 10);
}

unsigned g5p_usc_move2(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned bank,
                       unsigned src_group, unsigned header, unsigned dst_control)
{
   uint8_t raw[G5P_USC_MAX];
   CHECK(header_ok(header), g5p_usc_error);
   raw[0] = 0x98;
   raw[1] = header;
   if (!cdestination(raw + 2, dst_half, bank, dst_control))
      return 0;
   SEL(s, src_half, src_group, 0);
   raw[4] = s;
   raw[5] = 1;
   memcpy(raw + 6, mov_trailer, 4);
   return finish(out, raw, 10);
}

unsigned g5p_usc_immediate(uint8_t *out, unsigned dst_half, uint32_t value, unsigned bank,
                           int long_form)
{
   uint8_t raw[G5P_USC_MAX];
   if (!destination(raw + 2, dst_half, bank))
      return 0;
   bool wide = long_form < 0 ? value > 0xffff : long_form;
   CHECK(wide || value <= 0xffff, "short move cannot represent this value");
   raw[0] = 0x98;
   raw[1] = 1;
   raw[4] = 1;
   raw[5] = 1;
   raw[6] = 0x20;
   raw[7] = 1;
   unsigned n = 8;
   if (wide) {
      raw[n++] = 0;
      raw[n++] = 1;
      raw[6] |= 1;
   }
   for (unsigned bit = 0; bit < 32; bit++)
      if (imm_byte[bit] < n && (value >> bit) & 1)
         raw[imm_byte[bit]] |= 1u << imm_bit[bit];
   memcpy(raw + n, mov_trailer, 4);
   return finish(out, raw, n + 4);
}

unsigned g5p_usc_special(uint8_t *out, bool end)
{
   out[0] = 0;
   out[1] = end ? 0x80 : 0xc0;
   g5p_usc_error = NULL;
   return 2;
}

/* End of a task that writes no output: 5e 01 00 a0. The TA's end-of-stream
 * program (the geometry work record's TA program slot, W+0x88) is a branch to the next
 * 64-byte block, 0xc000 padding, and this word: Apple's A9 job layout has it
 * there, and the G9P driver builds the same program for the same register
 * on the A10 (pvr_arch_g9p_gfx.c, "Register 0x470's program"). */
unsigned g5p_usc_task_end(uint8_t *out)
{
   out[0] = 0x5e;
   out[1] = 0x01;
   out[2] = 0x00;
   out[3] = 0xa0;
   g5p_usc_error = NULL;
   return 4;
}

unsigned g5p_usc_branch(uint8_t *out, int delta, bool conditional)
{
   CHECK(!(delta % 2) && delta >= -32768 && delta <= 65534, "branch must be even, in [-32768,65534]");
   CHECK(!(delta < 0 && conditional), "conditional negative branch form not characterized");
   unsigned u = (unsigned)delta & 0xffff;
   uint8_t raw[6] = {delta < 0 ? 1 : 0, conditional ? 9 : 1, (u & 0xfe) | ((u >> 8) & 1),
                     ((u >> 9) << 1) | 1, delta < 0 ? 0xff : 0, delta < 0 ? 0xfe : 0};
   return finish(out, raw, 6);
}

unsigned g5p_usc_iterator(uint8_t *out, unsigned dst_half, unsigned slot, unsigned count,
                          bool narrow, bool saturate)
{
   if (g5p_usc_g4p)
      return iterator_g4p(out, dst_half, slot, count, narrow, saturate, 0);
   CHECK(count >= 1 && count <= 4 && !(dst_half % 2), "iterator count/alignment");
   CHECK(dst_half < 256, "iterator destination must be u8");
   CHECK(slot < 128, "iterator slot must be u7");
   uint8_t raw[4] = {narrow ? 0x0a : 8, ((count % 4) << 3) | 3, dst_half | (saturate ? 1 : 0), slot << 1};
   return finish(out, raw, 4);
}

unsigned g5p_usc_iterator_flags(uint8_t *out, unsigned dst_half, unsigned slot, unsigned count,
                                bool narrow, bool saturate, unsigned flags)
{
   CHECK(!(flags & ~(G5P_ITER_LINEAR | G5P_ITER_CENTROID | G5P_ITER_DRC1)), "unknown iterator flag");
   CHECK(!(narrow && (flags & (G5P_ITER_LINEAR | G5P_ITER_CENTROID))), "narrow iterator modes not witnessed");
   if (g5p_usc_g4p)
      return iterator_g4p(out, dst_half, slot, count, narrow, saturate, flags);
   if (!g5p_usc_iterator(out, dst_half, slot, count, narrow, saturate))
      return 0;
   if (flags & G5P_ITER_DRC1)
      out[0] |= 1;
   if (flags & G5P_ITER_LINEAR)
      out[1] &= ~2u;
   if (flags & G5P_ITER_CENTROID)
      out[1] |= 0x40;
   return 4;
}

unsigned g5p_usc_coeff_load(uint8_t *out, unsigned dst_half, unsigned coeff_dword, bool drc1)
{
   CHECK(!(dst_half % 2) && dst_half < 256, "coefficient load destination must be an even u8 half");
   CHECK(coeff_dword < 256 && coeff_dword % 4 == 2, "coefficient load reads a C term (4 * set + 2, u8)");
   if (g5p_usc_g4p) {
      /* The A8's flat read: 18 09 DD 01 AA 01 60 00 (AGXMetalA8's f_iter in an iOS capture: 18 09 00 01 0e 01 60 00,
       * set 3). The A9's 14 23 does not exist on the A8 in this form (on the iPod a flat varying's first component
       * read 0 with it). */
      CHECK(!drc1, "A8 flat read on the second dependent-read counter not captured");
      const uint8_t g4[8] = {0x18, 0x09, (uint8_t)dst_half, 0x01, (uint8_t)coeff_dword, 0x01, 0x60, 0x00};
      return finish_g4p(out, g4, 8);
   }
   uint8_t raw[8] = {drc1 ? 0x15 : 0x14, 0x23, dst_half, 0x01, coeff_dword, 0x01, 0x00, 0x02};
   return finish(out, raw, 8);
}

unsigned g5p_usc_coord_iterator(uint8_t *out, unsigned dst_half, unsigned slot, unsigned count)
{
   if (g5p_usc_g4p) /* the A8 iterates a texture coordinate with its plain iterator (Apple's f_tex) */
      return iterator_g4p(out, dst_half, slot, count, false, false, 0);
   CHECK((count == 2 || count == 3) && !(dst_half % 2), "coordinate iterator count/alignment");
   CHECK(dst_half < 64, "coordinate destination half must be u6");
   CHECK(slot < 128, "iterator slot must be u7");
   uint8_t raw[6] = {8, count == 2 ? 0x13 : 0x1b, dst_half, (slot << 1) | 1, 0, 0x80};
   return finish(out, raw, 6);
}

unsigned g5p_usc_emit(uint8_t *out, unsigned words, unsigned source_half)
{
   CHECK(source_half < 256, "output source half must be u8");
   if (g5p_usc_g4p) {
      /* A8: byte 3 bit 0 and the extension halfword 0x8200 (Apple's 1c 21 00 09 00 82 in every captured
       * fragment program; the A9's 1c 21 00 08) */
      CHECK(words == 1 || words == 2 || words == 4, "A8 emitpix captured for 1, 2 and 4 words (111c, 211c, 411c)");
      const uint8_t g4[6] = {0x1c, (uint8_t)((words << 4) | 1), (uint8_t)source_half, 9, 0x00, 0x82};
      return finish_g4p(out, g4, 6);
   }
   CHECK(words == 2 || words == 4, "only two/four output words characterized");
   uint8_t raw[4] = {0x1c, (words << 4) | 1, source_half, 8};
   return finish(out, raw, 4);
}

/* Multi-target emitpix. The word count is the total over all targets and the
 * field is three bits wide, so 8 wraps to 0 (Apple's `1c01 0008` for four
 * 8-bit targets). Only the homogeneous 1/2/4-target counts with an Apple A9
 * witness are accepted. g5p_usc_emit keeps its two/four contract: it is
 * replayed against the Python reference (tools/gpu-userspace/g5p_usc.py). */
unsigned g5p_usc_emit_mrt(uint8_t *out, unsigned words, unsigned source_half)
{
   CHECK(g5p_mrt_emit_words_witnessed(words), "only 1/2/4/8 terminal source words have an Apple witness");
   CHECK(source_half < 256, "output source half must be u8");
   if (g5p_usc_g4p)
      return g5p_usc_emit(out, words, source_half);
   uint8_t raw[4] = {0x1c, (g5p_mrt_emit_field(words) << 4) | 1, source_half, 8};
   return finish(out, raw, 4);
}

unsigned g5p_usc_vout(uint8_t *out, unsigned src_half, unsigned slot, bool temp, bool last)
{
   CHECK(!(src_half & 1), "vertex output source must be a 32-bit aligned half");
   CHECK(src_half < 256, "vertex output source half must be u8");
   CHECK(slot < 256, "vertex output slot must be u8");
   unsigned b1 = temp ? 0x15 : 0x11;
   if (last) {
      uint8_t raw[6] = {0x5e, b1, src_half, 0xc1, slot, 0};
      return finish(out, raw, 6);
   }
   if (slot == 0 && src_half == 0) {
      uint8_t raw[2] = {0x1e, temp ? 0x14 : 0x10};
      return finish(out, raw, 2);
   }
   if (slot == 0) {
      uint8_t raw[4] = {0x1e, b1, src_half, 0};
      return finish(out, raw, 4);
   }
   uint8_t raw[6] = {0x1e, b1, src_half, 1, slot, 0};
   return finish(out, raw, 6);
}

/* ---- device memory fence ----
 * Apple's compiler (metal-tt -arch applegpu_g5p) emits this eight-byte
 * memory instruction, then a wait, for threadgroup_barrier(mem_device), and
 * at the end of every device compare-exchange. It completes after the task's
 * earlier device stores. Its operand is a register pair holding a 64-bit
 * value; in compiler-built kernels that is the ComputeMemFenceAddr address
 * Apple's driver supplies.
 *
 *   b0      memory instruction class (0x10, as the loads and stores);
 *           bit 5: G5P_FENCE_FLAG_B0
 *   b1      0xc3: the fence operation of the class (the stores' b1 is the
 *           access size, 0x0b..0x3b)
 *   b2:b3   0x0100 | pair << 10: the operand pair r[2 pair]:r[2 pair + 1]
 *           (checked over four register allocations)
 *   b4:b5   0x0100; bit 11: G5P_FENCE_FLAG_W2
 *   b6:b7   0x1000 (the same in every Apple form)
 *
 * Compiler-built kernels set both flags (every barrier and compare-exchange
 * the compiler produced). The compute flush program iOS runs at the end of a
 * compute pass (g5p_end_barrier.h) clears both. No Apple symbol names the
 * flags, so the two forms are named after the programs that use them, and
 * only these two are accepted. */
#define G5P_MEM_CLASS 0x10u
#define G5P_MEM_OP_FENCE 0xc3u
#define G5P_FENCE_FLAG_B0 0x20u
#define G5P_FENCE_PAIR_W1(pair) (0x0100u | (pair) << 10)
#define G5P_FENCE_W2 0x0100u
#define G5P_FENCE_FLAG_W2 0x0800u
#define G5P_FENCE_W3 0x1000u
unsigned g5p_usc_mem_fence(uint8_t *out, unsigned pair, enum g5p_fence_form form)
{
   CHECK(pair < 32, "fence operand pair must name r0..r63");
   CHECK(form == G5P_FENCE_FORM_KERNEL || form == G5P_FENCE_FORM_FLUSH_PROGRAM, "unknown fence form");
   const unsigned flags_b0 = form == G5P_FENCE_FORM_KERNEL ? G5P_FENCE_FLAG_B0 : 0;
   const unsigned flags_w2 = form == G5P_FENCE_FORM_KERNEL ? G5P_FENCE_FLAG_W2 : 0;
   const unsigned w1 = G5P_FENCE_PAIR_W1(pair), w2 = G5P_FENCE_W2 | flags_w2;
   const uint8_t raw[8] = {G5P_MEM_CLASS | flags_b0, G5P_MEM_OP_FENCE,
                           w1 & 0xff, w1 >> 8, w2 & 0xff, w2 >> 8,
                           G5P_FENCE_W3 & 0xff, G5P_FENCE_W3 >> 8};
   return finish(out, raw, 8);
}

/* ---- zero-word emit ----
 * g5p_usc_emit()'s layout, `1c (words << 4)|1 source control`, with no
 * output words. Control bit 3 (0x08) is set in every Apple emit; bits 4-7
 * hold a code. The compute flush program uses code 0xb, the only one
 * accepted here. iOS has two compute flush programs: the plain one (an empty
 * @agc.flush) and one with a firmware interrupt (@agc.flush.fw_interrupt,
 * which adds llvm.agx1.nop.dbg(i16 2)); whether this emit is that interrupt
 * is not settled. */
#define G5P_EMIT_OP 0x1cu
#define G5P_EMIT_WORDS_B1(words) ((words) << 4 | 1u)
#define G5P_EMIT_CTL_EMIT 0x08u
#define G5P_EMIT_CODE_SHIFT 4
unsigned g5p_usc_emit_code(uint8_t *out, enum g5p_emit_code code)
{
   CHECK(code == G5P_EMIT_CODE_FLUSH_PROGRAM, "only the compute flush program's code");
   const uint8_t raw[4] = {G5P_EMIT_OP, G5P_EMIT_WORDS_B1(0), 0,
                           (uint8_t)(code << G5P_EMIT_CODE_SHIFT | G5P_EMIT_CTL_EMIT)};
   return finish(out, raw, 4);
}

unsigned g5p_usc_wait(uint8_t *out, unsigned control)
{
   CHECK(control == 0 || control == 2 || control == 4, "unwitnessed wait control");
   uint8_t raw[2] = {0x4d, control};
   return finish(out, raw, 2);
}

/* ---- resources ---- */
static bool half_register(unsigned value, const char *why)
{
   if (value > 62 || value & 1) {
      g5p_usc_error = why;
      return false;
   }
   return true;
}

unsigned g5p_usc_mov_shared(uint8_t *out, unsigned dst_half, unsigned src_dword, unsigned bank)
{
   CHECK(half_register(dst_half, "dst_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   CHECK(bank <= 1, "bank must be an integer in [0,1]");
   /* Dwords 64..255: the extension word's field (src_dword >> 5) is the one
    * Apple's A9 compiler writes for inline common-store operands up to dword
    * 255 (tools/mesa-a9-hypr-re f_u256all; check-hypr.py encoders). */
   CHECK(src_dword <= 255, "src_dword must be an integer in [0,255]");
   uint8_t raw[12] = {0x98, 1, dst_half, (bank << 1) | 1, 0x81 | ((src_dword & 31) << 2), 1,
                      (src_dword >> 5) << 1, 0x41, 0x54, 0x01, 0x38, 0x00};
   return finish(out, raw, 12);
}

/* Byte fields of the address add (g5p_usc_address_add_shift()) and later encoders. */
#define G5P_HEADER_FRAGMENT 0x01    /* the header's continuation into the destination parcel, nothing else */
#define G5P_HEADER_SRC_B_BANK1 0x10 /* source B in register bank 1 (g9p_w0()'s W0 bit 12; vertex code) */
#define G5P_PARCEL_CONTINUES 0x01
#define G5P_SRC_LAST_READ 0x80      /* a register source read for the last time (in the group bit's place) */
#define G5P_CLASS_ADDR_ADD 0x89     /* the address add's class byte (the binary class with the 16-bit result) */
#define G5P_ADDR_ADD_ROUTING_NONE 0x00
#define G5P_ADDR_ADD_CTL 0x06       /* set in every witnessed address add (observed role) */
#define G5P_ADDR_ADD_SHIFT_BIT 5    /* the shift, bits 5..7 of the control byte */

/* ---- ISA round 2: inline shared sources (g5p_usc_resources.py) ---- */
static unsigned alu_header(uint8_t *raw, unsigned cls, unsigned dst_half, unsigned bank, unsigned header,
                           unsigned source_extension, unsigned dst_control)
{
   CHECK(half_register(dst_half, "dst_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   CHECK(bank <= 1, "bank must be an integer in [0,1]");
   CHECK(header >= 1 && header <= 255, "header must be an integer in [1,255]");
   CHECK((header & 1) && (source_extension == 0 || source_extension == 4) &&
         (dst_control == 0 || dst_control == 0x80),
         "unmeasured ALU source class or destination control");
   raw[0] = cls | source_extension;
   raw[1] = header;
   raw[2] = dst_half;
   raw[3] = (bank << 1) | 1 | dst_control;
   return 4;
}

static bool shared_src(uint8_t *raw, unsigned index, unsigned group)
{
   if (index > 255 || group > 1) {
      g5p_usc_error = index > 255 ? "shared dword must be an integer in [0,255]"
                                  : "shared group must be an integer in [0,1]";
      return false;
   }
   raw[0] = 1 | (group << 7) | ((index & 31) << 2);
   raw[1] = 1;
   raw[2] = (index >> 5) << 1;
   raw[3] = 0x41;
   return true;
}

static unsigned add_route(uint8_t *raw, unsigned n, int route)
{
   raw[n++] |= route >= 0;
   if (route >= 0) {
      raw[n++] = route;
      raw[n++] = 0;
   }
   return n;
}

/* Names shared by the fragment encoders. */
#define G5P_CLASS_TERNARY_RESULT_HALF 0x8a /* g5p_usc_literal_ternary() with result_half */
#define G5P_TERNARY_CTL_RESULT_HALF 0x02

/* One byte of a packed UNORM8 colour, for the clear of 5..8 colour targets (g5p_mrt_programs.h): the f32 in
 * common-store dword src_dword, converted to an unsigned normalized byte, into byte 0 or byte 1 of the 16-bit
 * dst_half, bank 0. Two of them fill one half: the byte-1 form reads dst_half as B and keeps the byte that the
 * byte-0 form wrote. It is the float-to-integer conversion in the three-source class with the 16-bit result,
 * laid out as literal_prefix() followed by a common-store C:
 *   A        a short literal, the conversion's target format: G5P_SMALL_UN8, the format code of
 *            g5p_usc_convert_small() (g5p_usc_f2u16_half() puts G5P_SMALL_U16 in the same place);
 *   B        byte 0: unused (selector 0); byte 1: dst_half, group 1;
 *   C        the common-store dword (shared_src(), group 1);
 *   opcode   the float-to-integer conversion (convert_code[], as g5p_usc_f2u16_half());
 *   control  bit 0 the continuation into the last parcel; bit 1 the 16-bit result; bit 2 set by every
 *            conversion form (g5p_usc_convert2(), g5p_usc_f2u16_half(), g5p_usc_convert_small()); the byte-1
 *            form adds bits 4 and 6, which were only observed together (bit 6 also selects the source's high
 *            byte in g5p_usc_convert_small());
 *   routing  the low byte of the last parcel, observed per byte; its high byte, the source control, is 0.
 * The form was recovered from controlled probes; the clear was tested on the A9 at 5, 6, 7 and 8 targets. */
#define G5P_CONVERT_CTL 0x04
#define G5P_UNORM8_CTL_BYTE1 0x50
#define G5P_UNORM8_ROUTING_BYTE0 0x10
#define G5P_UNORM8_ROUTING_BYTE1 0x1c
#define G5P_SOURCE_CONTROL_NONE 0x00

unsigned g5p_usc_unorm8_byte(uint8_t *out, unsigned dst_half, unsigned byte, unsigned src_dword)
{
   CHECK(byte <= 1, "a half has bytes 0 and 1");
   const unsigned dst_bank = 0, c_group = 1;
   const unsigned b_half = byte ? dst_half : 0, b_group = byte ? 1 : 0;
   uint8_t raw[G5P_USC_MAX];
   unsigned n = literal_prefix(raw, G5P_CLASS_TERNARY_RESULT_HALF, dst_half, dst_bank, G5P_SMALL_UN8, b_half,
                               b_group, G5P_HEADER_FRAGMENT, 0);
   if (!n)
      return 0;
   if (!shared_src(raw + n, src_dword, c_group))
      return 0;
   n += 4;
   raw[n++] = convert_code[G5P_F2U32];
   raw[n++] = G5P_TERNARY_CTL_RESULT_HALF | G5P_CONVERT_CTL | (byte ? G5P_UNORM8_CTL_BYTE1 : 0) |
              G5P_PARCEL_CONTINUES;
   raw[n++] = byte ? G5P_UNORM8_ROUTING_BYTE1 : G5P_UNORM8_ROUTING_BYTE0;
   raw[n++] = G5P_SOURCE_CONTROL_NONE;
   return finish(out, raw, n);
}

/* Integer min/max of a short literal A and a common-store dword C, the 32-bit result in bank 0: AGXMetalA8's
 * RGBA8UI clear (iPod touch 6, iOS 12.4, a8-passcap rgba8u) clamps each clear channel with
 *   9a 01 DD CC 7f 01 26 01 SS 01 00 41 b2 21 b0 00      (umin(255, common dword SS))
 * laid out as g5p_usc_fma_literal(): literal_prefix() in the three-source class 0x9a, B unused (selector 0),
 * then C = the common-store dword (shared_src(), group 1), the min/max code (minmax_code[], as
 * g5p_usc_minmax_literal()), control 0x21 and the routing byte 0xb0 (literal A, common-store C; the register
 * forms route 0x08 and 0x7c), no source control. CC carries the destination control: Apple sets 0x80 on one of
 * its four clamps (the second), on a register its next clamp writes again. Only the short literal is witnessed. */
#define G5P_MINMAX_CTL 0x21
#define G5P_ROUTING_LITERAL_A_SHARED_C 0xb0
unsigned g5p_usc_minmax_literal_shared(uint8_t *out, enum g5p_minmax op, unsigned dst_half, unsigned dst_control,
                                       uint32_t value, unsigned c_dword)
{
   CHECK(op < G5P_MINMAX_COUNT, "unsupported integer min/max");
   CHECK(value <= 0xffff, "only the short literal is witnessed");
   CHECK(half_register(dst_half, "dst_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   uint8_t raw[G5P_USC_MAX];
   unsigned n = literal_prefix(raw, 0x9a, dst_half, 0, value, 0, 0, G5P_HEADER_FRAGMENT, 0);
   if (!n)
      return 0;
   if (!cdestination(raw + 2, dst_half, 0, dst_control))
      return 0;
   if (!shared_src(raw + n, c_dword, 1))
      return 0;
   n += 4;
   raw[n++] = minmax_code[op];
   raw[n++] = G5P_MINMAX_CTL;
   raw[n++] = G5P_ROUTING_LITERAL_A_SHARED_C;
   raw[n++] = G5P_SOURCE_CONTROL_NONE;
   return finish(out, raw, n);
}

/* Two bytes into one 16-bit half of bank 0, dst = B | C << shift: AGXMetalA8's RGBA8UI clear (a8-passcap rgba8u)
 * packs its clamped channels (each 0..255) pairwise with
 *   8a 01 DH 01 11 BB 20 01 CC 01 f1 1f 58 00            (dst half DH = half BB | half CC << 8)
 * literal_prefix() in the three-source class with the 16-bit result (0x8a) and the shift as A's short literal,
 * B and C register halves of group 1 (C's selector without the continuation, then the 0x01 parcel byte, as
 * g5p_usc_fma_literal()), opcode 0xf1, control 0x1f, routing 0x58, no source control. The captured operands are
 * 0..255, so whether the opcode ors, adds or inserts the shifted C is not distinguished; only shift 8 is witnessed. */
#define G5P_PACK_BYTE_OP 0xf1
#define G5P_PACK_BYTE_CTL 0x1f
#define G5P_PACK_BYTE_ROUTING 0x58
unsigned g5p_usc_pack_byte_half(uint8_t *out, unsigned dst_half, unsigned b_half, unsigned c_half,
                                unsigned shift)
{
   CHECK(shift == 8, "only the byte pack (shift 8) is witnessed");
   uint8_t raw[G5P_USC_MAX];
   unsigned n = literal_prefix(raw, G5P_CLASS_TERNARY_RESULT_HALF, dst_half, 0, shift, b_half, 1,
                               G5P_HEADER_FRAGMENT, 0);
   if (!n)
      return 0;
   SEL(c, c_half, 1, 0);
   raw[n++] = c;
   raw[n++] = G5P_PARCEL_CONTINUES;
   raw[n++] = G5P_PACK_BYTE_OP;
   raw[n++] = G5P_PACK_BYTE_CTL;
   raw[n++] = G5P_PACK_BYTE_ROUTING;
   raw[n++] = G5P_SOURCE_CONTROL_NONE;
   return finish(out, raw, n);
}

/* A move of a common-store dword into a 16-bit half of bank 0 (the end-of-tile program's state moves,
 * g5p_mrt_programs.h): alu_header() with the 16-bit move class (G5P_CLASS_MOVE_HALF, as g5p_usc_half_immediate())
 * and the fragment header, A = the common-store dword (shared_src(), group 1), then the half move's opcode and
 * control (G5P_HALF_MOVE_OP, G5P_HALF_MOVE_CTL: the bytes of g5p_usc_half_immediate()'s last parcel, named by
 * that observed role) and no source control. */
#define G5P_CLASS_MOVE_HALF 0x88
#define G5P_HALF_MOVE_OP 0x28
#define G5P_HALF_MOVE_CTL 0x39
unsigned g5p_usc_mov_shared_half(uint8_t *out, unsigned dst_half, unsigned src_dword)
{
   uint8_t raw[12];
   if (!alu_header(raw, G5P_CLASS_MOVE_HALF, dst_half, 0, G5P_HEADER_FRAGMENT, 0, 0) ||
       !shared_src(raw + 4, src_dword, 1))
      return 0;
   raw[8] = G5P_HALF_MOVE_OP;
   raw[9] = G5P_PARCEL_CONTINUES;
   raw[10] = G5P_HALF_MOVE_CTL;
   raw[11] = G5P_SOURCE_CONTROL_NONE;
   return finish(out, raw, 12);
}

unsigned g5p_usc_alu_shared(uint8_t *out, enum g5p_binop op, unsigned dst_half, unsigned a_dword,
                            unsigned b_half, unsigned bank, unsigned b_group, unsigned header,
                            unsigned source_extension, unsigned dst_control, bool negate_b,
                            int route, unsigned a_shared_group)
{
   /* Integer add/multiply: vertex v_iadd_k / v_imul_k, no routing parcel.
    * Subtract: m_isub_u (shared - B, no parcel) and m_isub_u2 (B - shared,
    * routing 0x70). */
   bool integer = op == G5P_BIN_IADD || op == G5P_BIN_IMUL;
   CHECK(((op == G5P_BIN_FADD || op == G5P_BIN_FMUL) && (route < 0 || route == 0x70)) ||
         (integer && route < 0 && !negate_b) ||
         (op == G5P_BIN_ISUB && (route < 0 || route == 0x70) && !negate_b),
         "unmeasured shared binary operation/routing");
   CHECK(half_register(b_half, "b_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   CHECK(b_group <= 1, "b_group must be an integer in [0,1]");
   uint8_t raw[G5P_USC_MAX], a[4];
   if (!alu_header(raw, 0x99, dst_half, bank, header, source_extension, dst_control) ||
       !shared_src(a, a_dword, a_shared_group))
      return 0;
   raw[4] = a[0];
   raw[5] = (b_half << 1) | (b_group << 7) | 1;
   raw[6] = a[2];
   raw[7] = a[3];
   raw[8] = binop_code[op];
   raw[9] = negate_b ? 0x40 : 0;
   return finish(out, raw, add_route(raw, 9, route));
}

/* Apple's vertex m_imin_u 9911000389810041ba210800: the shared head of
 * g5p_usc_alu_shared, then the register form's min/max tail. */
unsigned g5p_usc_minmax_shared(uint8_t *out, enum g5p_minmax op, unsigned dst_half, unsigned a_dword,
                               unsigned b_half, unsigned bank, unsigned b_group, unsigned header,
                               unsigned a_shared_group)
{
   CHECK(op < G5P_MINMAX_COUNT, "unsupported integer min/max");
   CHECK(half_register(b_half, "b_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   CHECK(b_group <= 1, "b_group must be an integer in [0,1]");
   uint8_t raw[G5P_USC_MAX], a[4];
   if (!alu_header(raw, 0x99, dst_half, bank, header, 0, 0) || !shared_src(a, a_dword, a_shared_group))
      return 0;
   raw[4] = a[0];
   raw[5] = (b_half << 1) | (b_group << 7) | 1;
   raw[6] = a[2];
   raw[7] = a[3];
   raw[8] = minmax_code[op];
   raw[9] = 0x21;
   raw[10] = 8;
   raw[11] = 0;
   return finish(out, raw, 12);
}

unsigned g5p_usc_fma_shared_ac(uint8_t *out, unsigned dst_half, unsigned a_dword, unsigned b_half,
                               unsigned c_dword, unsigned bank, unsigned b_group, unsigned header,
                               unsigned source_extension, unsigned dst_control, int route,
                               unsigned a_shared_group, unsigned c_shared_group)
{
   CHECK(route < 0 || route == 0x70 || route == 0x1c, "unmeasured FMA source routing");
   CHECK(half_register(b_half, "b_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   CHECK(b_group <= 1, "b_group must be an integer in [0,1]");
   uint8_t raw[G5P_USC_MAX], a[4], c[4];
   if (!alu_header(raw, 0x9a, dst_half, bank, header, source_extension, dst_control) ||
       !shared_src(a, a_dword, a_shared_group) || !shared_src(c, c_dword, c_shared_group))
      return 0;
   raw[4] = a[0];
   raw[5] = (b_half << 1) | (b_group << 7) | 1;
   raw[6] = a[2];
   raw[7] = a[3];
   memcpy(raw + 8, c, 4);
   raw[12] = 0x4c;
   raw[13] = 0;
   unsigned n = add_route(raw, 13, route);
   CHECK(n <= G5P_USC_MAX, "instruction too long");
   return finish(out, raw, n);
}

unsigned g5p_usc_fma_shared_a(uint8_t *out, unsigned dst_half, unsigned a_dword, unsigned b_half,
                              unsigned c_half, unsigned bank, unsigned b_group, unsigned c_group,
                              unsigned header, unsigned source_extension, unsigned dst_control,
                              int route, unsigned a_shared_group)
{
   CHECK(route < 0 || route == 0x58, "unmeasured shared-A FMA source routing");
   CHECK(half_register(b_half, "b_half requires a measured 32-bit aligned half index") &&
         half_register(c_half, "c_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   CHECK(b_group <= 1 && c_group <= 1, "source group must be an integer in [0,1]");
   uint8_t raw[G5P_USC_MAX], a[4];
   if (!alu_header(raw, 0x9a, dst_half, bank, header, source_extension, dst_control) ||
       !shared_src(a, a_dword, a_shared_group))
      return 0;
   raw[4] = a[0];
   raw[5] = (b_half << 1) | (b_group << 7) | 1;
   raw[6] = a[2];
   raw[7] = a[3];
   raw[8] = (c_half << 1) | (c_group << 7);
   raw[9] = 1;
   raw[10] = 0x4c;
   raw[11] = 0;
   return finish(out, raw, add_route(raw, 11, route));
}

unsigned g5p_usc_alu_shared_pair(uint8_t *out, enum g5p_binop op, unsigned dst_half, unsigned a_dword,
                                 unsigned b_dword, unsigned bank, unsigned a_shared_group,
                                 unsigned b_shared_group)
{
   CHECK(op == G5P_BIN_FADD || op == G5P_BIN_FMUL, "unmeasured shared pair operation");
   uint8_t raw[G5P_USC_MAX];
   if (!alu_header(raw, 0x9a, dst_half, bank, 1, 0, 0) || !shared_src(raw + 4, a_dword, a_shared_group) ||
       !shared_src(raw + 8, b_dword, b_shared_group))
      return 0;
   raw[12] = binop_code[op];
   raw[13] = 1;
   raw[14] = 0x10;
   raw[15] = 0;
   return finish(out, raw, 16);
}

static unsigned texture_base(uint8_t *raw, const struct g5p_tex *t)
{
   CHECK(half_register(t->dst_half, "dst_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   /* A9 f16/v16 capacity programs independently reconstruct these selectors.
    * The common-store selector is the same seven-bit field as sample_ext. */
   CHECK(t->image_qword <= 120, "image_qword must be an integer in [0,120]");
   CHECK(!(t->image_qword & 3), "measured image descriptor bases are four-qword aligned");
   CHECK(t->sampler_qword <= 124, "sampler_qword must be an integer in [0,124]");
   CHECK(t->coord_dword <= 31, "coord_dword must be an integer in [0,31]");
   CHECK(t->class_variant <= 1, "class_variant must be an integer in [0,1]");
   CHECK(t->dim < G5P_TEX_DIM_COUNT, "unmeasured texture dimension");
   CHECK(!t->read || t->dim == G5P_TEX_2D, "only 2d read has been isolated");
   CHECK(t->dim != G5P_TEX_2D_MS || (!t->read && !t->class_variant),
         "the 2D-multisample fetch is witnessed only as the narrow class0 sample");
   raw[0] = 0x1a | t->class_variant;
   raw[1] = dim_code[t->dim];
   raw[2] = t->dst_half | (t->read ? 0x40 : 0);
   raw[3] = (t->image_qword << 1) | 1;
   raw[4] = t->sampler_qword << 1;
   raw[5] = (t->coord_dword << 1) | (t->coord_control ? 0x80 : 0);
   return 6;
}

/*
 * The A8's 2D sample with an implicit LOD and a packed-f16 (narrow) result, as Apple's A8 compiler writes
 * f_tex's (iPod touch 6, iOS 12.4, a8-shadercap): 1a 59 00 41 00 01 88 01 00 01 00 80, where the A9 writes
 * 18 f5 00 01 08 80 for the same source (image qword 0, sampler qword 4, coordinate dword 0 under coordinate
 * control). Byte 2 is the destination half and byte 8 the coordinate dword << 1 (Metal's blit program has
 * eight samples, 1a 79 DD 41 00 01 ac 01 CC 01 14 80, with eight destinations and coordinates); byte 6 is
 * 0x80 | mode << 4 | sampler dword << 1 (sampler 4 here, 6 in the blit, whose mode 2 is an explicit LOD). Byte 1 bit 5 is a 32-bit result there, which
 * also carries an operand byte 10 not decoded, so only the narrow form is taken; the image qword is 0 in
 * every witness (bytes 3-4 41 00), so only image qword 0.
 */
static unsigned sample_g4p(uint8_t *out, const struct g5p_tex *t)
{
   /* Two result forms, both Apple's: narrow (byte 1 0x59: the f16 results f_tex writes straight to the output) and
    * 32-bit (byte 1 0x79: one 32-bit register per channel, read by ALU after the wait; AGXMetalA8's R32F and RGBA8UI
    * load backgrounds, passcap USC 0x4826 and 0x4da6, the latter feeding umin and byte packs). Metal's blit program
    * has the 32-bit form with byte 10 = 0x14, an operand not decoded and not used here. */
   CHECK((t->dim == G5P_TEX_2D || t->dim == G5P_TEX_CUBE) && !t->read && !t->class_variant && t->coord_control &&
            !t->gradient && !t->lod_control && !t->length,
         "A8 sample captured only as a 2D or cube sample with an implicit LOD");
   /* the cube sample: mode 4 (byte 6 0xc8 with the sampler at dword 4), the direction in three temps from the
    * coordinate dword, no cube-coordinate instruction (AGXMetalA8's f_tex_cube, passcap USC 0x6076: two FMAs and a
    * move build r0..r2, then 1a 59 00 41 00 01 c8 01 00 01 00 80). Only the narrow form is captured. */
   CHECK(t->dim != G5P_TEX_CUBE || t->narrow, "A8 cube sample captured only in the narrow form (result straight to the output)");
   /* the explicit level in the register after the coordinate (Apple's sample at a varying level, iOS capture, USC
    * heap 0x1_02328000 + 0x8814: the iterated coordinate in r0..r1, r2 = r0 * 4.0, then
    * 1a 59 00 41 00 01 a8 09 00 01 00 80): byte 6 mode 2, byte 7 0x09, byte 10 0. 2D only. */
   CHECK(!t->lod_reg || t->dim == G5P_TEX_2D, "A8 explicit-level sample captured for 2D only");
   const uint8_t lod_mode = t->lod_reg ? 2 << 4 : 0, lod_byte7 = t->lod_reg ? 0x09 : 0x01;
   /* The A8's multi-image sample, Apple's form (iOS 12.4, AGXMetalA8's programs sampling two to seven textures
    * with one shared sampler or one per texture, captured with a8-passcap; USC heap 0x1_002cc000 +
    * 0x8a08..0x97dc): byte 4 is the image's common dword * 2, and the sampler's four-dword common slot S
    * (dword 4S) is byte 1 bits 7:6 | byte 5 bits 2:1 << 2; byte 6 is 0x88 in every sample. Apple's four-texture
    * programs with one sampler and with four differ only in byte 1 bits 7:6 (S = 4 shared; 4, 5, 6, 7 per
    * texture); five textures with five samplers have S = 5..9 (byte 5 03, 05), and seven textures' shared
    * sampler is S = 7 (b1 c1/e9/f1, byte 5 03) at dword 28, where its PDS DMAs it. The single-image forms agree:
    * 0x59/0x79 are S = 1 (dword 4). Byte 1 bit 5 is the 32-bit result and bits 4:3 the components written minus
    * one (Apple writes .r as 0, .g as 1, .b as 2); this writes four. On the iPod touch 6 every multi-texture
    * test passed with it. */
   if (t->a8_slots) {
      const unsigned image_dword = t->image_qword, sampler_dword = t->sampler_qword; /* dwords (g5p_tex) */
      CHECK(t->dim == G5P_TEX_2D, "A8 multi-image sample: 2D only (Apple's form is captured for 2D)");
      CHECK(!(image_dword % 4) && image_dword <= 124, "A8 multi-image sample: image dword a multiple of 4 below 128");
      CHECK(!(sampler_dword % 4) && sampler_dword / 4 <= 11, "A8 multi-image sample: sampler slot 0..11");
      CHECK(t->coord_dword <= 31 && !(t->dst_half & 1) && t->dst_half < 64, "A8 sample: coordinate/destination range");
      const unsigned slot = sampler_dword / 4;
      if (t->vtx) {
         /* Apple's vertex form (g5p_tex.vtx) */
         CHECK(!t->narrow && !t->lod_reg && t->coord_dword <= 31, "A8 vertex sample: 32-bit, Apple's constant level");
         CHECK(t->lod_common && t->lod_common <= 63, "A8 vertex sample: its level constant's common dword");
         const uint8_t b6 = 0xa0 | (!t->read_flag ? 0x08 : t->int_result ? 0x04 : 0x0c);
         const uint8_t vraw[12] = {0x1a, (uint8_t)((slot & 3) << 6 | 0x3d), (uint8_t)t->dst_half, 0x41,
                                   (uint8_t)(2 * image_dword), (uint8_t)(1 | (slot >> 2) << 1), b6, 0x05,
                                   (uint8_t)(t->coord_dword << 1), 0x01, (uint8_t)(2 * t->lod_common), 0x80};
         return finish_g4p(out, vraw, sizeof(vraw));
      }
      const uint8_t raw[12] = {0x1a, (uint8_t)((slot & 3) << 6 | (t->narrow ? 0x19 : 0x39)), (uint8_t)t->dst_half,
                               0x41, (uint8_t)(2 * image_dword), (uint8_t)(1 | (slot >> 2) << 1),
                               (uint8_t)(0x88 | lod_mode | (t->read_flag ? 4 : 0)), lod_byte7,
                               (uint8_t)(t->coord_dword << 1), 0x01, 0x00, 0x80};
      return finish_g4p(out, raw, sizeof(raw));
   }
   /* Another image or sampler slot: the A8 form's image field has one witnessed value (bytes 3-4 41 00 at qword 0)
    * and its sampler field three bits of dwords. With PVR_A8_A9_FORMS the sample is the A9's own six-byte
    * instruction (g5p_usc_sample()'s A9 branch: image and sampler qwords in bytes 3 and 4, the coordinate from
    * registers), over the A8's state words at the same common-store positions. */
   if (t->image_qword != 0 || t->sampler_qword > 7 || (t->sampler_qword & 1)) {
      CHECK(g5p_usc_g4p_a9_forms, "A8 sample captured only with the image descriptor at common qword 0");
      CHECK(!t->lod_reg, "A8 explicit-level sample: no A9 form");
      CHECK(t->dim == G5P_TEX_2D, "A8 sample in another image slot: the A9 form is taken for 2D only");
      CHECK(!(t->sampler_qword & 1), "A8 sample in another image slot: the sampler must start a qword");
      struct g5p_tex a9 = *t;
      a9.sampler_qword = t->sampler_qword / 2; /* the A8 callers pass the sampler's dword */
      uint8_t raw[G5P_USC_MAX];
      if (!texture_base(raw, &a9))
         return 0;
      if (t->narrow)
         raw[0] &= ~2;
      return finish(out, raw, 6);
   }
   /* the A8 names the sampler by its common DWORD (Apple's 0x88 reads the sampler at dword 4 in tex_plain and in
    * the load background): the caller passes the dword in sampler_qword */
   /* byte 6 is 0x80 | mode << 4 | sampler dword << 1: Apple's tex_lod, tex_bias, tex_grad and tex_cube samples
    * (passcap USC 0x5388, 0x5588, 0x57bc, 0x6076) read the same sampler (dword 4) with bytes a8, 98, b8 and c8,
    * and the blit's explicit-LOD samples are ac (sampler dword 6). Bits 4-6 are the sample mode, so a sampler at
    * dword 8 or above would encode another mode: the field is three bits. */
   CHECK(t->sampler_qword <= 7, "A8 sample: sampler dword above 7 (byte 6 bits 4-6 are the sample mode)");
   /* witnessed sampler dwords are 0, 4 and 6; byte 6 bit 1 (an odd dword) is also set by Apple's array sample (8a)
    * and bit 0 by its 0x3000 sample (89), neither decoded, so an odd dword could alias a flag */
   CHECK(!(t->sampler_qword & 1), "A8 sample: odd sampler dword (byte 6 bit 1 is set by Apple's undecoded array sample)");
   CHECK(t->coord_dword <= 31 && !(t->dst_half & 1) && t->dst_half < 64, "A8 sample: coordinate/destination range");
   const uint8_t mode = t->dim == G5P_TEX_CUBE ? 4 : t->lod_reg ? 2 : 0;
   const uint8_t raw[12] = {0x1a, t->narrow ? 0x59 : 0x79, (uint8_t)t->dst_half, 0x41, 0x00, 0x01,
                            (uint8_t)(0x80 | mode << 4 | t->sampler_qword << 1 | (t->read_flag ? 4 : 0)),
                            lod_byte7, (uint8_t)(t->coord_dword << 1), 0x01, 0x00, 0x80};
   return finish_g4p(out, raw, sizeof(raw));
}

unsigned g5p_usc_sample(uint8_t *out, const struct g5p_tex *t)
{
   if (g5p_usc_g4p)
      return sample_g4p(out, t);
   CHECK(!t->read, "mode must be sample");
   uint8_t raw[G5P_USC_MAX];
   if (!texture_base(raw, t))
      return 0;
   CHECK(t->dim != G5P_TEX_2D_MS || (t->narrow && t->coord_control),
         "the 2D-multisample sample is witnessed only narrow with coordinate control");
   if (t->narrow) {
      CHECK(!t->class_variant && (t->dim == G5P_TEX_2D || t->dim == G5P_TEX_2D_MS),
            "narrow sample witnessed only as 2d and 2d-multisample class0");
      raw[0] &= ~2;
   }
   return finish(out, raw, 6);
}

/* A move of a special value: the layout of g5p_usc_immediate()'s short move, with the source kind "special
 * value" where the immediate sets "literal". Fields, by byte:
 *   0  the single-source class of g5p_usc_move() and g5p_usc_immediate();
 *   1  the header: only the continuation into the destination parcel;
 *   2  the destination half, 3 its bank (destination());
 *   4  bit 0: slot A is not a register; bits 1..6: the special value's index (where the immediate keeps its
 *      bits 0..5); bit 7: zero (the register form's group bit);
 *   5  slot B unused (selector 0) with the continuation;
 *   6  the source kind (G5P_MOV_SOURCE_*); the immediate's long-form bit 0 and literal bits stay zero;
 *   7  the continuation, the immediate's literal bits zero;
 *   8  the move's trailer (mov_trailer, as g5p_usc_move()).
 * The A9 compiler emits this form for two indices: G5P_SPECIAL_OWN_SAMPLE, and 24 for the primitive's facing,
 * which no encoder here takes. */
#define G5P_CLASS_MOVE 0x98
#define G5P_SRC_A_NOT_REGISTER 0x01
#define G5P_SRC_B_UNUSED 0x00
#define G5P_MOV_SOURCE_SPECIAL 0x10 /* observed with the two indices above; the immediate sets 0x20 */
#define G5P_SPECIAL_OWN_SAMPLE 12   /* the sample the task shades */

unsigned g5p_usc_ms_own_sample(uint8_t *out, unsigned dst_half)
{
   CHECK(half_register(dst_half, "the sample operand's destination must be a 32-bit aligned half index"),
         g5p_usc_error);
   const unsigned dst_bank = 0;
   uint8_t raw[12];
   raw[0] = G5P_CLASS_MOVE;
   raw[1] = G5P_HEADER_FRAGMENT;
   if (!destination(raw + 2, dst_half, dst_bank))
      return 0;
   raw[4] = G5P_SRC_A_NOT_REGISTER | G5P_SPECIAL_OWN_SAMPLE << 1;
   raw[5] = G5P_SRC_B_UNUSED | G5P_PARCEL_CONTINUES;
   raw[6] = G5P_MOV_SOURCE_SPECIAL;
   raw[7] = G5P_PARCEL_CONTINUES;
   memcpy(raw + 8, mov_trailer, sizeof(mov_trailer));
   return finish(out, raw, 12);
}

unsigned g5p_usc_texture_lod(uint8_t *out, const struct g5p_tex *t)
{
   /* The A8 takes only the narrow implicit-LOD 2D sample (sample_g4p()) */
   CHECK(!g5p_usc_g4p, "A8 texture sample form not captured");
   CHECK(t->lod_bank == 1 || t->lod_bank == 3, "only LOD source banks 1 (shared) and 3 (temp) are measured");
   CHECK(t->lod_dword <= 95, "lod_dword must be an integer in [0,95]");
   CHECK(!t->narrow, "narrow result is not witnessed with explicit LOD");
   CHECK(!t->gradient || (t->lod_bank == 3 && !t->read), "only temp gradient source on sample is measured");
   CHECK(!t->gradient || t->lod_dword == 0 || t->lod_dword == 3, "gradient base outside witnessed 0/3 is unresolved");
   unsigned high = (t->lod_dword >> 5) << 1;
   unsigned minimum = high ? 14 : t->lod_control ? 10 : 8;
   unsigned length = t->length ? t->length : minimum;
   CHECK((length == 8 || length == 10 || length == 14) && length >= minimum,
         "invalid continuation length for measured texture LOD form");
   uint8_t raw[G5P_USC_MAX];
   if (!texture_base(raw, t))
      return 0;
   CHECK(t->dim == G5P_TEX_2D, "explicit LOD dimension variants are not isolated");
   raw[5] |= 1;
   unsigned n = 6;
   raw[n++] = (t->lod_bank << 6) | (t->gradient ? 0x20 : 0) | (t->lod_dword & 31);
   raw[n++] = length > 8 ? 1 : 0;
   if (length >= 10) {
      raw[n++] = t->lod_control ? 0x80 : 0;
      raw[n++] = length > 10 ? 1 : 0;
   }
   if (length == 14) {
      raw[n++] = 0;
      raw[n++] = 1;
      raw[n++] = high;
      raw[n++] = 0;
   }
   return finish(out, raw, n);
}

unsigned g5p_usc_gather(uint8_t *out, const struct g5p_tex *t)
{
   /* The A8 takes only the narrow implicit-LOD 2D sample (sample_g4p()) */
   CHECK(!g5p_usc_g4p, "A8 texture sample form not captured");
   CHECK(t->dim == G5P_TEX_2D, "only 2d gather is measured");
   CHECK(!t->read && !t->narrow, "gather is a 2d sample form");
   uint8_t raw[G5P_USC_MAX];
   if (!texture_base(raw, t))
      return 0;
   raw[5] |= 1;
   raw[6] = 0;
   raw[7] = 0x40;
   return finish(out, raw, 8);
}

unsigned g5p_usc_sample_ext(uint8_t *out, const struct g5p_tex_ext *t)
{
   /* The A8 takes only the narrow implicit-LOD 2D sample (sample_g4p()) */
   /* a vertex sample takes the A9's recipe on the A8 (g5p_usc.h g5p_usc_g4p_vtex) */
   CHECK(!g5p_usc_g4p || (t->vertex && g5p_usc_g4p_vtex), "A8 texture sample form not captured");
   const bool coupled = t->tuple != G5P_TEX_TUPLE_ORDINARY;
   CHECK((unsigned)t->tuple <= G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL_COMPARE_OFFSET, "unqualified texture recipe");
   CHECK(!coupled || t->lod_shared == (t->tuple == G5P_TEX_TUPLE_COUPLED_SHARED_LEVEL),
         "unqualified independent texture continuation choices");
   static const uint8_t dim_low[G5P_TEX_DIM_COUNT] = {0x5, 0x7, 0xb, 0xd, 0x9};
   static const uint8_t lod_mode[] = {0, 0x40, 0x20, 0x60};
   CHECK(half_register(t->dst_half, "dst_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   /* A9 Metal v_images31 and v_units16_sample witness the complete 7-bit
    * descriptor selector, including image bit6 and sampler bits5/6. */
   CHECK(t->image_qword <= 120 && !(t->image_qword & 3), "image_qword must be a four-qword base in [0,120]");
   CHECK(t->sampler_qword <= 124, "sampler_qword must be an integer in [0,124]");
   CHECK(t->coord_dword <= 31 && t->lod_dword <= 31 && t->extras_half <= 63, "operand beyond r31");
   CHECK(t->dim < G5P_TEX_DIM_COUNT, "unmeasured texture dimension");
   CHECK(t->dim != G5P_TEX_2D_MS, "no general-form 2D-multisample fetch is witnessed");
   CHECK(t->mask == 0xf || (t->mask == 1 && t->compare), "only rgba and compare results are measured");
   CHECK(!t->read || (!t->compare && !t->offset && t->lod == G5P_TEX_LOD_LEVEL && t->dim != G5P_TEX_CUBE),
         "read takes a level and no compare/offset (cube reads are refused by Apple)");
   CHECK(t->lod != G5P_TEX_LOD_GRAD || t->dim == G5P_TEX_2D || t->dim == G5P_TEX_2D_ARRAY || t->dim == G5P_TEX_3D || t->dim == G5P_TEX_CUBE,
         "unmeasured gradient dimension");
   CHECK(!t->offset || t->dim != G5P_TEX_CUBE, "cube offsets do not exist");
   const bool xyz_tuple = t->tuple == G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL_OFFSET_3D;
   const bool compare_tuple = t->tuple == G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL_COMPARE ||
                              t->tuple == G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL_COMPARE_OFFSET;
   const bool offset_tuple = t->tuple == G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL_OFFSET ||
                             t->tuple == G5P_TEX_TUPLE_COUPLED_GRAD_OFFSET || xyz_tuple ||
                             t->tuple == G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL_COMPARE_OFFSET;
   const bool grad_tuple = t->tuple == G5P_TEX_TUPLE_COUPLED_GRAD ||
                           t->tuple == G5P_TEX_TUPLE_COUPLED_GRAD_OFFSET;
   CHECK(!coupled || (t->compare == compare_tuple && t->offset == offset_tuple &&
                     t->lod == (grad_tuple ? G5P_TEX_LOD_GRAD : G5P_TEX_LOD_LEVEL)),
         "vertex texture tuple requires its complete level/offset recipe");
   CHECK(!compare_tuple || (!t->read && !t->lod_shared && t->mask == 1 &&
                           t->dim == G5P_TEX_2D && (!t->offset || t->extras_half <= 61)),
         "vertex compare recipe requires 2D, a temporary level/reference and room for its offset");
   CHECK(!grad_tuple || (!t->read && !t->lod_shared &&
                        (t->dim == G5P_TEX_2D || t->dim == G5P_TEX_2D_ARRAY)),
         "vertex gradient recipe is a 2D or array sample with temporary derivatives");
   CHECK(!offset_tuple || (!t->read && !t->lod_shared &&
                          (xyz_tuple ? t->dim == G5P_TEX_3D :
                           t->dim == G5P_TEX_2D || t->dim == G5P_TEX_2D_ARRAY)),
         "vertex offset recipe requires its witnessed dimension and temporary operands");
   CHECK(!(t->tuple == G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL) || (coupled && !t->lod_shared),
         "vertex level bank requires the vertex tuple and a temporary level");
   uint8_t raw[14] = {0x1a, (t->mask << 4) | dim_low[t->dim], t->dst_half | (t->read ? 0x40 : 0),
                      (t->image_qword << 1) | 1, t->sampler_qword << 1,
                      (t->coord_dword << 1) | 0x80 | (t->offset ? 0x40 : 0)};
   bool extras = t->compare || t->offset;
   if (t->lod == G5P_TEX_LOD_NONE && !extras)
      return finish(out, raw, 6);
   raw[5] |= 1;
   /* Metal native-buffer and manual-fold reads in both stages use
    * common bank1 for level0. Existing register levels keep bank3. */
   raw[6] = t->lod ? (t->lod_shared ? 0 : 0x80) | lod_mode[t->lod] | t->lod_dword : 0;
   raw[7] = 1 | (t->compare ? 0x80 : 0);
   raw[8] = (t->lod ? 0x80 : 0) | (extras ? 0x40 | t->extras_half : 0);
   raw[9] = 0;
   if (coupled) {
      /* Complete native operand/result recipe, reconstructed against the
       * paired Metal programs and exact in the non-cube device cases.
       * Individual destination/coordinate continuation bits are unknown;
       * keep this suffix indivisible rather than guessing mixed forms. */
      raw[9] = (1u << 7) | 1u;
      raw[10] = 1u << 3;
      if (t->tuple == G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL || offset_tuple || grad_tuple || compare_tuple) {
         /* The level producer writes a bank-1 temp. A common-store level
          * (native buffer/manual fold) has no level-bank extension. */
         raw[11] = 1;
         raw[12] = 1u << 4;
         /* Complete native v_{2d,2d_array}_attribute_{pos,neg} recipe.
          * Its coordinate, result, level and packed-offset producers all
          * write bank1. No independent continuation choice is exposed. */
         if (offset_tuple || compare_tuple)
            raw[13] = 1u << 7;
         return finish(out, raw, 14);
      }
      return finish(out, raw, 12);
   }
   return finish(out, raw, 10);
}

/* A9 compute's array or cube sample with the level in a register (Apple's A9 compiler's form): the coupled
 * temporary-level tuple of g5p_usc_sample_ext(). Its fetch form (read) keeps the coordinate operand live: it
 * clears the coordinate's last-read bit, which every other sample form sets (texture_base()'s coordinate byte). */
#define G5P_TEX_COORD_BYTE 5
#define G5P_TEX_COORD_LAST_READ 0x80
unsigned g5p_usc_sample_register_lod(uint8_t *out, const struct g5p_tex_ext *t)
{
   /* The A8 takes only the narrow implicit-LOD 2D sample (sample_g4p()) */
   CHECK(!g5p_usc_g4p, "A8 texture sample form not captured");
   CHECK(t->tuple == G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL && t->lod == G5P_TEX_LOD_LEVEL && !t->lod_shared &&
         (t->dim == G5P_TEX_2D_ARRAY || t->dim == G5P_TEX_CUBE),
         "the register-LOD form is an array or cube sample at a level");
   const unsigned n = g5p_usc_sample_ext(out, t);
   if (n && t->read)
      out[G5P_TEX_COORD_BYTE] &= ~G5P_TEX_COORD_LAST_READ;
   return n;
}

unsigned g5p_usc_half_immediate(uint8_t *out, unsigned dst_half, uint16_t value)
{
   CHECK(dst_half <= 63, "dst_half beyond r31");
   CHECK(value < 1024, "16-bit immediate bits 10-15 are not witnessed in class 0x88");
   uint8_t raw[12] = {0x88, 0x01, dst_half, 0x01, 0x01, 0x01, 0x20, 0x01, 0x28, 0x01, 0x39, 0x00};
   for (unsigned b = 0; b < 10; b++)
      if (value >> b & 1)
         raw[imm_byte[b]] |= 1 << imm_bit[b];
   return finish(out, raw, 12);
}

unsigned g5p_usc_vertex_half_immediate(uint8_t *out, unsigned dst_half, uint16_t value)
{
   uint8_t raw[12];
   unsigned n = g5p_usc_half_immediate(raw, dst_half, value);
   if (!n || !destination(raw + 2, dst_half, 1))
      return 0;
   return finish(out, raw, n);
}

unsigned g5p_usc_half_immediate_xyz(uint8_t *out, unsigned dst_half, uint16_t value, unsigned bank)
{
   CHECK(value < 4096, "XYZ packed offset is three signed nibbles");
   uint8_t raw[12];
   unsigned n = g5p_usc_half_immediate(raw, dst_half, value & 1023);
   if (!n || !destination(raw + 2, dst_half, bank))
      return 0;
   /* Native v_{uniform,attribute}_axis2_bit{2,3} independently witness
    * literal bits10/11 at the existing split-immediate map positions. */
   for (unsigned b = 10; b < 12; b++)
      if ((value >> b) & 1)
         raw[imm_byte[b]] |= 1u << imm_bit[b];
   return finish(out, raw, n);
}

unsigned g5p_usc_cube_coords(uint8_t *out, unsigned dst_half, unsigned src_half)
{
   CHECK(half_register(dst_half, "dst_half requires a measured 32-bit aligned half index") &&
         half_register(src_half, "src_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   uint8_t raw[6] = {0x08, 0x7d, dst_half, (src_half << 1) | 1, 0x00, 0x80};
   return finish(out, raw, 6);
}

unsigned g5p_usc_cube_gradient_tuple(uint8_t *out)
{
   /* Native f_cube_grad_{float,int,uint} and cube0..3 controls: the
    * gradient form of the cube operation has an indivisible r0..r8 ABI.
    * Do not infer independent source/destination relocation fields. */
   unsigned n = g5p_usc_cube_coords(out, 0, 12);
   if (n) out[1] |= 2;
   return n;
}

unsigned g5p_usc_cube_coords_bank1(uint8_t *out, unsigned dst_half, unsigned src_half)
{
   /* v_cube_sample_{attribute,uniform_lod_attribute}: bank0 direction
    * producers precede this exact form; the coupled sample reads bank1.
    * The native live LOD immediately after its three outputs bounds the
    * result to three dwords. Source direction stays in bank0. */
   unsigned n = g5p_usc_cube_coords(out, dst_half, src_half);
   if (n) out[5] |= 1u << 2;
   return n;
}

unsigned g5p_usc_f2u16_half(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned group)
{
   CHECK(dst_half <= 63, "dst_half beyond r31");
   CHECK(half_register(src_half, "src_half requires a measured 32-bit aligned half index") && group <= 1,
         g5p_usc_error);
   uint8_t raw[10] = {0x89, 0x01, dst_half, 0x01, 0x09, (group << 7) | (src_half << 1) | 1, 0x20, 0x01,
                      0x60, 0x06};
   return finish(out, raw, 10);
}

static int mem_code(unsigned byte_count)
{
   switch (byte_count) {
   case 1: return 0x0a;
   case 2: return 0x12;
   case 4: return 0x22;
   case 8: return 0x2a;
   case 16: return 0x3a;
   default: return -1;
   }
}

unsigned g5p_usc_buffer_load(uint8_t *out, unsigned dst_half, unsigned byte_count,
                             unsigned base_common_dword, unsigned index, unsigned length)
{
   CHECK(half_register(dst_half, "dst_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   int code = mem_code(byte_count);
   CHECK(code >= 0, "unmeasured load size");
   CHECK(base_common_dword <= 2, "base_common_dword must be an integer in [0,2]");
   CHECK(!(base_common_dword & 1), "64-bit pointer base must be pair aligned");
   CHECK(index <= 255, "index must be an integer in [0,255]");
   unsigned minimum = index ? 6 : (dst_half || base_common_dword) ? 4 : 2;
   if (!length)
      length = minimum;
   CHECK((length == 2 || length == 4 || length == 6) && length >= minimum,
         "invalid fixed load continuation length");
   uint8_t raw[6] = {0x12, code | (length > 2), dst_half, (base_common_dword << 1) | (length > 4), index, 0};
   return finish(out, raw, length);
}

unsigned g5p_usc_address_load(uint8_t *out, unsigned dst_half, unsigned byte_count,
                              unsigned address_half, unsigned length)
{
   CHECK(half_register(dst_half, "dst_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   CHECK(half_register(address_half, "address_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   CHECK(byte_count == 4 || byte_count == 16, "only 4/16-byte addressed fragment loads are isolated");
   unsigned minimum = (dst_half || address_half) ? 4 : 2;
   if (!length)
      length = minimum;
   CHECK((length == 2 || length == 4) && length >= minimum, "invalid addressed load continuation length");
   uint8_t raw[4] = {0x10, mem_code(byte_count) | (length > 2), dst_half, address_half << 1};
   return finish(out, raw, length);
}

/* The A8 has no 64-bit address add: AGXMetalA8 never emits class 0x89 (none in the iPod touch 6 shader and pass
 * captures), the A10/A9 form faults in A8 compute, and Apple's A8 builds base + offset from 32-bit adds
 * with the carry's high half (g9p_addr()). That build needs a scratch register the fragment/vertex allocator does
 * not provide here yet, so graphics programs that index a buffer are refused on the A8. */
#define G5P_A8_NO_ADDR_ADD "the A8 has no 64-bit address add (Apple's 32-bit carry build is not ported to " \
                           "graphics programs)"

/* `89 11` (vertex) / `89 01` (fragment): pair = pointer pair at common dword
 * `base` + (index << shift). Bytes from Apple's own compiles of a constant
 * buffer indexed by a register (tools/mesa-a9-gles3tex/dynubo.metal: v_cexpr,
 * v_cbase `89 11 00 03 81 81 00 41 a1 01 00 86`, f_cflat `89 01 00 01 ...`);
 * the same fields as the A10 port's g9p_addr (dst, 0x81 | 4 * base,
 * 1 | index << 2 | kill << 7, shift in b11[7:5]). The shift field is
 * corpus-verified 0-4 (isa-g5p.md); kill marks the index's last read. */
unsigned g5p_usc_address_add_shift(uint8_t *out, unsigned pair_half, unsigned base_common_dword,
                                   unsigned index_half, unsigned shift, bool kill, bool vs)
{
   CHECK(half_register(pair_half, "pair_half requires a 64-bit aligned register pair") && !(pair_half & 2),
         "pair_half requires a 64-bit aligned register pair");
   CHECK(half_register(index_half, "index_half requires a measured 32-bit register"), g5p_usc_error);
   CHECK(base_common_dword < 256 && !(base_common_dword & 1), "pointer base must be an even common dword below 256");
   CHECK(shift <= 4, "only shifts 0-4 are corpus-verified");
   CHECK(!g5p_usc_g4p, G5P_A8_NO_ADDR_ADD);
   /* The layout of g5p_usc_alu_shared(): alu_header() (vertex code: destination bank 1, and B's bank bit in
    * the header), A = the pointer base (shared_src(), group 1; above dword 31 its high bits are the field that
    * the vertex XFB record's address add and every shared ALU operand use), B = the index with its last read,
    * the integer add, then the last parcel: no routing, and the control byte with the shift. */
   uint8_t raw[12], a[4];
   if (!alu_header(raw, G5P_CLASS_ADDR_ADD, pair_half, vs, G5P_HEADER_FRAGMENT | (vs ? G5P_HEADER_SRC_B_BANK1 : 0),
                   0, 0) ||
       !shared_src(a, base_common_dword, 1))
      return 0;
   raw[4] = a[0];
   raw[5] = (index_half << 1) | (kill ? G5P_SRC_LAST_READ : 0) | G5P_PARCEL_CONTINUES;
   raw[6] = a[2];
   raw[7] = a[3];
   raw[8] = binop_code[G5P_BIN_IADD];
   raw[9] = G5P_PARCEL_CONTINUES;
   raw[10] = G5P_ADDR_ADD_ROUTING_NONE;
   raw[11] = G5P_ADDR_ADD_CTL | shift << G5P_ADDR_ADD_SHIFT_BIT;
   return finish(out, raw, 12);
}

/* Vertex class-10 load from an address pair, six bytes: `10 3b 00 01 00 88`
 * (dynubo.metal v_cbase; G9P's g9p_mem_at vertex form). b4 is 0 here
 * and in G9P's; Apple's v_cexpr has 1 there (unresolved). */
unsigned g5p_usc_vertex_address_load(uint8_t *out, unsigned dst_half, unsigned byte_count,
                                     unsigned address_half)
{
   CHECK(half_register(dst_half, "dst_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   CHECK(half_register(address_half, "address_half requires a measured 32-bit aligned half index") &&
         !(address_half & 2), "address_half requires a 64-bit aligned register pair");
   CHECK(byte_count == 4 || byte_count == 16, "only 4/16-byte vertex loads are witnessed");
   uint8_t raw[6] = {0x10, mem_code(byte_count) | 1, dst_half, address_half << 1 | 1, 0x00, 0x88};
   return finish(out, raw, 6);
}

/* ---- ISA round 2: relations, selections, SFU cores, discard (g5p_usc.py) ---- */
static unsigned ctl_head(uint8_t *raw, unsigned cls, unsigned dst_half, struct g5p_ctl k)
{
   CHECK(k.source_extension == 0 || k.source_extension == 4, "unmeasured source class extension");
   CHECK(header_ok(k.header), g5p_usc_error);
   raw[0] = cls | k.source_extension;
   raw[1] = k.header;
   if (!cdestination(raw + 2, dst_half, k.bank, k.dst_control))
      return 0;
   return 4;
}

unsigned g5p_usc_compare_write(uint8_t *out, enum g5p_cmp op, unsigned dst_half, struct g5p_src a,
                               struct g5p_src b, struct g5p_src c, struct g5p_ctl k,
                               unsigned relation_control)
{
   CHECK(op < G5P_CMP_COUNT, "unsupported compare/write operation/class");
   CHECK(relation_control == 0 ||
         (relation_control == 0x10 && (op == G5P_FLT || op == G5P_FGT || op == G5P_FEQ)),
         "compare relation extension measured for float only");
   uint8_t raw[G5P_USC_MAX];
   if (!ctl_head(raw, 0x9a, dst_half, k))
      return 0;
   SEL(sa, a.half, a.group, 0);
   SEL(sb, b.half, b.group, 1);
   SEL(sc, c.half, c.group, 0);
   raw[4] = sa;
   raw[5] = sb;
   raw[6] = sc;
   raw[7] = 1;
   raw[8] = cmp_code[op];
   raw[9] = relation_control ? 0x61 : 0x60;
   unsigned n = 10;
   if (relation_control) {
      raw[n++] = 0;
      raw[n++] = relation_control;
   }
   return finish(out, raw, n);
}

unsigned g5p_usc_predicated_move(uint8_t *out, unsigned dst_half, struct g5p_src s, struct g5p_ctl k)
{
   uint8_t raw[G5P_USC_MAX];
   if (!ctl_head(raw, 0xf8, dst_half, k))
      return 0;
   SEL(ss, s.half, s.group, 0);
   raw[4] = ss;
   raw[5] = 1;
   memcpy(raw + 6, mov_trailer, 4);
   return finish(out, raw, 10);
}

unsigned g5p_usc_float_minmax_test(uint8_t *out, bool max, unsigned dst_half, struct g5p_src a,
                                   struct g5p_src b, unsigned bank, unsigned header)
{
   uint8_t raw[G5P_USC_MAX];
   CHECK(header_ok(header), g5p_usc_error);
   raw[0] = 0x9d;
   raw[1] = header;
   if (!destination(raw + 2, dst_half, bank))
      return 0;
   SEL(sa, a.half, a.group, 0);
   SEL(sb, b.half, b.group, 1);
   uint8_t tail[4] = {0x58, max ? 0x81 : 0x41, 8, 0};
   raw[4] = sa;
   raw[5] = sb;
   memcpy(raw + 6, tail, 4);
   return finish(out, raw, 10);
}

unsigned g5p_usc_minmax_merge(uint8_t *out, unsigned dst_half, struct g5p_src s, unsigned bank,
                              unsigned header)
{
   uint8_t raw[G5P_USC_MAX];
   CHECK(header_ok(header), g5p_usc_error);
   raw[0] = 0x98;
   raw[1] = header;
   if (!destination(raw + 2, dst_half, bank))
      return 0;
   SEL(ss, s.half, s.group, 0);
   uint8_t tail[5] = {1, 0x55, 0x61, 0x38, 0};
   raw[4] = ss;
   memcpy(raw + 5, tail, 5);
   return finish(out, raw, 10);
}

unsigned g5p_usc_sfu_core(uint8_t *out, enum g5p_sfu op, unsigned dst_half, struct g5p_src s,
                          struct g5p_ctl k, bool absolute)
{
   CHECK(op < G5P_SFU_COUNT && (!absolute || op == G5P_RSQRT_APPROX), "unmeasured SFU core");
   uint8_t raw[G5P_USC_MAX];
   if (!ctl_head(raw, 0x98, dst_half, k))
      return 0;
   SEL(ss, s.half, s.group, 0);
   raw[4] = ss;
   raw[5] = 1;
   raw[6] = sfu_code[op];
   raw[7] = absolute;
   unsigned n = 8;
   if (absolute) {
      raw[n++] = 0;
      raw[n++] = 0x80;
   }
   return finish(out, raw, n);
}

unsigned g5p_usc_square(uint8_t *out, unsigned dst_half, struct g5p_src s, struct g5p_ctl k)
{
   uint8_t raw[G5P_USC_MAX];
   if (!ctl_head(raw, 0x98, dst_half, k))
      return 0;
   SEL(ss, s.half, s.group, 0);
   uint8_t tail[5] = {1, 0x48, 1, 0x30, 0};
   raw[4] = ss;
   memcpy(raw + 5, tail, 5);
   return finish(out, raw, 10);
}

unsigned g5p_usc_discard_test(uint8_t *out, enum g5p_cmp op, unsigned dst_half, struct g5p_src a,
                              struct g5p_src b, unsigned relation_control)
{
   CHECK((op == G5P_FEQ || op == G5P_FLT || op == G5P_FGT) &&
         (relation_control == 0 || relation_control == 0x10), "unmeasured discard test");
   uint8_t raw[G5P_USC_MAX];
   raw[0] = 0x89;
   raw[1] = 1;
   if (!destination(raw + 2, dst_half, 0))
      return 0;
   SEL(sa, a.half, a.group, 0);
   SEL(sb, b.half, b.group, 1);
   raw[4] = sa;
   raw[5] = sb;
   raw[6] = op == G5P_FEQ ? 0x56 : op == G5P_FLT ? 0x5a : 0x5e;
   raw[7] = 0x42 | (relation_control ? 1 : 0);
   unsigned n = 8;
   if (relation_control) {
      raw[n++] = 0;
      raw[n++] = relation_control;
   }
   return finish(out, raw, n);
}

/* The 4x discard test is a packed ternary of g5p_usc_literal_ternary()'s layout: the 16-bit-result class,
 * A a long literal (literal_prefix()), B the tested value, C a short literal, then the opcode, the operation
 * control and a last parcel of routing and relation. Its values:
 *   A  -0.0f, the literal Apple's compiler compares against zero with (as the rsqrt and sign recipes);
 *   C  the kill's sample mask, all four samples;
 *   opcode  the float-equality compare of g5p_usc_compare_write();
 *   control  bit 1 the 16-bit result (as g5p_usc_binary()'s result_half flag); bit 4 observed with a short C
 *            literal (set in this form and in the cos-scale and bool-to-half recipes, clear in the rsqrt and
 *            sign recipes' long C literals; the rcp-refine recipe also sets it, with a register C); bit 0 the
 *            continuation into the last parcel;
 *   routing  the low byte, the routing of the literal forms (g5p_usc_binary_literal2());
 *   relation  the high byte, as relation_control in g5p_usc_discard_test() and g5p_usc_compare_write(): 0, so
 *             with this opcode the test is A != B, as Apple's compiler emits it for (v != 0.0f) discard. The ==
 *             relation (0x10) killed the samples a discard should keep on the device. Its continuation bit is
 *             clear: the parcel ends the instruction. */
#define G5P_F32_NEG_ZERO 0x80000000u
#define G5P_MS4_ALL_SAMPLES 0xfu
#define G5P_TERNARY_CTL_C_SHORT 0x10
#define G5P_ROUTING_LITERAL 0x70
#define G5P_RELATION_NE 0x00

unsigned g5p_usc_discard_test_ms4(uint8_t *out, unsigned dst_half, struct g5p_src a)
{
   CHECK(!a.neg && !a.abs, "the 4x discard test is witnessed without source modifiers");
   const unsigned dst_bank = 0;
   const int a_long = 1, c_long = 0;
   uint8_t raw[G5P_USC_MAX], c[G5P_USC_MAX];
   unsigned n = literal_prefix(raw, G5P_CLASS_TERNARY_RESULT_HALF, dst_half, dst_bank, G5P_F32_NEG_ZERO, a.half,
                               a.group, G5P_HEADER_FRAGMENT, a_long);
   if (!n)
      return 0;
   /* C's literal: its parcels from byte 4 on of a literal prefix (as g5p_usc_literal_ternary()). */
   unsigned m = literal_prefix(c, 0, 0, 0, G5P_MS4_ALL_SAMPLES, 0, 0, G5P_HEADER_FRAGMENT, c_long);
   if (!m)
      return 0;
   memcpy(raw + n, c + 4, m - 4);
   n += m - 4;
   raw[n++] = cmp_code[G5P_FEQ];
   raw[n++] = G5P_TERNARY_CTL_RESULT_HALF | G5P_TERNARY_CTL_C_SHORT | G5P_PARCEL_CONTINUES;
   raw[n++] = G5P_ROUTING_LITERAL;
   raw[n++] = G5P_RELATION_NE;
   return finish(out, raw, n);
}

unsigned g5p_usc_discard(uint8_t *out, int src_half)
{
   if (src_half < 0) {
      uint8_t raw[2] = {4, 0x68};
      return finish(out, raw, 2);
   }
   CHECK(src_half < 256, "discard source half must be u8");
   uint8_t raw[4] = {4, 0x61, src_half, 0};
   return finish(out, raw, 4);
}

unsigned g5p_usc_depth_feedback(uint8_t *out, unsigned src_half)
{
   CHECK(src_half < 256 && !(src_half & 1), "depth feedback source must be a dword half-index");
   /* DEPTHF, unlike coverage/discard feedback, consumes an f32 depth. */
   uint8_t raw[4] = {4, 0x21, src_half, 0};
   return finish(out, raw, sizeof(raw));
}

/* (opcode, operation-control, routing) tuples observed in the fast and
 * strict corpus (PACKED_TERNARY_CONTROLS); routing -1 = no parcel. */
static const struct { uint8_t op, ctl; int route; } lit3_seen[] = {
   {0x4c, 0, 0x70}, {0x4c, 0, 0x58}, {0x4c, 8, 0x70}, {0x4c, 0x10, 0x70}, {0x4c, 0x10, 0x58},
   {0x4c, 0x20, 0x70}, {0x4c, 0x28, 0x70}, {0x4c, 0x90, 0x58},
   {0x55, 0x12, 0x8070}, {0x55, 0x20, 0x70}, {0x55, 0x20, 0x8070}, {0x55, 0x40, 0x58}, {0x55, 0x40, 0x1058},
   {0x58, 0, 0x3070}, {0x58, 0, 0x2058}, {0x58, 0, 0x2070}, {0x58, 0x1a, 0x2070}, {0x58, 0x40, -1},
   {0x59, 0, 0x9070}, {0x59, 0, 0x70}, {0x59, 0x12, 0x70}, {0x59, 0x12, 0x8070}, {0x59, 0x20, 0x8070},
   {0x59, 0x20, 0x9070}, {0x59, 0x20, 0x1070}, {0x59, 0x32, 0x1070}, {0x59, 0x40, 0x1058},
   {0x59, 0x40, 0x58}, {0x59, 0x52, 0x1058}, {0x59, 0x52, 0xd058},
   {0x5c, 0, 0x2070}, {0x5c, 0x40, 0x4000}, {0x5c, 0x40, 0x5000}, {0x5c, 0x40, 0x1000}, {0x5c, 0x40, -1},
   {0x5d, 0, 0x70}, {0x5d, 0x20, 0x70}, {0x5d, 0x40, 0x1058}, {0x5d, 0x40, 0x58},
   {0xb2, 0x60, 0x58}, {0xb2, 0x60, 0x2058}, {0xb3, 0x60, 0x58}, {0xb3, 0x60, 0x2058},
   {0xb9, 0x20, -1}, {0xb9, 0x60, 0x58}, {0xb9, 0x60, 0x2058}, {0xb9, 0x72, 0x2070},
   {0xba, 0x60, 0x58}, {0xba, 0x60, 0x2058}, {0xbb, 0x60, 0x58}, {0xbb, 0x60, 0x2058},
   {0xbd, 0x60, -1}, {0xbd, 0x60, 0x70}, {0xbd, 0x60, 0x2070}, {0xbd, 0x6c, 0x70}, {0xbd, 0x6c, 0x2070},
};

unsigned g5p_usc_literal_ternary(uint8_t *out, const struct g5p_lit3 *t)
{
   static const uint8_t ops[] = {0x4c, 0x55, 0x58, 0x59, 0x5c, 0x5d, 0xb2, 0xb3, 0xb9, 0xba, 0xbb, 0xbd};
   bool known = false, seen = false;
   for (unsigned i = 0; i < sizeof(ops); i++)
      known |= ops[i] == t->opcode;
   CHECK(known && (t->k.source_extension == 0 || t->k.source_extension == 4),
         "unmeasured packed ternary opcode/class");
   CHECK(t->operation_control <= 255, "operation_control must be u8");
   CHECK(!(t->operation_control & 1), "continuation is derived from routing");
   for (unsigned i = 0; i < sizeof(lit3_seen) / sizeof(lit3_seen[0]); i++)
      seen |= lit3_seen[i].op == t->opcode && lit3_seen[i].ctl == t->operation_control &&
              lit3_seen[i].route == t->routing;
   CHECK(seen, "unobserved packed operation-control/routing pair");
   CHECK(t->routing < 0 || !(t->routing & 0x100), "routing parcel must terminate");
   uint8_t raw[24];
   unsigned n = literal_prefix(raw, (t->result_half ? 0x8a : 0x9a) | t->k.source_extension, t->dst_half,
                               t->k.bank, t->value, t->b_half, t->b_group, t->k.header, t->long_form);
   if (!n)
      return 0;
   if (!cdestination(raw + 2, t->dst_half, t->k.bank, t->k.dst_control))
      return 0;
   if (t->c_literal) {
      CHECK(!(t->c_value & 0x200), "C-literal bit9 is not independently mapped");
      uint8_t c[12];
      unsigned m = literal_prefix(c, 0, 0, 0, t->c_value, 0, 0, 1, t->c_long_form);
      if (!m)
         return 0;
      memcpy(raw + n, c + 4, m - 4);
      n += m - 4;
   } else {
      CHECK(t->c_long_form < 0, "C length applies only to a literal");
      SEL(sc, t->c_half, t->c_group, 0);
      raw[n++] = sc;
      raw[n++] = 1;
   }
   raw[n++] = t->opcode;
   raw[n++] = t->operation_control | (t->routing >= 0);
   if (t->routing >= 0) {
      raw[n++] = t->routing & 255;
      raw[n++] = t->routing >> 8;
   }
   CHECK(n <= G5P_USC_MAX, "instruction too long");
   return finish(out, raw, n);
}

/* Strict compute-speed fma4/fma4_fs witnesses. C bit9 remains unmapped;
 * callers retain the register-source path for those constants. */
unsigned g5p_usc_fma_f32_literals(uint8_t *out, unsigned dst_half, unsigned b_half,
                                   uint32_t multiplier, uint32_t addend, unsigned bank)
{
   const struct g5p_lit3 t = {
      .opcode = 0x4c, .dst_half = dst_half, .value = multiplier,
      .b_half = b_half, .b_group = 0, .c_literal = true, .c_value = addend,
      .k = {.bank = bank, .header = bank ? 0x11 : 1},
      .operation_control = 0, .routing = 0x70, .long_form = 1, .c_long_form = 1,
   };
   CHECK(bank <= 1, "invalid FMA register bank");
   return g5p_usc_literal_ternary(out, &t);
}

unsigned g5p_usc_repeat_fma(uint8_t *out, unsigned dst_half, struct g5p_src a, struct g5p_src b,
                            struct g5p_ctl k)
{
   uint8_t raw[G5P_USC_MAX];
   if (!ctl_head(raw, 0x99, dst_half, k))
      return 0;
   SEL(sa, a.half, a.group, 0);
   SEL(sb, b.half, b.group, 1);
   uint8_t tail[4] = {0x4c, 1, 0x0c, 0};
   raw[4] = sa;
   raw[5] = sb;
   memcpy(raw + 6, tail, 4);
   return finish(out, raw, 10);
}

unsigned g5p_usc_predicated_sfu(uint8_t *out, unsigned dst_half, struct g5p_src s, unsigned bank,
                                unsigned header)
{
   /* 0x21: the vertex form (v_rsq: the source is a bank-1 temp) */
   CHECK(header == 1 || header == 0x21, "only the predicated rsqrt core is characterized");
   uint8_t raw[G5P_USC_MAX];
   raw[0] = 0xf8;
   raw[1] = header;
   if (!destination(raw + 2, dst_half, bank))
      return 0;
   SEL(ss, s.half, s.group, 0);
   raw[4] = ss;
   raw[5] = 1;
   raw[6] = 0x81;
   raw[7] = 0;
   return finish(out, raw, 8);
}

unsigned g5p_usc_bool_to_half(uint8_t *out, unsigned dst_half, unsigned src_half)
{
   struct g5p_lit3 t = {.opcode = 0xb9, .dst_half = dst_half, .value = 0, .b_half = src_half, .b_group = 1,
                        .c_literal = true, .c_value = 0x3c00, .k = {.header = 1, .dst_control = 0x80},
                        .operation_control = 0x72, .routing = 0x2070, .long_form = 0, .c_long_form = 0,
                        .result_half = true};
   return g5p_usc_literal_ternary(out, &t);
}

unsigned g5p_usc_cond_select(uint8_t *out, unsigned dst_half, struct g5p_src cond, struct g5p_src t,
                             struct g5p_src f, unsigned bank, unsigned source_extension, unsigned header)
{
   CHECK(source_extension == 0 || source_extension == 4, "unmeasured conditional select class");
   uint8_t raw[G5P_USC_MAX];
   CHECK(header_ok(header), g5p_usc_error);
   raw[0] = 0x9a | source_extension;
   raw[1] = header;
   if (!destination(raw + 2, dst_half, bank))
      return 0;
   SEL(sc, cond.half, cond.group, 0);
   SEL(st, t.half, t.group, 1);
   SEL(sf, f.half, f.group, 0);
   uint8_t tail[5] = {1, 0x54, 5, 0, 0x20};
   raw[4] = sc;
   raw[5] = st;
   raw[6] = sf;
   memcpy(raw + 7, tail, 5);
   return finish(out, raw, 12);
}

/* ---- ISA round 2: raster position, tile access, f16 blend arithmetic ---- */
unsigned g5p_usc_position_xy(uint8_t *out, unsigned dst_half, unsigned words, unsigned source_half)
{
   CHECK((words == 1 || words == 2) && (source_half == 0 || source_half == 2) &&
         !(words == 2 && source_half), "unmeasured raster XY range");
   CHECK(dst_half < 256, "position destination must be u8");
   uint8_t raw[6] = {0x1c, (words << 4) | 1, dst_half, source_half ? 0x69 : 0x68, source_half, 0};
   return finish(out, raw, source_half ? 6 : 4);
}

/* The screen-linear position iterator: one coefficient set (set 0, or set 1
 * with set1), or sets 0 and 1 together (both). The last byte is the set << 1,
 * as in the varying iterator: Apple puts W at set 0 and Z at set 1, or Z at
 * set 0 when no W is iterated (Apple's compiles f_raster_pos_z/w). */
unsigned g5p_usc_position_zw(uint8_t *out, unsigned dst_half, bool set1, bool both, int long_form)
{
   CHECK(!(both && set1), "invalid position iterator");
   CHECK(!(dst_half & 1), "position iterator needs a dword-aligned destination");
   CHECK(dst_half < 256, "position destination must be u8");
   bool minimum = dst_half || set1 || both;
   bool wide = long_form < 0 ? minimum : long_form;
   CHECK(!(minimum && !wide), "short position iterator omits nonzero fields");
   if (g5p_usc_g4p && g5p_usc_g4p_pos_iter) {
      /* The A8: Apple's A8 compiler reads the position's W and Z with its varying iterator `18 ..`, screen-linear
       * (byte 3 without 0x80), one f32 component. Set 0 is `18 09 DD 00` with no slot byte, as Apple reads W next
       * to a perspective varying. Set 1 (Z after W) is inferred: the slot byte set << 2 after byte 3 = 0x01, the
       * f32 screen-linear shape of Apple's varying reads at other sets (Apple's own set-1 position read is the f16
       * pair `18 11 02 01 00 01 04 00`). The coefficient side (DOUTI items, TA_OUTPUT_SEL with unclamped Z) is the
       * A9's, the same words as Apple's A8 output. */
      CHECK(!both, "A8 position iterator: both sets in one read not captured in f32");
      const uint8_t g4[6] = {0x18, 0x09, (uint8_t)dst_half, set1 ? 0x01 : 0x00, 0x04, 0x00};
      return finish_g4p(out, g4, set1 ? 6 : 4);
   }
   uint8_t raw[4] = {8, (both ? 0x10 : 8) | wide, dst_half, set1 ? 2 : 0};
   return finish(out, raw, wide ? 4 : 2);
}

unsigned g5p_usc_tile_read(uint8_t *out, unsigned dst_half, unsigned words)
{
   CHECK(words == 1 || words == 2 || words == 4, "unmeasured tile-read word count");
   CHECK(dst_half < 256, "tile destination must be u8");
   uint8_t raw[4] = {0x1c, (words << 4) | 1, dst_half, 0x28};
   return finish(out, raw, 4);
}

unsigned g5p_usc_half_binary(uint8_t *out, bool mul, unsigned dst_half, struct g5p_src a, struct g5p_src b,
                             struct g5p_ctl k, int source_control)
{
   CHECK(source_control < 0 || source_control == 0x40 || source_control == 0x80, "unmeasured half binary form");
   uint8_t raw[G5P_USC_MAX];
   if (!ctl_head(raw, 0x89, dst_half, k))
      return 0;
   SEL(sa, a.half, a.group, 0);
   SEL(sb, b.half, b.group, 1);
   raw[4] = sa;
   raw[5] = sb;
   raw[6] = mul ? 8 : 4;
   raw[7] = source_control >= 0;
   unsigned n = 8;
   if (source_control >= 0) {
      raw[n++] = 0;
      raw[n++] = source_control;
   }
   return finish(out, raw, n);
}

unsigned g5p_usc_half_fma(uint8_t *out, unsigned dst_half, struct g5p_src a, struct g5p_src b,
                          struct g5p_src c, struct g5p_ctl k, unsigned source_control)
{
   CHECK(source_control == 0x80, "unmeasured half FMA control");
   uint8_t raw[G5P_USC_MAX];
   if (!ctl_head(raw, 0x8a, dst_half, k))
      return 0;
   SEL(sa, a.half, a.group, 0);
   SEL(sb, b.half, b.group, 1);
   SEL(sc, c.half, c.group, 0);
   uint8_t tail[5] = {1, 0x0c, 1, 0, 0x80};
   raw[4] = sa;
   raw[5] = sb;
   raw[6] = sc;
   memcpy(raw + 7, tail, 5);
   return finish(out, raw, 12);
}

unsigned g5p_usc_half_one_minus(uint8_t *out, unsigned dst_half, unsigned src_half)
{
   uint8_t raw[G5P_USC_MAX];
   unsigned n = literal_prefix(raw, 0x8d, dst_half, 0, 0x3c00, src_half, 0, 0x41, 0);
   if (!n)
      return 0;
   raw[3] |= 0x80;
   uint8_t tail[4] = {4, 0x81, 0x70, 0};
   memcpy(raw + n, tail, 4);
   return finish(out, raw, n + 4);
}

unsigned g5p_usc_tile_component(uint8_t *out, unsigned dst_half, unsigned src_half)
{
   uint8_t raw[G5P_USC_MAX];
   raw[0] = 0x89;
   raw[1] = 1;
   if (!destination(raw + 2, dst_half, 0))
      return 0;
   SEL(ss, src_half, 1, 1);
   raw[4] = 1;
   raw[5] = ss;
   raw[6] = 0x20;
   raw[7] = 1;
   raw[8] = 0x28;
   raw[9] = 0x81;
   raw[10] = 0x4d;
   raw[11] = 0;
   return finish(out, raw, 12);
}

/* ---- Native-f16 blend-epilogue forms (g5p_usc.py, blend-forms.md) ---- */
static bool plain_src(struct g5p_src s)
{
   if (s.neg || s.abs) {
      g5p_usc_error = "source modifiers are named arguments in the half blend forms";
      return false;
   }
   return true;
}

#define PLAIN(s) CHECK(plain_src(s), g5p_usc_error)

/* Literal prefix with destination control and class extension (the
 * reference's _literal_prefix followed by _controlled_destination). */
static unsigned half_literal_head(uint8_t *raw, unsigned cls, unsigned dst_half, uint16_t value,
                                  struct g5p_src b, struct g5p_ctl k)
{
   CHECK(k.source_extension == 0 || k.source_extension == 4, "unmeasured half source class extension");
   CHECK(k.dst_control == 0 || k.dst_control == 0x80, "unmeasured destination control");
   unsigned n = literal_prefix(raw, cls | k.source_extension, dst_half, k.bank, value, b.half, b.group,
                               k.header, 0);
   if (n)
      raw[3] |= k.dst_control;
   return n;
}

/* _common_source: S(s,g) with byte 2 control 0x40 (every half read). */
static bool half_common(uint8_t *raw, unsigned a_dword, unsigned group)
{
   if (a_dword > 255 || group > 1) {
      g5p_usc_error = a_dword > 255 ? "common dword must be u8" : "common group must be u1";
      return false;
   }
   raw[0] = 1 | (group << 7) | ((a_dword & 31) << 2);
   raw[1] = ((a_dword >> 5) << 1) | 0x40;
   raw[2] = 0x41;
   return true;
}

unsigned g5p_usc_tile_read_component(uint8_t *out, unsigned dst_half, unsigned words,
                                     unsigned first_component)
{
   /* ios-witness-r1 mrtm-blend1: two words from component4;
    * mrtm-3bm: three from component6. Component indices are f16 halves. */
   CHECK(words >= 1 && words <= 4 && !(first_component & 1) && first_component + 2 * words <= 16,
         "tile-read range outside four RGBA targets");
   CHECK(dst_half < 256, "tile destination must be u8");
   uint8_t raw[6] = {0x1c, (words << 4) | 1, dst_half, 0x29, first_component, 0};
   return finish(out, raw, 6);
}

unsigned g5p_usc_tile_component2(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned dst_control)
{
   uint8_t raw[G5P_USC_MAX];
   unsigned n = g5p_usc_tile_component(raw, dst_half, src_half);
   if (!n)
      return 0;
   if (!cdestination(raw + 2, dst_half, 0, dst_control))
      return 0;
   return finish(out, raw, n);
}

unsigned g5p_usc_half_binary2(uint8_t *out, bool mul, unsigned dst_half, struct g5p_src a,
                              struct g5p_src b, struct g5p_ctl k, int source_control, bool negate_b)
{
   CHECK(source_control < 0 || source_control == 0x40 || source_control == 0x80 || source_control == 0xc0,
         "unmeasured half binary form");
   PLAIN(a);
   PLAIN(b);
   uint8_t raw[G5P_USC_MAX];
   if (!ctl_head(raw, 0x89, dst_half, k))
      return 0;
   SEL(sa, a.half, a.group, 0);
   SEL(sb, b.half, b.group, 1);
   raw[4] = sa;
   raw[5] = sb;
   raw[6] = mul ? 8 : 4;
   raw[7] = (negate_b ? 0x20 : 0) | (source_control >= 0);
   unsigned n = 8;
   if (source_control >= 0) {
      raw[n++] = 0;
      raw[n++] = source_control;
   }
   return finish(out, raw, n);
}

unsigned g5p_usc_half_fma2(uint8_t *out, unsigned dst_half, struct g5p_src a, struct g5p_src b,
                           struct g5p_src c, struct g5p_ctl k, int source_control)
{
   CHECK(source_control < 0 || source_control == 0x80, "unmeasured half FMA control");
   PLAIN(a);
   PLAIN(b);
   PLAIN(c);
   uint8_t raw[G5P_USC_MAX];
   if (!ctl_head(raw, 0x8a, dst_half, k))
      return 0;
   SEL(sa, a.half, a.group, 0);
   SEL(sb, b.half, b.group, 1);
   SEL(sc, c.half, c.group, 0);
   raw[4] = sa;
   raw[5] = sb;
   raw[6] = sc;
   raw[7] = 1;
   raw[8] = 0x0c;
   raw[9] = source_control >= 0;
   unsigned n = 10;
   if (source_control >= 0) {
      raw[n++] = 0;
      raw[n++] = source_control;
   }
   return finish(out, raw, n);
}

unsigned g5p_usc_half_square(uint8_t *out, unsigned dst_half, struct g5p_src s, struct g5p_ctl k,
                             unsigned source_control)
{
   CHECK(source_control == 0 || source_control == 0xc0, "unmeasured half square control");
   PLAIN(s);
   uint8_t raw[G5P_USC_MAX];
   if (!ctl_head(raw, 0x88, dst_half, k))
      return 0;
   SEL(ss, s.half, s.group, 0);
   uint8_t tail[6] = {ss, 1, 0x08, 1, 0x30, source_control};
   memcpy(raw + 4, tail, 6);
   return finish(out, raw, 10);
}

unsigned g5p_usc_half_binary_literal(uint8_t *out, unsigned dst_half, uint16_t value, struct g5p_src b,
                                     struct g5p_ctl k, bool negate_b, unsigned control)
{
   CHECK(control == 0 || control == 2, "unmeasured half literal form");
   PLAIN(b);
   uint8_t raw[G5P_USC_MAX];
   unsigned n = half_literal_head(raw, 0x89, dst_half, value, b, k);
   if (!n)
      return 0;
   uint8_t tail[4] = {0x04, (negate_b ? 0x80 : 0) | control | 1, 0x70, 0};
   memcpy(raw + n, tail, 4);
   return finish(out, raw, n + 4);
}

unsigned g5p_usc_half_binary_shared(uint8_t *out, bool mul, unsigned dst_half, unsigned a_dword,
                                    unsigned a_shared_group, struct g5p_src b, struct g5p_ctl k,
                                    bool negate_b, unsigned routing, unsigned source_control)
{
   CHECK(routing == 0 || routing == 0x70, "unmeasured half shared routing/control");
   CHECK(source_control == 0x80 || source_control == 0xc0, "unmeasured half shared routing/control");
   PLAIN(b);
   uint8_t raw[G5P_USC_MAX], a[3];
   if (!half_common(a, a_dword, a_shared_group) || !ctl_head(raw, 0x89, dst_half, k))
      return 0;
   SEL(sb, b.half, b.group, 1);
   uint8_t tail[8] = {a[0], sb, a[1], a[2], mul ? 8 : 4, (negate_b ? 0x20 : 0) | 1, routing, source_control};
   memcpy(raw + 4, tail, 8);
   return finish(out, raw, 12);
}

unsigned g5p_usc_half_move_shared(uint8_t *out, unsigned dst_half, unsigned a_dword,
                                  unsigned a_shared_group, struct g5p_ctl k)
{
   uint8_t raw[G5P_USC_MAX], a[3];
   if (!half_common(a, a_dword, a_shared_group) || !ctl_head(raw, 0x88, dst_half, k))
      return 0;
   uint8_t tail[8] = {a[0], 1, a[1], a[2], 0x28, 1, 0x39, 0};
   memcpy(raw + 4, tail, 8);
   return finish(out, raw, 12);
}

unsigned g5p_usc_half_fma_literal(uint8_t *out, unsigned dst_half, uint16_t a_value, struct g5p_src b,
                                  uint16_t c_value, struct g5p_ctl k, unsigned routing,
                                  unsigned source_control)
{
   CHECK(!(c_value & 0x200), "C-literal bit9 is not independently mapped");
   CHECK(routing == 0x70 && source_control == 0x80, "unmeasured half literal FMA control");
   PLAIN(b);
   uint8_t raw[G5P_USC_MAX], c[G5P_USC_MAX];
   unsigned n = half_literal_head(raw, 0x8a, dst_half, a_value, b, k);
   if (!n || !literal_prefix(c, 0, 0, 0, c_value, 0, 0, 1, 0))
      return 0;
   memcpy(raw + n, c + 4, 4);
   n += 4;
   uint8_t tail[4] = {0x0c, 1, routing, source_control};
   memcpy(raw + n, tail, 4);
   return finish(out, raw, n + 4);
}

unsigned g5p_usc_half_minmax(uint8_t *out, bool max, unsigned opcode, unsigned dst_half, struct g5p_src a,
                             struct g5p_src b, struct g5p_ctl k)
{
   if (!opcode)
      opcode = max ? 0x38 : 0x30;
   CHECK(opcode == 0x30 || (opcode == 0x38 && max), "unmeasured half min/max encoding");
   PLAIN(a);
   PLAIN(b);
   uint8_t raw[G5P_USC_MAX];
   if (!ctl_head(raw, 0x89, dst_half, k))
      return 0;
   SEL(sa, a.half, a.group, 0);
   SEL(sb, b.half, b.group, 1);
   uint8_t tail[6] = {sa, sb, opcode, (opcode == 0x30 && max ? 0x80 : 0x20) | 1, 0x0a, 0};
   memcpy(raw + 4, tail, 6);
   return finish(out, raw, 10);
}

unsigned g5p_usc_half_minmax_fixup(uint8_t *out, unsigned dst_half, struct g5p_src a, struct g5p_src b,
                                   struct g5p_ctl k)
{
   PLAIN(a);
   PLAIN(b);
   uint8_t raw[G5P_USC_MAX];
   if (!ctl_head(raw, 0x89, dst_half, k))
      return 0;
   SEL(sa, a.half, a.group, 0);
   SEL(sb, b.half, b.group, 1);
   uint8_t tail[6] = {sa, sb, 0x28, 0x21, 0x3a, 0};
   memcpy(raw + 4, tail, 6);
   return finish(out, raw, 10);
}

unsigned g5p_usc_half_lerp(uint8_t *out, unsigned dst_half, struct g5p_src a, struct g5p_src b,
                           struct g5p_src c, struct g5p_ctl k, bool negate_c, unsigned source_control)
{
   CHECK(source_control == 0x80, "unmeasured half lerp control");
   PLAIN(a);
   PLAIN(b);
   PLAIN(c);
   uint8_t raw[G5P_USC_MAX];
   if (!ctl_head(raw, 0x8a, dst_half, k))
      return 0;
   SEL(sa, a.half, a.group, 0);
   SEL(sb, b.half, b.group, 1);
   SEL(sc, c.half, c.group, 0);
   uint8_t tail[8] = {sa, sb, sc, 1, 0x14, (negate_c ? 8 : 0) | 1, 0, source_control};
   memcpy(raw + 4, tail, 8);
   return finish(out, raw, 12);
}

unsigned g5p_usc_half_dot2(uint8_t *out, unsigned dst_half, struct g5p_src a, struct g5p_src b,
                           struct g5p_src c, struct g5p_src d, struct g5p_ctl k, bool subtract,
                           int source_control)
{
   CHECK(source_control < 0 || source_control == 0x20 || source_control == 0x80, "unmeasured half dot2 control");
   PLAIN(a);
   PLAIN(b);
   PLAIN(c);
   PLAIN(d);
   uint8_t raw[G5P_USC_MAX];
   if (!ctl_head(raw, 0x8b, dst_half, k))
      return 0;
   SEL(sa, a.half, a.group, 0);
   SEL(sb, b.half, b.group, 1);
   SEL(sc, c.half, c.group, 0);
   SEL(sd, d.half, d.group, 1);
   uint8_t tail[6] = {sa, sb, sc, sd, subtract ? 0x12 : 0x10, source_control >= 0};
   memcpy(raw + 4, tail, 6);
   unsigned n = 10;
   if (source_control >= 0) {
      raw[n++] = 0;
      raw[n++] = source_control;
   }
   return finish(out, raw, n);
}

/* ---- Vertex-stage stores and integer forms (XFB probes, xfb.md) ---- */
unsigned g5p_usc_address_add(uint8_t *out, unsigned dst_half, unsigned ptr_dword, unsigned ptr_group,
                             unsigned off_half, unsigned off_group, unsigned shift)
{
   CHECK(half_register(dst_half, "dst_half requires a measured 32-bit aligned half index") &&
         half_register(off_half, "off_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   CHECK(!(ptr_dword & 1), "64-bit pointer base must be pair aligned");
   CHECK(off_group <= 1, "off_group must be an integer in [0,1]");
   CHECK(shift <= 4, "address shifts 0..4 (uchar..uint4) are witnessed");
   CHECK(!g5p_usc_g4p, G5P_A8_NO_ADDR_ADD);
   uint8_t raw[G5P_USC_MAX], a[4];
   if (!alu_header(raw, 0x89, dst_half, 1, 0x11, 0, 0) || !shared_src(a, ptr_dword, ptr_group))
      return 0;
   raw[4] = a[0];
   raw[5] = (off_half << 1) | (off_group << 7) | 1;
   raw[6] = a[2];
   raw[7] = a[3];
   raw[8] = 0xa1;
   raw[9] = 1;
   raw[10] = 0;
   raw[11] = (shift << 5) | 6;
   return finish(out, raw, 12);
}

unsigned g5p_usc_global_store(uint8_t *out, unsigned bytes, unsigned data_half, unsigned addr_half,
                              unsigned dword_offset)
{
   CHECK(bytes == 4 || bytes == 8 || bytes == 12 || bytes == 16, "stores of 1..4 dwords are witnessed");
   CHECK(dword_offset <= 3 && (bytes == 4 || !dword_offset),
         "immediate offsets 0..3 dwords on 4-byte stores are witnessed");
   CHECK(half_register(data_half, "data_half requires a measured 32-bit aligned half index") &&
         half_register(addr_half, "addr_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   uint8_t raw[6] = {0x10, 0x23 + 8 * (bytes / 4 - 1), 0x80 | data_half, (addr_half << 1) | 1, dword_offset, 0x88};
   return finish(out, raw, 6);
}

unsigned g5p_usc_iadd_literal(uint8_t *out, unsigned dst_half, uint32_t value, unsigned b_half,
                              unsigned bank, unsigned b_group)
{
   return g5p_usc_iadd_literal2(out, dst_half, value, b_half, bank, b_group, 0x11);
}

/* The header names B's bank: 0x11 bank 1, 0x01 bank 0 (Apple's v_loop counter
 * increment, 9901040103892001a1097000: bank 0 to bank 0). */
unsigned g5p_usc_iadd_literal2(uint8_t *out, unsigned dst_half, uint32_t value, unsigned b_half,
                               unsigned bank, unsigned b_group, unsigned header)
{
   CHECK(half_register(b_half, "b_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   CHECK(header == 0x11 || header == 0x01, "iadd literal headers 0x11 and 0x01 are witnessed");
   bool wide = value > 0xffff;
   uint8_t raw[G5P_USC_MAX];
   unsigned n = literal_prefix(raw, 0x99, dst_half, bank, value, b_half, b_group, header, wide);
   if (!n)
      return 0;
   raw[n++] = binop_code[G5P_BIN_IADD];
   raw[n++] = wide ? 1 : 9;
   raw[n++] = 0x70;
   raw[n++] = 0;
   return finish(out, raw, n);
}

unsigned g5p_usc_shl_literal(uint8_t *out, unsigned dst_half, unsigned src_half, unsigned bank,
                             unsigned src_group, unsigned shift)
{
   CHECK(shift >= 1 && shift <= 4, "vertex shift-add amounts 1..4 are witnessed");
   uint8_t raw[G5P_USC_MAX];
   unsigned n = literal_prefix(raw, 0x99, dst_half, bank, 0, src_half, src_group, 0x11, 0);
   if (!n)
      return 0;
   raw[n++] = binop_code[G5P_BIN_IADD];
   raw[n++] = 5;
   raw[n++] = 0;
   raw[n++] = shift << 5;
   return finish(out, raw, n);
}

/* ---- Control flow: the execution-mask stack, its compares and jumps ----
 * Copyright 2026 Henrik Klee. Ported from his G9P
 * encoders (OrchardLinux/mesa src/imagination/g9p/g9p_isa.h: g9p_push_if,
 * g9p_else, g9p_pop_levels, g9p_loop_begin, g9p_loop_break, g9p_jump,
 * g9p_jump_patch, g9p_cmp_reg, g9p_cmp_imm; commit d91427625ff; MIT), the
 * fragment-bank forms only. Apple's A9 compiler (applegpu_g5p) emits these
 * words, the same as for A10 (docs/gpu/experiments/control-flow.md). Two
 * changes from his: the source's last-read bit is the caller's (Apple sets it
 * only on the program's last stack operation, he sets it on every pop), and
 * the first push may read the immediate 0 (Apple's first push; his
 * g9p_push_init). The stack is a counter in the upper half of register st;
 * a lane runs while it is 0. */
static unsigned put_words(uint8_t *out, const uint16_t *w, unsigned n)
{
   uint8_t raw[G5P_USC_MAX];
   CHECK(2 * n <= G5P_USC_MAX, "too many words");
   for (unsigned i = 0; i < n; i++) {
      raw[2 * i] = w[i] & 0xff;
      raw[2 * i + 1] = w[i] >> 8;
   }
   return finish(out, raw, 2 * n);
}

unsigned g5p_usc_stack(uint8_t *out, enum g5p_stack_op op, unsigned st, bool imm_src, bool invert,
                       unsigned levels, bool last)
{
   uint16_t w3;
   CHECK(st < 128, "mask-stack register above r127");
   switch (op) {
   case G5P_ST_PUSH:
      w3 = 0x46d1 | invert << 13;
      break;
   case G5P_ST_ELSE:
      w3 = 0x46d4;
      break;
   case G5P_ST_POP:
      CHECK(levels >= 1 && levels <= 3, "pop: 1 to 3 levels");
      w3 = 0x06d2 | levels << 14;
      break;
   case G5P_ST_LOOP:
      w3 = 0x86d0;
      break;
   case G5P_ST_BREAK:
      w3 = 0x86d7 | invert << 13;
      break;
   default:
      REJECT("unknown mask-stack operation");
   }
   const uint16_t dst = 0x0100 | (2 * st + 1);
   if (imm_src) {
      CHECK(op == G5P_ST_PUSH || op == G5P_ST_LOOP, "only a push or a loop reads the immediate 0");
      const uint16_t w[] = {0x01c8, dst, 0x0101, 0x0120, w3};
      return put_words(out, w, 5);
   }
   if (st >= 32) {
      /* a source above r31: r & 31 in the field, bit 0 set, r >> 5 in an
       * extension word (G9P's g9p_mov_from_high(); Apple's f_k16) */
      const uint16_t w[] = {0x01c8, dst, 0x0101 | (2 * (st & 31) + 1) << 1 | (last ? 0x80 : 0),
                            0x0100 | (st >> 5) << 1, w3};
      return put_words(out, w, 5);
   }
   const uint16_t w[] = {0x01c8, dst, 0x0100 | (2 * st + 1) << 1 | (last ? 0x80 : 0), w3};
   return put_words(out, w, 4);
}

/* The offset's low half is in word 1 (bit 8, the next-word flag's position, in
 * bit 0; offsets are even). Word 2 holds bits 16..31 except bit 24: bit 8 of
 * the last word is the next-word flag, clear, so bit 24 must equal bit 25 and
 * the offset is within 16 MiB either way. Apple's A9 compiler, jumps of +0x1a05a
 * and +0x10c94 over long bodies (tools/mesa-a9-codesize-re f_if1536, f_loop512):
 * word 2 is offset >> 16 with bit 16 in bit 0. The port had bit 24 there, the
 * same bits whenever bits 16 and 24 agree (every offset within 64 KiB). */
unsigned g5p_usc_jump(uint8_t *out, bool any, int offset)
{
   CHECK(!(offset & 1) && offset >= -0x1000000 && offset < 0x1000000, "jump offset must be even and within 16 MiB");
   const uint32_t off = offset;
   const uint16_t w[] = {any ? 0x0101 : 0x0900, (off & 0xfefe) | 0x0100 | ((off >> 8) & 1),
                         (off >> 16) & 0xfeff};
   return put_words(out, w, 3);
}

unsigned g5p_usc_jump_patch(uint8_t *jump, int offset)
{
   uint8_t raw[G5P_USC_MAX] = {0};
   unsigned n = g5p_usc_jump(raw, jump[1] == 0x01, offset);
   if (n)
      memcpy(jump + 2, raw + 2, 4);
   return n;
}

unsigned g5p_usc_cf_compare(uint8_t *out, enum g5p_cmp op, unsigned a, unsigned a_group, unsigned b,
                            unsigned b_group)
{
   CHECK(op < G5P_CMP_COUNT && a < 32 && b < 32, "compare: registers r0..r31");
   const bool f = op == G5P_FLT || op == G5P_FGT || op == G5P_FEQ;
   const uint16_t w[] = {0x0181, 0x0100 | a << 2 | b << 10 | (a_group & 1) << 7 | (b_group & 1) << 15,
                         (f ? 0x6000 : 0xa000) | cmp_code[op]};
   return put_words(out, w, 3);
}

unsigned g5p_usc_cf_compare_imm(uint8_t *out, enum g5p_cmp op, unsigned reg, unsigned group, unsigned imm)
{
   CHECK(op >= G5P_ILT && op < G5P_CMP_COUNT && reg < 32 && imm < 1024,
         "compare with an immediate: integer relation, r0..r31, immediate < 1024");
   const uint16_t w[] = {0x0181 | ((imm >> 9) & 1) << 13,
                         0x0101 | (imm & 0x3f) << 1 | reg << 10 | (group & 1) << 15,
                         0x0120 | ((imm >> 6) & 7) << 1, 0xa100 | cmp_code[op], 0x0070};
   return put_words(out, w, 5);
}

/* ---- Derivatives ----
 * Apple's A9 output (tools/mesa-a9-hypr-re: f_ddx_var, f_ddy_var, f_ddxy_live,
 * f_fwidth, f_ddx_tex, f_ddxy_vec): 98|ext header dst src 44 01 00 02 (x) /
 * 00 04 (y). A10 sets bit 0 of the op word (0145), as it does in the output
 * conversion. */
unsigned g5p_usc_derivative(uint8_t *out, bool y, unsigned dst_half, unsigned src_half, unsigned src_group,
                            unsigned header, unsigned source_extension, unsigned dst_control)
{
   CHECK(source_extension == 0 || source_extension == 4, "unmeasured derivative class");
   uint8_t raw[G5P_USC_MAX];
   CHECK(header_ok(header), g5p_usc_error);
   raw[0] = 0x98 | source_extension;
   raw[1] = header;
   if (!cdestination(raw + 2, dst_half, 0, dst_control))
      return 0;
   SEL(s, src_half, src_group, 0);
   raw[4] = s;
   raw[5] = 1;
   raw[6] = 0x44;
   raw[7] = 1;
   raw[8] = 0;
   raw[9] = y ? 4 : 2;
   return finish(out, raw, 10);
}

/* ---- Registers above r31 ----
 * Copyright 2026 Henrik Klee. The two-source ALU form
 * of his g9p_alu2() with his high-register operands (g9p_mov_to_high(),
 * g9p_mov_from_high(): OrchardLinux/mesa src/imagination/g9p/g9p_isa.h,
 * d1ab45daed3; MIT), fragment bank: a destination up to r127 is the usual
 * half index; a source above r31 sets bit 0 of the operand word, keeps r & 31
 * in its 5-bit field and puts r >> 5 in an extension word (A at bit 1, B at
 * bit 9). Apple's A9 compiler emits the same words (f_k8, f_k16, f_kv16 use
 * r32..r69; control-flow.md). op: the binop code (fadd 44, fmul 48, iadd a1,
 * or ce, ...). G9P's moves to and from a home are op ce (or) with a == b. */
unsigned g5p_usc_alu2_wide(uint8_t *out, unsigned op, unsigned dst, unsigned a, unsigned a_group, unsigned b,
                           unsigned b_group)
{
   CHECK(dst < 128 && a < 128 && b < 128 && op < 256, "registers r0..r127");
   const uint16_t srcs = 0x0100 | (a & 31) << 2 | (b & 31) << 10 | (a_group & 1) << 7 | (b_group & 1) << 15;
   if (a < 32 && b < 32) {
      const uint16_t w[] = {0x0199, 0x0100 | 2 * dst, srcs, op};
      return put_words(out, w, 4);
   }
   const uint16_t w[] = {0x0199, 0x0100 | 2 * dst, srcs | 1, 0x0100 | (a >> 5) << 1 | (b >> 5) << 9, op};
   return put_words(out, w, 5);
}

#define G5P_CLASS_BINARY 0x99        /* g5p_usc_binary()'s class byte with the 32-bit result */
#define G5P_HEADER_SRC_A_BANK1 0x20  /* source A in register bank 1 (g9p_w0()'s W0 bit 13) */
unsigned g5p_usc_vs_or_wide(uint8_t *out, unsigned dst_half, unsigned dst_bank, unsigned a, unsigned a_group,
                            unsigned b, unsigned b_group)
{
   CHECK(dst_half < 256 && !(dst_half & 1) && dst_bank <= 1 && a < 128 && b < 128, "registers r0..r127");
   /* g5p_usc_binary()'s layout for A | B with both sources' bank bits in the header; a source above r31 keeps
    * r & 31 in its selector, sets A's selector bit 0 and puts r >> 5 in an extension parcel (A at bit 1, B at
    * bit 9 of the parcel, as g5p_usc_alu2_wide()). */
   const bool wide = a >= 32 || b >= 32;
   uint8_t raw[10];
   raw[0] = G5P_CLASS_BINARY;
   raw[1] = G5P_HEADER_FRAGMENT | G5P_HEADER_SRC_A_BANK1 | G5P_HEADER_SRC_B_BANK1;
   if (!destination(raw + 2, dst_half, dst_bank))
      return 0;
   SEL(sa, 2 * (a & 31), a_group & 1, wide);
   SEL(sb, 2 * (b & 31), b_group & 1, G5P_PARCEL_CONTINUES);
   raw[4] = sa;
   raw[5] = sb;
   unsigned n = 6;
   if (wide) {
      raw[n++] = (a >> 5) << 1;
      raw[n++] = (b >> 5) << 1 | G5P_PARCEL_CONTINUES;
   }
   raw[n++] = binop_code[G5P_BIN_OR];
   raw[n++] = 0;
   return finish(out, raw, n);
}

/* Graphics storage differential host compiles, compute-gfx-storage-isa.md.
 * Unlike compute/vertex class10, fragment stores use the pixel side-effect
 * port70/71 and the observed active-fragment predicate. */
unsigned g5p_usc_fs_store_predicate(uint8_t *out)
{
   const uint16_t w[]={0x0182,0x0101,0x0120,0x0129,0x0110,0xa1bd,0x00b0};
   return put_words(out,w,7);
}
unsigned g5p_usc_fs_store(uint8_t *out,unsigned bytes,unsigned data_half,unsigned addr_half,unsigned slot)
{
   CHECK(bytes==4 || bytes==8 || bytes==16,"fragment stores:4/8/16bytes witnessed");
   CHECK(data_half<64 && !(data_half&1) && addr_half<62 && !(addr_half&3) && slot<=1,"fragment store register/slot fields");
   unsigned size=bytes==4?4:bytes==8?5:7;
   const uint16_t w[]={size<<11|0x0370|slot,0x0080|data_half|addr_half<<9};
   return put_words(out,w,2);
}

unsigned g5p_usc_vertex_texture_level(uint8_t *out, unsigned dst_half, unsigned src_half,
                                      unsigned group)
{
   CHECK(dst_half <= 63 && half_register(src_half, "vertex level source requires a dword"), g5p_usc_error);
   uint8_t raw[G5P_USC_MAX];
   /* -0.0h + source preserves signed zero and narrows to half. The named
    * literal prefix and result-half flag reproduce v_*_sample_attribute. */
   unsigned n = literal_prefix(raw, 0x89, dst_half, 1, 0x8000, src_half, group, 0x11, 0);
   if (!n)
      return 0;
   raw[n++] = binop_code[G5P_BIN_FADD];
   raw[n++] = 9 | 2; /* f16 literal, f16 result */
   raw[n++] = 0x70;
   raw[n++] = 0;
   return finish(out, raw, n);
}

unsigned g5p_usc_vertex_array_layer(uint8_t *out, unsigned dst_half, unsigned src_half)
{
   unsigned n = g5p_usc_f2u16_half(out, dst_half, src_half, 1);
   if (n) {
      out[1] |= 1u << 4; /* vertex source routing */
      out[3] |= 1u << 1; /* bank-1 destination */
   }
   return n;
}
