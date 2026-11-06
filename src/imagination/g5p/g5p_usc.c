/* SPDX-License-Identifier: MIT
 * G5P USC field encoders, ported from tools/gpu-userspace/g5p_usc.py and
 * g5p_usc_resources.py. Field checks follow the reference one for one: a
 * value the reference rejects with ValueError returns 0 here.
 */
#include "g5p_usc.h"
#include "g5p_mrt_out.h"

#include <string.h>

_Thread_local const char *g5p_usc_error;

const char *const g5p_binop_name[G5P_BIN_COUNT] = {"fadd", "fmul", "iadd", "imul", "and", "or", "xor", "ishr",
                                                       "isub"};
const char *const g5p_minmax_name[G5P_MINMAX_COUNT] = {"umin", "umax", "imin", "imax"};
const char *const g5p_convert_name[G5P_CONVERT_COUNT] = {"f2i32", "f2u32", "i2f32", "u2f32", "f2i32_rte"};
const char *const g5p_unary_name[G5P_UNARY_COUNT] = {"floor", "log2", "exp2", "not", "popcount", "bitreverse"};
const char *const g5p_tex_dim_name[G5P_TEX_DIM_COUNT] = {"2d", "2d_array", "3d", "cube"};
const char *const g5p_cmp_name[G5P_CMP_COUNT] = {"flt", "fgt", "feq", "ilt", "igt", "ult", "ugt", "ieq"};
const char *const g5p_sfu_name[G5P_SFU_COUNT] = {"rcp_approx", "rsqrt_approx", "sin_core", "cos_core"};

static const uint8_t binop_code[G5P_BIN_COUNT] = {0x44, 0x48, 0xa1, 0xa2, 0xc8, 0xce, 0xc6, 0xfc, 0xa5};
static const uint8_t minmax_code[G5P_MINMAX_COUNT] = {0xb2, 0xb3, 0xba, 0xbb};
static const uint8_t convert_tag[G5P_CONVERT_COUNT] = {0x13, 0x11, 0x17, 0x15, 0x17};
static const uint8_t convert_code[G5P_CONVERT_COUNT] = {0x60, 0x60, 0x68, 0x68, 0x60};
static const uint8_t cmp_code[G5P_CMP_COUNT] = {0x59, 0x5d, 0x55, 0xbe, 0xbf, 0xb6, 0xb7, 0xbd};
static const uint8_t sfu_code[G5P_SFU_COUNT] = {0x80, 0x81, 0x82, 0x86};
static const uint8_t unary_code[G5P_UNARY_COUNT] = {0x40, 0x84, 0x85, 0xc5, 0xe8, 0xe4};
static const uint8_t dim_code[G5P_TEX_DIM_COUNT] = {0xf5, 0xf7, 0xfb, 0xfd};
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
 * program (GET_RENDER_INFO's TA program slot, W+0x88) is a branch to the next
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
   uint8_t raw[8] = {drc1 ? 0x15 : 0x14, 0x23, dst_half, 0x01, coeff_dword, 0x01, 0x00, 0x02};
   return finish(out, raw, 8);
}

unsigned g5p_usc_coord_iterator(uint8_t *out, unsigned dst_half, unsigned slot, unsigned count)
{
   CHECK((count == 2 || count == 3) && !(dst_half % 2), "coordinate iterator count/alignment");
   CHECK(dst_half < 64, "coordinate destination half must be u6");
   CHECK(slot < 128, "iterator slot must be u7");
   uint8_t raw[6] = {8, count == 2 ? 0x13 : 0x1b, dst_half, (slot << 1) | 1, 0, 0x80};
   return finish(out, raw, 6);
}

unsigned g5p_usc_emit(uint8_t *out, unsigned words, unsigned source_half)
{
   CHECK(words == 2 || words == 4, "only two/four output words characterized");
   CHECK(source_half < 256, "output source half must be u8");
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
   CHECK(g5p_mrt_emit_words_witnessed(words), "only 2/4/8 output words (1/2/4 targets) have an Apple witness");
   CHECK(source_half < 256, "output source half must be u8");
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
   CHECK(t->image_qword <= 28, "image_qword must be an integer in [0,28]");
   CHECK(!(t->image_qword & 3), "measured image descriptor bases are four-qword aligned");
   CHECK(t->sampler_qword <= 32, "sampler_qword must be an integer in [0,32]");
   CHECK(t->coord_dword <= 31, "coord_dword must be an integer in [0,31]");
   CHECK(t->class_variant <= 1, "class_variant must be an integer in [0,1]");
   CHECK(t->dim < G5P_TEX_DIM_COUNT, "unmeasured texture dimension");
   CHECK(!t->read || t->dim == G5P_TEX_2D, "only 2d read has been isolated");
   raw[0] = 0x1a | t->class_variant;
   raw[1] = dim_code[t->dim];
   raw[2] = t->dst_half | (t->read ? 0x40 : 0);
   raw[3] = (t->image_qword << 1) | 1;
   raw[4] = t->sampler_qword << 1;
   raw[5] = (t->coord_dword << 1) | (t->coord_control ? 0x80 : 0);
   return 6;
}

unsigned g5p_usc_sample(uint8_t *out, const struct g5p_tex *t)
{
   CHECK(!t->read, "mode must be sample");
   uint8_t raw[G5P_USC_MAX];
   if (!texture_base(raw, t))
      return 0;
   if (t->narrow) {
      CHECK(!t->class_variant && t->dim == G5P_TEX_2D, "narrow sample witnessed only as 2d class0");
      raw[0] &= ~2;
   }
   return finish(out, raw, 6);
}

unsigned g5p_usc_texture_lod(uint8_t *out, const struct g5p_tex *t)
{
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
   const bool coupled = t->tuple != G5P_TEX_TUPLE_ORDINARY;
   CHECK(t->tuple <= G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL, "unqualified texture recipe");
   CHECK(!coupled || t->lod_shared == (t->tuple == G5P_TEX_TUPLE_COUPLED_SHARED_LEVEL),
         "unqualified independent texture continuation choices");
   static const uint8_t dim_low[G5P_TEX_DIM_COUNT] = {0x5, 0x7, 0xb, 0xd};
   static const uint8_t lod_mode[] = {0, 0x40, 0x20, 0x60};
   CHECK(half_register(t->dst_half, "dst_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   /* A9 Metal v_images31 and v_units16_sample witness the complete 7-bit
    * descriptor selector, including image bit6 and sampler bits5/6. */
   CHECK(t->image_qword <= 120 && !(t->image_qword & 3), "image_qword must be a four-qword base in [0,120]");
   CHECK(t->sampler_qword <= 124, "sampler_qword must be an integer in [0,124]");
   CHECK(t->coord_dword <= 31 && t->lod_dword <= 31 && t->extras_half <= 63, "operand beyond r31");
   CHECK(t->dim < G5P_TEX_DIM_COUNT, "unmeasured texture dimension");
   CHECK(t->mask == 0xf || (t->mask == 1 && t->compare), "only rgba and compare results are measured");
   CHECK(!t->read || (!t->compare && !t->offset && t->lod == G5P_TEX_LOD_LEVEL && t->dim != G5P_TEX_CUBE),
         "read takes a level and no compare/offset (cube reads are refused by Apple)");
   CHECK(t->lod != G5P_TEX_LOD_GRAD || t->dim == G5P_TEX_2D || t->dim == G5P_TEX_2D_ARRAY,
         "gradients are measured on 2D and 2D arrays only");
   CHECK(!t->offset || t->dim != G5P_TEX_CUBE, "cube offsets do not exist");
   CHECK(!coupled || (!t->compare && !t->offset && t->lod == G5P_TEX_LOD_LEVEL),
         "vertex texture tuple requires explicit level, no compare or offset");
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
       * paired Metal programs and exact in r1c's non-cube device rungs.
       * Individual destination/coordinate continuation bits are unknown;
       * keep this suffix indivisible rather than guessing mixed forms. */
      raw[9] = (1u << 7) | 1u;
      raw[10] = 1u << 3;
      if ((t->tuple == G5P_TEX_TUPLE_COUPLED_TEMP_LEVEL)) {
         /* The level producer writes a bank-1 temp. A common-store level
          * (native buffer/manual fold) has no level-bank extension. */
         raw[11] = 1;
         raw[12] = 1u << 4;
         return finish(out, raw, 14);
      }
      return finish(out, raw, 12);
   }
   return finish(out, raw, 10);
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

unsigned g5p_usc_cube_coords(uint8_t *out, unsigned dst_half, unsigned src_half)
{
   CHECK(half_register(dst_half, "dst_half requires a measured 32-bit aligned half index") &&
         half_register(src_half, "src_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   uint8_t raw[6] = {0x08, 0x7d, dst_half, (src_half << 1) | 1, 0x00, 0x80};
   return finish(out, raw, 6);
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
   CHECK(base_common_dword < 32 && !(base_common_dword & 1), "pointer base must be an even common dword below 32");
   CHECK(shift <= 4, "only shifts 0-4 are corpus-verified");
   uint8_t raw[12] = {0x89, vs ? 0x11 : 0x01, pair_half, vs ? 0x03 : 0x01, 0x81 | 4 * base_common_dword,
                      (kill ? 0x80 : 0) | 1 | index_half << 1, 0x00, 0x41, 0xa1, 0x01, 0x00, 0x06 | shift << 5};
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

unsigned g5p_usc_position_zw(uint8_t *out, unsigned dst_half, bool z, bool both, int long_form)
{
   CHECK(!(both && z), "invalid position iterator");
   CHECK(!(dst_half & 1), "position iterator needs a dword-aligned destination");
   CHECK(dst_half < 256, "position destination must be u8");
   bool minimum = dst_half || z || both;
   bool wide = long_form < 0 ? minimum : long_form;
   CHECK(!(minimum && !wide), "short position iterator omits nonzero fields");
   uint8_t raw[4] = {8, (both ? 0x10 : 8) | wide, dst_half, z ? 2 : 0};
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
   CHECK(words == 1 && first_component == 2, "only the one-word component2 (alpha) tile read is witnessed");
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
   CHECK(half_register(b_half, "b_half requires a measured 32-bit aligned half index"), g5p_usc_error);
   bool wide = value > 0xffff;
   uint8_t raw[G5P_USC_MAX];
   unsigned n = literal_prefix(raw, 0x99, dst_half, bank, value, b_half, b_group, 0x11, wide);
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
