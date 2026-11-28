/* SPDX-License-Identifier: MIT */
/*
 * G3: the Apple A7's compute backend (g3_compute.h). Every USC group form
 * here ran on the A7 (J71) in the a7-cdm R5/R6 rungs; g3_selftest()
 * re-encodes those rungs' programs byte for byte.
 */
#include "g3_compute.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "pco_isa.h"
#include "util/macros.h"
#include "util/u_math.h"

/* ---- USC: Rogue instruction groups in the A7's forms ---- */

/* The registers a program names: the unified store (USC "vtxin" bank) and the constant table */
#define G3_BANK_SPECIAL PCO_REGBANK_SPECIAL
#define G3_BANK_UNIFIED PCO_REGBANK_VTXIN

struct g3_reg {
   enum pco_regbank bank;
   unsigned index;
};

#define VI(i) ((struct g3_reg){ G3_BANK_UNIFIED, (i) })
/* Rogue's constant table holds the integers 0..31 at indices 0..31 (pco_const_imms.c); sp4 and sp16 ran */
#define SP(i) ((struct g3_reg){ G3_BANK_SPECIAL, (i) })

/*
 * One group: header, body, then pco's padding: an odd length gets 0xff, and
 * align_pad more bytes (0xf0 | n / 2, then 0xff) end the group on that many
 * bytes (pco calc_align_padding()). da is the offset of the ALU part's end
 * from the group start, minus the header (the header's size is added here).
 */
static unsigned
g3_group(uint8_t *out, bool control, const struct pco_igrp_hdr_main_brief *main_hdr,
         const struct pco_igrp_hdr_control *ctrl_hdr, const uint8_t *body, unsigned body_size,
         unsigned da_after_header, unsigned align_pad)
{
   uint8_t hdr[8] = { 0 };
   const unsigned hlen = control ? 3 : 2;
   unsigned total = hlen + body_size;
   const bool odd = total & 1;

   total += odd;
   total += align_pad;
   if (control) {
      struct pco_igrp_hdr_control h = *ctrl_hdr;
      h.length = (total / 2) % 16;
      h.da = 0;
      ASSERTED unsigned n = _pco_igrp_hdr_control_encode(hdr, h);
      assert(n == hlen);
   } else {
      struct pco_igrp_hdr_main_brief h = *main_hdr;
      h.length = (total / 2) % 16;
      h.da = hlen + da_after_header;
      ASSERTED unsigned n = _pco_igrp_hdr_main_brief_encode(hdr, h);
      assert(n == hlen);
   }
   unsigned at = 0;
   memcpy(out + at, hdr, hlen);
   at += hlen;
   memcpy(out + at, body, body_size);
   at += body_size;
   if (odd)
      out[at++] = 0xff;
   if (align_pad) {
      out[at++] = 0xf0 | (align_pad / 2);
      for (unsigned i = 1; i < align_pad; i++)
         out[at++] = 0xff;
   }
   assert(at == total);
   return total;
}

/*
 * {d0, d1} = s0 * s1 + {s2, s3} (signed madd64), both halves in one group,
 * in Apple's A7 form (ds.A7 +0xdd0: {vi6,vi7} = sp4 * sp16 + {vi6,vi7}):
 * the upper source s3 in the A7's 2-byte form (pco's 1up_3b11i without its
 * third byte, which is zero for a 3-bit bank and an index below 256).
 */
static unsigned
g3_imadd64(uint8_t *out, struct g3_reg s0, struct g3_reg s1, struct g3_reg s2, struct g3_reg s3,
           struct g3_reg d0, struct g3_reg d1)
{
   uint8_t body[32] = { 0 }, up[4] = { 0 };
   unsigned n = 0;

   assert(s0.index < 128 && s1.index < 128 && s2.index < 64 && s3.index < 256);
   assert(d0.index < 256 && d1.index < 256);
   n += pco_main_int32_64_encode(body + n, .s = true, .s2neg = false,
                                 .int32_64_op = PCO_INT32_64_OP_MADD64);
   n += pco_src_3lo_2b7i_2b7i_2b6i_2m_encode(body + n, .sb0 = s0.bank, .s0 = s0.index, .sb1 = s1.bank,
                                             .s1 = s1.index, .sb2 = s2.bank, .s2 = s2.index,
                                             .is0 = PCO_IS0_SEL_S3);
   ASSERTED unsigned u = pco_src_1up_3b11i_encode(up, .sb3 = s3.bank, .s3 = s3.index);
   assert(u == 3 && up[2] == 0);
   memcpy(body + n, up, 2); /* the A7's 2-byte single upper source */
   n += 2;
   n += pco_iss_iss_encode(body + n, .is5 = PCO_IS5_SEL_FTE, .is4 = PCO_IS4_SEL_FT0);
   n += pco_dst_2_3b8i_3b8i_encode(body + n, .db0 = d0.bank, .d0 = d0.index, .db1 = d1.bank,
                                   .d1 = d1.index);
   const struct pco_igrp_hdr_main_brief h = { .oporg = PCO_OPORG_P0, .w0p = true, .w1p = true };
   return g3_group(out, false, &h, NULL, body, n, 0, 0);
}

/*
 * st.32 [addr:addr+1] <- addr+2 .. addr+1+burst: the A7's store (ds.A7
 * +0x208): pco's 3-byte st_immbl, MCU write-through, the data field (s0)
 * encoded as the address + 2 as Apple does, the address pair in s1.
 */
static unsigned
g3_st(uint8_t *out, struct g3_reg addr, unsigned burst)
{
   uint8_t body[32] = { 0 };
   unsigned n = 0, be;

   assert(addr.bank == G3_BANK_UNIFIED && addr.index + 2 < 128 && burst >= 1 && burst <= 4);
   be = pco_backend_st_immbl_encode(body, .drc = 0, .srcseladd = PCO_SRCSEL_S1, .burstlen = burst,
                                    .cachemode_st = PCO_CACHEMODE_ST_WRITE_THROUGH,
                                    .srcseldata = PCO_SRCSEL_S0, .dsize = PCO_DSIZE_32BIT);
   assert(be == 3);
   n = be;
   n += pco_src_2lo_2b7i_2b7i_2m_encode(body + n, .sb0 = addr.bank, .s0 = addr.index + 2,
                                        .sb1 = addr.bank, .s1 = addr.index, .is0 = PCO_IS0_SEL_S0);
   n += pco_src_1up_1b6i_encode(body + n, .sb3 = PCO_REGBANK_SPECIAL, .s3 = 0);
   n += pco_iss_iss_encode(body + n);
   const struct pco_igrp_hdr_main_brief h = { .oporg = PCO_OPORG_BE };
   return g3_group(out, false, &h, NULL, body, n, be, 0);
}

/* wdf: wait for the data fence of every store before it (03 80 6a ff f1 ff, 04 80 6a ff f2 ff ff ff ran) */
static unsigned
g3_wdf(uint8_t *out, unsigned align_pad)
{
   const struct pco_igrp_hdr_control h = { .ctrlop = PCO_CTRLOP_WDF };
   return g3_group(out, true, NULL, &h, NULL, 0, 0, align_pad);
}

const uint8_t g3_usc_nop_end[8] = { 0x04, 0x80, 0xee, 0x00, 0xf2, 0xff, 0xff, 0xff };

/* nop.end, the program's last group, on an 8-byte boundary and 8 bytes long */
static unsigned
g3_nop_end(uint8_t *out)
{
   const struct pco_igrp_hdr_control h = { .miscctl = 1, .ctrlop = PCO_CTRLOP_NOP };
   const uint8_t nop = 0; /* the nop's operand byte (pco I_NOP) */
   return g3_group(out, true, NULL, &h, &nop, 1, 0, 4);
}

/*
 * A program's groups: wdf then nop.end end it; the wdf pads so that nop.end
 * starts on 8 bytes. Only its 4-, 6- and 8-byte forms ran on the A7, so a
 * 10-byte one is a 4-byte wdf and a 6-byte one (a second wdf waits for
 * nothing).
 */
static unsigned
g3_end(uint8_t *code, unsigned at)
{
   uint8_t tmp[16];
   const unsigned plain = g3_wdf(tmp, 0);
   unsigned pad = (8 - (at + plain) % 8) % 8;
   if (pad == 6) {
      at += g3_wdf(code + at, 0);
      pad = 2;
   }
   at += g3_wdf(code + at, pad);
   assert(at % 8 == 0);
   at += g3_nop_end(code + at);
   return at;
}

/* ---- the program's registers ---- */

/*
 * Store 0 keeps the R6 rungs' places (compute/R6.md): its base vi0:vi1, the
 * lane vi4, its address vi6:vi7 and its data vi8.. (address + 2), the value
 * addend vi2:vi3. Store j > 0: base, addend, address, data in a block of 10
 * from vi12.
 */
#define G3_LANE 4u
static unsigned g3_base(unsigned j) { return j ? 12 + 10 * (j - 1) : 0; }
static unsigned g3_addend(unsigned j) { return j ? 12 + 10 * (j - 1) + 2 : 2; }
static unsigned g3_addr(unsigned j) { return j ? 12 + 10 * (j - 1) + 4 : 6; }
static unsigned g3_data(unsigned j) { return g3_addr(j) + 2; }

static bool
g3_const(const struct g3_affine *a)
{
   return !a->lid && !a->wg;
}

/* value = lane exactly: the PDS writes the lane straight into the data register */
static bool
g3_plain_lane(const struct g3_affine *a)
{
   return a->lid == 1 && !a->wg && !a->c;
}

static void
g3_write(struct g3_program *p, enum g3_pds_src src, unsigned store, unsigned pair, unsigned ao)
{
   assert(p->num_writes < G3_MAX_PDS_WRITES);
   p->writes[p->num_writes++] = (struct g3_pds_write){ src, store, pair, ao };
}

static bool
refuse(struct g3_program *p, const char *fmt, ...) PRINTFLIKE(2, 3);

static bool
refuse(struct g3_program *p, const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(p->refusal, sizeof(p->refusal), fmt, ap);
   va_end(ap);
   return false;
}

/* The USC code and the PDS writes of p->stores */
static bool
g3_emit(struct g3_program *p)
{
   unsigned at = 0, top = 0;
   bool lane = false;

   p->num_writes = 0;
   for (unsigned j = 0; j < p->num_stores; j++) {
      const struct g3_store *s = &p->stores[j];
      const struct g3_affine *v = &s->value[0];
      const bool v_alu = !g3_const(v) && !g3_plain_lane(v);
      const unsigned data = g3_data(j);

      top = MAX2(top, data + ALIGN_POT(s->components, 2));
      /* the address: the PDS's (a per-workgroup constant), or lane * s + base */
      if (!s->offset.lid) {
         g3_write(p, G3_PDS_ADDRESS, j, 0, g3_addr(j));
      } else {
         g3_write(p, G3_PDS_ADDRESS, j, 0, g3_base(j));
         if (!g3_plain_lane(v))
            lane = true;
      }
      if (g3_const(v)) {
         for (unsigned c = 0; c < s->components; c += 2)
            g3_write(p, G3_PDS_CONSTANTS, j, c / 2, data + c);
      } else if (g3_plain_lane(v)) {
         g3_write(p, G3_PDS_LANE, j, 0, data);
      } else {
         g3_write(p, G3_PDS_ADDEND, j, 0, g3_addend(j));
         lane = true;
      }
      if (v_alu) {
         /* {data, data+1} = lane * a + {addend, 0} */
         at += g3_imadd64(p->code + at, VI(G3_LANE), SP(v->lid), VI(g3_addend(j)), VI(g3_addend(j) + 1),
                          VI(data), VI(data + 1));
      }
      if (s->offset.lid) {
         /* {addr, addr+1} = index * s + {base, base+1}; the index is the data register when it holds the lane */
         const unsigned index = g3_plain_lane(v) ? data : G3_LANE;
         at += g3_imadd64(p->code + at, VI(index), SP(s->offset.lid), VI(g3_base(j)), VI(g3_base(j) + 1),
                          VI(g3_addr(j)), VI(g3_addr(j) + 1));
      }
   }
   if (lane)
      g3_write(p, G3_PDS_LANE, 0, 0, G3_LANE);
   /* every register the stores read is written before the first st and left alone until the wdf
    * (a store may read its data after the next group, compute/R6.md) */
   for (unsigned j = 0; j < p->num_stores; j++)
      at += g3_st(p->code + at, VI(g3_addr(j)), p->stores[j].components);
   at = g3_end(p->code, at);
   assert(at <= sizeof(p->code));
   p->code_size = at;
   p->unified_regs = MAX2(top, lane ? G3_LANE + 1 : 0);
   return true;
}

/* ---- the PDS program and data ---- */

#define G3_PDS_DOUTW 1u
#define G3_PDS_DOUTU 2u
#define G3_PDS_SPACE_DATA 0u
#define G3_PDS_SPACE_TEMPS 2u
#define G3_PDS_END (1u << 26)
#define G3_PDS_CC (1u << 27)
#define G3_PDS_HALT 0xd3000000u
#define G3_PDS_BRA_NOT_IF1(target) (0xc2c00000u | (target))
/* Rogue's DOUT layout (the A8's, g5p_pds_dout_g4p()) */
static uint32_t
g3_dout(uint32_t type, uint32_t src64, uint32_t space, uint32_t ctrl, uint32_t flags)
{
   return 0xf0000000u | flags | ctrl << 16 | (space << 5 | src64) << 8 | type;
}

/* DOUTW source 1: LAST (bit 31), the unified register << 13, BSIZE 2 = 64 bits, 0 = the lower 32 */
#define G3_DOUTW_LAST (1u << 31)
#define G3_DOUTW_AO_SHIFT 13
#define G3_DOUTW_ALL64 2u
#define G3_DOUTW_LOWER32 0u

static unsigned
g3_pairs(const struct g3_program *p)
{
   unsigned n = 0;
   for (unsigned i = 0; i < p->num_writes; i++)
      n += p->writes[i].src != G3_PDS_LANE;
   return n;
}

unsigned
g3_pds_code(const struct g3_program *p, uint32_t code[G3_MAX_PDS_CODE])
{
   const unsigned ctrl0 = 2 + 2 * g3_pairs(p);
   unsigned n = 0, q = 1;

   /* the coefficient (IF1) part does nothing: no common store on the A7 */
   code[n++] = G3_PDS_BRA_NOT_IF1(2);
   code[n++] = G3_PDS_HALT;
   for (unsigned i = 0; i < p->num_writes; i++) {
      if (p->writes[i].src == G3_PDS_LANE)
         code[n++] = g3_dout(G3_PDS_DOUTW, 0, G3_PDS_SPACE_TEMPS, ctrl0 + i, 0);
      else
         code[n++] = g3_dout(G3_PDS_DOUTW, q++, G3_PDS_SPACE_DATA, ctrl0 + i, 0);
   }
   code[n++] = g3_dout(G3_PDS_DOUTU, 0, G3_PDS_SPACE_DATA, 0, G3_PDS_END | G3_PDS_CC);
   code[n++] = G3_PDS_HALT;
   assert(n <= G3_MAX_PDS_CODE);
   return n;
}

unsigned
g3_pds_data(const struct g3_program *p, uint32_t usc_offset, const uint64_t *buffer, uint32_t wgid,
            uint32_t data[G3_MAX_PDS_DATA])
{
   const unsigned ctrl0 = 2 + 2 * g3_pairs(p);
   const unsigned n = ALIGN_POT(ctrl0 + p->num_writes, 4);
   unsigned q = 1;

   assert(n <= G3_MAX_PDS_DATA);
   memset(data, 0, n * sizeof(*data));
   data[0] = usc_offset;
   data[1] = 0x10;
   for (unsigned i = 0; i < p->num_writes; i++) {
      const struct g3_pds_write *w = &p->writes[i];
      const struct g3_store *s = &p->stores[w->store];
      uint64_t v = 0;

      switch (w->src) {
      case G3_PDS_LANE:
         break;
      case G3_PDS_ADDRESS:
         /* 32-bit wrap of t * wgid + c, sign-extended as a byte offset */
         v = buffer[w->store] + (int64_t)(int32_t)(s->offset.wg * wgid + s->offset.c);
         break;
      case G3_PDS_ADDEND:
         v = (uint32_t)(s->value[0].wg * wgid + s->value[0].c);
         break;
      case G3_PDS_CONSTANTS: {
         const unsigned c = 2 * w->pair;
         v = (uint32_t)(s->value[c].wg * wgid + s->value[c].c);
         if (c + 1 < s->components)
            v |= (uint64_t)(uint32_t)(s->value[c + 1].wg * wgid + s->value[c + 1].c) << 32;
         break;
      }
      }
      if (w->src != G3_PDS_LANE) {
         data[2 * q] = (uint32_t)v;
         data[2 * q + 1] = (uint32_t)(v >> 32);
         q++;
      }
      data[ctrl0 + i] = (i + 1 == p->num_writes ? G3_DOUTW_LAST : 0) |
                        (uint32_t)w->ao << G3_DOUTW_AO_SHIFT |
                        (w->src == G3_PDS_LANE ? G3_DOUTW_LOWER32 : G3_DOUTW_ALL64);
   }
   return n;
}

uint32_t
g3_cdm_kernel0(const struct g3_program *p, unsigned pds_dwords)
{
   /* unified store in 16-byte units (4 registers) << 12, PDS temps 1 (the lane in q0), PDS data units
    * counted directly (the A8's form), any cluster, no fence: R6's 0x410e */
   const unsigned units = MAX2(DIV_ROUND_UP(p->unified_regs, 4), 1);
   return units << 12 | 0x100 | (pds_dwords / 4) << 2 | 2;
}

uint32_t
g3_cdm_kernel8(const struct g3_program *p)
{
   /* size_x - 1 << 18; max_instances 0 (= 32) as the 32- and 16-lane rungs ran, 8 for 1x1x1 (0x40000000) */
   return (p->local_size - 1) << 18 | (p->local_size == 1 ? 8u << 27 : 0);
}

/* ---- the self-test: the R6 programs that ran on the J71 (cdm/r6.h) ---- */

static const uint8_t r6_store32[40] = {
   0x27, 0x06, 0xeb, 0x84, 0x44, 0x30, 0x80, 0x81, 0x04, 0xc0, 0x06, 0x87, 0x22, 0xff,
   0x55, 0x20, 0xf2, 0x24, 0x08, 0x88, 0xc6, 0x18, 0x00, 0x00, 0x04, 0x80, 0x6a, 0xff,
   0xf2, 0xff, 0xff, 0xff, 0x04, 0x80, 0xee, 0x00, 0xf2, 0xff, 0xff, 0xff,
};
static const uint8_t r6_lanes[40] = {
   0x27, 0x06, 0xeb, 0x88, 0x44, 0x30, 0x80, 0x81, 0x04, 0xc0, 0x06, 0x87, 0x22, 0xff,
   0x55, 0x20, 0xf2, 0x24, 0x08, 0x88, 0xc6, 0x18, 0x00, 0x00, 0x04, 0x80, 0x6a, 0xff,
   0xf2, 0xff, 0xff, 0xff, 0x04, 0x80, 0xee, 0x00, 0xf2, 0xff, 0xff, 0xff,
};
static const uint8_t r6_combo[64] = {
   0x27, 0x06, 0xeb, 0x88, 0x44, 0x30, 0x80, 0x81, 0x04, 0xc0, 0x06, 0x87, 0x22, 0xff,
   0x55, 0x20, 0xf2, 0x24, 0x08, 0x88, 0xc6, 0x18, 0x00, 0x00, 0x27, 0x06, 0xeb, 0x90,
   0x50, 0x20, 0x86, 0x87, 0x04, 0xc0, 0x0a, 0x8b, 0x22, 0xff, 0x55, 0x20, 0xf2, 0x24,
   0x08, 0x8c, 0xca, 0x18, 0x00, 0x00, 0x04, 0x80, 0x6a, 0xff, 0xf2, 0xff, 0xff, 0xff,
   0x04, 0x80, 0xee, 0x00, 0xf2, 0xff, 0xff, 0xff,
};

static bool
same(const char *what, const uint8_t *got, unsigned got_size, const uint8_t *want, unsigned want_size,
     char *why, unsigned size)
{
   if (got_size == want_size && !memcmp(got, want, want_size))
      return true;
   unsigned n = snprintf(why, size, "%s: %u bytes", what, got_size);
   for (unsigned i = 0; i < got_size && n < size; i++)
      n += snprintf(why + n, size - n, "%s%02x", i ? "" : " ", got[i]);
   return false;
}

bool
g3_selftest(char *why, unsigned size)
{
   uint8_t code[128];
   unsigned at;

   /* store32: {vi6,vi7} = vi4 * 4 + {vi0,vi1}; st [vi6] <- vi8 */
   at = g3_imadd64(code, VI(4), SP(4), VI(0), VI(1), VI(6), VI(7));
   at += g3_st(code + at, VI(6), 1);
   at = g3_end(code, at);
   if (!same("store32", code, at, r6_store32, sizeof(r6_store32), why, size))
      return false;
   /* lanes: the index from vi8 */
   at = g3_imadd64(code, VI(8), SP(4), VI(0), VI(1), VI(6), VI(7));
   at += g3_st(code + at, VI(6), 1);
   at = g3_end(code, at);
   if (!same("lanes", code, at, r6_lanes, sizeof(r6_lanes), why, size))
      return false;
   /* combo: lanes, then {vi10,vi11} = 16 * 16 + {vi6,vi7}; st [vi10] <- vi12 */
   at = g3_imadd64(code, VI(8), SP(4), VI(0), VI(1), VI(6), VI(7));
   at += g3_st(code + at, VI(6), 1);
   at += g3_imadd64(code + at, SP(16), SP(16), VI(6), VI(7), VI(10), VI(11));
   at += g3_st(code + at, VI(10), 1);
   at = g3_end(code, at);
   if (!same("combo", code, at, r6_combo, sizeof(r6_combo), why, size))
      return false;
   at = g3_nop_end(code);
   if (!same("nop.end", code, at, g3_usc_nop_end, sizeof(g3_usc_nop_end), why, size))
      return false;
   if (why && size)
      why[0] = 0;
   return true;
}

/* ---- NIR ---- */

struct g3_ctx {
   struct g3_program *p;
   unsigned local[3];
};

static bool
g3_affine_of(struct g3_ctx *c, nir_scalar s, struct g3_affine *r, unsigned depth)
{
   struct g3_program *p = c->p;

   if (depth > 64)
      return refuse(p, "an expression deeper than 64");
   s = nir_scalar_chase_movs(s);
   if (s.def->bit_size != 32)
      return refuse(p, "a %u-bit value (32-bit integers only)", s.def->bit_size);
   if (nir_scalar_is_const(s)) {
      *r = (struct g3_affine){ 0, 0, nir_scalar_as_uint(s) };
      return true;
   }
   if (nir_scalar_is_intrinsic(s)) {
      const nir_intrinsic_op op = nir_scalar_intrinsic_op(s);
      switch (op) {
      case nir_intrinsic_load_local_invocation_index:
         *r = (struct g3_affine){ 1, 0, 0 };
         return true;
      case nir_intrinsic_load_local_invocation_id:
         /* y and z are 0: the workgroup is local_size x 1 x 1 */
         *r = (struct g3_affine){ s.comp == 0, 0, 0 };
         return true;
      case nir_intrinsic_load_workgroup_id:
         /* a dispatch has one workgroup along y and z (pvr_arch_g3_compute.c) */
         *r = (struct g3_affine){ 0, s.comp == 0, 0 };
         return true;
      case nir_intrinsic_load_global_invocation_id:
         *r = (struct g3_affine){ s.comp == 0, s.comp == 0 ? c->local[0] : 0, 0 };
         return true;
      case nir_intrinsic_load_workgroup_size:
         *r = (struct g3_affine){ 0, 0, c->local[s.comp] };
         return true;
      case nir_intrinsic_load_vulkan_descriptor:
         /* 32bit_index_offset: (index, offset); the offset is 0, the descriptor's address includes the
          * binding's offset (pvr_arch_descriptor_set.c) */
         if (s.comp == 1) {
            *r = (struct g3_affine){ 0, 0, 0 };
            return true;
         }
         return refuse(p, "a buffer index in a value");
      default:
         return refuse(p, "intrinsic %s in a value", nir_intrinsic_infos[op].name);
      }
   }
   if (!nir_scalar_is_alu(s))
      return refuse(p, "a value that is not an ALU result, a constant or a system value");

   const nir_op op = nir_scalar_alu_op(s);
   struct g3_affine a, b;
   switch (op) {
   case nir_op_iadd:
   case nir_op_isub:
      if (!g3_affine_of(c, nir_scalar_chase_alu_src(s, 0), &a, depth + 1) ||
          !g3_affine_of(c, nir_scalar_chase_alu_src(s, 1), &b, depth + 1))
         return false;
      if (op == nir_op_isub)
         b = (struct g3_affine){ -b.lid, -b.wg, -b.c };
      *r = (struct g3_affine){ a.lid + b.lid, a.wg + b.wg, a.c + b.c };
      return true;
   case nir_op_ineg:
      if (!g3_affine_of(c, nir_scalar_chase_alu_src(s, 0), &a, depth + 1))
         return false;
      *r = (struct g3_affine){ -a.lid, -a.wg, -a.c };
      return true;
   case nir_op_imul:
      if (!g3_affine_of(c, nir_scalar_chase_alu_src(s, 0), &a, depth + 1) ||
          !g3_affine_of(c, nir_scalar_chase_alu_src(s, 1), &b, depth + 1))
         return false;
      if (!g3_const(&a) && !g3_const(&b))
         return refuse(p, "a product of two varying values");
      if (!g3_const(&b)) {
         struct g3_affine t = a;
         a = b;
         b = t;
      }
      *r = (struct g3_affine){ a.lid * b.c, a.wg * b.c, a.c * b.c };
      return true;
   case nir_op_ishl:
      if (!g3_affine_of(c, nir_scalar_chase_alu_src(s, 0), &a, depth + 1) ||
          !g3_affine_of(c, nir_scalar_chase_alu_src(s, 1), &b, depth + 1))
         return false;
      if (!g3_const(&b))
         return refuse(p, "a shift by a varying amount");
      *r = (struct g3_affine){ a.lid << (b.c & 31), a.wg << (b.c & 31), a.c << (b.c & 31) };
      return true;
   case nir_op_u2u32:
   case nir_op_i2i32:
      if (nir_scalar_chase_alu_src(s, 0).def->bit_size != 32)
         return refuse(p, "a conversion from %u bits", nir_scalar_chase_alu_src(s, 0).def->bit_size);
      return g3_affine_of(c, nir_scalar_chase_alu_src(s, 0), r, depth + 1);
   default:
      return refuse(p, "ALU op %s", nir_op_infos[op].name);
   }
}

/* The descriptor set and binding a buffer index comes from */
static bool
g3_binding_of(struct g3_ctx *c, nir_def *index, unsigned *set, unsigned *binding)
{
   nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(index, 0));

   if (nir_scalar_is_intrinsic(s) && nir_scalar_intrinsic_op(s) == nir_intrinsic_load_vulkan_descriptor) {
      nir_intrinsic_instr *d = nir_def_as_intrinsic(s.def);
      s = nir_scalar_chase_movs(nir_get_scalar(d->src[0].ssa, 0));
   }
   if (!nir_scalar_is_intrinsic(s) || nir_scalar_intrinsic_op(s) != nir_intrinsic_vulkan_resource_index)
      return refuse(c->p, "a storage buffer that is not a plain descriptor binding");
   nir_intrinsic_instr *r = nir_def_as_intrinsic(s.def);
   if (!nir_src_is_const(r->src[0]) || nir_src_as_uint(r->src[0]))
      return refuse(c->p, "an element of a descriptor array");
   if (nir_intrinsic_desc_type(r) != nir_descriptor_type_storage_buffer)
      return refuse(c->p, "a buffer descriptor of type %u (storage buffers only)", nir_intrinsic_desc_type(r));
   *set = nir_intrinsic_desc_set(r);
   *binding = nir_intrinsic_binding(r);
   return true;
}

static bool
g3_store(struct g3_ctx *c, nir_intrinsic_instr *st)
{
   struct g3_program *p = c->p;
   nir_def *value = st->src[0].ssa;
   const unsigned mask = nir_intrinsic_write_mask(st);
   unsigned set, binding;

   if (p->num_stores == G3_MAX_STORES)
      return refuse(p, "more than %u stores", G3_MAX_STORES);
   if (value->bit_size != 32)
      return refuse(p, "a %u-bit store", value->bit_size);
   if (!mask || (mask & (mask + 1)) || util_last_bit(mask) > G3_MAX_COMPONENTS)
      return refuse(p, "a store with write mask %#x", mask);
   if (!g3_binding_of(c, st->src[1].ssa, &set, &binding))
      return false;
   if (set > 255 || binding > 255)
      return refuse(p, "set %u binding %u", set, binding);

   struct g3_store *s = &p->stores[p->num_stores];
   memset(s, 0, sizeof(*s));
   s->set = set;
   s->binding = binding;
   s->components = util_last_bit(mask);
   if (!g3_affine_of(c, nir_get_scalar(st->src[2].ssa, 0), &s->offset, 0))
      return false;
   if ((s->offset.lid | s->offset.wg | s->offset.c) & 3)
      return refuse(p, "an offset that is not a multiple of 4 bytes");
   if (s->offset.lid >= 32)
      return refuse(p, "a per-invocation stride of %#x bytes (0..31: Rogue's constant table)", s->offset.lid);
   for (unsigned i = 0; i < s->components; i++) {
      if (!g3_affine_of(c, nir_get_scalar(value, i), &s->value[i], 0))
         return false;
   }
   if (!g3_const(&s->value[0]) && s->components > 1)
      return refuse(p, "a vector store of an invocation-dependent value");
   if (s->value[0].lid >= 32)
      return refuse(p, "a value lane factor of %#x (0..31: Rogue's constant table)", s->value[0].lid);
   for (unsigned i = 1; i < s->components; i++) {
      if (!g3_const(&s->value[i]))
         return refuse(p, "an invocation-dependent component %u", i);
   }
   p->uses_wgid |= s->offset.wg || s->value[0].wg;
   p->num_stores++;
   return true;
}

static bool
g3_instr_ok(struct g3_ctx *c, nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_alu:
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      /* judged where a store uses it */
      return true;
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
      switch (in->intrinsic) {
      case nir_intrinsic_store_ssbo:
         return g3_store(c, in);
      case nir_intrinsic_vulkan_resource_index:
      case nir_intrinsic_load_vulkan_descriptor:
      case nir_intrinsic_load_local_invocation_index:
      case nir_intrinsic_load_local_invocation_id:
      case nir_intrinsic_load_workgroup_id:
      case nir_intrinsic_load_global_invocation_id:
      case nir_intrinsic_load_workgroup_size:
         return true;
      default:
         return refuse(c->p, "intrinsic %s", nir_intrinsic_infos[in->intrinsic].name);
      }
   }
   default:
      return refuse(c->p, "a NIR instruction of type %u", instr->type);
   }
}

static void
g3_lower(nir_shader *nir)
{
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   nir_remove_non_entrypoints(nir);
   NIR_PASS(_, nir, nir_opt_deref);
   NIR_PASS(_, nir, nir_lower_variable_initializers, ~0);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_system_values);
   const nir_lower_compute_system_values_options csv = { 0 };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo, nir_address_format_32bit_index_offset);
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
   } while (progress);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp | nir_var_shader_temp, NULL);
}

bool
g3_compile_compute(nir_shader *nir, struct g3_program *out)
{
   struct g3_ctx c = { .p = out };

   memset(out, 0, sizeof(*out));
   if (!g3_selftest(out->refusal, sizeof(out->refusal)))
      return false;
   if (nir->info.stage != MESA_SHADER_COMPUTE)
      return refuse(out, "a %s shader (compute only)", mesa_shader_stage_name(nir->info.stage));
   if (nir->info.workgroup_size_variable)
      return refuse(out, "a variable workgroup size");
   for (unsigned i = 0; i < 3; i++)
      c.local[i] = nir->info.workgroup_size[i];
   if (c.local[1] != 1 || c.local[2] != 1 || !c.local[0] || c.local[0] > 32)
      return refuse(out, "a %ux%ux%u workgroup (Nx1x1 with N <= 32: one task)", c.local[0], c.local[1],
                    c.local[2]);
   out->local_size = c.local[0];

   g3_lower(nir);

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   unsigned blocks = 0;
   nir_foreach_block (block, impl)
      blocks++;
   if (blocks != 1)
      return refuse(out, "control flow (%u blocks)", blocks);
   nir_foreach_block (block, impl) {
      nir_foreach_instr (instr, block) {
         if (!g3_instr_ok(&c, instr))
            return false;
      }
   }
   return g3_emit(out);
}

void
g3_describe(const struct g3_program *p, char *buf, unsigned size)
{
   unsigned n = snprintf(buf, size, "local %u, %u store(s)", p->local_size, p->num_stores);
   for (unsigned j = 0; j < p->num_stores && n < size; j++) {
      const struct g3_store *s = &p->stores[j];
      n += snprintf(buf + n, size - n, "; %u.%u[%d*lid%+d*wg%+d] = ", s->set, s->binding, (int)s->offset.lid,
                    (int)s->offset.wg, (int)s->offset.c);
      for (unsigned i = 0; i < s->components && n < size; i++)
         n += snprintf(buf + n, size - n, "%s%d*lid%+d*wg%+d", i ? "," : "", (int)s->value[i].lid,
                       (int)s->value[i].wg, (int)s->value[i].c);
   }
}
