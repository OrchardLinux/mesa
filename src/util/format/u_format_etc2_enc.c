/* SPDX-License-Identifier: MIT */
/*
 * A fast ETC2 encoder for 8-bit RGBA texels.
 *
 * Made for transcoding at texture upload (mesa/st: S3TC content on a GPU that
 * samples ETC2 but not S3TC), so it favours speed: no exhaustive search, and
 * the searches go over a block's distinct colours (a block of S3TC has at most
 * four). Per 4x4 block of colour it tries
 *  - the ETC1 modes: the block split into two halves, each half a base colour
 *    (the half's mean, RGB555 + a 3-bit difference, or 2x RGB444 when the
 *    halves are too far apart for that) and the best of the eight modifier
 *    tables. The split that leaves less spread in its halves, and the other
 *    one when it is close;
 *  - planar, when that leaves an error: a least-squares plane per channel;
 *  - T and H, when that still leaves an error: the block as two clusters of
 *    colour, where ETC1's one colour per half fails (an edge between two
 *    hues, and S3TC's own two endpoint colours when they differ in hue);
 * and keeps the one with the smallest squared error. A punchthrough block
 * with a transparent texel has the differential mode, T and H, each with
 * three colours and the hole. Alpha of the RGBA8 formats is an EAC block:
 * every table with the multipliers that fit the block's range.
 *
 * The bit layouts are those of the Khronos data format specification's ETC2
 * section; the decoders in this tree (texcompress_etc.c, texcompress_etc_tmp.h)
 * are the reference this file is tested against.
 */

#include "u_format_etc2_enc.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* An ETC1 block's error above which planar is tried, and T/H after it */
#ifndef ETC2_ENC_PLANAR_GATE
#define ETC2_ENC_PLANAR_GATE 24
#endif
#ifndef ETC2_ENC_TH_GATE
#define ETC2_ENC_TH_GATE 96
#endif
/* The second split of the ETC1 modes is tried when its estimated cost is
 * within NUM/DEN of the first's
 */
#ifndef ETC2_ENC_FLIP_MARGIN_NUM
#define ETC2_ENC_FLIP_MARGIN_NUM 3
#define ETC2_ENC_FLIP_MARGIN_DEN 2
#endif
/* T/H: the least squared distance between the two farthest texels */
#define ETC2_ENC_TH_MIN_SPLIT 300

#ifdef UTIL_FORMAT_ETC2_ENC_DEBUG
/* For the host check (not built into Mesa): the squared error the encoder
 * expects of the blocks it wrote. The decoders must agree, to the unit.
 */
uint64_t util_format_etc2_enc_debug_rgb_sse, util_format_etc2_enc_debug_alpha_sse;
#define ETC2_ENC_DEBUG_ADD(what, v) (util_format_etc2_enc_debug_##what##_sse += (v))
#else
#define ETC2_ENC_DEBUG_ADD(what, v) ((void)(v))
#endif

struct etc2_enc_src {
   uint8_t px[16][3]; /* texel i = y * 4 + x */
   uint16_t opaque;   /* bit i: texel i is not a punchthrough hole */

   /* The distinct colours of the opaque texels (a block of S3TC has at most
    * four): the searches go over these, weighted, and not over 16 texels.
    */
   uint8_t pal[16][3];
   uint8_t pal_n[16];  /* texels of each colour */
   uint8_t pal_of[16]; /* a texel's colour (not set for holes) */
   unsigned npal;
};

static void
etc2_src_palette(struct etc2_enc_src *src)
{
   uint32_t key[16];
   unsigned n = 0;

   for (unsigned i = 0; i < 16; i++) {
      if (!(src->opaque >> i & 1))
         continue;

      const uint32_t k = src->px[i][0] | src->px[i][1] << 8 | src->px[i][2] << 16;
      unsigned j;

      for (j = 0; j < n && key[j] != k; j++)
         ;
      if (j == n) {
         key[n] = k;
         memcpy(src->pal[n], src->px[i], 3);
         src->pal_n[n++] = 0;
      }
      src->pal_n[j]++;
      src->pal_of[i] = j;
   }
   src->npal = n;
}

/* {small, large}: a table is +small, +large, -small, -large */
static const uint8_t etc1_mod[8][2] = {
   { 2, 8 }, { 5, 17 }, { 9, 29 }, { 13, 42 }, { 18, 60 }, { 24, 80 }, { 33, 106 }, { 47, 183 },
};

static const uint8_t etc2_dist[8] = { 3, 6, 11, 16, 23, 32, 41, 64 };

static const int8_t eac_mod[16][8] = {
   { -3, -6, -9, -15, 2, 5, 8, 14 }, { -3, -7, -10, -13, 2, 6, 9, 12 }, { -2, -5, -8, -13, 1, 4, 7, 12 },
   { -2, -4, -6, -13, 1, 3, 5, 12 }, { -3, -6, -8, -12, 2, 5, 7, 11 },  { -3, -7, -9, -11, 2, 6, 8, 10 },
   { -4, -7, -8, -11, 3, 6, 7, 10 }, { -3, -5, -8, -11, 2, 4, 7, 10 },  { -2, -6, -8, -10, 1, 5, 7, 9 },
   { -2, -5, -8, -10, 1, 4, 7, 9 },  { -2, -4, -8, -10, 1, 3, 7, 9 },   { -2, -5, -7, -10, 1, 4, 6, 9 },
   { -3, -4, -7, -10, 2, 3, 6, 9 },  { -1, -2, -3, -10, 0, 1, 2, 9 },   { -4, -6, -8, -9, 3, 5, 7, 8 },
   { -3, -5, -7, -9, 2, 4, 6, 8 },
};

/* The texels of the two halves: split left/right (flip 0) or top/bottom (1) */
static const uint8_t etc1_half_px[2][2][8] = {
   { { 0, 1, 4, 5, 8, 9, 12, 13 }, { 2, 3, 6, 7, 10, 11, 14, 15 } },
   { { 0, 1, 2, 3, 4, 5, 6, 7 }, { 8, 9, 10, 11, 12, 13, 14, 15 } },
};

static inline int
clamp255(int v)
{
   return v < 0 ? 0 : v > 255 ? 255 : v;
}

static inline int
sq(int v)
{
   return v * v;
}

static inline uint32_t
rgb_dist(const uint8_t *a, const uint8_t *b)
{
   return sq(a[0] - b[0]) + sq(a[1] - b[1]) + sq(a[2] - b[2]);
}

/* Two bits per texel: the low bits in bits 0-15 and the high bits in 16-31
 * of a big-endian word, texel (x, y) at bit x * 4 + y.
 */
static void
etc2_pack_indices(const uint8_t idx[16], uint8_t dst[4])
{
   uint32_t w = 0;

   for (unsigned i = 0; i < 16; i++) {
      const unsigned k = (i & 3) * 4 + (i >> 2);

      w |= (uint32_t)(idx[i] & 1) << k | (uint32_t)(idx[i] >> 1) << (16 + k);
   }
   dst[0] = w >> 24;
   dst[1] = w >> 16;
   dst[2] = w >> 8;
   dst[3] = w;
}

/* A block's first three bytes are, in the differential mode, a 5-bit base
 * (bits 7-3) and a signed 3-bit difference (bits 2-0) per channel. A sum of
 * the two outside [0, 31] marks the other modes: red T, green H, blue planar.
 * These build such a byte around the payload bits.
 */

/* Overflowing: payload v1 in bits 4-3 and v2 in bits 1-0 */
static inline uint8_t
etc2_overflow_byte(unsigned v1, unsigned v2)
{
   return (v1 + v2 >= 4 ? 0xe0 : 0x04) | v1 << 3 | v2;
}

/* Not overflowing: payload in bits 6-0 */
static inline uint8_t
etc2_in_range_byte(unsigned low7)
{
   static const int8_t diff[8] = { 0, 1, 2, 3, -4, -3, -2, -1 };

   return (int)(low7 >> 3) + diff[low7 & 7] < 0 ? 0x80 | low7 : low7;
}

/*
 * The best modifier table of one half for a base colour, and the half's
 * texel indices. A modifier m adds to all three channels, so for a texel at
 * d from the base, with s = dr + dg + db, the error is |d|^2 + m (3 m - 2 s):
 * the table is chosen on that scalar (ignoring the clamp to [0, 255]), the
 * returned error is the decoder's.
 *
 * transparent: a punchthrough block with a hole: its tables are 0, +large,
 * hole, -large.
 */
static uint32_t
etc1_fit_half(const struct etc2_enc_src *src, const uint8_t *list, const uint8_t base[3],
              bool transparent, uint8_t *table, uint8_t idx[16])
{
   uint8_t cnt[16] = { 0 };
   int s[8], as[8], w[8], far = 0;
   uint8_t id[8], pidx[16];
   unsigned n = 0;

   for (unsigned j = 0; j < 8; j++) {
      const unsigned i = list[j];

      if (src->opaque >> i & 1)
         cnt[src->pal_of[i]]++;
      else
         idx[i] = 2;
   }
   for (unsigned k = 0; k < src->npal; k++) {
      if (!cnt[k])
         continue;
      s[n] = src->pal[k][0] - base[0] + src->pal[k][1] - base[1] + src->pal[k][2] - base[2];
      as[n] = abs(s[n]);
      far = as[n] > far ? as[n] : far;
      w[n] = cnt[k];
      id[n++] = k;
   }

   int best = INT_MAX;
   unsigned bt = 0;

   for (unsigned t = 0; t < 8; t++) {
      const int a = transparent ? 0 : etc1_mod[t][0], b = etc1_mod[t][1];
      const int th = 3 * (a + b);
      int e = 0;

      for (unsigned j = 0; j < n; j++) {
         const int m = 2 * as[j] > th ? b : a;

         e += w[j] * m * (3 * m - 2 * as[j]);
      }
      if (e < best) {
         best = e;
         bt = t;
      }
      /* No texel takes this table's large modifier and its small one is
       * past every texel's best (s / 3): the tables after it (both modifiers
       * larger) only add error.
       */
      if (2 * far <= th && 3 * a >= far)
         break;
   }

   const int a = transparent ? 0 : etc1_mod[bt][0], b = etc1_mod[bt][1];
   const int th = 3 * (a + b);
   uint32_t err = 0;

   for (unsigned j = 0; j < n; j++) {
      const uint8_t *p = src->pal[id[j]];
      const bool large = 2 * as[j] > th, neg = s[j] < 0;
      const int m = neg ? -(large ? b : a) : (large ? b : a);

      if (transparent)
         pidx[id[j]] = large ? (neg ? 3 : 1) : 0;
      else
         pidx[id[j]] = (neg ? 2 : 0) | large;
      err += w[j] * (sq(clamp255(base[0] + m) - p[0]) + sq(clamp255(base[1] + m) - p[1]) +
                     sq(clamp255(base[2] + m) - p[2]));
   }
   for (unsigned j = 0; j < 8; j++) {
      const unsigned i = list[j];

      if (src->opaque >> i & 1)
         idx[i] = pidx[src->pal_of[i]];
   }
   *table = bt;
   return err;
}

/* The mean colour of each half's opaque texels (a half of holes takes the
 * other half's): for a block with holes
 */
static void
etc1_means(const struct etc2_enc_src *src, unsigned flip, int mean[2][3])
{
   int sum[2][3] = { { 0 } };
   unsigned n[2] = { 0, 0 };

   for (unsigned h = 0; h < 2; h++) {
      for (unsigned j = 0; j < 8; j++) {
         const unsigned i = etc1_half_px[flip][h][j];

         if (!(src->opaque >> i & 1))
            continue;
         sum[h][0] += src->px[i][0];
         sum[h][1] += src->px[i][1];
         sum[h][2] += src->px[i][2];
         n[h]++;
      }
   }
   for (unsigned h = 0; h < 2; h++) {
      const unsigned k = n[h] ? h : !h;

      for (unsigned c = 0; c < 3; c++)
         mean[h][c] = n[k] ? (sum[k][c] + (int)n[k] / 2) / (int)n[k] : 0;
   }
}

/*
 * One ETC1 candidate: the split, and differential (RGB555 + 3-bit difference,
 * the difference clamped into its range; *clamped tells) or individual
 * (2x RGB444) base colours. Writes dst and returns the error when it beats
 * best.
 */
static uint32_t
etc1_try(const struct etc2_enc_src *src, unsigned flip, bool diff, bool punchthrough, const int mean[2][3],
         uint32_t best, uint8_t dst[8], bool *clamped)
{
   const bool transparent = src->opaque != 0xffff;
   uint8_t base[2][3], bytes[3], table[2], idx[16];
   bool cl = false;

   for (unsigned c = 0; c < 3; c++) {
      if (diff) {
         int q0 = (mean[0][c] * 31 + 127) / 255, q1 = (mean[1][c] * 31 + 127) / 255;
         const int d = q1 - q0;

         if (d > 3) {
            q0 += (d - 3) / 2;
            q1 = q0 + 3;
            cl = true;
         } else if (d < -4) {
            q0 -= (-4 - d) / 2;
            q1 = q0 - 4;
            cl = true;
         }
         base[0][c] = q0 << 3 | q0 >> 2;
         base[1][c] = q1 << 3 | q1 >> 2;
         bytes[c] = q0 << 3 | ((q1 - q0) & 7);
      } else {
         const int q0 = (mean[0][c] * 15 + 127) / 255, q1 = (mean[1][c] * 15 + 127) / 255;

         base[0][c] = q0 * 17;
         base[1][c] = q1 * 17;
         bytes[c] = q0 << 4 | q1;
      }
   }
   if (clamped)
      *clamped = cl;

   uint32_t err = 0;

   for (unsigned h = 0; h < 2; h++) {
      err += etc1_fit_half(src, etc1_half_px[flip][h], base[h], transparent, &table[h], idx);
      if (err >= best)
         return best;
   }

   /* Bit 1: differential, or in a punchthrough format (always differential):
    * no holes in this block.
    */
   memcpy(dst, bytes, 3);
   dst[3] = table[0] << 5 | table[1] << 2 | (punchthrough ? !transparent : diff) << 1 | flip;
   etc2_pack_indices(idx, dst + 4);
   return err;
}

static inline int
div_round(int n, int d)
{
   return (n >= 0 ? n + d / 2 : n - d / 2) / d;
}

/*
 * Planar: three colours O (at texel 0,0), H (at 4,0) and V (at 0,4) in
 * RGB676, interpolated. The least-squares plane of each channel: with S the
 * sum of the texels, Sx the sum weighted by 2x - 3 and Sy by 2y - 3, the
 * plane is O = (5 S - 3 Sx - 3 Sy) / 80, H = O + Sx / 10, V = O + Sy / 10.
 */
static uint32_t
etc2_try_planar(const struct etc2_enc_src *src, uint32_t best, uint8_t dst[8])
{
   int o[3], h[3], v[3]; /* as the decoder expands them */
   unsigned qo[3], qh[3], qv[3];

   for (unsigned c = 0; c < 3; c++) {
      int s = 0, sx = 0, sy = 0;

      for (unsigned i = 0; i < 16; i++) {
         const int p = src->px[i][c];

         s += p;
         sx += (2 * (int)(i & 3) - 3) * p;
         sy += (2 * (int)(i >> 2) - 3) * p;
      }

      const int po = clamp255(div_round(5 * s - 3 * sx - 3 * sy, 80));
      const int ph = clamp255(div_round(5 * s + 5 * sx - 3 * sy, 80));
      const int pv = clamp255(div_round(5 * s - 3 * sx + 5 * sy, 80));

      if (c == 1) {
         qo[c] = (po * 127 + 127) / 255;
         qh[c] = (ph * 127 + 127) / 255;
         qv[c] = (pv * 127 + 127) / 255;
         o[c] = qo[c] << 1 | qo[c] >> 6;
         h[c] = qh[c] << 1 | qh[c] >> 6;
         v[c] = qv[c] << 1 | qv[c] >> 6;
      } else {
         qo[c] = (po * 63 + 127) / 255;
         qh[c] = (ph * 63 + 127) / 255;
         qv[c] = (pv * 63 + 127) / 255;
         o[c] = qo[c] << 2 | qo[c] >> 4;
         h[c] = qh[c] << 2 | qh[c] >> 4;
         v[c] = qv[c] << 2 | qv[c] >> 4;
      }
   }

   uint32_t err = 0;

   for (unsigned i = 0; i < 16; i++) {
      const int x = i & 3, y = i >> 2;

      for (unsigned c = 0; c < 3; c++)
         err += sq(clamp255((x * (h[c] - o[c]) + y * (v[c] - o[c]) + 4 * o[c] + 2) >> 2) - src->px[i][c]);
      if (err >= best)
         return best;
   }

   /* RO6 GO7 BO6 RH6 GH7 BH6 RV6 GV7 BV6, red and green in range, blue
    * overflowing, bit 1 of byte 3 set (differential, or opaque)
    */
   dst[0] = etc2_in_range_byte(qo[0] << 1 | qo[1] >> 6);
   dst[1] = etc2_in_range_byte((qo[1] & 0x3f) << 1 | qo[2] >> 5);
   dst[2] = etc2_overflow_byte((qo[2] >> 3) & 3, (qo[2] >> 1) & 3);
   dst[3] = (qo[2] & 1) << 7 | (qh[0] >> 1) << 2 | 0x2 | (qh[0] & 1);
   dst[4] = qh[1] << 1 | qh[2] >> 5;
   dst[5] = (qh[2] & 0x1f) << 3 | qv[0] >> 3;
   dst[6] = (qv[0] & 7) << 5 | qv[1] >> 2;
   dst[7] = (qv[1] & 3) << 6 | qv[2];
   return err;
}

/* The nearest paint colour for every colour of the block. In a block with
 * holes, index 2 is the hole.
 */
static uint32_t
etc2_th_assign(const struct etc2_enc_src *src, const uint8_t paint[4][3], bool holes, uint32_t limit,
               uint8_t pidx[16])
{
   uint32_t err = 0;

   for (unsigned k = 0; k < src->npal; k++) {
      uint32_t e = rgb_dist(src->pal[k], paint[0]);
      unsigned best = 0;

      for (unsigned j = 1; j < 4; j++) {
         if (holes && j == 2)
            continue;

         const uint32_t e2 = rgb_dist(src->pal[k], paint[j]);

         if (e2 < e) {
            e = e2;
            best = j;
         }
      }
      pidx[k] = best;
      err += e * src->pal_n[k];
      if (err >= limit)
         break;
   }
   return err;
}

static unsigned
etc2_dist_index(int d)
{
   unsigned best = 0;

   for (unsigned j = 1; j < 8; j++) {
      if (abs(etc2_dist[j] - d) < abs(etc2_dist[best] - d))
         best = j;
   }
   return best;
}

/*
 * T and H: two base colours in RGB444 and a distance d, any texel any of four
 * paint colours. H: c0 + d, c0 - d, c1 + d, c1 - d. T: c0, c1 + d, c1, c1 - d.
 * The base colours are the centres of the block's two clusters (2-means from
 * its two farthest texels), d a few table entries around the clusters'
 * deviation along (1, 1, 1).
 */
static uint32_t
etc2_try_th(const struct etc2_enc_src *src, uint32_t best, uint8_t dst[8])
{
   const bool holes = src->opaque != 0xffff;
   uint8_t cen[2][3];
   int sum[3] = { 0, 0, 0 }, total = 0;

   if (src->npal < 2)
      return best;
   for (unsigned k = 0; k < src->npal; k++) {
      for (unsigned c = 0; c < 3; c++)
         sum[c] += src->pal[k][c] * src->pal_n[k];
      total += src->pal_n[k];
   }
   for (unsigned c = 0; c < 3; c++)
      cen[0][c] = (sum[c] + total / 2) / total;

   unsigned far0 = 0, far1 = 0;
   uint32_t d0 = 0, d1 = 0;

   for (unsigned k = 0; k < src->npal; k++) {
      const uint32_t d = rgb_dist(src->pal[k], cen[0]);

      if (d > d0) {
         d0 = d;
         far0 = k;
      }
   }
   for (unsigned k = 0; k < src->npal; k++) {
      const uint32_t d = rgb_dist(src->pal[k], src->pal[far0]);

      if (d > d1) {
         d1 = d;
         far1 = k;
      }
   }
   if (d1 < ETC2_ENC_TH_MIN_SPLIT)
      return best;
   memcpy(cen[0], src->pal[far0], 3);
   memcpy(cen[1], src->pal[far1], 3);

   for (unsigned iter = 0; iter < 2; iter++) {
      int s[2][3] = { { 0 } };
      int n[2] = { 0, 0 };

      for (unsigned k = 0; k < src->npal; k++) {
         const unsigned g = rgb_dist(src->pal[k], cen[1]) < rgb_dist(src->pal[k], cen[0]);

         for (unsigned c = 0; c < 3; c++)
            s[g][c] += src->pal[k][c] * src->pal_n[k];
         n[g] += src->pal_n[k];
      }
      if (!n[0] || !n[1])
         return best;
      for (unsigned g = 0; g < 2; g++) {
         for (unsigned c = 0; c < 3; c++)
            cen[g][c] = (s[g][c] + n[g] / 2) / n[g];
      }
   }

   uint8_t q[2][3], c8[2][3];

   for (unsigned k = 0; k < 2; k++) {
      for (unsigned c = 0; c < 3; c++) {
         q[k][c] = (cen[k][c] * 15 + 127) / 255;
         c8[k][c] = q[k][c] * 17;
      }
   }

   /* each cluster's deviation along (1, 1, 1), summed over its texels, x3 */
   int dev[2] = { 0, 0 }, n[2] = { 0, 0 };

   for (unsigned k = 0; k < src->npal; k++) {
      const uint8_t *p = src->pal[k];
      const unsigned g = rgb_dist(p, c8[1]) < rgb_dist(p, c8[0]);

      dev[g] += abs(p[0] - c8[g][0] + p[1] - c8[g][1] + p[2] - c8[g][2]) * src->pal_n[k];
      n[g] += src->pal_n[k];
   }

   uint8_t paint[4][3], pidx[16], bidx[16], idx[16];
   int mode = -1; /* 0: T with cluster 0 the single colour, 1: T with cluster 1, 2: H */
   unsigned bd = 0;

   /* H: the low bit of the distance's index is not stored: it is whether the
    * first base colour is the larger of the two (as RGB444 numbers). So the
    * distance tells which cluster comes first.
    */
   const unsigned v0 = q[0][0] << 8 | q[0][1] << 4 | q[0][2], v1 = q[1][0] << 8 | q[1][1] << 4 | q[1][2];
   const unsigned larger = v1 > v0;

   if (v0 != v1) {
      const unsigned j = etc2_dist_index((dev[0] + dev[1]) / (3 * total));

      for (unsigned d = j ? j - 1 : 0; d <= (j < 7 ? j + 1 : 7); d++) {
         const unsigned first = (d & 1) ? larger : !larger;

         for (unsigned c = 0; c < 3; c++) {
            paint[0][c] = clamp255(c8[first][c] + etc2_dist[d]);
            paint[1][c] = clamp255(c8[first][c] - etc2_dist[d]);
            paint[2][c] = clamp255(c8[!first][c] + etc2_dist[d]);
            paint[3][c] = clamp255(c8[!first][c] - etc2_dist[d]);
         }

         const uint32_t e = etc2_th_assign(src, paint, holes, best, pidx);

         if (e < best) {
            best = e;
            mode = 2;
            bd = d;
            memcpy(bidx, pidx, 16);
         }
      }
   }

   /* T: the flatter cluster is the single colour; three levels cover the
    * other, so d is about 1.5 of its mean deviation
    */
   const unsigned f = (int64_t)dev[0] * n[1] > (int64_t)dev[1] * n[0];
   const unsigned jt = etc2_dist_index(n[!f] ? dev[!f] / (2 * n[!f]) : 0);

   for (unsigned d = jt ? jt - 1 : 0; d <= (jt < 7 ? jt + 1 : 7); d++) {
      for (unsigned c = 0; c < 3; c++) {
         paint[0][c] = c8[f][c];
         paint[1][c] = clamp255(c8[!f][c] + etc2_dist[d]);
         paint[2][c] = c8[!f][c];
         paint[3][c] = clamp255(c8[!f][c] - etc2_dist[d]);
      }

      const uint32_t e = etc2_th_assign(src, paint, holes, best, pidx);

      if (e < best) {
         best = e;
         mode = f;
         bd = d;
         memcpy(bidx, pidx, 16);
      }
   }

   if (mode < 0)
      return best;

   for (unsigned i = 0; i < 16; i++)
      idx[i] = (src->opaque >> i & 1) ? bidx[src->pal_of[i]] : 2;

   /* bit 1 of byte 3: differential, or in a punchthrough format: no holes */
   const unsigned bit1 = holes ? 0 : 0x2;

   if (mode < 2) {
      /* T: R1 around the overflow, G1 B1, R2 G2, B2 d[2:1] bit1 d[0] */
      const uint8_t *c0 = q[mode], *c1 = q[!mode];

      dst[0] = etc2_overflow_byte(c0[0] >> 2, c0[0] & 3);
      dst[1] = c0[1] << 4 | c0[2];
      dst[2] = c1[0] << 4 | c1[1];
      dst[3] = c1[2] << 4 | (bd >> 1) << 2 | bit1 | (bd & 1);
   } else {
      /* H: R1 G1[3:1] in range, G1[0] B1[3:1] around the overflow, B1[0] R2
       * G2[3:1], G2[0] B2 d[2] bit1 d[1]
       */
      const unsigned first = (bd & 1) ? larger : !larger;
      const uint8_t *c0 = q[first], *c1 = q[!first];

      dst[0] = etc2_in_range_byte(c0[0] << 3 | c0[1] >> 1);
      dst[1] = etc2_overflow_byte((c0[1] & 1) << 1 | c0[2] >> 3, (c0[2] >> 1) & 3);
      dst[2] = (c0[2] & 1) << 7 | c1[0] << 3 | c1[1] >> 1;
      dst[3] = (c1[1] & 1) << 7 | c1[2] << 3 | (bd >> 2) << 2 | bit1 | ((bd >> 1) & 1);
   }
   etc2_pack_indices(idx, dst + 4);
   return best;
}

/*
 * A block without holes: the mean colours of the halves of both splits, which
 * split to try first, and whether the other one is worth trying. What a split
 * cannot represent is the spread of its halves across (1, 1, 1) (one base
 * colour per half; the modifiers only move along it), so the split with the
 * smaller such spread goes first, the spread along (1, 1, 1) counting an
 * eighth (the modifier tables cover most of it).
 *
 * Per half of 8 texels, with P the sum of the colours, N the sum of their
 * squared lengths, s = r + g + b of a texel, S and S2 the sums of s and s^2:
 * 24 x the spread across = 24 N - 3 |P|^2 - (8 S2 - S^2), and 8 S2 - S^2 is
 * 24 x the spread along.
 */
static unsigned
etc1_pick_flip(const struct etc2_enc_src *src, int mean[2][2][3], bool *both)
{
   int p[4][3] = { { 0 } }, n[4] = { 0 }, s2[4] = { 0 };

   /* quadrants: bit 0 right, bit 1 bottom */
   for (unsigned i = 0; i < 16; i++) {
      const unsigned q = ((i >> 1) & 1) | ((i >> 3) << 1);
      const int r = src->px[i][0], g = src->px[i][1], b = src->px[i][2];

      p[q][0] += r;
      p[q][1] += g;
      p[q][2] += b;
      n[q] += r * r + g * g + b * b;
      s2[q] += (r + g + b) * (r + g + b);
   }

   /* halves: left, right (flip 0); top, bottom (flip 1) */
   static const uint8_t half[4][2] = { { 0, 2 }, { 1, 3 }, { 0, 1 }, { 2, 3 } };
   int64_t cost[2] = { 0, 0 };

   for (unsigned h = 0; h < 4; h++) {
      const unsigned a = half[h][0], b = half[h][1];
      const int64_t pr = p[a][0] + p[b][0], pg = p[a][1] + p[b][1], pb = p[a][2] + p[b][2];
      const int64_t s = pr + pg + pb;
      const int64_t along = 8 * (int64_t)(s2[a] + s2[b]) - s * s;
      const int64_t across = 24 * (int64_t)(n[a] + n[b]) - 3 * (pr * pr + pg * pg + pb * pb) - along;

      cost[h / 2] += across + along / 8;
      mean[h / 2][h & 1][0] = (pr + 4) / 8;
      mean[h / 2][h & 1][1] = (pg + 4) / 8;
      mean[h / 2][h & 1][2] = (pb + 4) / 8;
   }

   const unsigned first = cost[1] < cost[0];

   *both = cost[!first] * ETC2_ENC_FLIP_MARGIN_DEN <= cost[first] * ETC2_ENC_FLIP_MARGIN_NUM;
   return first;
}

static void
etc2_encode_color(const struct etc2_enc_src *src, bool punchthrough, uint8_t dst[8])
{
   const bool opaque = src->opaque == 0xffff;
   const bool solid = opaque && src->npal == 1;
   uint32_t best = UINT32_MAX;
   int mean[2][2][3];
   unsigned first = 0;
   bool both = true;

   if (opaque) {
      first = etc1_pick_flip(src, mean, &both);
      both &= !solid;
   } else {
      etc1_means(src, 0, mean[0]);
      etc1_means(src, 1, mean[1]);
   }

   for (unsigned k = 0; k < (both ? 2 : 1); k++) {
      const unsigned flip = first ^ k;
      bool clamped;

      best = etc1_try(src, flip, true, punchthrough, mean[flip], best, dst, &clamped);
      /* punchthrough formats have no individual mode */
      if (clamped && !punchthrough)
         best = etc1_try(src, flip, false, punchthrough, mean[flip], best, dst, NULL);
   }

   /* planar has no holes */
   if (opaque && best > ETC2_ENC_PLANAR_GATE)
      best = etc2_try_planar(src, best, dst);
   if (best > ETC2_ENC_TH_GATE)
      best = etc2_try_th(src, best, dst);
   ETC2_ENC_DEBUG_ADD(rgb, best);
}

/*
 * EAC alpha: a base, a multiplier, one of sixteen tables of eight modifiers
 * and sixteen 3-bit indices; a texel is base + multiplier * modifier, clamped.
 * For every table, the multipliers that stretch it to about the block's range.
 */
static void
eac_encode_alpha(const uint8_t a[16], uint8_t dst[8])
{
   uint8_t vals[16], cnt[16], idx[16];
   unsigned n = 0, lo = 255, hi = 0;
   unsigned bbase = a[0], bmult = 1, btable = 13;

   /* the block's distinct values (at most 8 from DXT5) */
   for (unsigned i = 0; i < 16; i++) {
      unsigned k;

      for (k = 0; k < n && vals[k] != a[i]; k++)
         ;
      if (k == n) {
         vals[n] = a[i];
         cnt[n++] = 0;
      }
      cnt[k]++;
      lo = a[i] < lo ? a[i] : lo;
      hi = a[i] > hi ? a[i] : hi;
   }

   if (n == 1) {
      /* table 13 has a modifier of 0, at index 4 */
      memset(idx, 4, 16);
   } else {
      uint32_t best = UINT32_MAX;

      for (unsigned t = 0; t < 16; t++) {
         const int tlo = eac_mod[t][3], thi = eac_mod[t][7];
         const int m0 = div_round(hi - lo, thi - tlo);

         for (int m = m0 > 1 ? m0 - 1 : 1; m <= m0 + 1 && m <= 15; m++) {
            const int base = clamp255(div_round((int)(lo + hi) - m * (tlo + thi), 2));
            uint32_t err = 0;

            for (unsigned k = 0; k < n && err < best; k++) {
               int e = INT_MAX;

               for (unsigned j = 0; j < 8; j++) {
                  const int d = sq(clamp255(base + m * eac_mod[t][j]) - vals[k]);

                  e = d < e ? d : e;
               }
               err += e * cnt[k];
            }
            if (err < best) {
               best = err;
               bbase = base;
               bmult = m;
               btable = t;
            }
         }
      }
      ETC2_ENC_DEBUG_ADD(alpha, best);
      for (unsigned i = 0; i < 16; i++) {
         int e = INT_MAX;

         for (unsigned j = 0; j < 8; j++) {
            const int d = sq(clamp255((int)bbase + (int)bmult * eac_mod[btable][j]) - a[i]);

            if (d < e) {
               e = d;
               idx[i] = j;
            }
         }
      }
   }

   /* base, multiplier | table, then 48 bits of indices, texel (0, 0) first,
    * columns before rows
    */
   uint64_t bits = 0;

   for (unsigned i = 0; i < 16; i++)
      bits |= (uint64_t)idx[i] << (((3 - (i >> 2)) + (3 - (i & 3)) * 4) * 3);
   dst[0] = bbase;
   dst[1] = bmult << 4 | btable;
   for (unsigned i = 0; i < 6; i++)
      dst[2 + i] = bits >> (40 - 8 * i);
}

bool
util_format_etc2_can_encode(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_ETC2_RGB8:
   case PIPE_FORMAT_ETC2_SRGB8:
   case PIPE_FORMAT_ETC2_RGB8A1:
   case PIPE_FORMAT_ETC2_SRGB8A1:
   case PIPE_FORMAT_ETC2_RGBA8:
   case PIPE_FORMAT_ETC2_SRGBA8:
      return true;
   default:
      return false;
   }
}

void
util_format_etc2_encode_rgba8(enum pipe_format format,
                              uint8_t *dst_row, unsigned dst_stride,
                              const uint8_t *src_row, unsigned src_stride,
                              unsigned width, unsigned height)
{
   const bool alpha = format == PIPE_FORMAT_ETC2_RGBA8 || format == PIPE_FORMAT_ETC2_SRGBA8;
   const bool punchthrough = format == PIPE_FORMAT_ETC2_RGB8A1 || format == PIPE_FORMAT_ETC2_SRGB8A1;

   for (unsigned by = 0; by < height; by += 4) {
      uint8_t *dst = dst_row + (size_t)(by / 4) * dst_stride;

      for (unsigned bx = 0; bx < width; bx += 4) {
         struct etc2_enc_src src;
         uint8_t a[16];

         src.opaque = 0xffff;
         for (unsigned i = 0; i < 16; i++) {
            const unsigned x = bx + (i & 3) < width ? bx + (i & 3) : width - 1;
            const unsigned y = by + (i >> 2) < height ? by + (i >> 2) : height - 1;
            const uint8_t *p = src_row + (size_t)y * src_stride + (size_t)x * 4;

            memcpy(src.px[i], p, 3);
            a[i] = p[3];
            if (punchthrough && p[3] < 128)
               src.opaque &= ~(1u << i);
         }
         etc2_src_palette(&src);

         if (alpha) {
            eac_encode_alpha(a, dst);
            dst += 8;
         }
         etc2_encode_color(&src, punchthrough, dst);
         dst += 8;
      }
   }
}
