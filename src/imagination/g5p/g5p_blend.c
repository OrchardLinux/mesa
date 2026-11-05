/* SPDX-License-Identifier: MIT
 * General blend epilogue (docs/gpu/experiments/hyprland-gpu.md §11.2 stage 2,
 * blend-forms.md): per channel op(src * Fs, dst * Fd) for every VkBlendFactor
 * and VkBlendOp, built only from the native-f16 forms of Apple's 45 blend
 * probes (applegpu_g5p, one Metal blend setting each).
 *
 * The program is built in three steps:
 *  1. lowering: a DAG of value nodes per channel, with Apple's shapes
 *     (src_X: factor * src, dst_X: dst * factor, the source-over FMA, the
 *     SourceAlpha/OneMinusSourceAlpha lerp and dot2, the blend-colour text
 *     form (c - d) * s + d, the alpha min/max pair);
 *  2. the tile-value rule: a destination component is read straight from
 *     tile_component (its readers carry source-control bit 7/6/5/4 for slot
 *     A/B/C/D) unless one reader cannot carry the bit (a negated B, a literal,
 *     min/max, the output itself, a slot without a witness), in which case
 *     the -0.0 step runs before tile_component and no reader has the bit;
 *  3. order and registers: then the bits that depend on them, by the rules
 *     that reproduce all of Apple's 45 programs:
 *     - source group 0 exactly when the value is read again later (an
 *       instruction or the final emit), 1 on its last read;
 *     - destination control 0x80 exactly when an instruction more than two
 *       places later reads the result (the emit excluded);
 *     - class extension 4 when a source is group 0 and read again by an
 *       instruction; header 0x41 when that source is slot B and A is not
 *       group 0.
 * The colour is in halves 0..3 (saturated f16), the tile read goes to half 4.
 * The blend constants are four f32 common dwords from `const_base`.
 */
#include "g5p_usc.h"

#include <stdio.h>
#include <string.h>

enum { F_ZERO, F_ONE, F_SC, F_OMSC, F_DC, F_OMDC, F_SA, F_OMSA, F_DA, F_OMDA, F_CC, F_OMCC, F_CA, F_OMCA, F_SAS };
enum { O_ADD, O_SUB, O_RSUB, O_MIN, O_MAX };

enum bop {
   B_SRC,    /* source colour half `lane` */
   B_RAW,    /* raw tile half of `lane` */
   B_TC,     /* tile_component of RAW (stepped: the -0.0 step first) */
   B_ONEM,   /* 1 - x: f16 literal 3c00 + -B */
   B_MOVC,   /* common dword into a half */
   B_ONEMC,  /* -1 * x + 1 (x = MOVC) */
   B_MUL,    /* A * B */
   B_SQUARE, /* A * A */
   B_MULC,   /* c[dword] * B */
   B_SUBC,   /* c[dword] - B */
   B_ADD,    /* A + B, A - B (neg) */
   B_FMA,    /* A * B + C */
   B_LERP,   /* A + B * (C - A), A + B * (-C - A) (neg) */
   B_DOT2,   /* A * B - C * D */
   B_MINMAX, /* min/max(A, B); pair: the alpha form, FIXUP follows */
   B_FIXUP,  /* the pair's second instruction: A = MINMAX, B = MINMAX's A */
   B_COPY,   /* 0 + B: defer output assignment until all source reads finish */
   B_NEG,    /* 0 - B */
   B_ZERO,   /* literal zero, without modifying the original source alpha */
};

#define MAXN 64
#define MAXI 96

struct bnode {
   enum bop op;
   int s[4];      /* operand nodes, -1 none */
   unsigned lane; /* SRC/RAW/TC: channel; MOVC/MULC/SUBC: common dword */
   bool neg, max, pair;
   int final;     /* output channel, -1 */
   bool stepped;  /* TC */
   int half;
   int def;       /* flat index of the defining instruction */
   int last;      /* flat index of the last read (-1 none) */
};

struct binstr {
   int node;
   bool step; /* the -0.0 step of a stepped TC */
};

struct bctx {
   struct bnode n[MAXN];
   unsigned nn;
   struct binstr ins[MAXI];
   unsigned ni;
   int raw[4];
   int out[4];         /* the node in half L at the emit */
   unsigned const_base;
   bool tile_alpha_only;
   bool general; /* canonical fallback; measured shapes keep their exact bytes */
   char err[160];
};

#ifdef G5P_BLEND_TEST
static void g5p_blend_test_program(const struct bctx *, const struct g5p_blend_info *);
#endif

#define ERR(c, ...) do { if (!(c)->err[0]) snprintf((c)->err, sizeof((c)->err), __VA_ARGS__); } while (0)
#define FAIL(c, ...) do { ERR(c, __VA_ARGS__); return -1; } while (0)

static int node(struct bctx *c, enum bop op, int a, int b, int x, int d, unsigned lane, bool neg, bool max,
                bool pair, bool cse)
{
   if (c->err[0] || a < -1 || b < -1 || x < -1 || d < -1)
      return -1;
   if (cse)
      for (unsigned i = 0; i < c->nn; i++) {
         struct bnode *e = &c->n[i];
         if (e->op == op && e->s[0] == a && e->s[1] == b && e->s[2] == x && e->s[3] == d && e->lane == lane &&
             e->neg == neg && e->max == max && e->pair == pair && e->final < 0)
            return i;
      }
   if (c->nn == MAXN)
      FAIL(c, "blend: too many values");
   c->n[c->nn] = (struct bnode){op, {a, b, x, d}, lane, neg, max, pair, -1, false, -1, -1, -1};
   return c->nn++;
}

static int src(struct bctx *c, unsigned l)
{
   return node(c, B_SRC, -1, -1, -1, -1, l, false, false, false, true);
}

static int tile(struct bctx *c, unsigned l, bool final)
{
   if (c->raw[l] < 0)
      c->raw[l] = node(c, B_RAW, -1, -1, -1, -1, l, false, false, false, true);
   return node(c, B_TC, c->raw[l], -1, -1, -1, l, false, false, false, !final);
}

/* A factor: a node, a common dword, or the constants 0 and 1. */
enum { V_ZERO, V_ONE, V_NODE, V_COMMON };
struct fv {
   int kind, n;
   unsigned dword;
};

static struct fv V(int n)
{
   return (struct fv){n < 0 ? -1 : V_NODE, n, 0};
}

static int onem(struct bctx *c, int x)
{
   return node(c, B_ONEM, -1, x, -1, -1, 0, true, false, false, true);
}

static int onemc(struct bctx *c, unsigned dword)
{
   int m = node(c, B_MOVC, -1, -1, -1, -1, dword, false, false, false, true);
   return node(c, B_ONEMC, -1, m, -1, -1, 0, false, false, false, true);
}

static struct fv factor(struct bctx *c, unsigned f, unsigned l)
{
   switch (f) {
   case F_ZERO: return (struct fv){V_ZERO};
   case F_ONE: return (struct fv){V_ONE};
   case F_SC: return V(src(c, l));
   case F_OMSC: return V(onem(c, src(c, l)));
   case F_DC: return V(tile(c, l, false));
   case F_OMDC: return V(onem(c, tile(c, l, false)));
   case F_SA: return V(src(c, 3));
   case F_OMSA: return V(onem(c, src(c, 3)));
   case F_DA: return V(tile(c, 3, false));
   case F_OMDA: return V(onem(c, tile(c, 3, false)));
   case F_CC: return (struct fv){V_COMMON, -1, c->const_base + l};
   case F_OMCC: return V(onemc(c, c->const_base + l));
   case F_CA: return (struct fv){V_COMMON, -1, c->const_base + 3};
   case F_OMCA: return V(onemc(c, c->const_base + 3));
   case F_SAS:
      if (l == 3)
         return (struct fv){V_ONE};
      /* Apple's min(As, 1 - Ad): the pair with A = 1 - Ad (src/dst_SourceAlphaSaturated) */
      {
         int a = onem(c, tile(c, 3, false));
         int m = node(c, B_MINMAX, a, src(c, 3), -1, -1, 0, false, false, true, true);
         return V(node(c, B_FIXUP, m, a, -1, -1, 0, false, false, false, true));
      }
   default:
      ERR(c, "blend: factor %u", f);
      return (struct fv){-1, -1};
   }
}

/* x * f: src_X (A = factor, B = source) and dst_X (A = destination, B = factor). */
static struct fv product(struct bctx *c, int x, struct fv f, bool dst)
{
   switch (f.kind) {
   case V_ZERO:
   case V_ONE: return f.kind == V_ONE ? V(x) : f;
   case V_COMMON: return V(node(c, B_MULC, -1, x, -1, -1, f.dword, false, false, false, true));
   case V_NODE:
      if (f.n == x)
         return V(node(c, B_SQUARE, x, -1, -1, -1, 0, false, false, false, true));
      return V(dst ? node(c, B_MUL, x, f.n, -1, -1, 0, false, false, false, true)
                   : node(c, B_MUL, f.n, x, -1, -1, 0, false, false, false, true));
   default: return (struct fv){-1, -1};
   }
}

static bool used(struct bctx *c, int v)
{
   for (unsigned u = 0; u < c->nn; u++)
      for (unsigned s = 0; s < 4; s++)
         if (c->n[u].s[s] == v)
            return true;
   return false;
}

static int final_of(struct bctx *c, int v, unsigned l)
{
   if (v < 0)
      return -1;
   if (c->general) {
      c->out[l] = v;
      return v;
   }
   /* The destination as the output while a factor reads it too: Apple keeps
    * one tile_component in the output half (its source colour sits elsewhere,
    * dst_SourceAlphaSaturated); ours would read the tile twice. Unwitnessed. */
   if (c->n[v].op == B_TC && (c->n[v].final >= 0 || used(c, v)))
      FAIL(c, "blend: channel %u outputs the destination that a factor also reads (a second tile read, "
              "no Apple witness)", l);
   if (c->n[v].op != B_SRC) {
      if (c->n[v].final >= 0)
         FAIL(c, "blend: channel %u output is channel %d's", l, c->n[v].final);
      c->n[v].final = l;
   } else if (c->n[v].lane != l) {
      FAIL(c, "blend: channel %u output is source channel %u", l, c->n[v].lane);
   }
   c->out[l] = v;
   return v;
}

static int fin(struct bctx *c, enum bop op, int a, int b, int x, int d, bool neg, bool max, bool pair,
               unsigned l)
{
   int v = node(c, op, a, b, x, d, 0, neg, max, pair, false);
   return final_of(c, v, l);
}

static void channel(struct bctx *c, unsigned l, unsigned fs, unsigned fd, unsigned op)
{
   int S = src(c, l);
   if (c->general) {
      int v;
      if (op == O_MIN || op == O_MAX) {
         int d = tile(c, l, false);
         /* Preserve Apple's paired alpha min/max and SAS factor form. */
         v = node(c, B_MINMAX, l == 3 ? d : S, l == 3 ? S : d,
                  -1, -1, 0, false, op == O_MAX, l == 3, false);
         if (l == 3)
            v = node(c, B_FIXUP, v, d, -1, -1, 0, false, false, false, false);
      } else {
         struct fv sf = factor(c, fs, l), df = factor(c, fd, l);
         struct fv s = product(c, S, sf, false);
         struct fv d = df.kind == V_ZERO ? df : product(c, tile(c, l, false), df, true);
         bool sz = s.kind == V_ZERO, dz = d.kind == V_ZERO;
         if (sz && dz)
            v = node(c, B_ZERO, -1, -1, -1, -1, 0, false, false, false, true);
         else if (sz || dz) {
            v = sz ? d.n : s.n;
            if ((op == O_SUB && sz) || (op == O_RSUB && dz))
               v = node(c, B_NEG, -1, v, -1, -1, 0, false, false, false, true);
         } else {
            /* Equal products still need distinct source slots. Keep the
             * witnessed binary form instead of reading one half twice.
             */
            if (s.n == d.n)
               d.n = node(c, B_COPY, -1, d.n, -1, -1, 0, false, false, false, false);
            v = node(c, B_ADD, op == O_RSUB ? d.n : s.n,
                     op == O_RSUB ? s.n : d.n, -1, -1, 0,
                     op != O_ADD, false, false, false);
         }
      }
      final_of(c, v, l);
      return;
   }
   if (op == O_MIN || op == O_MAX) { /* factors ignored */
      bool max = op == O_MAX;
      if (l == 3) {
         int d = tile(c, 3, false);
         int m = node(c, B_MINMAX, d, S, -1, -1, 0, false, max, true, false);
         fin(c, B_FIXUP, m, d, -1, -1, false, false, false, l);
      } else {
         fin(c, B_MINMAX, S, tile(c, l, false), -1, -1, false, max, false, l);
      }
      return;
   }
   if (fs == F_SA && fd == F_OMSA && l < 3) { /* op_*_sa (on alpha, As would fill two slots) */
      if (op == O_SUB)
         fin(c, B_DOT2, src(c, 3), S, tile(c, l, false), onem(c, src(c, 3)), true, false, false, l);
      else
         fin(c, B_LERP, tile(c, l, false), src(c, 3), S, -1, op == O_RSUB, false, false, l);
      return;
   }
   if (op == O_ADD && (fs == F_CC || fs == F_CA) && fd == F_OMSC) { /* qs_text: (c - d) * s + d */
      int d = tile(c, l, false);
      int x = node(c, B_SUBC, -1, d, -1, -1, c->const_base + (fs == F_CC ? l : 3), true, false, false, true);
      fin(c, B_FMA, x, S, d, -1, false, false, false, l);
      return;
   }
   struct fv vs = factor(c, fs, l), vd = factor(c, fd, l);
   if (c->err[0])
      return;
   bool dzero = vd.kind == V_ZERO, szero = vs.kind == V_ZERO;
   if (szero && dzero) {
      ERR(c, "blend: channel %u is constant 0", l);
      return;
   }
   if (op == O_ADD && vs.kind == V_ONE && vd.kind == V_NODE && c->n[vd.n].op != B_TC && vd.n != S) { /* premult */
      fin(c, B_FMA, tile(c, l, false), vd.n, S, -1, false, false, false, l);
      return;
   }
   struct fv ts = szero ? vs : product(c, S, vs, false);
   struct fv td = dzero ? vd : product(c, tile(c, l, false), vd, true);
   if (c->err[0])
      return;
   switch (op) {
   case O_ADD:
      if (dzero || szero)
         final_of(c, dzero ? ts.n : td.n, l);
      else
         fin(c, B_ADD, td.n, ts.n, -1, -1, false, false, false, l);
      return;
   case O_SUB:
      if (dzero)
         final_of(c, ts.n, l);
      else if (szero)
         ERR(c, "blend: channel %u: -dst*Fd has no witnessed negation", l);
      else
         fin(c, B_ADD, ts.n, td.n, -1, -1, true, false, false, l);
      return;
   case O_RSUB:
      if (szero)
         final_of(c, td.n, l);
      else if (dzero)
         ERR(c, "blend: channel %u: -src*Fs has no witnessed negation", l);
      else
         fin(c, B_ADD, td.n, ts.n, -1, -1, true, false, false, l);
      return;
   default:
      ERR(c, "blend: op %u", op);
   }
}

/* Slots of each operation that carry the tile-value bit (witnessed). */
static bool carries(const struct bnode *e, unsigned slot)
{
   switch (e->op) {
   case B_MUL: return slot <= 1;
   case B_SQUARE: return slot == 0;
   case B_MULC: return slot == 1;
   case B_ADD: return slot == 0;
   case B_FMA: return slot == 0;
   case B_LERP: return slot == 0;
   case B_DOT2: return slot == 2;
   default: return false;
   }
}

static bool unprocessed(struct bctx *c, int v)
{
   return v >= 0 && c->n[v].op == B_TC && !c->n[v].stepped;
}

/* Read of node v at flat index i: is it read again (by an instruction, or by the emit)? */
static bool read_again(struct bctx *c, int v, int i, bool *by_instr)
{
   *by_instr = c->n[v].last > i;
   bool emit = false;
   for (unsigned l = 0; l < 4; l++)
      emit |= c->out[l] == v;
   return *by_instr || emit;
}

static void reads(struct bctx *c, unsigned i, int v)
{
   if (v >= 0 && (int)i > c->n[v].last)
      c->n[v].last = i;
}

static unsigned blend_build(uint8_t *out, unsigned max, const struct g5p_blend_key *k, unsigned const_base,
                            struct g5p_blend_info *info, const char **err, bool general)
{
   static _Thread_local struct bctx cx;
   struct bctx *c = &cx;
   memset(c, 0, sizeof(*c));
   memset(info, 0, sizeof(*info));
   c->const_base = const_base;
   c->general = general;
   for (unsigned l = 0; l < 4; l++)
      c->raw[l] = c->out[l] = -1;
   *err = NULL;

   /* channels that blend to constant 0 take source 0 and pass it through
    * (the compiler zeroes them before the epilogue) */
   uint8_t fs[4], fd[4], op[4];
   for (unsigned l = 0; l < 4; l++) {
      fs[l] = l < 3 ? k->src_rgb : k->src_a;
      fd[l] = l < 3 ? k->dst_rgb : k->dst_a;
      op[l] = l < 3 ? k->op_rgb : k->op_a;
      if (fs[l] > F_SAS || fd[l] > F_SAS || op[l] > O_MAX) {
         snprintf(c->err, sizeof(c->err), "blend factors %u/%u op %u: outside VkBlendFactor/VkBlendOp core",
                  fs[l], fd[l], op[l]);
         *err = c->err;
         return 0;
      }
      if (!general && op[l] <= O_RSUB && fs[l] == F_ZERO && fd[l] == F_ZERO) {
         info->zero_lanes |= 1u << l;
         fs[l] = F_ONE;
      }
   }
   bool uses_sa = false;
   for (unsigned l = 0; l < 3; l++)
      uses_sa |= op[l] <= O_RSUB && !(info->zero_lanes & (1u << l)) &&
                 (fs[l] == F_SA || fs[l] == F_OMSA || fs[l] == F_SAS || fd[l] == F_SA || fd[l] == F_OMSA ||
                  fd[l] == F_SAS);
   if ((info->zero_lanes & 8) && uses_sa) {
      *err = "blend: alpha blends to 0 while the colour factors read the source alpha";
      return 0;
   }
   if (info->zero_lanes == 0xf) {
#ifdef G5P_BLEND_TEST
      g5p_blend_test_program(c, info);
#endif
      return 0; /* no epilogue: the colour is constant 0 */
   }

   /* 1. lower: colour channels, then alpha (its output overwrites the source alpha) */
   for (unsigned l = 0; l < 4; l++)
      channel(c, l, fs[l], fd[l], op[l]);
   if (general && !c->err[0]) {
      /* Every equation reads the original colour/alpha, including when the
       * alpha equation is zero. Only now may the four output halves change.
       */
      for (unsigned l = 0; l < 4; l++) {
         int v = node(c, B_COPY, -1, c->out[l], -1, -1, 0, false, false, false, false);
         if (v < 0)
            break;
         c->n[v].final = l;
         c->out[l] = v;
      }
   }
   for (unsigned l = 0; l < 4 && !c->err[0]; l++)
      if (c->out[l] < 0)
         snprintf(c->err, sizeof(c->err), "blend: channel %u has no output", l);
   if (c->err[0]) {
      *err = c->err;
      return 0;
   }

   /* Apple's programs never read one value in two slots (x * x is the square) */
   for (unsigned v = 0; v < c->nn; v++)
      for (unsigned a = 0; a < 4; a++)
         for (unsigned b = a + 1; b < 4; b++)
            if (c->n[v].s[a] >= 0 && c->n[v].s[a] == c->n[v].s[b]) {
               snprintf(c->err, sizeof(c->err), "blend: one value in two operand slots (no Apple witness)");
               *err = c->err;
               return 0;
            }

   /* 2. the tile-value rule */
   for (unsigned v = 0; v < c->nn; v++) {
      if (c->n[v].op != B_TC)
         continue;
      bool all = c->n[v].final < 0;
      for (unsigned u = 0; u < c->nn && all; u++)
         for (unsigned s = 0; s < 4; s++)
            if (c->n[u].s[s] == (int)v && !carries(&c->n[u], s))
               all = false;
      c->n[v].stepped = general || !all;
   }
   for (unsigned v = 0; v < c->nn; v++) {
      if (c->n[v].op == B_LERP && !unprocessed(c, c->n[v].s[0])) {
         *err = "blend: the lerp's destination is also read where the tile bit is not witnessed";
         return 0;
      }
      if (c->n[v].op == B_MULC || c->n[v].op == B_SUBC || c->n[v].op == B_MOVC)
         info->uses_constants = true;
   }

   /* 3. order: tile read, then the values in creation order */
   bool any_raw = false, colour_raw = false;
   for (unsigned l = 0; l < 4; l++) {
      any_raw |= c->raw[l] >= 0;
      colour_raw |= l < 3 && c->raw[l] >= 0;
   }
   c->tile_alpha_only = any_raw && !colour_raw;
   unsigned base = any_raw ? 2 : 0; /* tile_read, wait */
   for (unsigned v = 0; v < c->nn; v++) {
      struct bnode *e = &c->n[v];
      if (e->op == B_SRC || e->op == B_RAW) {
         e->def = 0;
         continue;
      }
      if (e->op == B_TC && e->stepped)
         c->ins[c->ni++] = (struct binstr){v, true};
      e->def = base + c->ni;
      c->ins[c->ni++] = (struct binstr){v, false};
   }
   for (unsigned i = 0; i < c->ni; i++) {
      struct bnode *e = &c->n[c->ins[i].node];
      if (e->op == B_TC) {
         if (c->ins[i].step || !e->stepped)
            reads(c, base + i, e->s[0]);
         continue;
      }
      for (unsigned s = 0; s < 4; s++)
         reads(c, base + i, e->s[s]);
   }

   /* registers: the colour in 0..3, the raw tile from 4, values in the lowest free half from 4 */
   int holder[32];
   for (unsigned h = 0; h < 32; h++)
      holder[h] = -1;
   for (unsigned v = 0; v < c->nn; v++) {
      struct bnode *e = &c->n[v];
      if (e->op == B_SRC)
         e->half = e->lane;
      else if (e->op == B_RAW)
         e->half = c->tile_alpha_only ? 5 : 4 + e->lane;
      else
         continue;
      holder[e->half] = v;
   }
   for (unsigned l = 0; l < 4; l++)
      if (holder[l] < 0)
         holder[l] = -2; /* a source channel nothing reads: still emitted if passed through */
   unsigned top = any_raw ? 7 : 3;
   for (unsigned i = 0; i < c->ni; i++) {
      int fi = base + i;
      int v = c->ins[i].node;
      struct bnode *e = &c->n[v];
      if (e->op == B_TC && !c->ins[i].step && e->stepped)
         continue; /* tile_component into the step's half */
      /* free what dies here (never an emitted half) */
      for (unsigned h = 0; h < 32; h++) {
         int u = holder[h];
         if (u < 0)
            continue;
         bool emitted = false;
         for (unsigned l = 0; l < 4; l++)
            emitted |= c->out[l] == u;
         if (!emitted && c->n[u].last <= fi && c->n[u].def <= fi)
            holder[h] = -1;
      }
      int h = -1;
      if (e->final >= 0) {
         h = e->final;
         if (holder[h] >= 0 && holder[h] != v) {
            snprintf(c->err, sizeof(c->err), "blend: channel %d output overwrites a live value", e->final);
            *err = c->err;
            return 0;
         }
      } else {
         for (unsigned x = 4; x < 32 && h < 0; x++)
            if (holder[x] < 0)
               h = x;
         if (h < 0) {
            *err = "blend: no free half";
            return 0;
         }
      }
      e->half = h;
      holder[h] = v;
      if ((unsigned)h > top)
         top = h;
   }
   info->max_half = top;
   info->reads_tile = any_raw;

   /* encode */
   unsigned n = 0;
   uint8_t raw[G5P_USC_MAX];
#define PUT(len, what)                                                                                         \
   do {                                                                                                        \
      unsigned _l = (len);                                                                                     \
      if (!_l || n + _l > max) {                                                                               \
         snprintf(c->err, sizeof(c->err), "blend: %s (%s)", what, _l ? "too long" : g5p_usc_error);           \
         *err = c->err;                                                                                        \
         return 0;                                                                                             \
      }                                                                                                        \
      memcpy(out + n, raw, _l);                                                                                \
      n += _l;                                                                                                 \
   } while (0)
   if (any_raw) {
      PUT(c->tile_alpha_only ? g5p_usc_tile_read_component(raw, 4, 1, 2) : g5p_usc_tile_read(raw, 4, 2),
          "tile read");
      PUT(g5p_usc_wait(raw, 0), "wait");
   }
   for (unsigned i = 0; i < c->ni; i++) {
      int fi = base + i;
      int v = c->ins[i].node;
      struct bnode *e = &c->n[v];
      struct g5p_src S[4];
      bool g0[4] = {0}, again_i[4] = {0};
      unsigned sc = 0;
      for (unsigned s = 0; s < 4; s++) {
         int u = e->s[s];
         if (u < 0)
            continue;
         bool by_i;
         g0[s] = read_again(c, u, fi, &by_i);
         again_i[s] = g0[s] && by_i;
         S[s] = (struct g5p_src){.half = c->n[u].half, .group = !g0[s]};
         if (unprocessed(c, u))
            sc |= 0x80 >> s;
      }
      /* destination control: read by an instruction more than two places later */
      bool far = false;
      for (unsigned j = i + 1; j < c->ni; j++) {
         struct bnode *r = &c->n[c->ins[j].node];
         bool rd = false;
         if (r->op == B_TC)
            rd = (c->ins[j].step || !r->stepped) && r->s[0] == v;
         else
            for (unsigned s = 0; s < 4; s++)
               rd |= r->s[s] == v;
         if (rd && j > i + 2)
            far = true;
      }
      struct g5p_ctl k = G5P_CTL;
      k.dst_control = far ? 0x80 : 0;
      bool ext = false;
      for (unsigned s = 0; s < 4; s++)
         ext |= again_i[s];
      k.source_extension = ext ? 4 : 0;
      k.header = again_i[1] && !g0[0] ? 0x41 : 1;
      switch (e->op) {
      case B_TC:
         if (c->ins[i].step) {
            /* the -0.0 step: raw tile half (read once) into this value's half */
            bool by_i;
            struct g5p_src b = {.half = c->n[e->s[0]].half, .group = !read_again(c, e->s[0], fi, &by_i)};
            struct g5p_ctl ks = G5P_CTL;
            PUT(g5p_usc_half_binary_literal(raw, e->half, 0x8000, b, ks, false, 2), "tile step");
         } else {
            unsigned from = e->stepped ? (unsigned)e->half : (unsigned)c->n[e->s[0]].half;
            if (!e->stepped && c->n[e->s[0]].last > fi) {
               *err = "blend: a raw tile half read again after its tile_component (no witnessed group)";
               return 0;
            }
            PUT(g5p_usc_tile_component2(raw, e->half, from, k.dst_control), "tile component");
         }
         break;
      case B_ONEM:
         PUT(g5p_usc_half_binary_literal(raw, e->half, 0x3c00, S[1], k, true, 0), "1 - x");
         break;
      case B_COPY:
      case B_NEG:
         PUT(g5p_usc_half_binary_literal(raw, e->half, 0, S[1], k, e->op == B_NEG, 0),
             e->op == B_NEG ? "negate" : "output copy");
         break;
      case B_ZERO:
         PUT(g5p_usc_half_immediate(raw, e->half, 0), "zero");
         break;
      case B_MOVC:
         PUT(g5p_usc_half_move_shared(raw, e->half, e->lane, 0, k), "blend constant move");
         break;
      case B_ONEMC:
         PUT(g5p_usc_half_fma_literal(raw, e->half, 0xbc00, S[1], 0x3c00, k, 0x70, 0x80), "1 - constant");
         break;
      case B_MUL:
      case B_ADD:
         PUT(g5p_usc_half_binary2(raw, e->op == B_MUL, e->half, S[0], S[1], k, sc ? (int)sc : -1, e->neg),
             e->op == B_MUL ? "multiply" : "add");
         break;
      case B_SQUARE:
         PUT(g5p_usc_half_square(raw, e->half, S[0], k, sc ? 0xc0 : 0), "square");
         break;
      case B_MULC:
      case B_SUBC: {
         /* routing 0x70: Apple's dst_BlendAlpha, channels 2-3 (single witness) */
         unsigned routing = e->op == B_MULC && e->lane == c->const_base + 3 && unprocessed(c, e->s[1]) &&
                                  c->n[e->s[1]].lane >= 2 ? 0x70 : 0;
         PUT(g5p_usc_half_binary_shared(raw, e->op == B_MULC, e->half, e->lane, 0, S[1], k, e->neg, routing,
                                        0x80 | (sc & 0x40)),
             "blend constant op");
         break;
      }
      case B_FMA:
         PUT(g5p_usc_half_fma2(raw, e->half, S[0], S[1], S[2], k, sc ? (int)sc : -1), "fma");
         break;
      case B_LERP:
         PUT(g5p_usc_half_lerp(raw, e->half, S[0], S[1], S[2], k, e->neg, sc), "lerp");
         break;
      case B_DOT2:
         PUT(g5p_usc_half_dot2(raw, e->half, S[0], S[1], S[2], S[3], k, e->neg, sc ? (int)sc : -1), "dot2");
         break;
      case B_MINMAX:
         PUT(g5p_usc_half_minmax(raw, e->max, e->pair ? 0x30 : 0, e->half, S[0], S[1], k), "min/max");
         break;
      case B_FIXUP:
         PUT(g5p_usc_half_minmax_fixup(raw, e->half, S[0], S[1], k), "min/max pair");
         break;
      default:
         *err = "blend: unexpected value";
         return 0;
      }
   }
#undef PUT
#ifdef G5P_BLEND_TEST
   g5p_blend_test_program(c, info);
#endif
   return n;
}

unsigned g5p_blend_general(uint8_t *out, unsigned max, const struct g5p_blend_key *k, unsigned const_base,
                           struct g5p_blend_info *info, const char **err)
{
   unsigned n = blend_build(out, max, k, const_base, info, err, false);
   if (!*err)
      return n;
   /* Retain byte identity for every existing measured equation. Other core
    * equations use the same tile read/convert/native-f16 ALU/write protocol.
    */
   return blend_build(out, max, k, const_base, info, err, true);
}
