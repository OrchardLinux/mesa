/* SPDX-License-Identifier: MIT */
/* ds-ladder: depth/stencil under blending and discard, OpenGL ES 3.0 (meant for Zink on the Vulkan ICD), pixel-exact
 * read-back against a CPU reference. The ds_* rungs of the Apple A9 (G5P) Linux port's blend ladder (Qt Quick's and
 * Hyprland's window shapes); the colour-only blend rungs are not in this copy.
 * One rung per process: EGL surfaceless (EGL_MESA_platform_surfaceless, no window system), an ES 3.0 context.
 *
 *   ds-ladder --list            the rung names, in ladder order
 *   ds-ladder RUNG              run one rung; prints "RUNG <name> PASS|FAIL ..." and exits 0/1
 *   ds-ladder --offline RUNG    the same on a driver that executes nothing: compile and draw only, values not judged
 *
 * Every rung: a 256x128 RGBA8 target with a GL_DEPTH24_STENCIL8 renderbuffer (Zink renders it as D32F_S8 when the
 * device offers no D24S8). Draw 1 (blending off) copies the destination pattern D, partial alpha everywhere (alpha
 * 0..255 across the image). Draw 2 samples the source pattern S 1:1 with the rung's blend state; the left half is
 * drawn with blend constant C1, then glBlendColor(C2) and the right half (scissor) with the same program and blend
 * state. PASS = every pixel within 1 of round(255 * clamp(op(s * Fs, d * Fd))) per channel. On failure: the worst
 * pixels and a per-channel error histogram.
 *
 * Rungs:
 *   ctl              no depth/stencil, blending off (draw 2 replaces): proves the harness
 *   ds_premult, ds_qs_text   Qt Quick's shapes: draw 1 depth ALWAYS + write, draw 2 LEQUAL without write, blended
 *                    ONE/ONE_MINUS_SRC_ALPHA (premultiplied) or CONSTANT_COLOR/ONE_MINUS_SRC_COLOR (Qt's text)
 *   *_load           the same with glFinish between the draws: draw 2's pass LOADs colour, depth and stencil
 *   ds_clip_premult  a colour-mask-0 full-screen stencil write (Qt's stencil clip), draw 2 stencil EQUAL 1
 *   ds_clear_rect    draw 1 writes depth 0.5 everywhere, a scissored glClear of depth to 0.25 inside the clear
 *                    rectangle, then draw 2 LEQUAL at 0.5: rejected inside the rectangle (the pixel keeps draw 1's
 *                    pattern), blended outside
 *   ds_discard*      a discarding program under a depth/stencil test (enum punch_mode below): depth test + write,
 *                    test only, the write order (a discarded fragment must not write depth, so a later draw behind it
 *                    still shows; also across a pass break, and with two discard conditions), stencil REPLACE with
 *                    colour mask 0 under blending (Hyprland's blur mask), with colour, both faces */
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 256
#define H 128

struct rung {
   const char *name;
   GLenum sr, dr, er, sa, da, ea;
   int off;    /* blending off */
   int ds;     /* a D24S8 attachment, Qt Quick's: draw 1 depth ALWAYS + write, draw 2 LEQUAL, no write */
   int load;   /* glFinish between draw 1 and draw 2: draw 2's pass LOADs */
   int clip;   /* (ds) a colour-mask-0 full-screen stencil write (Qt's stencil clip), draw 2 stencil EQUAL 1 */
   int dclear; /* (ds) a scissored depth clear to 0.25 over the clear rectangle after draw 1; draw 2 LEQUAL at 0.5
                * is rejected there */
   int punch;  /* a discarding program under a depth/stencil test (enum punch_mode, run_punch()) */
};
/* Discard under depth/stencil, all in a D24S8 target. The discarding program draws the source pattern S and discards
 * where S.a < 0.5 (Hyprland's discardAlpha; P_ORDER2: also where S.r > 0.75). Depths in window space: the destination
 * draw 0.5 everywhere and 0.1 in the left quarter (x < 64), the discarding draw 0.25, the probe draw 0.4. "kept" = a
 * pixel the discarding program does not discard. */
enum punch_mode {
   P_DEPTH = 1,      /* ds_discard_depth: LESS + write. x < 64: D (depth fails); kept: S; discarded: D */
   P_DEPTH_TEST,     /* ds_discard_depth_test: LESS, no write: the same pixels */
   P_ORDER,          /* ds_discard_depth_write_order: P_DEPTH, then the CLR probe at 0.4 LESS: discarded pixels kept
                      * depth 0.5, so the probe shows there (CLR); kept pixels have 0.25 (S stays); x < 64: D */
   P_ORDER_LOAD,     /* ds_discard_depth_write_order_load: the same with glFinish before the probe (depth stored and
                      * loaded between passes) */
   P_ORDER2,         /* ds_discard2_depth_write_order: P_ORDER with the two-condition program */
   P_STENCIL,        /* ds_discard_stencil: Hyprland's blur mask draw: colour mask 0 with premultiplied blending on,
                      * stencil ALWAYS 1 REPLACE, no depth test; then CLR under stencil EQUAL 1: kept CLR, discarded D */
   P_STENCIL_COLOUR, /* ds_discard_stencil_colour: the same draw writing colour (blending off), then CLR under
                      * stencil NOTEQUAL 1: kept S, discarded CLR */
   P_FACES,          /* ds_discard_stencil_faces: colour mask 0, the left half front-facing (ref 1), the right half
                      * back-facing (ref 2), both REPLACE; then CLR under EQUAL 1 and CLR2 under EQUAL 2: kept left CLR,
                      * kept right CLR2, discarded D */
   P_BLEND_STENCIL,  /* ds_discard_blend_stencil: the titlebars plugin's shape: premultiplied blending, colour on, stencil
                      * ALWAYS 1 REPLACE; then CLR under stencil NOTEQUAL 1: kept S over D, discarded CLR (a discarded
                      * fragment that wrote stencil would hide the probe) */
   P_BLEND_ORDER,    /* ds_discard_blend_depth_write_order: premultiplied blending + LESS + write, then the CLR probe at
                      * 0.4: kept S over D, discarded CLR, x < 64 D */
};
#define QT(n, s, d, ...) {n, s, d, GL_FUNC_ADD, s, d, GL_FUNC_ADD, 0, __VA_ARGS__}
#define PD(n, m) {n, GL_ONE, GL_ZERO, GL_FUNC_ADD, GL_ONE, GL_ZERO, GL_FUNC_ADD, 1, .ds = 1, .punch = m}

static const struct rung rungs[] = {
   {"ctl", GL_ONE, GL_ZERO, GL_FUNC_ADD, GL_ONE, GL_ZERO, GL_FUNC_ADD, 1},
   QT("ds_premult", GL_ONE, GL_ONE_MINUS_SRC_ALPHA, .ds = 1),
   QT("ds_premult_load", GL_ONE, GL_ONE_MINUS_SRC_ALPHA, .ds = 1, .load = 1),
   QT("ds_qs_text", GL_CONSTANT_COLOR, GL_ONE_MINUS_SRC_COLOR, .ds = 1),
   QT("ds_qs_text_load", GL_CONSTANT_COLOR, GL_ONE_MINUS_SRC_COLOR, .ds = 1, .load = 1),
   QT("ds_clip_premult", GL_ONE, GL_ONE_MINUS_SRC_ALPHA, .ds = 1, .clip = 1),
   QT("ds_clear_rect", GL_ONE, GL_ONE_MINUS_SRC_ALPHA, .ds = 1, .dclear = 1),
   PD("ds_discard_depth", P_DEPTH),
   PD("ds_discard_depth_test", P_DEPTH_TEST),
   PD("ds_discard_depth_write_order", P_ORDER),
   PD("ds_discard_depth_write_order_load", P_ORDER_LOAD),
   PD("ds_discard2_depth_write_order", P_ORDER2),
   PD("ds_discard_stencil", P_STENCIL),
   PD("ds_discard_stencil_colour", P_STENCIL_COLOUR),
   PD("ds_discard_stencil_faces", P_FACES),
   PD("ds_discard_blend_stencil", P_BLEND_STENCIL),
   PD("ds_discard_blend_depth_write_order", P_BLEND_ORDER),
};
#define NRUNGS (sizeof(rungs) / sizeof(rungs[0]))

static const float C1[4] = {0.25f, 0.75f, 0.5f, 0.375f}, C2[4] = {0.9f, 0.1f, 0.6f, 0.8f};
static const unsigned char CLR[4] = {51, 102, 153, 204}; /* the probe: 0.2 0.4 0.6 0.8 */
static const unsigned char CLR2[4] = {204, 51, 0, 255};  /* the second probe: 0.8 0.2 0 1 */
static int in_clear_rect(int x, int y) { return x >= 40 && x < 200 && y >= 24 && y < 100; }

static unsigned char dpat(int x, int y, int c) { return (unsigned char)(x * 7 + y * 13 + c * 50 + 3); }
static unsigned char spat(int x, int y, int c) { return (unsigned char)(x * 11 + y * 5 + c * 90 + 30); }

static float factor(GLenum f, int c, const float *s, const float *d, const float *k)
{
   int a = 3;
   switch (f) {
   case GL_ZERO: return 0;
   case GL_ONE: return 1;
   case GL_SRC_COLOR: return s[c];
   case GL_ONE_MINUS_SRC_COLOR: return 1 - s[c];
   case GL_DST_COLOR: return d[c];
   case GL_ONE_MINUS_DST_COLOR: return 1 - d[c];
   case GL_SRC_ALPHA: return s[a];
   case GL_ONE_MINUS_SRC_ALPHA: return 1 - s[a];
   case GL_DST_ALPHA: return d[a];
   case GL_ONE_MINUS_DST_ALPHA: return 1 - d[a];
   case GL_CONSTANT_COLOR: return k[c];
   case GL_ONE_MINUS_CONSTANT_COLOR: return 1 - k[c];
   case GL_CONSTANT_ALPHA: return k[a];
   case GL_ONE_MINUS_CONSTANT_ALPHA: return 1 - k[a];
   case GL_SRC_ALPHA_SATURATE: return c == 3 ? 1 : fminf(s[a], 1 - d[a]);
   default: abort();
   }
}

static unsigned expect(const struct rung *r, int c, const float *s, const float *d, const float *k)
{
   GLenum sf = c < 3 ? r->sr : r->sa, df = c < 3 ? r->dr : r->da, e = c < 3 ? r->er : r->ea;
   float v;
   if (r->off)
      v = s[c];
   else if (e == GL_MIN)
      v = fminf(s[c], d[c]);
   else if (e == GL_MAX)
      v = fmaxf(s[c], d[c]);
   else {
      float ts = s[c] * factor(sf, c, s, d, k), td = d[c] * factor(df, c, s, d, k);
      v = e == GL_FUNC_ADD ? ts + td : e == GL_FUNC_SUBTRACT ? ts - td : td - ts;
   }
   v = fminf(fmaxf(v, 0), 1);
   return (unsigned)lrintf(v * 255.0f);
}

/* body: the main() of the fragment program (uv, w, t declared); NULL = texelFetch of t at the pixel */
static GLuint shader_program(const char *body)
{
   char fsbuf[1024];
   const char *vs = "#version 300 es\nlayout(location=0) in vec4 p;\nvoid main() { gl_Position = p; }\n";
   const char *fs = "#version 300 es\nprecision highp float;\nuniform sampler2D t;\nout vec4 o;\n"
                    "void main() { o = texelFetch(t, ivec2(gl_FragCoord.xy), 0); }\n";
   if (body) {
      snprintf(fsbuf, sizeof(fsbuf), "#version 300 es\nprecision highp float;\nuniform sampler2D t;\nout vec4 o;\n"
               "void main() {\n   vec2 uv = gl_FragCoord.xy / vec2(256.0, 128.0);\n   %s\n}\n", body);
      fs = fsbuf;
   }
   GLuint p = glCreateProgram();
   const char *src[2] = {vs, fs};
   GLenum type[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
   for (int i = 0; i < 2; i++) {
      GLuint s = glCreateShader(type[i]);
      glShaderSource(s, 1, &src[i], NULL);
      glCompileShader(s);
      GLint ok;
      glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
      if (!ok) {
         char log[512];
         glGetShaderInfoLog(s, sizeof(log), NULL, log);
         printf("shader: %s\n", log);
      }
      glAttachShader(p, s);
   }
   glBindAttribLocation(p, 0, "p");
   glLinkProgram(p);
   return p;
}

static GLuint pattern_tex(int dst)
{
   unsigned char *img = malloc(W * H * 4);
   for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++)
         for (int c = 0; c < 4; c++)
            img[(y * W + x) * 4 + c] = dst ? dpat(x, y, c) : spat(x, y, c);
   GLuint t;
   glGenTextures(1, &t);
   glBindTexture(GL_TEXTURE_2D, t);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, img);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   free(img);
   return t;
}

static void draw(GLuint prog, GLuint tex)
{
   glUseProgram(prog);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, tex);
   glUniform1i(glGetUniformLocation(prog, "t"), 0);
   glDrawArrays(GL_TRIANGLES, 0, 3);
}

static GLuint target_fb(int ds)
{
   GLuint rt, fb;
   glGenTextures(1, &rt);
   glBindTexture(GL_TEXTURE_2D, rt);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt, 0);
   if (ds) {
      GLuint rb;
      glGenRenderbuffers(1, &rb);
      glBindRenderbuffer(GL_RENDERBUFFER, rb);
      glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, W, H);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rb);
   }
   GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
   if (st != GL_FRAMEBUFFER_COMPLETE)
      printf("framebuffer status 0x%x (want 0x%x)\n", st, GL_FRAMEBUFFER_COMPLETE);
   return fb;
}

static void report(const struct rung *r, int offline, unsigned bad, unsigned maxerr, GLenum err, int nkept,
                   unsigned hist[4][4])
{
   int ok = !bad && !err;
   printf("RUNG %s %s bad_pixels=%u max_error=%u glError=0x%x", r->name, offline ? "OFFLINE" : ok ? "PASS" : "FAIL",
          bad, maxerr, err);
   if (nkept >= 0)
      printf(" kept=%d/%d", nkept, W * H);
   printf(" err_hist(0/1/2/3+) r=%u/%u/%u/%u g=%u/%u/%u/%u b=%u/%u/%u/%u a=%u/%u/%u/%u\n", hist[0][0], hist[0][1],
          hist[0][2], hist[0][3], hist[1][0], hist[1][1], hist[1][2], hist[1][3], hist[2][0], hist[2][1], hist[2][2],
          hist[2][3], hist[3][0], hist[3][1], hist[3][2], hist[3][3]);
}

/* ---- discard under depth/stencil (enum punch_mode) ---- */
static int pd_kept(int mode, int x, int y)
{
   return spat(x, y, 3) >= 128 && (mode != P_ORDER2 || spat(x, y, 0) <= 191);
}

/* A full-target triangle at window depth z (NDC 2z - 1), or (quad >= 0) the left (0) / right (1) half as two
 * triangles, counter-clockwise (front-facing) unless cw. */
static void pd_draw(GLuint prog, GLuint tex, float z, int quad, int cw)
{
   float nz = 2.0f * z - 1.0f, v[24];
   int n;
   if (quad < 0) {
      const float t[] = {-1, -1, 3, -1, -1, 3};
      for (int i = 0; i < 3; i++)
         v[4 * i] = t[2 * i], v[4 * i + 1] = t[2 * i + 1], v[4 * i + 2] = nz, v[4 * i + 3] = 1;
      n = 3;
   } else {
      float x0 = quad ? 0.0f : -1.0f, x1 = quad ? 1.0f : 0.0f;
      const float ccw[12] = {x0, -1, x1, -1, x1, 1, x0, -1, x1, 1, x0, 1};
      for (int i = 0; i < 6; i++) {
         int k = cw ? (i / 3) * 3 + (2 - i % 3) : i; /* reversed winding per triangle */
         v[4 * i] = ccw[2 * k], v[4 * i + 1] = ccw[2 * k + 1], v[4 * i + 2] = nz, v[4 * i + 3] = 1;
      }
      n = 6;
   }
   glBufferData(GL_ARRAY_BUFFER, n * 16, v, GL_STREAM_DRAW);
   glUseProgram(prog);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, tex);
   glUniform1i(glGetUniformLocation(prog, "t"), 0);
   glDrawArrays(GL_TRIANGLES, 0, n);
}

static int run_punch(const struct rung *r, int offline)
{
   const int m = r->punch;
   GLuint prog = shader_program(NULL), dt = pattern_tex(1), st = pattern_tex(0), vb;
   GLuint pd = shader_program(m == P_ORDER2 ? "vec4 s = texelFetch(t, ivec2(gl_FragCoord.xy), 0);\n"
                                              "   if (s.a < 0.5) discard;\n   if (s.r > 0.75) discard;\n   o = s;"
                                            : "vec4 s = texelFetch(t, ivec2(gl_FragCoord.xy), 0);\n"
                                              "   if (s.a < 0.5) discard;\n   o = s;");
   GLuint clr = shader_program("o = vec4(0.2, 0.4, 0.6, 0.8);"), clr2 = shader_program("o = vec4(0.8, 0.2, 0.0, 1.0);");
   target_fb(1);
   glGenBuffers(1, &vb);
   glBindBuffer(GL_ARRAY_BUFFER, vb);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glViewport(0, 0, W, H);
   glClearColor(0, 0, 0, 0);
   glClearDepthf(1.0f);
   glClearStencil(0);
   glDepthMask(GL_TRUE);
   glStencilMask(0xff);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
   /* the destination: D at depth 0.5, 0.1 in the left quarter */
   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_ALWAYS);
   pd_draw(prog, dt, 0.5f, -1, 0);
   glEnable(GL_SCISSOR_TEST);
   glScissor(0, 0, W / 4, H);
   pd_draw(prog, dt, 0.1f, -1, 0);
   glDisable(GL_SCISSOR_TEST);
   const int depth = m == P_DEPTH || m == P_DEPTH_TEST || m == P_ORDER || m == P_ORDER_LOAD || m == P_ORDER2 ||
                     m == P_BLEND_ORDER;
   if (depth) {
      glDepthFunc(GL_LESS);
      glDepthMask(m == P_DEPTH_TEST ? GL_FALSE : GL_TRUE);
      if (m == P_BLEND_ORDER) {
         glEnable(GL_BLEND);
         glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
      }
      pd_draw(pd, st, 0.25f, -1, 0);
      glDisable(GL_BLEND);
      glDepthMask(GL_FALSE);
      if (m == P_ORDER_LOAD)
         glFinish(); /* the probe's pass LOADs colour and depth */
      if (m == P_ORDER || m == P_ORDER_LOAD || m == P_ORDER2 || m == P_BLEND_ORDER)
         pd_draw(clr, st, 0.4f, -1, 0);
   } else {
      glDisable(GL_DEPTH_TEST);
      glEnable(GL_STENCIL_TEST);
      if (m == P_FACES) {
         glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
         glStencilFuncSeparate(GL_FRONT, GL_ALWAYS, 1, 0xff);
         glStencilFuncSeparate(GL_BACK, GL_ALWAYS, 2, 0xff);
         glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
         pd_draw(pd, st, 0.25f, 0, 0);
         pd_draw(pd, st, 0.25f, 1, 1);
         glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
         glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
         glStencilFunc(GL_EQUAL, 1, 0xff);
         pd_draw(clr, st, 0.4f, -1, 0);
         glStencilFunc(GL_EQUAL, 2, 0xff);
         pd_draw(clr2, st, 0.4f, -1, 0);
      } else {
         glStencilFunc(GL_ALWAYS, 1, 0xff);
         glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
         if (m == P_STENCIL) /* Hyprland: glColorMask off while GL_BLEND stays on */
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
         if (m == P_STENCIL || m == P_BLEND_STENCIL) {
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
         }
         pd_draw(pd, st, 0.25f, -1, 0);
         glDisable(GL_BLEND);
         glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
         glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
         glStencilFunc(m == P_STENCIL ? GL_EQUAL : GL_NOTEQUAL, 1, 0xff);
         pd_draw(clr, st, 0.4f, -1, 0);
      }
      glDisable(GL_STENCIL_TEST);
   }
   glDisable(GL_DEPTH_TEST);
   unsigned char *px = malloc(W * H * 4);
   glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
   GLenum err = glGetError();
   unsigned bad = 0, maxerr = 0, hist[4][4] = {{0}};
   int nkept = 0;
   for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
         int kept = pd_kept(m, x, y);
         nkept += kept;
         unsigned w[4];
         for (int c = 0; c < 4; c++) {
            unsigned d = dpat(x, y, c), s = spat(x, y, c);
            /* premultiplied source-over: s + d (1 - s.a) */
            unsigned over = (unsigned)lrintf(fminf(s / 255.0f + d / 255.0f * (1 - spat(x, y, 3) / 255.0f), 1) * 255.0f);
            switch (m) {
            case P_DEPTH: case P_DEPTH_TEST: w[c] = x < W / 4 || !kept ? d : s; break;
            case P_ORDER: case P_ORDER_LOAD: case P_ORDER2: w[c] = x < W / 4 ? d : kept ? s : CLR[c]; break;
            case P_STENCIL: w[c] = kept ? CLR[c] : d; break;
            case P_STENCIL_COLOUR: w[c] = kept ? s : CLR[c]; break;
            case P_BLEND_STENCIL: w[c] = kept ? over : CLR[c]; break;
            case P_BLEND_ORDER: w[c] = x < W / 4 ? d : kept ? over : CLR[c]; break;
            default: w[c] = !kept ? d : x < W / 2 ? CLR[c] : CLR2[c]; break; /* P_FACES */
            }
         }
         int pbad = 0;
         for (int c = 0; c < 4; c++) {
            unsigned g = px[(y * W + x) * 4 + c], e = g > w[c] ? g - w[c] : w[c] - g;
            hist[c][e > 3 ? 3 : e]++;
            if (e > maxerr)
               maxerr = e;
            if (e > 1)
               pbad = 1;
         }
         if (pbad && bad++ < 6 && !offline)
            printf("  (%d,%d) got %02x%02x%02x%02x want %02x%02x%02x%02x kept=%d\n", x, y, px[(y * W + x) * 4],
                   px[(y * W + x) * 4 + 1], px[(y * W + x) * 4 + 2], px[(y * W + x) * 4 + 3], w[0], w[1], w[2], w[3],
                   kept);
      }
   free(px);
   report(r, offline, bad, maxerr, err, nkept, hist);
   return offline ? err != 0 : !(!bad && !err);
}

static int run(const struct rung *r, int offline)
{
   if (r->punch)
      return run_punch(r, offline);
   GLuint prog = shader_program(NULL), dt = pattern_tex(1), st = pattern_tex(0), vb;
   target_fb(r->ds);
   static const float tri[] = {-1, -1, 0, 1, 3, -1, 0, 1, -1, 3, 0, 1};
   glGenBuffers(1, &vb);
   glBindBuffer(GL_ARRAY_BUFFER, vb);
   glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   glViewport(0, 0, W, H);
   glDisable(GL_BLEND);
   glDisable(GL_SCISSOR_TEST);
   if (r->ds) { /* Qt: clear colour, depth and stencil, then the opaque pass with depth write */
      glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
      glClearDepthf(1.0f);
      glClearStencil(0);
      glDepthMask(GL_TRUE);
      glStencilMask(0xff);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
      glEnable(GL_DEPTH_TEST);
      glDepthFunc(GL_ALWAYS);
   }
   draw(prog, dt); /* the destination pattern */
   if (r->dclear) { /* depth 0.25 inside the clear rectangle; draw 1 wrote 0.5 */
      glEnable(GL_SCISSOR_TEST);
      glScissor(40, 24, 160, 76);
      glClearDepthf(0.25f);
      glClear(GL_DEPTH_BUFFER_BIT);
      glDisable(GL_SCISSOR_TEST);
   }
   if (r->load)
      glFinish(); /* the pass ends; draw 2's pass must LOAD colour, depth and stencil */
   if (r->ds) { /* Qt's alpha pass: depth test, no depth write */
      glDepthFunc(GL_LEQUAL);
      glDepthMask(GL_FALSE);
   }
   if (r->clip) { /* Qt's stencil clip: colour mask 0, stencil REPLACE 1, full screen */
      glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
      glEnable(GL_STENCIL_TEST);
      glStencilFunc(GL_ALWAYS, 1, 0xff);
      glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
      draw(prog, st);
      glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
      glStencilFunc(GL_EQUAL, 1, 0xff);
      glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
   }
   if (!r->off) {
      glEnable(GL_BLEND);
      glBlendFuncSeparate(r->sr, r->dr, r->sa, r->da);
      glBlendEquationSeparate(r->er, r->ea);
   }
   glEnable(GL_SCISSOR_TEST);
   glScissor(0, 0, W / 2, H);
   glBlendColor(C1[0], C1[1], C1[2], C1[3]);
   draw(prog, st);
   glScissor(W / 2, 0, W / 2, H);
   glBlendColor(C2[0], C2[1], C2[2], C2[3]);
   draw(prog, st);
   glDisable(GL_SCISSOR_TEST);
   glDisable(GL_BLEND);
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_STENCIL_TEST);
   unsigned char *px = malloc(W * H * 4);
   glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
   GLenum err = glGetError();
   unsigned bad = 0, maxerr = 0, hist[4][4] = {{0}};
   for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
         float s[4], d[4];
         for (int c = 0; c < 4; c++) {
            s[c] = spat(x, y, c) / 255.0f;
            d[c] = dpat(x, y, c) / 255.0f;
         }
         const float *k = x < W / 2 ? C1 : C2;
         unsigned w[4];
         int pbad = 0;
         for (int c = 0; c < 4; c++) {
            w[c] = r->dclear && in_clear_rect(x, y) ? dpat(x, y, c) : expect(r, c, s, d, k);
            unsigned g = px[(y * W + x) * 4 + c], e = g > w[c] ? g - w[c] : w[c] - g;
            hist[c][e > 3 ? 3 : e]++;
            if (e > maxerr)
               maxerr = e;
            if (e > 1)
               pbad = 1;
         }
         if (pbad && bad++ < 6 && !offline)
            printf("  (%d,%d) got %02x%02x%02x%02x want %02x%02x%02x%02x  s %02x%02x%02x%02x d %02x%02x%02x%02x\n", x, y,
                   px[(y * W + x) * 4], px[(y * W + x) * 4 + 1], px[(y * W + x) * 4 + 2], px[(y * W + x) * 4 + 3],
                   w[0], w[1], w[2], w[3], spat(x, y, 0), spat(x, y, 1), spat(x, y, 2), spat(x, y, 3), dpat(x, y, 0),
                   dpat(x, y, 1), dpat(x, y, 2), dpat(x, y, 3));
      }
   free(px);
   report(r, offline, bad, maxerr, err, -1, hist);
   return offline ? err != 0 : !(!bad && !err);
}

int main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IOLBF, 0);
   int ai = 1, offline = 0;
   if (argc > 1 && !strcmp(argv[1], "--list")) {
      for (unsigned i = 0; i < NRUNGS; i++)
         printf("%s\n", rungs[i].name);
      return 0;
   }
   if (argc > 1 && !strcmp(argv[1], "--offline")) {
      offline = 1;
      ai = 2;
   }
   if (argc != ai + 1) {
      fprintf(stderr, "usage: %s --list | [--offline] RUNG\n", argv[0]);
      return 2;
   }
   const struct rung *r = NULL;
   for (unsigned i = 0; i < NRUNGS; i++)
      if (!strcmp(rungs[i].name, argv[ai]))
         r = &rungs[i];
   if (!r) {
      printf("RUNG %s FAIL unknown rung\n", argv[ai]);
      return 2;
   }
   PFNEGLGETPLATFORMDISPLAYEXTPROC gpd = (void *)eglGetProcAddress("eglGetPlatformDisplayEXT");
   EGLDisplay dpy = gpd ? gpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL) : EGL_NO_DISPLAY;
   EGLint maj, min;
   if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, &maj, &min) || !eglBindAPI(EGL_OPENGL_ES_API)) {
      printf("RUNG %s FAIL no surfaceless EGL display (0x%x)\n", r->name, eglGetError());
      return 1;
   }
   const EGLint attr[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 0, EGL_NONE};
   EGLContext c = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attr);
   if (c == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, c)) {
      printf("RUNG %s FAIL no ES 3.0 context (0x%x)\n", r->name, eglGetError());
      return 1;
   }
   const char *rdr = (const char *)glGetString(GL_RENDERER), *ver = (const char *)glGetString(GL_VERSION);
   printf("GL_RENDERER %s\nGL_VERSION %s\n", rdr ? rdr : "-", ver ? ver : "-");
   int rc = run(r, offline);
   eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(dpy, c);
   eglTerminate(dpy);
   return rc;
}
