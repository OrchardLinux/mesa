/* SPDX-License-Identifier: MIT
 * Generated-by: LLM.
 * WebRender ps_quad_blend vertex-stage pixel oracle (r39-7.1).
 *
 * Runs Firefox's own ps_quad_blend vertex shader (the corpus text, argv[1])
 * with a selector fragment program that reads every vertex output, so the
 * linker keeps all of them and the vertex program is the register-pressure
 * shape G5P refused before r39-7.1. 16 instances cover every op-table arm
 * (1 grayscale, 2 hue-rotate, 4 saturate, 5 sepia, 7 colour matrix from the
 * GPU buffer, 10 flood/offset, 11 component transfer, 0 none), quad-edge
 * inflation flags and the clip-to-task path (inverse transform).
 *
 * Thirteen readbacks (one per selector, the last one coverage) of a 32x32 RGBA8 target. Floats are
 * encoded as clamp(v*k + 0.5) with a per-selector k keeping them mid-range;
 * integers as (v & 255)/255. Components the
 * shader leaves undefined for an instance's op (e.g. v_color_mat for op 10/11,
 * v_op_table_address_vec.y unless op 11, v_flags.xy, v_amount.y) are masked.
 *
 *   quad-blend-numeric VS --emit FILE   llvmpipe reference: write expected
 *   quad-blend-numeric VS --check FILE  G5P: compare (+-2 per channel)
 * The expected file is produced once on the host and pinned by SHA256 in the
 * runner manifest; the reference arm re-checks llvmpipe against it.
 */
#include <EGL/egl.h>
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 32
#define SELS 13
#define INST 16

static const char *fs_body =
   "uniform int uSel;\n"
   "flat in vec4 vTransformBounds; flat in vec4 v_color; flat in ivec4 v_flags; in vec2 vLocalPos;\n"
   "flat in vec4 v_uv0_sample_bounds; in vec2 v_uv0; flat in ivec2 v_op_table_address_vec; flat in vec2 v_amount;\n"
   "flat in mat4 v_color_mat; flat in vec4 v_color_offset; flat in vec4 v_funcs;\n"
   "out vec4 o;\n"
   "vec4 f(vec4 v, float k){return clamp(v*k+0.5,0.0,1.0);}\n"
   "void main(){vec4 r;\n"
   " if(uSel==0) r=f(vTransformBounds-16.0,1.0/96.0);\n"
   " else if(uSel==1) r=f(v_color,0.4);\n"
   " else if(uSel==2) r=vec4(v_flags & ivec4(255))/255.0;\n"
   " else if(uSel==3) r=vec4(f(vec4(vLocalPos-16.0,0.0,0.0),1.0/160.0).xy,f(vec4(v_uv0,0.0,0.0),0.4).xy);\n"
   " else if(uSel==4) r=f(v_uv0_sample_bounds,0.4);\n"
   " else if(uSel==5) r=vec4(vec2(v_op_table_address_vec & ivec2(255))/255.0,f(vec4(v_amount,0.0,0.0),0.4).xy);\n"
   " else if(uSel<10) r=f(v_color_mat[uSel-6],0.3);\n"
   " else if(uSel==10) r=f(v_color_offset,0.3);\n"
   " else if(uSel==11) r=f(v_funcs-8.0,1.0/20.0);\n"
   " else r=vec4(1.0);\n"
   " o=r;}\n";

static const int ops[INST] = {1, 2, 4, 5, 7, 10, 11, 0, 2, 7, 1, 5, 4, 10, 11, 7};

static GLuint shader(GLenum type, const char *src)
{
   GLuint sh = glCreateShader(type);
   glShaderSource(sh, 1, &src, NULL);
   glCompileShader(sh);
   GLint ok;
   glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[16384];
      glGetShaderInfoLog(sh, sizeof(log), NULL, log);
      fprintf(stderr, "shader: %s\n", log);
      exit(2);
   }
   return sh;
}

static char *slurp(const char *path)
{
   FILE *f = fopen(path, "rb");
   if (!f)
      exit(2);
   fseek(f, 0, SEEK_END);
   long n = ftell(f);
   fseek(f, 0, SEEK_SET);
   char *s = calloc(1, n + 1);
   if (fread(s, 1, n, f) != (size_t)n)
      exit(2);
   fclose(f);
   return s;
}

/* the instance covering pixel (x, y) in the reference (selector 12 writes
 * 1.0 wherever a quad is rasterized; each instance stays inside its 8x8
 * tile, inflated or shrunk by at most 2 px), or -1 for background */
static int owner(const unsigned char *coverage, int x, int y)
{
   return coverage[(y * W + x) * 4 + 3] ? (y / 8) * 4 + x / 8 : -1;
}

static int defined(int sel, int ch, int inst)
{
   int op = ops[inst];
   int mat = op == 1 || op == 2 || op == 4 || op == 5 || op == 7;
   if (sel == 2)
      return ch >= 2;                      /* v_flags.zw */
   if (sel == 5)
      return ch == 0 || (ch == 1 && op == 11) || ch == 2; /* op.x, op.y (11), amount.x */
   if (sel >= 6 && sel <= 9)
      return mat;
   if (sel == 10)
      return mat || op == 10;
   return 1;
}

int main(int argc, char **argv)
{
   if (argc != 4 || (strcmp(argv[2], "--emit") && strcmp(argv[2], "--check")))
      return 2;
   const int emit = !strcmp(argv[2], "--emit");
   char *vs = slurp(argv[1]);
   const int es = !strncmp(vs, "#version 300 es", 15); /* the pinned GLES corpus; else GL 3.2 (optimized) */
   char fs[8192];
   snprintf(fs, sizeof(fs), "%s%s", es ? "#version 300 es\nprecision highp float;\nprecision highp int;\n" : "#version 150\n", fs_body);
   EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (!eglInitialize(d, &major, &minor) || !eglBindAPI(es ? EGL_OPENGL_ES_API : EGL_OPENGL_API))
      return 2;
   const EGLint attr[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, es ? 0 : 2, EGL_NONE};
   EGLContext ctx = eglCreateContext(d, NULL, EGL_NO_CONTEXT, attr);
   if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx))
      return 2;
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   printf("QB-NUMERIC renderer=%s reference=%d api=%s\n", renderer, emit, es ? "gles3" : "gl32");
   if (!emit && (!strstr(renderer, "zink") || !strstr(renderer, "PowerVR")))
      return 2;

   GLuint p = glCreateProgram(), a = shader(GL_VERTEX_SHADER, vs), b = shader(GL_FRAGMENT_SHADER, fs);
   glAttachShader(p, a);
   glAttachShader(p, b);
   glBindAttribLocation(p, 0, "aPosition");
   glBindAttribLocation(p, 1, "aData");
   glLinkProgram(p);
   GLint ok;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[16384];
      glGetProgramInfoLog(p, sizeof(log), NULL, log);
      fprintf(stderr, "link: %s\n", log);
      return 2;
   }
   glUseProgram(p);

   /* data textures in WebRender's layouts */
   static float gpuf[1024 * 2 * 4];
   static int gpui[1024 * 4];
   float palette[8 * 4] = {0}, tasks[2 * 4] = {0, 0, W, W, 0, 0, 0, 0};
   for (int c = 0; c < 4; c++)
      palette[c * 4 + c] = palette[(4 + c) * 4 + c] = 1.0f; /* identity and inverse */
   for (int i = 0; i < INST; i++) {
      const int tx = i % 4, ty = i / 4, base = 16 + 8 * i, mat = 400 + 8 * i;
      const float x0 = tx * 8 + 2, y0 = ty * 8 + 2, x1 = x0 + 4, y1 = y0 + 4;
      float *f = &gpuf[base * 4];
      const float prim[4] = {x0, y0, x1, y1}, uv[4] = {4 + i, 2, 60 - i, 30};
      const float xf[4] = {i & 1 ? 0.5f : 1.0f, i & 2 ? 2.0f : 1.0f, i & 4 ? 1.0f : 0.0f, i & 8 ? -3.0f : 0.0f};
      const float colour[4] = {0.0625f * i, 1.0f - 0.03125f * i, 0.5f, 0.25f + 0.03f * i};
      memcpy(f + 0, prim, 16);
      memcpy(f + 4, prim, 16);
      memcpy(f + 8, uv, 16);
      memcpy(f + 12, xf, 16);
      memcpy(f + 16, colour, 16);
      for (int k = 0; k < 20; k++) /* op 7 matrix (4 texels) and offset (texel 5) */
         gpuf[mat * 4 + k] = 0.05f * ((k * 7 + i * 3) % 17) - 0.4f;
      const int amount = 4096 * (i + 1); /* 0.0625 .. 1.0 */
      const int w = ops[i] == 7 || ops[i] == 10 ? mat : ops[i] == 11 ? 37 + i : amount;
      gpui[i * 4 + 0] = 0;                                   /* transform 0 */
      gpui[i * 4 + 1] = 0;                                   /* z */
      gpui[i * 4 + 2] = ((0x1234 + 0x111 * i) & 0xffff) << 16 | ops[i];
      gpui[i * 4 + 3] = w;
   }
   GLuint tex[5];
   glGenTextures(5, tex);
   const struct { const char *name; GLenum ifmt, fmt, type; int w, h; const void *data; } t[5] = {
      {"sGpuBufferF", GL_RGBA32F, GL_RGBA, GL_FLOAT, 1024, 2, gpuf},
      {"sGpuBufferI", GL_RGBA32I, GL_RGBA_INTEGER, GL_INT, 1024, 1, gpui},
      {"sTransformPalette", GL_RGBA32F, GL_RGBA, GL_FLOAT, 8, 1, palette},
      {"sRenderTasks", GL_RGBA32F, GL_RGBA, GL_FLOAT, 2, 1, tasks},
      {"sColor0", GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 64, 32, NULL},
   };
   for (int k = 0; k < 5; k++) {
      glActiveTexture(GL_TEXTURE0 + k);
      glBindTexture(GL_TEXTURE_2D, tex[k]);
      glTexImage2D(GL_TEXTURE_2D, 0, t[k].ifmt, t[k].w, t[k].h, 0, t[k].fmt, t[k].type, t[k].data);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glUniform1i(glGetUniformLocation(p, t[k].name), k);
   }
   const float ortho[16] = {2.0f / W, 0, 0, 0, 0, 2.0f / W, 0, 0, 0, 0, 1, 0, -1, -1, 0, 1};
   glUniformMatrix4fv(glGetUniformLocation(p, "uTransform"), 1, GL_FALSE, ortho);

   GLuint vao, vbo[2];
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(2, vbo);
   const float quad[8] = {0, 0, 1, 0, 0, 1, 1, 1};
   glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
   glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(0);
   int data[INST * 4];
   for (int i = 0; i < INST; i++) {
      /* edge flags 1 left, 2 top, 4 right, 8 bottom; never both opposite
       * edges, so a shrink by 2 px keeps the 4 px quad visible and an
       * inflation stays inside its 8x8 tile */
      int e = (i * 5) & 15;
      if (e & 1)
         e &= ~4;
      if (e & 2)
         e &= ~8;
      const int quadflags = i % 5, clip = i % 3 == 0 ? 2 : 0, bit16 = i & 1 ? 16 : 0;
      data[i * 4 + 0] = i;
      data[i * 4 + 1] = 16 + 8 * i;
      data[i * 4 + 2] = (clip | bit16) << 24 | e << 16 | quadflags << 8 | 255;
      data[i * 4 + 3] = 0;
   }
   glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
   glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
   glVertexAttribIPointer(1, 4, GL_INT, 0, 0);
   glVertexAttribDivisor(1, 1);
   glEnableVertexAttribArray(1);

   GLuint fb, rb;
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glGenRenderbuffers(1, &rb);
   glBindRenderbuffer(GL_RENDERBUFFER, rb);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, W, W);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return 2;
   glViewport(0, 0, W, W);

   static unsigned char got[SELS][W * W * 4];
   for (int sel = 0; sel < SELS; sel++) {
      glUniform1i(glGetUniformLocation(p, "uSel"), sel);
      glClearColor(0, 0, 0, 0);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, INST);
      glReadPixels(0, 0, W, W, GL_RGBA, GL_UNSIGNED_BYTE, got[sel]);
   }
   const GLenum error = glGetError();
   if (emit) {
      FILE *f = fopen(argv[3], "wb");
      if (!f || fwrite(got, sizeof(got), 1, f) != 1)
         return 2;
      fclose(f);
      unsigned covered = 0;
      for (int y = 0; y < W; y++)
         for (int x = 0; x < W; x++)
            covered += got[0][(y * W + x) * 4 + 3] != 0;
      printf("QB-NUMERIC EMIT covered_pixels=%u gl_error=%#x reference=1 PASS bad_channels=0\n", covered, error);
      eglTerminate(d);
      return error ? 1 : 0;
   }
   static unsigned char want[SELS][W * W * 4];
   FILE *f = fopen(argv[3], "rb");
   if (!f || fread(want, sizeof(want), 1, f) != 1)
      return 2;
   fclose(f);
   unsigned bad = 0, checked = 0, bad_sel[SELS] = {0};
   for (int sel = 0; sel < SELS; sel++)
      for (int y = 0; y < W; y++)
         for (int x = 0; x < W; x++) {
            const int i = owner(want[SELS - 1], x, y);
            for (int ch = 0; ch < 4; ch++) {
               if (i >= 0 && !defined(sel, ch, i))
                  continue;
               const int o = (y * W + x) * 4 + ch, diff = abs((int)got[sel][o] - (int)want[sel][o]);
               checked++;
               if (diff > 2) {
                  bad++;
                  bad_sel[sel]++;
               }
            }
         }
   printf("QB-NUMERIC per-selector bad:");
   for (int sel = 0; sel < SELS; sel++)
      printf(" %u", bad_sel[sel]);
   printf("\nQB-NUMERIC ps-quad-blend %s bad_channels=%u checked=%u gl_error=%#x reference=0\n",
          bad || error ? "FAIL" : "PASS", bad, checked, error);
   printf("QB-NUMERIC RESULT %s failed=%u\n", bad || error ? "FAIL" : "PASS", bad || error);
   eglTerminate(d);
   return bad || error ? 1 : 0;
}
