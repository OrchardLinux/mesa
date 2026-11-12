/* SPDX-License-Identifier: MIT
 * Generated-by: LLM.
 * Aquarium-style pixel oracles: discard + depth/stencil write + blend.
 *
 * WebGL Aquarium's plant/seaweed program samples a texture, discards on
 * `diffuse.a < 0.3` and draws with depth test/write and
 * glBlendFunc(SRC_ALPHA, ONE_MINUS_SRC_ALPHA) (one function for RGB and
 * alpha: a general blend whose zero source keeps the destination). G5P's
 * base program turns those discards into a colour select, which is exact
 * for colour but would let a discarded fragment write depth/stencil; under
 * a depth/stencil write the driver must place the punch-through variant
 * (one hardware kill before the general epilogue, PUNCH_THROUGH pass).
 *
 * Each case draws the discarding blended program, then a second opaque
 * program that only passes where the first one did NOT write depth/stencil:
 *  - discarded pixels must show the second colour (no depth/stencil leak);
 *  - kept pixels must show the exact blend of texel over the clear colour.
 * On r39-6 the first draw refused at record time and killed the command
 * buffer; that is FAIL here (no pixels), as is any refusal or fatal Zink
 * error, checked by run-numeric.py.
 *
 * Run on real G5P; --reference permits llvmpipe for oracle validation.
 * Fake-node zero readback must fail these nonzero image expectations.
 */
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 32
static const char *vs =
   "#version 300 es\nprecision highp float;\nuniform float uZ; out vec2 uv;\n"
   "void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);"
   "uv=p;gl_Position=vec4(p*2.0-1.0,uZ,1);}";
/* Aquarium's shape: sample, discard on low alpha, light the kept colour */
static const char *fs_aquarium =
   "#version 300 es\nprecision mediump float;\nuniform sampler2D diffuse; uniform float uLight;\n"
   "in vec2 uv; out vec4 o;\n"
   "void main(){vec4 d=texture(diffuse,uv);if(d.a<0.3){discard;}o=vec4(d.rgb*uLight,d.a);}";
/* two conditions: the punch variant merges them into one feedback token */
static const char *fs_two =
   "#version 300 es\nprecision mediump float;\nuniform sampler2D diffuse; uniform float uLight;\n"
   "in vec2 uv; out vec4 o;\n"
   "void main(){vec4 d=texture(diffuse,uv);if(d.a<0.3)discard;if(gl_FragCoord.y<8.0)discard;"
   "o=vec4(d.rgb*uLight,d.a);}";
static const char *fs_green =
   "#version 300 es\nprecision mediump float;\nout vec4 o;\nvoid main(){o=vec4(0,1,0,1);}";

static const unsigned char texels[8] = {255, 0, 0, 26, /* alpha 0.102: discarded */
                                        64, 128, 192, 192};
static const double clear_c[4] = {51 / 255.0, 102 / 255.0, 153 / 255.0, 1.0};

static GLuint shader(GLenum type, const char *src)
{
   GLuint sh = glCreateShader(type);
   glShaderSource(sh, 1, &src, NULL);
   glCompileShader(sh);
   GLint ok;
   glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[8192];
      glGetShaderInfoLog(sh, sizeof(log), NULL, log);
      fprintf(stderr, "shader: %s\n", log);
      exit(2);
   }
   return sh;
}
static GLuint program(const char *v, const char *f)
{
   GLuint p = glCreateProgram(), a = shader(GL_VERTEX_SHADER, v), b = shader(GL_FRAGMENT_SHADER, f);
   glAttachShader(p, a);
   glAttachShader(p, b);
   glLinkProgram(p);
   glDeleteShader(a);
   glDeleteShader(b);
   GLint ok;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[8192];
      glGetProgramInfoLog(p, sizeof(log), NULL, log);
      fprintf(stderr, "link: %s\n", log);
      exit(2);
   }
   return p;
}

/* kept: texel 1 blended SRC_ALPHA / ONE_MINUS_SRC_ALPHA over the clear */
static double kept(int c)
{
   double a = texels[7] / 255.0, s = texels[4 + c] / 255.0;
   if (c == 3)
      return a * a + clear_c[3] * (1 - a);
   return s * a + clear_c[c] * (1 - a);
}

static unsigned check(const char *name, int two)
{
   unsigned char px[W * W * 4];
   memset(px, 0, sizeof(px));
   glReadPixels(0, 0, W, W, GL_RGBA, GL_UNSIGNED_BYTE, px);
   GLenum error = glGetError();
   unsigned bad = 0, bad_discarded = 0, bad_kept = 0;
   for (int y = 0; y < W; y++)
      for (int x = 0; x < W; x++) {
         const int discarded = x < W / 2 || (two && y < 8);
         for (int c = 0; c < 4; c++) {
            double v = discarded ? (c == 1 || c == 3 ? 1.0 : 0.0) : kept(c);
            int e = (int)lround(fmin(1, fmax(0, v)) * 255);
            int d = abs((int)px[4 * (y * W + x) + c] - e) > 1;
            bad += d;
            if (discarded)
               bad_discarded += d;
            else
               bad_kept += d;
         }
      }
   const unsigned char *f = px, *l = px + 4 * (W * W - 1);
   printf("WR-NUMERIC %s %s bad_channels=%u discarded_bad=%u kept_bad=%u gl_error=%#x "
          "first=(%u,%u,%u,%u) last=(%u,%u,%u,%u)\n",
          name, (!bad && !error) ? "PASS" : "FAIL", bad, bad_discarded, bad_kept, error, f[0], f[1], f[2], f[3],
          l[0], l[1], l[2], l[3]);
   return bad || error;
}

int main(int argc, char **argv)
{
   const char *names[] = {"aquarium-discard-depth-blend", "aquarium-two-discards-depth-blend",
                          "aquarium-discard-stencil-blend"};
   int selected = -1, reference = 0;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--reference"))
         reference = 1;
      else if (!strcmp(argv[i], "--case") && i + 1 < argc) {
         const char *name = argv[++i];
         for (int j = 0; j < 3; j++)
            if (!strcmp(name, names[j]))
               selected = j;
      } else
         return 2;
   }
   if (selected < 0)
      return 2;
   EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major, minor;
   if (!eglInitialize(d, &major, &minor) || !eglBindAPI(EGL_OPENGL_ES_API))
      return 2;
   EGLint attr[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
   EGLContext ctx = eglCreateContext(d, NULL, EGL_NO_CONTEXT, attr);
   if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx))
      return 2;
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   const char *bank0 = getenv("PVR_G5P_VS_BANK0_FIRST");
   printf("WR-NUMERIC IDENTITY renderer=%s reference=%d bank0_first=%s\n", renderer, reference,
          bank0 ? bank0 : "default");
   if (!reference && (!strstr(renderer, "zink") || !strstr(renderer, "PowerVR")))
      return 2;

   GLuint fb, rb[2];
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glGenRenderbuffers(2, rb);
   glBindRenderbuffer(GL_RENDERBUFFER, rb[0]);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, W, W);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb[0]);
   glBindRenderbuffer(GL_RENDERBUFFER, rb[1]);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, W, W);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rb[1]);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      return 2;
   glViewport(0, 0, W, W);

   GLuint tex;
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_2D, tex);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, texels);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   glClearColor(clear_c[0], clear_c[1], clear_c[2], clear_c[3]);
   glClearDepthf(1.0f);
   glClearStencil(0);
   glDepthMask(GL_TRUE);
   glStencilMask(0xff);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

   const int stencil = selected == 2;
   GLuint p = program(vs, selected == 1 ? fs_two : fs_aquarium);
   glUseProgram(p);
   glUniform1i(glGetUniformLocation(p, "diffuse"), 0);
   glUniform1f(glGetUniformLocation(p, "uLight"), 1.0f);
   glUniform1f(glGetUniformLocation(p, "uZ"), 0.0f); /* window depth 0.5 */
   if (stencil) {
      glDisable(GL_DEPTH_TEST);
      glEnable(GL_STENCIL_TEST);
      glStencilFunc(GL_ALWAYS, 1, 0xff);
      glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
   } else {
      glEnable(GL_DEPTH_TEST);
      glDepthFunc(GL_LESS);
   }
   glEnable(GL_BLEND);
   glBlendEquation(GL_FUNC_ADD);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); /* Aquarium's */
   glDrawArrays(GL_TRIANGLES, 0, 3);

   /* second draw: passes only where the first wrote no depth/stencil */
   glDisable(GL_BLEND);
   GLuint q = program(vs, fs_green);
   glUseProgram(q);
   glUniform1f(glGetUniformLocation(q, "uZ"), 0.5f); /* window depth 0.75 */
   if (stencil) {
      glStencilFunc(GL_EQUAL, 0, 0xff);
      glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
   }
   glDrawArrays(GL_TRIANGLES, 0, 3);
   unsigned failed = check(names[selected], selected == 1);
   glDeleteProgram(p);
   glDeleteProgram(q);
   printf("WR-NUMERIC RESULT %s cases=1 failed=%u reference=%d\n", failed ? "FAIL" : "PASS", failed, reference);
   eglTerminate(d);
   return failed ? 1 : 0;
}
