/* SPDX-License-Identifier: MIT */
/* es3-smoke: a small OpenGL ES 3.0 smoke (meant for Zink on the Vulkan ICD), checked against fixed expected values.
 * Ported from the Apple A9 (G5P) Linux port, where all three features pass on hardware through Zink.
 * EGL surfaceless (EGL_MESA_platform_surfaceless, no window system, no config), an ES 3.0 context, then three ES 3.0
 * features with exact read-backs:
 *   mrt     four RGBA8 colour attachments written by one draw (glDrawBuffers, layout(location) outputs);
 *   xfb     transform feedback of one vec4 per vertex with rasterizer discard (glMapBufferRange read-back);
 *   zs      a GL_DEPTH24_STENCIL8 texture (Zink: D32F_S8 when D24S8 is not offered; attachment + sampled) written as
 *           the depth attachment (z=.5 triangle over a 1.0 clear), then sampled in place: plain (texture().r) and
 *           with compare LEQUAL against .6 (sampler2DShadow), read back inside and outside the triangle.
 * Shaders are branch-free and their vertex stages only pass attributes through (the A9 driver of the time refused
 * branching fragment programs and vertex arithmetic it had not seen on hardware). With --offline (a driver that
 * executes nothing) pixel and buffer values are reported, not judged: only the context, link, completeness and GL
 * errors are. --only NAME[,NAME] runs only the named features (mrt, xfb, zs). Output: one
 * "SMOKE <name> PASS|FAIL|OFFLINE ..." line per feature, then RESULT. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int offline, fails;
static const char *only;

static int wanted(const char *name)
{
   size_t n = strlen(name);
   for (const char *p = only; p && *p; p = strchr(p, ',') ? strchr(p, ',') + 1 : NULL)
      if (!strncmp(p, name, n) && (p[n] == ',' || !p[n]))
         return 1;
   return !only;
}

static void verdict(const char *name, int ok, int judged, const char *detail)
{
   GLenum e = glGetError();
   if (e)
      ok = 0;
   printf("SMOKE %-4s %s %s glError=0x%x\n", name, !ok ? "FAIL" : judged ? "PASS" : "OFFLINE", detail, e);
   fails += !ok;
}

static GLuint program(const char *vs, const char *fs, const char *xfb)
{
   const char *src[2] = {vs, fs};
   GLuint p = glCreateProgram();
   for (int i = 0; i < 2; i++) {
      GLuint s = glCreateShader(i ? GL_FRAGMENT_SHADER : GL_VERTEX_SHADER);
      glShaderSource(s, 1, &src[i], NULL);
      glCompileShader(s);
      GLint ok = 0;
      glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
      if (!ok) {
         char log[1024] = "";
         glGetShaderInfoLog(s, sizeof(log), NULL, log);
         printf("compile %s: %s\n", i ? "fs" : "vs", log);
      }
      glAttachShader(p, s);
      glDeleteShader(s);
   }
   if (xfb)
      glTransformFeedbackVaryings(p, 1, &xfb, GL_INTERLEAVED_ATTRIBS);
   glLinkProgram(p);
   GLint ok = 0;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[1024] = "";
      glGetProgramInfoLog(p, sizeof(log), NULL, log);
      printf("link: %s\n", log);
      glDeleteProgram(p);
      return 0;
   }
   return p;
}

static GLuint texture(GLenum internal, int w, int h)
{
   GLuint t;
   glGenTextures(1, &t);
   glBindTexture(GL_TEXTURE_2D, t);
   glTexStorage2D(GL_TEXTURE_2D, 1, internal, w, h);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   return t;
}

/* n vertices of 8 floats: a vec4 position at attribute 0 and a vec4 at attribute 1 (uv, or the XFB payload) */
static GLuint vertices(const float *p, int n)
{
   GLuint vao, vbo;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, n * 32, p, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 32, 0);
   glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, (void *)16);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);
   return vao;
}

/* a 64x64 RGBA8 colour target bound as the draw framebuffer (a surfaceless context has no default one) */
static GLuint target(void)
{
   GLuint fb, t = texture(GL_RGBA8, 64, 64);
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);
   return fb;
}

static void pixel(int x, int y, unsigned char px[4])
{
   memset(px, 0, 4);
   glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
}

/* a triangle over the whole target; uv = pos * .5 + .5 per vertex, computed here */
static const float full[3 * 8] = {-1, -1, 0, 1, 0, 0, 0, 0, 3, -1, 0, 1, 2, 0, 0, 0, -1, 3, 0, 1, 0, 2, 0, 0};
static const char *vs_pos = "#version 300 es\nlayout(location=0) in vec4 p;\nlayout(location=1) in vec4 t;\n"
                            "out vec2 uv;\nvoid main(){ uv = t.xy; gl_Position = p; }\n";

static void smoke_mrt(void)
{
   /* UNORM8 = the float x 255 rounded to nearest: .25 -> 63.75 -> 0x40, .75 -> 191.25 -> 0xbf; .5 -> 127.5 is a tie,
    * so target 0's green may be 0x7f or 0x80 (GL and Vulkan leave the tie to the implementation) */
   static const unsigned char want[4][4] = {{0x40, 0x80, 0xbf, 0xff}, {0xff, 0x00, 0x00, 0xff},
                                             {0x00, 0xff, 0x00, 0xff}, {0x00, 0x00, 0xff, 0xff}};
   GLuint tex[4], fb;
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   const GLenum bufs[4] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
   for (int i = 0; i < 4; i++) {
      tex[i] = texture(GL_RGBA8, 64, 64);
      glFramebufferTexture2D(GL_FRAMEBUFFER, bufs[i], GL_TEXTURE_2D, tex[i], 0);
   }
   glDrawBuffers(4, bufs);
   GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
   GLuint p = program(vs_pos, "#version 300 es\nprecision mediump float;\nin vec2 uv;\n"
                              "layout(location=0) out vec4 o0;\nlayout(location=1) out vec4 o1;\n"
                              "layout(location=2) out vec4 o2;\nlayout(location=3) out vec4 o3;\n"
                              "void main(){ o0 = vec4(0.25, 0.5, 0.75, 1.0); o1 = vec4(1.0, 0.0, 0.0, 1.0);\n"
                              "  o2 = vec4(0.0, 1.0, 0.0, 1.0); o3 = vec4(0.0, 0.0, 1.0, 1.0); }\n", NULL);
   glViewport(0, 0, 64, 64);
   glClearColor(0, 0, 0, 0);
   glClear(GL_COLOR_BUFFER_BIT);
   glUseProgram(p);
   GLuint vao = vertices(full, 3);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   char d[64] = ""; /* four " tN=rrggbbaa" */
   int ok = st == GL_FRAMEBUFFER_COMPLETE && p, n = 0;
   for (int i = 0; i < 4; i++) {
      unsigned char px[4];
      glReadBuffer(bufs[i]);
      pixel(32, 32, px);
      n += snprintf(d + n, sizeof(d) - n, " t%d=%02x%02x%02x%02x", i, px[0], px[1], px[2], px[3]);
      if (!offline && memcmp(px, want[i], 4) && !(i == 0 && px[1] == 0x7f && !memcmp(px + 2, want[0] + 2, 2) &&
                                                   px[0] == want[0][0]))
         ok = 0;
   }
   char det[192];
   snprintf(det, sizeof(det), "status=0x%x program=%u%s (want 40{80|7f}bfff ff0000ff 00ff00ff 0000ffff)", st, p, d);
   verdict("mrt", ok, !offline, det);
   glDeleteVertexArrays(1, &vao);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void smoke_xfb(void)
{
   /* the captured vec4 is attribute 1: 0.5, 1.5, ... 11.5 */
   static const float tri[3 * 8] = {-0.5f, -0.5f, 0, 1, 0.5f, 1.5f, 2.5f, 3.5f, 0.5f, -0.5f, 0, 1, 4.5f, 5.5f, 6.5f, 7.5f,
                                    0, 0.5f, 0, 1, 8.5f, 9.5f, 10.5f, 11.5f};
   GLuint p = program("#version 300 es\nlayout(location=0) in vec4 p;\nlayout(location=1) in vec4 a;\nout vec4 v;\n"
                      "void main(){ v = a; gl_Position = p; }\n",
                      "#version 300 es\nprecision mediump float;\nout vec4 o;\nvoid main(){ o = vec4(1.0); }\n", "v");
   GLuint fb = target();
   glViewport(0, 0, 64, 64);
   GLuint vao = vertices(tri, 3), buf;
   glGenBuffers(1, &buf);
   glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
   static const float poison[12] = {-7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7};
   glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof(poison), poison, GL_STREAM_READ);
   glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf);
   glUseProgram(p);
   glEnable(GL_RASTERIZER_DISCARD);
   glBeginTransformFeedback(GL_TRIANGLES);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glEndTransformFeedback();
   glDisable(GL_RASTERIZER_DISCARD);
   float got[12] = {0};
   const float *m = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(got), GL_MAP_READ_BIT);
   if (m) {
      memcpy(got, m, sizeof(got));
      glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
   }
   int ok = p && m, exact = 1;
   for (int i = 0; i < 12; i++)
      exact &= got[i] == 0.5f + i;
   if (!offline)
      ok &= exact;
   char det[320];
   snprintf(det, sizeof(det), "program=%u mapped=%d v0=(%g,%g,%g,%g) v2=(%g,%g,%g,%g) exact=%d "
            "(want v0=(0.5,1.5,2.5,3.5) v2=(8.5,9.5,10.5,11.5))", p, m != NULL, got[0], got[1], got[2], got[3],
            got[8], got[9], got[10], got[11], exact);
   verdict("xfb", ok, !offline, det);
   glDeleteVertexArrays(1, &vao);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDeleteFramebuffers(1, &fb);
}

static void smoke_zs(void)
{
   /* pass 1: colour + D24S8 texture as the depth/stencil attachment, z=.5 triangle over a 1.0 clear */
   static const float tri[3 * 8] = {-0.75f, -0.75f, 0, 1, 0, 0, 0, 0, 0.75f, -0.75f, 0, 1, 0, 0, 0, 0,
                                    0, 0.75f, 0, 1, 0, 0, 0, 0};
   GLuint ds = texture(GL_DEPTH24_STENCIL8, 64, 64), col = texture(GL_RGBA8, 64, 64), out = texture(GL_RGBA8, 64, 64);
   GLuint fb1, fb2;
   glGenFramebuffers(1, &fb1);
   glBindFramebuffer(GL_FRAMEBUFFER, fb1);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, col, 0);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, ds, 0);
   GLenum st1 = glCheckFramebufferStatus(GL_FRAMEBUFFER);
   GLuint p1 = program(vs_pos, "#version 300 es\nprecision mediump float;\nout vec4 o;\n"
                               "void main(){ o = vec4(1.0, 0.0, 0.0, 1.0); }\n", NULL);
   glViewport(0, 0, 64, 64);
   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_LESS);
   glClearColor(0, 0, 0, 1);
   glClearDepthf(1.0f);
   glClearStencil(0);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
   glUseProgram(p1);
   GLuint vao = vertices(tri, 3);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glDisable(GL_DEPTH_TEST);

   /* pass 2: sample the depth texture in place, plain and with compare */
   glGenFramebuffers(1, &fb2);
   glBindFramebuffer(GL_FRAMEBUFFER, fb2);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
   GLenum st2 = glCheckFramebufferStatus(GL_FRAMEBUFFER);
   GLuint plain = program(vs_pos, "#version 300 es\nprecision highp float;\nin vec2 uv;\nout vec4 o;\n"
                                  "uniform highp sampler2D d;\nvoid main(){ o = vec4(texture(d, uv).r); }\n", NULL);
   GLuint shadow = program(vs_pos, "#version 300 es\nprecision highp float;\nin vec2 uv;\nout vec4 o;\n"
                                   "uniform highp sampler2DShadow d;\n"
                                   "void main(){ o = vec4(texture(d, vec3(uv, 0.6))); }\n", NULL);
   GLuint fvao = vertices(full, 3);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, ds);
   unsigned char pin[2][4], pout[2][4];
   for (int k = 0; k < 2; k++) {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, k ? GL_COMPARE_REF_TO_TEXTURE : GL_NONE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
      glUseProgram(k ? shadow : plain);
      glUniform1i(glGetUniformLocation(k ? shadow : plain, "d"), 0);
      glClearColor(0, 0, 1, 1);
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      pixel(32, 24, pin[k]);  /* inside the triangle: depth .5 */
      pixel(4, 60, pout[k]);  /* outside: the 1.0 clear */
   }
   /* plain: .5 -> 0x7f/0x80, 1.0 -> 0xff; compare LEQUAL .6: inside .6 <= .5 false -> 0, outside -> 0xff */
   int ok = st1 == GL_FRAMEBUFFER_COMPLETE && st2 == GL_FRAMEBUFFER_COMPLETE && p1 && plain && shadow;
   if (!offline)
      ok &= (pin[0][0] == 0x7f || pin[0][0] == 0x80) && pout[0][0] == 0xff && pin[1][0] == 0 && pout[1][0] == 0xff;
   char det[320];
   snprintf(det, sizeof(det), "status=0x%x/0x%x plain in=%02x out=%02x shadow in=%02x out=%02x "
            "(want plain 7f|80/ff, shadow 00/ff)", st1, st2, pin[0][0], pout[0][0], pin[1][0], pout[1][0]);
   verdict("zs", ok, !offline, det);
   glDeleteVertexArrays(1, &vao);
   glDeleteVertexArrays(1, &fvao);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

int main(int argc, char **argv)
{
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--offline"))
         offline = 1;
      else if (!strcmp(argv[i], "--only") && i + 1 < argc)
         only = argv[++i];
      else {
         fprintf(stderr, "usage: %s [--offline] [--only mrt,xfb,zs]\n", argv[0]);
         return 2;
      }
   }
   setvbuf(stdout, NULL, _IOLBF, 0); /* SMOKE lines between the driver's stderr lines */
   PFNEGLGETPLATFORMDISPLAYEXTPROC gpd = (void *)eglGetProcAddress("eglGetPlatformDisplayEXT");
   EGLDisplay dpy = gpd ? gpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL) : EGL_NO_DISPLAY;
   EGLint maj, min;
   if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, &maj, &min) || !eglBindAPI(EGL_OPENGL_ES_API)) {
      printf("RESULT FAIL no surfaceless EGL display (0x%x)\n", eglGetError());
      return 1;
   }
   const EGLint attr[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 0, EGL_NONE};
   EGLContext c = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attr);
   if (c == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, c)) {
      printf("RESULT FAIL no ES 3.0 context (0x%x)\n", eglGetError());
      return 1;
   }
   const char *ver = (const char *)glGetString(GL_VERSION), *sl = (const char *)glGetString(GL_SHADING_LANGUAGE_VERSION);
   int ctx_ok = ver && !strncmp(ver, "OpenGL ES 3.", 12) && sl && strstr(sl, "GLSL ES 3.");
   const char *rdr = (const char *)glGetString(GL_RENDERER);
   printf("GL_VERSION %s\nGL_SHADING_LANGUAGE_VERSION %s\nGL_RENDERER %s\n", ver ? ver : "-", sl ? sl : "-",
          rdr ? rdr : "-");
   verdict("ctx", ctx_ok, 1, ver ? ver : "-");
   if (wanted("mrt")) {
      printf("BEGIN mrt\n");
      smoke_mrt();
   }
   if (wanted("xfb")) {
      printf("BEGIN xfb\n");
      smoke_xfb();
   }
   if (wanted("zs")) {
      printf("BEGIN zs\n");
      smoke_zs();
   }
   printf("RESULT %s %d failure(s)%s\n", fails ? "FAIL" : "PASS", fails, offline ? " (offline: values not judged)" : "");
   eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   eglDestroyContext(dpy, c);
   eglTerminate(dpy);
   return fails ? 1 : 0;
}
