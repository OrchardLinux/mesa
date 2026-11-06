/* SPDX-License-Identifier: MIT */
/* Generated-by: LLM */
/* 32-bit integer colour targets the way dEQP's ShaderExecutor (glsShaderExecUtil.cpp FragmentOutExecutor) uses
 * them on GL ES: a numValues x 1 renderbuffer of GL_R32UI / RG32UI / RGBA32UI (float outputs bit-cast to uint) or
 * R32I / RG32I / RGBA32I, one FBO attachment, glCheckFramebufferStatus, one 1-pixel point per value whose fragment
 * program writes known bits, then glReadPixels(GL_RGBA_INTEGER) and an exact compare.
 *
 *   es-int-rt FORMAT [W]     FORMAT: r32ui rg32ui rgba32ui r32i rg32i rgba32i; W values (default 97: one row,
 *                            not a multiple of anything)
 * Prints INTRT-STATUS (the FBO status) and "result=PASS|FAIL format=.. complete=.. mismatches=..". Exit 0 = PASS. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *w) { printf("result=FAIL %s\n", w); exit(1); }

static GLuint shader(GLenum t, const char *src)
{
   GLuint s = glCreateShader(t);
   glShaderSource(s, 1, &src, NULL);
   glCompileShader(s);
   GLint ok;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[1024];
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      printf("compile: %s\n", log);
      fail("compile");
   }
   return s;
}

/* value v of pixel x, channel c: distinct high and low bits, sign bit set on odd x (signed formats see negatives) */
static uint32_t want(unsigned x, unsigned c) {
   static const uint32_t v[]={0,1,2,3,4,127,128,255,256,1023,1024,32767,32768,65534,65535,65536,0xffffffffu,0xffffff80u,0xffffff7fu,0xffff8000u,0xffff7fffu,0x80000000u};
   return v[(x+3*c)%22];
}

int main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IOLBF, 0);
   if (argc < 2)
      return 2;
   static const struct { const char *name; GLenum fmt; unsigned comps; int is_signed; } F[] = {
      { "rgba16ui", GL_RGBA16UI, 4, 0 },
      { "rg16ui", GL_RG16UI, 2, 0 },
      { "rgba8ui", GL_RGBA8UI, 4, 0 },
      { "rgb10_a2ui", GL_RGB10_A2UI, 4, 0 },
      { "rgba8i", GL_RGBA8I, 4, 1 },
      { "rg16i", GL_RG16I, 2, 1 },
      { "rgba16i", GL_RGBA16I, 4, 1 },
      { "r16ui", GL_R16UI, 1, 0 }, { "r16i", GL_R16I, 1, 1 },
      { "rg8ui", GL_RG8UI, 2, 0 }, { "rg8i", GL_RG8I, 2, 1 },
      { "r32ui", GL_R32UI, 1, 0 },    { "rg32ui", GL_RG32UI, 2, 0 }, { "rgba32ui", GL_RGBA32UI, 4, 0 },
      { "r32i", GL_R32I, 1, 1 },      { "rg32i", GL_RG32I, 2, 1 },   { "rgba32i", GL_RGBA32I, 4, 1 },
   };
   int f = -1;
   for (unsigned i = 0; i < sizeof(F) / sizeof(F[0]); i++)
      if (!strcmp(argv[1], F[i].name))
         f = (int)i;
   if (f < 0)
      return 2;
   const unsigned W = argc > 2 ? (unsigned)atoi(argv[2]) : 97, C = F[f].comps;

   PFNEGLGETPLATFORMDISPLAYEXTPROC get = (void *)eglGetProcAddress("eglGetPlatformDisplayEXT");
   EGLDisplay d = get(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   EGLint n;
   EGLConfig cfg;
   const EGLint a[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE };
   const EGLint ca[] = { EGL_CONTEXT_MAJOR_VERSION_KHR, 3, EGL_NONE };
   if (!eglInitialize(d, NULL, NULL) || !eglBindAPI(EGL_OPENGL_ES_API) || !eglChooseConfig(d, a, &cfg, 1, &n) || !n)
      fail("EGL init");
   EGLContext c = eglCreateContext(d, cfg, EGL_NO_CONTEXT, ca);
   if (c == EGL_NO_CONTEXT || !eglMakeCurrent(d, EGL_NO_SURFACE, EGL_NO_SURFACE, c))
      fail("ES3 context");
   printf("renderer=%s format=%s W=%u\n", glGetString(GL_RENDERER), F[f].name, W);

   GLuint rb, fb;
   glGenFramebuffers(1, &fb);
   glBindFramebuffer(GL_FRAMEBUFFER, fb);
   glGenRenderbuffers(1, &rb);
   glBindRenderbuffer(GL_RENDERBUFFER, rb);
   glRenderbufferStorage(GL_RENDERBUFFER, F[f].fmt, (GLsizei)W, 1);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
   const GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
   printf("INTRT-STATUS format=%s status=%#x complete=%d\n", F[f].name, st, st == GL_FRAMEBUFFER_COMPLETE);
   if (st != GL_FRAMEBUFFER_COMPLETE) {
      printf("result=FAIL format=%s complete=0 mismatches=-1\n", F[f].name);
      return 1;
   }

   /* the values come in as a flat uvec4 attribute (bit patterns), the output takes the target's components */
   const char *ty = F[f].is_signed ? (C == 1 ? "int" : C == 2 ? "ivec2" : "ivec4")
                                   : (C == 1 ? "uint" : C == 2 ? "uvec2" : "uvec4");
   const char *sw = C == 1 ? ".x" : C == 2 ? ".xy" : "";
   char vs[512], fs[512];
   snprintf(vs, sizeof(vs),
            "#version 300 es\nin vec2 p;in uvec4 v;flat out uvec4 o;"
            "void main(){gl_Position=vec4(p,0.0,1.0);gl_PointSize=1.0;o=v;}");
   snprintf(fs, sizeof(fs), "#version 300 es\nflat in highp uvec4 o;layout(location=0) out highp %s c;void main(){c=%s(o%s);}",
            ty, ty, sw);
   GLuint prog = glCreateProgram();
   glAttachShader(prog, shader(GL_VERTEX_SHADER, vs));
   glAttachShader(prog, shader(GL_FRAGMENT_SHADER, fs));
   glBindAttribLocation(prog, 0, "p");
   glBindAttribLocation(prog, 1, "v");
   glLinkProgram(prog);
   GLint ok;
   glGetProgramiv(prog, GL_LINK_STATUS, &ok);
   if (!ok)
      fail("link");
   float *pos = malloc(sizeof(float) * 2 * W);
   uint32_t *val = malloc(sizeof(uint32_t) * 4 * W);
   for (unsigned x = 0; x < W; x++) {
      pos[2 * x] = -1.0f + 2.0f * (((float)x + 0.5f) / (float)W);
      pos[2 * x + 1] = 0.0f;
      for (unsigned k = 0; k < 4; k++)
         val[4 * x + k] = want(x, k);
   }
   glUseProgram(prog);
   glViewport(0, 0, (GLsizei)W, 1);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, pos);
   glEnableVertexAttribArray(1);
   glVertexAttribIPointer(1, 4, GL_UNSIGNED_INT, 0, val);
   const GLuint zero[4] = { 0, 65535, 65536, 0xffffffffu };
   if(F[f].is_signed)glClearBufferiv(GL_COLOR,0,(const GLint*)zero);
   else glClearBufferuiv(GL_COLOR,0,zero);
   unsigned allbad=0;
   for(unsigned phase=0;phase<3;phase++) {
      if(phase==1) glDrawArrays(GL_POINTS,0,(GLsizei)W);
      if(phase==2) {
         for(unsigned x=0;x<W/2;x++)for(unsigned k=0;k<4;k++)val[4*x+k]=want(x+W,k);
         glDrawArrays(GL_POINTS,0,(GLsizei)(W/2));
      }
      glFinish();
      if(glGetError()!=GL_NO_ERROR) fail("draw");
      uint32_t *px=calloc(4*W,sizeof(uint32_t));
      glReadPixels(0,0,(GLsizei)W,1,GL_RGBA_INTEGER,F[f].is_signed?GL_INT:GL_UNSIGNED_INT,px);
      if(glGetError()!=GL_NO_ERROR) fail("readback");
      unsigned bad=0;
      for(unsigned x=0;x<W;x++)for(unsigned k=0;k<4;k++) {
         uint32_t w=k<C?(phase==0?zero[k]:want(x+(phase==2&&x<W/2?W:0),k)):(k==3?1:0);
         if((F[f].fmt==GL_RGBA16UI || F[f].fmt==GL_RG16UI || F[f].fmt==GL_R16UI) && w>65535 && k<C)w=65535;
         if((F[f].fmt==GL_RGBA8UI || F[f].fmt==GL_RG8UI) && k<C && w>255)w=255;
         if(F[f].fmt==GL_RGB10_A2UI && w>(k==3?3u:1023u))w=k==3?3u:1023u;
         if((F[f].fmt==GL_RGBA8I || F[f].fmt==GL_RG8I || F[f].fmt==GL_R16I || F[f].fmt==GL_RG16I || F[f].fmt==GL_RGBA16I) && k<C) {
            int lo=(F[f].fmt==GL_RGBA8I || F[f].fmt==GL_RG8I)?-128:-32768, hi=(F[f].fmt==GL_RGBA8I || F[f].fmt==GL_RG8I)?127:32767;
            int32_t signed_w=(int32_t)w;
            w=(uint32_t)(signed_w<lo?lo:signed_w>hi?hi:signed_w);
         }
         if(px[4*x+k]!=w && bad++<8)
            printf("PACKED-RT pixel phase=%u x=%u c=%u got=%08x want=%08x\n",phase,x,k,px[4*x+k],w);
      }
      free(px);allbad+=bad;
      printf("PACKED-RT case format=%s phase=%u bad=%u\n",F[f].name,phase,bad);
   }
   printf("PACKED-RT RESULT %s format=%s cases=3 bad=%u\n",allbad?"FAIL":"PASS",F[f].name,allbad);
   return allbad?1:0;
}
