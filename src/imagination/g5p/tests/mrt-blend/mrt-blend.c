/* SPDX-License-Identifier: MIT
 * Generated-by: LLM. Independent per-target RGBA8 mask/blend pixel oracle.
 * Destination RGB is exact0/1, source is its complement; source alpha=.5,
 * destination alpha=1. Blending produces exact .5 RGB / .75 alpha before
 * UNORM rounding. No tolerance, image masking or recording mode.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

enum { W=37,H=29 };
typedef void (*enable_i)(GLenum,GLuint);
typedef void (*mask_i)(GLuint,GLboolean,GLboolean,GLboolean,GLboolean);
typedef void (*func_i)(GLuint,GLenum,GLenum,GLenum,GLenum);
static void die(const char *what) { fprintf(stderr,"MRT-BLEND FAIL %s error=%x\n",what,glGetError()); exit(1); }
static GLuint shader(GLenum type,const char *text) {
 GLuint s=glCreateShader(type);glShaderSource(s,1,&text,NULL);glCompileShader(s);
 GLint ok;glGetShaderiv(s,GL_COMPILE_STATUS,&ok);if(!ok)die("shader");return s;
}
int main(void) {
 setvbuf(stdout,NULL,_IOLBF,0);
 PFNEGLGETPLATFORMDISPLAYEXTPROC get=(void*)eglGetProcAddress("eglGetPlatformDisplayEXT");
 EGLDisplay d=get(EGL_PLATFORM_SURFACELESS_MESA,EGL_DEFAULT_DISPLAY,NULL);
 EGLint n;EGLConfig cfg;const EGLint a[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT,EGL_NONE};
 const EGLint ca[]={EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE};
 if(!eglInitialize(d,NULL,NULL)||!eglBindAPI(EGL_OPENGL_ES_API)||!eglChooseConfig(d,a,&cfg,1,&n)||!n)die("EGL");
 EGLContext c=eglCreateContext(d,cfg,EGL_NO_CONTEXT,ca);
 if(c==EGL_NO_CONTEXT||!eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,c))die("context");
 enable_i enable=(void*)eglGetProcAddress("glEnableiEXT"),disable=(void*)eglGetProcAddress("glDisableiEXT");
 mask_i mask=(void*)eglGetProcAddress("glColorMaskiEXT");func_i func=(void*)eglGetProcAddress("glBlendFuncSeparateiEXT");
 if(!enable||!disable||!mask||!func)die("indexed entrypoints");
 const char *vs="#version 300 es\nvoid main(){vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexID],0,1);}";
 unsigned cases=0,totalbad=0;
 for(unsigned nt=2;nt<=4;nt++) {
  GLuint fb,rb[4];GLenum bufs[4];glGenFramebuffers(1,&fb);glBindFramebuffer(GL_FRAMEBUFFER,fb);glGenRenderbuffers(nt,rb);
  char fs[2048];int at=snprintf(fs,sizeof(fs),"#version 300 es\nprecision highp float;\n");
  for(unsigned t=0;t<nt;t++) {
   glBindRenderbuffer(GL_RENDERBUFFER,rb[t]);glRenderbufferStorage(GL_RENDERBUFFER,GL_RGBA8,W,H);
   bufs[t]=GL_COLOR_ATTACHMENT0+t;glFramebufferRenderbuffer(GL_FRAMEBUFFER,bufs[t],GL_RENDERBUFFER,rb[t]);
   at+=snprintf(fs+at,sizeof(fs)-at,"layout(location=%u) out vec4 o%u;\n",t,t);
  }
  glDrawBuffers(nt,bufs);if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)die("FBO");
  at+=snprintf(fs+at,sizeof(fs)-at,"void main(){");
  for(unsigned t=0;t<nt;t++)at+=snprintf(fs+at,sizeof(fs)-at,"o%u=vec4(%u.,%u.,%u.,.5);",t,t&1,(t+1)&1,t&1);
  snprintf(fs+at,sizeof(fs)-at,"}");
  GLuint prog=glCreateProgram();glAttachShader(prog,shader(GL_VERTEX_SHADER,vs));glAttachShader(prog,shader(GL_FRAGMENT_SHADER,fs));
  glLinkProgram(prog);GLint ok;glGetProgramiv(prog,GL_LINK_STATUS,&ok);if(!ok)die("link");glUseProgram(prog);glViewport(0,0,W,H);
  for(unsigned mode=0;mode<5;mode++) {
   unsigned masks[4];int blends[4];
   for(unsigned t=0;t<nt;t++) {
    disable(GL_BLEND,t);mask(t,1,1,1,1);
    const GLfloat clear[]={ (float)((t+1)&1),(float)(t&1),(float)((t+1)&1),1 };
    glClearBufferfv(GL_COLOR,t,clear);
   }
   /* Full clears and the draw share one render; N-target LOAD remains
    * a separate named unsupported shape in this topic. */
   for(unsigned t=0;t<nt;t++) {
    masks[t]=mode==1?(t&1?10:5):mode==3?(t&1?3:12):mode==4?0:15;
    blends[t]=(mode==2||mode==3)&&(t&1);
    mask(t,!!(masks[t]&1),!!(masks[t]&2),!!(masks[t]&4),!!(masks[t]&8));
    func(t,GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    if(blends[t])enable(GL_BLEND,t);
   }
   glDrawArrays(GL_TRIANGLES,0,3);glFinish();if(glGetError()!=GL_NO_ERROR)die("draw");
   unsigned bad=0;
   for(unsigned t=0;t<nt;t++) {
    uint8_t pixels[W*H*4],want[4];
    for(unsigned k=0;k<4;k++)want[k]=!(masks[t]&(1u<<k))?(k==3?255:(((t+k+1)&1)?255:0))
                                  :blends[t]?(k==3?191:128):(k==3?128:(((t+k)&1)?255:0));
    glReadBuffer(bufs[t]);glReadPixels(0,0,W,H,GL_RGBA,GL_UNSIGNED_BYTE,pixels);
    if(glGetError()!=GL_NO_ERROR)die("readback");
    for(unsigned p=0;p<W*H;p++)for(unsigned k=0;k<4;k++)if(pixels[4*p+k]!=want[k]) {
     if(!bad)printf("MRT-BLEND first targets=%u mode=%u target=%u pixel=%u channel=%u got=%u want=%u\n",nt,mode,t,p,k,pixels[4*p+k],want[k]);
     bad++;
    }
   }
   printf("MRT-BLEND case targets=%u mode=%u bad=%u\n",nt,mode,bad);totalbad+=bad;cases++;
  }
  glDeleteProgram(prog);glDeleteRenderbuffers(nt,rb);glDeleteFramebuffers(1,&fb);
 }
 printf("MRT-BLEND RESULT %s cases=%u bad=%u\n",totalbad?"FAIL":"PASS",cases,totalbad);return totalbad?1:0;
}
