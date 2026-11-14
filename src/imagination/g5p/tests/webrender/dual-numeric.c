/* SPDX-License-Identifier: MIT
 * Generated-by: LLM. Independent CPU colour and DS visibility oracles.
 * --reference permits llvmpipe. Fake readback is never pixel evidence.
 */
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define W 32
static const char *vs="#version 150\nout vec2 uv;uniform float depth;void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);uv=p;gl_Position=vec4(p*2.-1.,depth,1);}";
static GLuint shader(GLenum kind,const char *s){GLuint a=glCreateShader(kind);glShaderSource(a,1,&s,NULL);glCompileShader(a);GLint ok;glGetShaderiv(a,GL_COMPILE_STATUS,&ok);if(!ok){char log[4096];glGetShaderInfoLog(a,sizeof(log),NULL,log);fprintf(stderr,"compile %s\n",log);exit(2);}return a;}
static GLuint program(const char *f){GLuint p=glCreateProgram();glAttachShader(p,shader(GL_VERTEX_SHADER,vs));glAttachShader(p,shader(GL_FRAGMENT_SHADER,f));glLinkProgram(p);GLint ok;glGetProgramiv(p,GL_LINK_STATUS,&ok);if(!ok){char log[4096];glGetProgramInfoLog(p,sizeof(log),NULL,log);fprintf(stderr,"link %s\n",log);exit(2);}glUseProgram(p);return p;}

static unsigned check(int mode){unsigned char pix[W*W*4];glReadPixels(0,0,W,W,GL_RGBA,GL_UNSIGNED_BYTE,pix);GLenum error=glGetError();unsigned bad=0;for(int y=0;y<W;y++)for(int x=0;x<W;x++)for(int c=0;c<4;c++){double s=(double[]){.125,.25,.375,.5}[c],b=(double[]){.25,.125,.0625,.5}[c],s1=(double[]){(x+.5)/W,(y+.5)/W,.25,.75}[c];double e=(mode?s*s1:s)+b*(1-s1);bad+=abs(pix[(y*W+x)*4+c]-(int)lround(e*255))>1;}printf("WR3-DUAL mode=%d %s bad_channels=%u gl_error=%x\n",mode,bad||error?"FAIL":"PASS",bad,error);return !!(bad||error);}
int main(int argc,char **argv){int ref=argc==2&&!strcmp(argv[1],"--reference");EGLDisplay d=eglGetDisplay(EGL_DEFAULT_DISPLAY);EGLint ma,mi;if(!eglInitialize(d,&ma,&mi)||!eglBindAPI(EGL_OPENGL_API))return 2;EGLint attr[]={EGL_CONTEXT_MAJOR_VERSION,3,EGL_NONE};EGLContext ctx=eglCreateContext(d,NULL,EGL_NO_CONTEXT,attr);if(ctx==EGL_NO_CONTEXT||!eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,ctx))return 2;const char *renderer=(const char*)glGetString(GL_RENDERER);printf("WR3-DUAL renderer=%s reference=%d\n",renderer,ref);if(!ref&&(!strstr(renderer,"zink")||!strstr(renderer,"PowerVR")))return 2;GLuint vao;glGenVertexArrays(1,&vao);glBindVertexArray(vao);GLuint fb,rb;glGenFramebuffers(1,&fb);glBindFramebuffer(GL_FRAMEBUFFER,fb);glGenRenderbuffers(1,&rb);glBindRenderbuffer(GL_RENDERBUFFER,rb);glRenderbufferStorage(GL_RENDERBUFFER,GL_RGBA8,W,W);glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_RENDERBUFFER,rb);if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)return 2;glViewport(0,0,W,W);program("#version 150\n#extension GL_ARB_explicit_attrib_location : require\nin vec2 uv;layout(location=0,index=0) out vec4 a;layout(location=0,index=1) out vec4 b;void main(){a=vec4(.125,.25,.375,.5);b=vec4(uv,.25,.75);}");unsigned failed=0;for(int mode=0;mode<2;mode++){glDisable(GL_BLEND);glClearColor(.25,.125,.0625,.5);glClear(GL_COLOR_BUFFER_BIT);glEnable(GL_BLEND);glBlendFuncSeparate(mode?0x88F9:GL_ONE,0x88FA,mode?0x8589:GL_ONE,0x88FB);glDrawArrays(GL_TRIANGLES,0,3);failed+=check(mode);}printf("WR3-DUAL RESULT %s failed=%u\n",failed?"FAIL":"PASS",failed);eglTerminate(d);return !!failed;}
