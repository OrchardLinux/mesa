/* SPDX-License-Identifier: MIT
 * Independent pixel oracles for WebRender-related compiler lowerings.
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
 "#version 300 es\nprecision highp float; precision highp int;\n"
 "uniform float uValue; flat out int mode; out vec2 uv;\n"
 "void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);"
 "uv=p;mode=int(uValue);gl_Position=vec4(p*2.0-1.0,0,1);}";
static const char *vs_w =
 "#version 300 es\nprecision highp float; flat out int mode; out vec2 uv;\n"
 "void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);"
 "float w=gl_VertexID==0?1.0:(gl_VertexID==1?2.0:4.0);"
 "uv=p;mode=1;gl_Position=vec4((p*2.0-1.0)*w,0,w);}";
static const char *vs_tex =
 "#version 300 es\nprecision highp float; precision highp int;\n"
 "uniform highp sampler2D tex; uniform float uValue; out vec4 color;\n"
 "void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);"
 "color=texelFetch(tex,ivec2(0),0)*(1.0/uValue);gl_Position=vec4(p*2.0-1.0,0,1);}";

static GLuint shader(GLenum type, const char *src)
{
 GLuint sh=glCreateShader(type); glShaderSource(sh,1,&src,NULL); glCompileShader(sh);
 GLint ok; glGetShaderiv(sh,GL_COMPILE_STATUS,&ok);
 if(!ok){char log[8192];glGetShaderInfoLog(sh,sizeof(log),NULL,log);fprintf(stderr,"shader: %s\n",log);exit(2);}
 return sh;
}
static GLuint program(const char *v,const char *f)
{
 GLuint p=glCreateProgram(), a=shader(GL_VERTEX_SHADER,v), b=shader(GL_FRAGMENT_SHADER,f);
 glAttachShader(p,a);glAttachShader(p,b);glLinkProgram(p);glDeleteShader(a);glDeleteShader(b);
 GLint ok;glGetProgramiv(p,GL_LINK_STATUS,&ok);
 if(!ok){char log[8192];glGetProgramInfoLog(p,sizeof(log),NULL,log);fprintf(stderr,"link: %s\n",log);exit(2);}
 glUseProgram(p);glUniform1f(glGetUniformLocation(p,"uValue"),1.0f);return p;
}

static const char *vs_z="#version 300 es\nprecision highp float;void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);float w=gl_VertexID==0?1.:(gl_VertexID==1?2.:4.);float z=gl_VertexID==0?-.75:(gl_VertexID==1?.25:.75);gl_Position=vec4((p*2.-1.)*w,z*w,w);}";
static unsigned zcheck(int range){unsigned char p[W*W*4];glReadPixels(0,0,W,W,GL_RGBA,GL_UNSIGNED_BYTE,p);GLenum e=glGetError();unsigned bad=0;for(int y=0;y<W;y++)for(int x=0;x<W;x++)for(int c=0;c<4;c++){double z=.125+(x+.5)/(4*W)+3*(y+.5)/(8*W);if(range)z=.2+.6*z;double expected=c==3?1:z;bad+=abs(p[(y*W+x)*4+c]-(int)lround(expected*255))>1;}printf("WR3-Z range=%d %s bad_channels=%u gl_error=%x\n",range,bad||e?"FAIL":"PASS",bad,e);return !!(bad||e);}
int main(int argc,char **argv){int ref=argc==2&&!strcmp(argv[1],"--reference");EGLDisplay d=eglGetDisplay(EGL_DEFAULT_DISPLAY);EGLint ma,mi;if(!eglInitialize(d,&ma,&mi)||!eglBindAPI(EGL_OPENGL_ES_API))return 2;EGLint attr[]={EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE};EGLContext ctx=eglCreateContext(d,NULL,EGL_NO_CONTEXT,attr);if(ctx==EGL_NO_CONTEXT||!eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,ctx))return 2;const char *r=(const char*)glGetString(GL_RENDERER);printf("WR3-Z renderer=%s reference=%d\n",r,ref);if(!ref&&(!strstr(r,"zink")||!strstr(r,"PowerVR")))return 2;GLuint fb,rb;glGenFramebuffers(1,&fb);glBindFramebuffer(GL_FRAMEBUFFER,fb);glGenRenderbuffers(1,&rb);glBindRenderbuffer(GL_RENDERBUFFER,rb);glRenderbufferStorage(GL_RENDERBUFFER,GL_RGBA8,W,W);glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_RENDERBUFFER,rb);if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)return 2;glViewport(0,0,W,W);program(vs_z,"#version 300 es\nprecision highp float;out vec4 o;void main(){o=vec4(vec3(gl_FragCoord.z),1);}");unsigned failed=0;for(int i=0;i<2;i++){glDepthRangef(i?.2:0,i?.8:1);glClearColor(0,0,0,0);glClear(GL_COLOR_BUFFER_BIT);glDrawArrays(GL_TRIANGLES,0,3);failed+=zcheck(i);}printf("WR3-Z RESULT %s failed=%u\n",failed?"FAIL":"PASS",failed);eglTerminate(d);return !!failed;}
