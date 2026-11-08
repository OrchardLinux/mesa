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
static char *read_file(const char *path){FILE *f=fopen(path,"rb");if(!f)exit(2);fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);char *s=calloc(n+1,1);if(fread(s,1,n,f)!=(size_t)n)exit(2);fclose(f);return s;}
int main(int argc,char **argv){
 if(argc!=9)return 2;
 EGLDisplay d=eglGetDisplay(EGL_DEFAULT_DISPLAY);EGLint a,b; if(!eglInitialize(d,&a,&b)||!eglBindAPI(EGL_OPENGL_ES_API))return 2;
 EGLint attr[]={EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE};EGLContext ctx=eglCreateContext(d,NULL,EGL_NO_CONTEXT,attr);
 if(ctx==EGL_NO_CONTEXT||!eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,ctx))return 2;
 const char *r=(const char*)glGetString(GL_RENDERER);int ref=!strcmp(argv[8],"--reference");
 printf("WR2-NUMERIC renderer=%s reference=%d\n",r,ref);
 if(!ref&&(!strstr(r,"zink")||!strstr(r,"PowerVR")))return 2;
 GLuint fb,rb;glGenFramebuffers(1,&fb);glBindFramebuffer(GL_FRAMEBUFFER,fb);glGenRenderbuffers(1,&rb);glBindRenderbuffer(GL_RENDERBUFFER,rb);glRenderbufferStorage(GL_RENDERBUFFER,GL_RGBA8,W,W);glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_RENDERBUFFER,rb);if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)return 2;
 GLuint tex;glGenTextures(1,&tex);glBindTexture(GL_TEXTURE_2D,tex);float data[64*4];for(int i=0;i<64;i++)for(int c=0;c<4;c++)data[4*i+c]=((i*3+c*5)%29+1)/32.f;
 glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,64,1,0,GL_RGBA,GL_FLOAT,data);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
 char *v=read_file(argv[1]),*f=read_file(argv[2]);GLuint p=program(v,f);free(v);free(f);glUniform1i(glGetUniformLocation(p,"tex"),0);glUniform1i(glGetUniformLocation(p,"uOffset"),atoi(argv[3]));
 glViewport(0,0,W,W);glClearColor(0,0,0,0);glClear(GL_COLOR_BUFFER_BIT);glDrawArrays(GL_TRIANGLES,0,3);unsigned char pixels[W*W*4];glReadPixels(0,0,W,W,GL_RGBA,GL_UNSIGNED_BYTE,pixels);unsigned bad=0;for(int i=0;i<W*W*4;i++)bad+=abs((int)pixels[i]-(int)lround(atof(argv[4+i%4])*255))>1;GLenum err=glGetError();printf("WR2-NUMERIC %s bad_channels=%u gl_error=%x\n",bad||err?"FAIL":"PASS",bad,err);eglTerminate(d);return bad||err?1:0;
}
