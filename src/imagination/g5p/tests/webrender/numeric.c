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
static unsigned check(const char *name,int kind)
{
 unsigned char pixels[W*W*4]; memset(pixels,0,sizeof(pixels));
 glReadPixels(0,0,W,W,GL_RGBA,GL_UNSIGNED_BYTE,pixels);
 GLenum error=glGetError();unsigned bad=0;
 for(int y=0;y<W;y++)for(int x=0;x<W;x++)for(int c=0;c<4;c++){
   double value=0;
   switch(kind){
   case 0: value=c==0?0.5:c==1?0.25:c==2?0.75:1;break;
   case 1: value=c<3?1.0-(x+.5)/(4.0*W)-3.0*(y+.5)/(8.0*W):1;break;
   case 2: value=(double[]){64,128,192,255}[c]/255.0;break;
   case 3: value=(double[]){.25,.125,.0625,.5}[c];
           if(x>=W/2)value+=(double[]){.125,.25,.125,.25}[c];break;
   case 4: value=(double[]){.25,.125,.0625,.5}[c];
           if(x>=W/2)value*=1-(double[]){.125,.25,.125,.25}[c];break;
   }
   int expected=(int)lround(fmin(1,fmax(0,value))*255);
   bad+=abs((int)pixels[4*(y*W+x)+c]-expected)>1;
 }
 printf("WR-NUMERIC %s %s bad_channels=%u gl_error=%#x\n",name,(!bad&&!error)?"PASS":"FAIL",bad,error);
 return bad||error;
}
int main(int argc,char **argv)
{
 EGLDisplay d=eglGetDisplay(EGL_DEFAULT_DISPLAY);EGLint major,minor;
 if(!eglInitialize(d,&major,&minor)||!eglBindAPI(EGL_OPENGL_ES_API))return 2;
 EGLint attr[]={EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE};
 EGLContext ctx=eglCreateContext(d,NULL,EGL_NO_CONTEXT,attr);
 if(ctx==EGL_NO_CONTEXT||!eglMakeCurrent(d,EGL_NO_SURFACE,EGL_NO_SURFACE,ctx))return 2;
 const char *renderer=(const char*)glGetString(GL_RENDERER);
 int reference=argc==2&&!strcmp(argv[1],"--reference");
 const char *bank0=getenv("PVR_G5P_VS_BANK0_FIRST");
 printf("WR-NUMERIC IDENTITY renderer=%s reference=%d bank0_first=%s\n",renderer,reference,bank0?bank0:"default");
 if(!reference&&(!strstr(renderer,"zink")||!strstr(renderer,"PowerVR")))return 2;
 GLuint fb,rb;glGenFramebuffers(1,&fb);glBindFramebuffer(GL_FRAMEBUFFER,fb);
 glGenRenderbuffers(1,&rb);glBindRenderbuffer(GL_RENDERBUFFER,rb);glRenderbufferStorage(GL_RENDERBUFFER,GL_RGBA8,W,W);
 glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_RENDERBUFFER,rb);
 if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)return 2;
 glViewport(0,0,W,W);unsigned failed=0;GLuint p;
 p=program(vs,"#version 300 es\nprecision highp float;flat in int mode;in vec2 uv;out vec4 o;"
  "void main(){if(mode==1)o=vec4(dFdx(uv.x)*16.0,dFdy(uv.y)*8.0,.75,1);else o=vec4(1,0,1,1);}");
 glClearColor(0,0,0,0);glClear(GL_COLOR_BUFFER_BIT);glDrawArrays(GL_TRIANGLES,0,3);
 failed+=check("flat-branch-derivatives",0);glDeleteProgram(p);
 for(int smooth=0;smooth<2;smooth++){
  p=program(vs_w,smooth?
   "#version 300 es\nprecision highp float;in vec2 uv;flat in int mode;out vec4 o;void main(){o=vec4(vec3(gl_FragCoord.w)+uv.x*0.00001,1);}":
   "#version 300 es\nprecision highp float;flat in int mode;out vec4 o;void main(){o=vec4(vec3(gl_FragCoord.w),1);}");
  glClear(GL_COLOR_BUFFER_BIT);glDrawArrays(GL_TRIANGLES,0,3);
  failed+=check(smooth?"position-w-smooth":"position-w-no-smooth",1);glDeleteProgram(p);
 }
 GLuint tex;glGenTextures(1,&tex);glBindTexture(GL_TEXTURE_2D,tex);
 unsigned char data[]={64,128,192,255};glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,1,1,0,GL_RGBA,GL_UNSIGNED_BYTE,data);
 glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
 p=program(vs_tex,"#version 300 es\nprecision highp float;in vec4 color;out vec4 o;void main(){o=color;}");
 glUniform1i(glGetUniformLocation(p,"tex"),0);glClear(GL_COLOR_BUFFER_BIT);glDrawArrays(GL_TRIANGLES,0,3);
 failed+=check("vertex-fetch-reciprocal",2);glDeleteProgram(p);
 for(int multiply=0;multiply<2;multiply++){
  p=program(vs,"#version 300 es\nprecision highp float;out vec4 o;void main(){if(gl_FragCoord.x<16.0)discard;o=vec4(.125,.25,.125,.25);}");
  glClearColor(.25,.125,.0625,.5);glClear(GL_COLOR_BUFFER_BIT);glEnable(GL_BLEND);
  glBlendEquation(GL_FUNC_ADD);glBlendFunc(multiply?GL_ZERO:GL_ONE,multiply?GL_ONE_MINUS_SRC_COLOR:GL_ONE);
  glDrawArrays(GL_TRIANGLES,0,3);glDisable(GL_BLEND);
  failed+=check(multiply?"discard-multiply-identity":"discard-add-identity",multiply?4:3);glDeleteProgram(p);
 }
 printf("WR-NUMERIC RESULT %s cases=6 failed=%u reference=%d\n",failed?"FAIL":"PASS",failed,reference);
 eglTerminate(d);return failed?1:0;
}
