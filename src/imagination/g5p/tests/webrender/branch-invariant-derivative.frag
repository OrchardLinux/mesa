#version 450
layout(location=0) in vec2 uv;
layout(set=0,binding=0) uniform U { int n; } u;
layout(location=0) out vec4 color;
void main(){float r=0.;if((int(gl_FragCoord.x)&1)==0){for(int j=0;j<u.n;j++)r+=dFdx(uv.x)*8.;color=vec4(r,.25,.75,1);}else color=vec4(.125,.5,.25,1);}
