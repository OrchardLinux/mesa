#version 450
layout(location=0) in vec2 uv;
layout(set=0,binding=0) uniform U { float a[16]; int n; } u;
layout(location=0) out vec4 color;
void main(){int k=int(uv.x*64.);float r=0.;if(k>=0 && k<16){for(int j=0;j<u.n;j++)r+=dFdx(u.a[k]);}color=vec4(r,.25,.75,1);}
