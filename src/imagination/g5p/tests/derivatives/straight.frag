#version 450
layout(location=0) in vec4 v;
layout(set=0,binding=0) uniform U {int n;float k;}u;
layout(location=0) out vec4 color;
void main(){float x=v.x,r=0;r=dFdx(x)+dFdy(x);color=vec4(r,x,v.z,1);}
