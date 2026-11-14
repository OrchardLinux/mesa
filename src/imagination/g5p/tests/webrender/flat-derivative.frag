#version 450
layout(location=0) flat in int mode;
layout(location=1) in vec2 uv;
layout(location=0) out vec4 color;
void main(){if(mode==1)color=vec4(dFdx(uv.x),dFdy(uv.y),0.75,1.0);else color=vec4(0.25);}
