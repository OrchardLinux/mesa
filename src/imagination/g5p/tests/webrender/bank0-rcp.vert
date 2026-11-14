#version 450
layout(location=0) in vec4 position;
layout(set=0,binding=0) uniform U {float divisor;};
void main(){gl_Position=position*(1.0/divisor);}
