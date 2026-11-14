#version 450
layout(location=0) out vec4 color;
void main(){if(gl_FragCoord.x<16.0)discard;color=vec4(.125,.25,.125,.25);}
