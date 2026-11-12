#version 450
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
void main(){color=vec4(vec3(gl_FragCoord.w)+uv.x*0.00001,1);}
