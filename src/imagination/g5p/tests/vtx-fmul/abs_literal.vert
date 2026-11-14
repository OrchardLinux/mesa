#version 450
// Generated-by: LLM
layout(location=0) in vec4 a;
layout(location=0) out vec4 color;
void main(){color=vec4(abs(a.x)*0.5,a.y,a.z,1.0);gl_Position=a;}
