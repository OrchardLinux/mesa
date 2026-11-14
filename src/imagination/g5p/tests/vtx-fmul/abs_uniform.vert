#version 450
// Generated-by: LLM
layout(location=0) in vec4 a;
layout(set=0,binding=0) uniform Params { vec4 u; };
layout(location=0) out vec4 color;
void main(){ color=vec4(a.x * abs(u.x),a.y,a.z,1.0); gl_Position=a;}
