#version 450
layout(location=0) out vec4 color;
void main(){color=vec4(vec3(gl_FragCoord.w),1.0);}
