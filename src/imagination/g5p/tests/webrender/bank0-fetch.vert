#version 450
layout(location=0) in vec4 position;
layout(location=1) in vec2 coord;
layout(set=0,binding=0) uniform sampler2D image;
layout(location=0) out vec4 color;
void main(){color=textureLod(image,coord*0.5+0.25,0.0);gl_Position=position;}
