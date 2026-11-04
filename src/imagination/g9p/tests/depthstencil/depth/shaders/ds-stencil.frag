// SPDX-License-Identifier: MIT
#version 450
// The stencil aspect read as an unsigned integer: R = stencil / 255.
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
layout(set=0, binding=0) uniform usampler2D stencil;
void main(){color=vec4(float(texture(stencil,uv).r)/255.0,0.0,0.0,1.0);}
