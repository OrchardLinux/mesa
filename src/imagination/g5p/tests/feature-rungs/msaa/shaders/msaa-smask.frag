// SPDX-License-Identifier: MIT
#version 450
// msaa.md: Apple's g3b-samplemask program, [[sample_mask]] = 0x5, in GLSL (colour from the vertex).
layout(location=0) in vec4 v_color;
layout(location=0) out vec4 color;
void main(){color=v_color;gl_SampleMask[0]=5;}
