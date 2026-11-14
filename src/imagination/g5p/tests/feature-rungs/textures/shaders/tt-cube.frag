// SPDX-License-Identifier: MIT
#version 450
// tt-vk cube (A9 notes: experiments/textypes.md), with shaders/common/uv.vert.
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
layout(set=0, binding=0) uniform samplerCube tex;
// face 3 (-Y), as gles3 f_cube
void main(){vec2 c=uv*2.0-1.0; color=texture(tex,vec3(c.x,-1.0,c.y));}
