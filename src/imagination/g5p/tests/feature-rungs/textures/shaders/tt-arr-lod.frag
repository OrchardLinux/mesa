// SPDX-License-Identifier: MIT
#version 450
// tt-vk arr-layer (A9 notes: experiments/textypes.md), with shaders/common/uv.vert.
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
layout(set=0, binding=0) uniform sampler2DArray tex;
void main(){color=textureLod(tex,vec3(uv,uv.x*2.0),0.0);}
