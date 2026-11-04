// SPDX-License-Identifier: MIT
#version 450
// tt-vk npot-lod (A9 notes: experiments/textypes.md), with shaders/common/uv.vert.
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
layout(set=0, binding=0) uniform sampler2D tex;
void main(){color=textureLod(tex,uv,2.0);}
