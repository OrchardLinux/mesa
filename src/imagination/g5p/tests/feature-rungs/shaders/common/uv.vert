// SPDX-License-Identifier: MIT
#version 450
// vec2 attributes, constant position components, one vec2 varying.
layout(location=0) in vec2 pos;
layout(location=1) in vec2 uv;
layout(location=0) out vec2 v_uv;
void main(){gl_Position=vec4(pos,0.0,1.0); v_uv=uv;}
