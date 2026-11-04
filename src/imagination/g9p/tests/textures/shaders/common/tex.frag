// SPDX-License-Identifier: MIT
#version 450
// Hyprland passthru.frag shape: the iOS `tex` capture's fragment program.
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
layout(set=0, binding=0) uniform sampler2D tex;
void main(){color=texture(tex,uv);}
