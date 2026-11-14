// SPDX-License-Identifier: MIT
#version 450
// A depth texture read as is (nearest): R = the stored depth.
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
layout(set=0, binding=0) uniform sampler2D depth;
void main(){color=vec4(texture(depth,uv).r,0.0,0.0,1.0);}
