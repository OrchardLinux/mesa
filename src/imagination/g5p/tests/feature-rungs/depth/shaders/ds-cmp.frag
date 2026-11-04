// SPDX-License-Identifier: MIT
#version 450
// A depth compare (the sampler's compareOp against reference 0.4): R = 0 or 1.
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
layout(set=0, binding=0) uniform sampler2DShadow depth;
void main(){color=vec4(texture(depth,vec3(uv,0.4)),0.0,0.0,1.0);}
