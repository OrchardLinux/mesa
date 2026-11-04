// SPDX-License-Identifier: MIT
#version 450
// The scene of Apple's iOS mrt2 capture: constant R, G.
layout(location=0) out vec4 o0;
layout(location=1) out vec4 o1;
void main(){o0=vec4(1,0,0,1);o1=vec4(0,1,0,1);}
