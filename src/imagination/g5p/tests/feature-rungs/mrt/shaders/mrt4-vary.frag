// SPDX-License-Identifier: MIT
#version 450
// Four targets from one varying (R from the VS): R, G, B, Y by swizzle. Exercises the
// register-form writes into output halves 4..15, which no Apple program witnesses ([I]).
layout(location=0) in vec4 v_color;
layout(location=0) out vec4 o0;
layout(location=1) out vec4 o1;
layout(location=2) out vec4 o2;
layout(location=3) out vec4 o3;
void main(){o0=v_color;o1=v_color.yxzw;o2=v_color.yzxw;o3=v_color.xxzw;}
