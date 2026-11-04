// SPDX-License-Identifier: MIT
#version 450
// Position pass-through plus one constant varying (red) for mrt4-vary.frag.
layout(location=0) in vec4 position;
layout(location=0) out vec4 v_color;
void main(){gl_Position=position;v_color=vec4(1,0,0,1);}
