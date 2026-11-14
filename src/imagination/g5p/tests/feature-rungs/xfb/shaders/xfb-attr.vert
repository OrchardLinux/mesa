// SPDX-License-Identifier: MIT
#version 450
// The record beside a fetched attribute: the vertex index register follows
// the attribute registers. The position is captured too, into buffer 1 with
// stride 32: 16 bytes of padding per record that nothing may touch.
layout(location = 0) in vec4 pos;
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 16) flat out uvec4 rec;
layout(location = 1, xfb_buffer = 1, xfb_offset = 0, xfb_stride = 32) out vec4 posx;
void main()
{
    uint v = uint(gl_VertexIndex);
    rec = uvec4(v, ~v, floatBitsToUint(float(v)), floatBitsToUint(1.0));
    posx = pos;
    gl_Position = pos;
}
