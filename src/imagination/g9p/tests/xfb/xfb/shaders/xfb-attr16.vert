// SPDX-License-Identifier: MIT
#version 450
// x1 split: xfb-attr.vert with buffer 1 at stride 16, so both records share one offset (index << 4, the record path
// that lands on hardware). It tests the second buffer's address add and stores alone.
layout(location = 0) in vec4 pos;
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 16) flat out uvec4 rec;
layout(location = 1, xfb_buffer = 1, xfb_offset = 0, xfb_stride = 16) out vec4 posx;
void main()
{
    uint v = uint(gl_VertexIndex);
    rec = uvec4(v, ~v, floatBitsToUint(float(v)), floatBitsToUint(1.0));
    posx = pos;
    gl_Position = pos;
}
