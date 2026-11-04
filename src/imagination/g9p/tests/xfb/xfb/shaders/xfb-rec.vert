// SPDX-License-Identifier: MIT
#version 450
// xfb-test's suite (A9 notes: experiments/xfb.md): one 16-byte record per
// vertex, (id, ~id, float(id), 1.0f), no vertex attributes (the PDS writes
// the vertex index). Degenerate positions: capture only.
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 16) flat out uvec4 rec;
void main()
{
    uint v = uint(gl_VertexIndex);
    rec = uvec4(v, ~v, floatBitsToUint(float(v)), floatBitsToUint(1.0));
    gl_Position = vec4(float(v) * 0.125 - 0.875, 0.0, 0.0, 1.0);
}
