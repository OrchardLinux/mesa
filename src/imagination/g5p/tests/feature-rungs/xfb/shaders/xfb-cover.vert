// SPDX-License-Identifier: MIT
#version 450
// --run3 / --ctl (A9 notes: experiments/xfb.md, device run 3): xfb-rec.vert's
// record, but the 3 vertices form the full-screen triangle (-1,-1), (3,-1),
// (-1,3), computed from gl_VertexIndex alone (no attributes: the fetch-less
// PDS and its vertex-index DOUTW). x = v(8 - 4v) - 1, y = v(2v - 2) - 1, written with
// negative literals: the A9 vertex backend had no fadd with a negated source.
// A covered target proves the VS ran with the right index; a wrong index
// (constant or garbage) gives a degenerate or partial triangle.
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 16) flat out uvec4 rec;
void main()
{
    uint v = uint(gl_VertexIndex);
    float f = float(v);
    rec = uvec4(v, ~v, floatBitsToUint(f), floatBitsToUint(1.0));
    gl_Position = vec4(f * (f * -4.0 + 8.0) + -1.0, f * (f * 2.0 + -2.0) + -1.0, 0.0, 1.0);
}
