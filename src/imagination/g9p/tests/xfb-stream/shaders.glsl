// SPDX-License-Identifier: MIT
// xfb-vk's shaders: one file, the stage chosen with -D (build.sh).
#version 450
#if defined(VS_XFB)
// Two capture buffers: buffer 0 records 16 of every 32 bytes (stride padding stays untouched), buffer 1 is packed.
// Every captured value is exact in float: positions are multiples of 1/32, doubled, plus a small integer.
layout(location = 0) in vec4 position;
layout(location = 0) flat out vec4 v_color;
layout(xfb_buffer = 0, xfb_offset = 0, xfb_stride = 32, location = 1) out vec4 cap_a;
layout(xfb_buffer = 1, xfb_offset = 0, xfb_stride = 16, location = 2) out vec4 cap_b;
void main()
{
	gl_Position = position;
	float v = float(gl_VertexIndex);
	cap_a = 2.0 * position + vec4(v);
	cap_b = vec4(v, position.x, 1.0, -position.y);
	v_color = vec4(0.2, 0.4, 0.6, 0.8);
}
#elif defined(FS_FLAT)
layout(location = 0) flat in vec4 v_color;
layout(location = 0) out vec4 o;
void main() { o = v_color; }
#endif
