// SPDX-License-Identifier: MIT
#version 450
// Duplicate-store detection: the capture buffer is also this draw's vertex
// buffer (same memory, stride 16). A vertex shaded once stores prev.y + 1
// = 1 over the zero fill; a vertex shaded again after its first store
// landed reads 1 and stores 2.
layout(location = 0) in vec4 prev;
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 16) out vec4 rec;
void main()
{
    rec = vec4(float(uint(gl_VertexIndex)), prev.y + 1.0, 0.0, 1.0);
    gl_Position = vec4(float(uint(gl_VertexIndex)) * 0.125 - 0.875, 0.0, 0.0, 1.0);
}
