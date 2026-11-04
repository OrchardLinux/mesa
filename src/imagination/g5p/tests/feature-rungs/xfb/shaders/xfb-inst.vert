// SPDX-License-Identifier: MIT
#version 450
// Instanced capture: records are instance-major (instance * vertices + i).
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 16) flat out uvec4 rec;
void main()
{
    uint v = uint(gl_VertexIndex), i = uint(gl_InstanceIndex);
    rec = uvec4(v, i, v + 7u, floatBitsToUint(1.0));
    gl_Position = vec4(float(v) * 0.125 - 0.875, 0.0, 0.0, 1.0);
}
