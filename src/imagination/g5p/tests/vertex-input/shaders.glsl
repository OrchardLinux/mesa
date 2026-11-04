// SPDX-License-Identifier: MIT
// The shaders of vertex-input, one file: build.sh compiles each stage with
// glslangValidator -D<STAGE>. Every colour a rung draws is constant over its
// primitive, so the target's pixels are exact.

#version 450

#if defined(COLOR_VERT)
// a position and a colour per vertex
layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 v_color;
void main() { gl_Position = vec4(position, 0.0, 1.0); v_color = color; }

#elif defined(INSTANCE_VERT)
// instance i moves 16 pixels right and takes green 32 (i + 1) / 255
layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 v_color;
void main()
{
   gl_Position = vec4(position.x + 0.5 * float(gl_InstanceIndex), position.y, 0.0, 1.0);
   v_color = vec4(color.r, float(32 * (gl_InstanceIndex + 1)) / 255.0, color.ba);
}

#elif defined(VERTEX_INDEX_VERT)
// blue 40 (gl_VertexIndex / 4) / 255: one value per quad of 4 vertices
layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 v_color;
void main()
{
   gl_Position = vec4(position, 0.0, 1.0);
   v_color = vec4(color.rg, float(40 * (gl_VertexIndex >> 2)) / 255.0, color.a);
}

#elif defined(ATTRIB16_VERT)
// 16 attributes: a position and 15 floats, summed in fives into red, green and blue
layout(location = 0) in vec2 position;
layout(location = 1) in float a1;
layout(location = 2) in float a2;
layout(location = 3) in float a3;
layout(location = 4) in float a4;
layout(location = 5) in float a5;
layout(location = 6) in float a6;
layout(location = 7) in float a7;
layout(location = 8) in float a8;
layout(location = 9) in float a9;
layout(location = 10) in float a10;
layout(location = 11) in float a11;
layout(location = 12) in float a12;
layout(location = 13) in float a13;
layout(location = 14) in float a14;
layout(location = 15) in float a15;
layout(location = 0) out vec4 v_color;
void main()
{
   gl_Position = vec4(position, 0.0, 1.0);
   v_color = vec4(a1 + a2 + a3 + a4 + a5, a6 + a7 + a8 + a9 + a10, a11 + a12 + a13 + a14 + a15, 1.0);
}

#elif defined(INTOP_VERT)
// the int-* rungs: one integer operation per shader (OP_*). Quad q = gl_VertexIndex >> 2 of one column, instance i
// moves it 8 pixels right per instance. s, t in -4..3 (signed), u, v in 0..7 (unsigned). Each channel is one form of
// the operation: register pair, small literal, push constant, long literal.
layout(location = 0) in vec2 position;
layout(push_constant) uniform PC { int kneg4; int k1; uint k5; int k3; } pc;
layout(location = 0) out vec4 v_color;
void main()
{
   int q = gl_VertexIndex >> 2, i = gl_InstanceIndex;
   int s = q + pc.kneg4, t = i + pc.kneg4;
   uint u = uint(q), v = uint(i);
   gl_Position = vec4(position.x + 0.25 * float(i), position.y, 0.0, 1.0);
#if defined(OP_IMIN)
   v_color = vec4(ivec4(min(s, t), min(s, 1), min(s, pc.k1), min(t, -2)) + 8) / 255.0;
#elif defined(OP_IMAX)
   v_color = vec4(ivec4(max(s, t), max(s, 1), max(s, pc.k1), max(t, -2)) + 8) / 255.0;
#elif defined(OP_UMIN)
   v_color = vec4(uvec4(min(u, v), min(u, 5u), min(u, pc.k5), min(u << 17, 0x50000u) >> 12)) / 255.0;
#elif defined(OP_UMAX)
   v_color = vec4(uvec4(max(u, v), max(u, 3u), max(u, pc.k5), max(u << 17, 0x50000u) >> 12)) / 255.0;
#elif defined(OP_ISUB)
   v_color = vec4(ivec4(s - t + 8, 100 - q, pc.k3 - q + 8, q - pc.k3 + 8)) / 255.0;
#elif defined(OP_INEG)
   v_color = (vec4(float(-s), float(abs(s)), float(-t), float(-q)) + 8.0) / 255.0;
#endif
}

#elif defined(FMT_VERT)
// the fmt-* rungs: a position and a colour of the rung's vertex formats, colour * scale + bias
layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;
layout(push_constant) uniform PC { vec4 scale; vec4 bias; } pc;
layout(location = 0) out vec4 v_color;
void main() { gl_Position = vec4(position, 0.0, 1.0); v_color = color * pc.scale + pc.bias; }

#elif defined(FMT_INT_VERT)
// fmt-int32: 32-bit unsigned and signed integer attributes
layout(location = 0) in vec2 position;
layout(location = 1) in uvec2 u;
layout(location = 2) in ivec2 i;
layout(push_constant) uniform PC { vec4 scale; vec4 bias; } pc;
layout(location = 0) out vec4 v_color;
void main() { gl_Position = vec4(position, 0.0, 1.0); v_color = vec4(vec2(u), vec2(i)) * pc.scale + pc.bias; }

#elif defined(INST_ATTR_VERT)
// the inst-* rungs: a per-vertex position, a per-instance colour and offset
layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec2 offset;
layout(location = 0) out vec4 v_color;
void main() { gl_Position = vec4(position + offset, 0.0, 1.0); v_color = color; }

#elif defined(COLOR_FRAG)
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 color;
void main() { color = v_color; }

#endif
