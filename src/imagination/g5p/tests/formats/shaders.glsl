// SPDX-License-Identifier: MIT
// The formats test's shaders: one file, one -D per stage and -D constants
// per variant (build.sh). Constants are compiled in, not specialization
// constants, so a backend without those runs every variant.
#version 450
#ifndef SCALE
#define SCALE 1.0
#endif
#ifndef BIAS
#define BIAS 0.0
#endif
#ifndef SHIFT
#define SHIFT 0
#endif

#if defined(QUAD_VERT)
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 0) out vec2 v_uv;
void main()
{
   v_uv = uv;
   gl_Position = vec4(pos, 0.0, 1.0);
}

#elif defined(SAMP_FRAG) || defined(SAMPLOD_FRAG)
// A float texture: texel * SCALE + BIAS into the UNORM8 target (level 1 for SAMPLOD)
layout(set = 0, binding = 0) uniform sampler2D t;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o;
void main()
{
#if defined(SAMPLOD_FRAG)
   o = textureLod(t, v_uv, 1.0) * SCALE + BIAS;
#else
   o = texture(t, v_uv) * SCALE + BIAS;
#endif
}

#elif defined(USAMP_FRAG) || defined(UFETCH_FRAG)
// An unsigned integer texture: byte SHIFT/8 of each channel
layout(set = 0, binding = 0) uniform usampler2D t;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o;
void main()
{
#if defined(UFETCH_FRAG)
   uvec4 v = texelFetch(t, ivec2(v_uv * 8.0), 0);
#else
   uvec4 v = texture(t, v_uv);
#endif
   o = vec4((v >> uint(SHIFT)) & 0xffu) / 255.0;
}

#elif defined(ISAMP_FRAG)
// A signed integer texture: byte SHIFT/8 of each channel (two's complement)
layout(set = 0, binding = 0) uniform isampler2D t;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o;
void main()
{
   ivec4 v = texture(t, v_uv);
   o = vec4((v >> SHIFT) & 0xff) / 255.0;
}

#elif defined(LAY2D_FRAG) || defined(LAYARR_FRAG) || defined(LAY3D_FRAG)
// The layout rungs: an RGBA8 texture read at level LOD. 2D: over the whole
// target. Arrays and 3D: the target is a 4x4 grid of cells, cell
// c = x + 4 y reads layer c, or slice c of DEPTH at its centre; inside a cell
// the coordinates run over the level
#ifndef LOD
#define LOD 0.0
#endif
#ifndef DEPTH
#define DEPTH 1.0
#endif
#if defined(LAY2D_FRAG)
layout(set = 0, binding = 0) uniform sampler2D t;
#elif defined(LAYARR_FRAG)
layout(set = 0, binding = 0) uniform sampler2DArray t;
#else
layout(set = 0, binding = 0) uniform sampler3D t;
#endif
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o;
void main()
{
#if defined(LAY2D_FRAG)
   o = textureLod(t, v_uv, LOD);
#else
   vec2 cell = floor(v_uv * 4.0);
   vec2 st = fract(v_uv * 4.0);
   float c = cell.x + 4.0 * cell.y;
#if defined(LAYARR_FRAG)
   o = textureLod(t, vec3(st, c), LOD);
#else
   o = textureLod(t, vec3(st, (c + 0.5) / DEPTH), LOD);
#endif
#endif
}

#elif defined(SOLID_FRAG)
// A constant colour: COLOR, a vec4
layout(location = 0) out vec4 o;
void main()
{
   o = COLOR;
}
#endif
