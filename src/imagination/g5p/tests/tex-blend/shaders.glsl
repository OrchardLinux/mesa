// SPDX-License-Identifier: MIT
// The shaders of tex-blend, one file: build.sh compiles each stage with
// glslangValidator -D<STAGE>. SOLID_FRAG's colour is tex-blend.c's SRC.

#version 450

#if defined(PASS_VERT)
layout(location = 0) in vec4 position;
void main() { gl_Position = position; }

#elif defined(UV_VERT)
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 uv;
layout(location = 0) out vec2 v_uv;
void main() { gl_Position = vec4(position, 0.0, 1.0); v_uv = uv; }

#elif defined(TEX_FRAG)
layout(set = 0, binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 color;
void main() { color = texture(tex, v_uv); }

#elif defined(TEXLOD_FRAG)
// level 2 of a mip chain, whatever the derivatives
layout(set = 0, binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 color;
void main() { color = textureLod(tex, v_uv, 2.0); }

#elif defined(TEXARR_FRAG)
// layer 2 of a 2D array
layout(set = 0, binding = 0) uniform sampler2DArray tex;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 color;
void main() { color = texture(tex, vec3(v_uv, 2.0)); }

#elif defined(TEXCUBE_FRAG)
// the -Z face of a cube: s = 1 - u, t = 1 - v
layout(set = 0, binding = 0) uniform samplerCube tex;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 color;
void main() { color = texture(tex, vec3(2.0 * v_uv - 1.0, -1.0)); }

#elif defined(TEXARR0_FRAG)
// diagnostic: layer 0 of the 2D array
layout(set = 0, binding = 0) uniform sampler2DArray tex;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 color;
void main() { color = texture(tex, vec3(v_uv, 0.0)); }

#elif defined(TEXCUBEX_FRAG)
// diagnostic: the +X face of a cube (face 0): s = 1 - u, t = 1 - v
layout(set = 0, binding = 0) uniform samplerCube tex;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 color;
void main() { color = texture(tex, vec3(1.0, 2.0 * v_uv.y - 1.0, 2.0 * v_uv.x - 1.0)); }

#elif defined(TEXLOD0_FRAG)
// diagnostic: level 0 of a mip chain through an explicit LOD
layout(set = 0, binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 color;
void main() { color = textureLod(tex, v_uv, 0.0); }

#elif defined(SOLID_FRAG)
layout(location = 0) out vec4 color;
void main() { color = vec4(0.5, 0.3, 0.1, 0.6); }

#endif
