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

#elif defined(SOLID_FRAG)
layout(location = 0) out vec4 color;
void main() { color = vec4(0.5, 0.3, 0.1, 0.6); }

#endif
