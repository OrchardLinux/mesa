// SPDX-License-Identifier: MIT
// The test's three shaders; build.sh compiles each with one of these defines.
#version 450
#if defined(PASS_VERT)
layout(location = 0) in vec4 position;
void main() { gl_Position = position; }
#elif defined(SOLID_FRAG)
layout(location = 0) out vec4 color;
void main() { color = vec4(1, 0, 0, 1); }
#elif defined(GREEN_FRAG)
layout(location = 0) out vec4 color;
void main() { color = vec4(0, 1, 0, 1); }
#endif
