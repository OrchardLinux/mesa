#!/bin/sh
# SPDX-License-Identifier: MIT
# Build triangle: compile the shaders with glslangValidator into C arrays
# (generated here, never committed), then the program against the system's
# Vulkan loader.
#   ./build.sh [OUT_DIR]     (default: the current directory)
# Needs glslangValidator, a C compiler and the Vulkan headers/loader.
set -eu
src=$(cd "$(dirname "$0")" && pwd)
out=${1:-.}
mkdir -p "$out"
gen="$out/triangle-spirv.h"
: >"$gen.tmp"
for s in g5p_pass_vert:PASS_VERT:vert g5p_solid_frag:SOLID_FRAG:frag g5p_green_frag:GREEN_FRAG:frag; do
    var=${s%%:*} rest=${s#*:}
    def=${rest%%:*} stage=${rest#*:}
    glslangValidator -S "$stage" -D"$def" -V --vn "$var" -o "$out/$var.h" "$src/shaders.glsl" >/dev/null
    cat "$out/$var.h" >>"$gen.tmp"
    rm -f "$out/$var.h"
done
mv "$gen.tmp" "$gen"
${CC:-cc} -std=gnu11 -O1 -Wall -Werror -I"$out" -o "$out/triangle" "$src/triangle.c" -lvulkan -lm
echo "built $out/triangle"
