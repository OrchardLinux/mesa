#!/bin/sh
# SPDX-License-Identifier: MIT
# Build tex-blend: compile the shaders with glslangValidator into C arrays
# (generated here, never committed), then the program against the system's
# Vulkan loader.
#   ./build.sh [OUT_DIR]     (default: the current directory)
# Needs glslangValidator, a C compiler and the Vulkan headers/loader.
set -eu
src=$(cd "$(dirname "$0")" && pwd)
out=${1:-.}
mkdir -p "$out"
gen="$out/tex-blend-spirv.h"
: >"$gen.tmp"
for s in pass.vert:PASS_VERT:vert uv.vert:UV_VERT:vert tex.frag:TEX_FRAG:frag solid.frag:SOLID_FRAG:frag; do
    name=${s%%:*} rest=${s#*:}
    def=${rest%%:*} stage=${rest#*:}
    var=$(echo "$name" | tr . _)
    glslangValidator -S "$stage" -D"$def" -V --vn "spv_$var" -o "$out/$name.h" "$src/shaders.glsl" >/dev/null
    cat "$out/$name.h" >>"$gen.tmp"
    rm -f "$out/$name.h"
done
mv "$gen.tmp" "$gen"
${CC:-cc} -std=gnu11 -O1 -Wall -Werror -I"$out" -o "$out/tex-blend" "$src/tex-blend.c" -lvulkan -lm
echo "built $out/tex-blend"
