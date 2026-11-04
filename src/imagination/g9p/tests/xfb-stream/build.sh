#!/bin/sh
# SPDX-License-Identifier: MIT
# Build xfb-vk: the shaders with glslangValidator into C arrays (generated here, never committed), then the
# program against the system's Vulkan loader.
#   ./build.sh [OUT_DIR]     (default: the current directory)
set -eu
src=$(cd "$(dirname "$0")" && pwd)
out=${1:-.}
mkdir -p "$out"
gen="$out/xfb-spirv.h"
: >"$gen.tmp"
for s in VS_XFB:vert FS_FLAT:frag; do
    def=${s%%:*} stage=${s#*:}
    var=$(echo "$def" | tr 'A-Z' 'a-z')
    glslangValidator -S "$stage" -D"$def" -V --vn "spv_$var" -o "$out/$var.h" "$src/shaders.glsl" >/dev/null
    cat "$out/$var.h" >>"$gen.tmp"
    rm -f "$out/$var.h"
done
mv "$gen.tmp" "$gen"
${CC:-cc} -std=gnu11 -O1 -Wall -Werror -I"$out" -o "$out/xfb-vk" "$src/xfb-vk.c" -lvulkan -lm
echo "built $out/xfb-vk"
