#!/bin/sh
# SPDX-License-Identifier: MIT
# Build formats: compile the shaders with glslangValidator into C arrays
# (generated here, never committed), then the program against the system's
# Vulkan loader.
#   ./build.sh [OUT_DIR]     (default: the current directory)
# Needs glslangValidator, a C compiler and the Vulkan headers/loader.
set -eu
src=$(cd "$(dirname "$0")" && pwd)
out=${1:-.}
mkdir -p "$out"
gen="$out/formats-spirv.h"
: >"$gen.tmp"
# name:stage:defines (+-separated); the C names are spv_<name with . and - as _>
SN_BIAS='(128.0/255.0)'
for s in quad.vert:vert:QUAD_VERT \
         samp-un.frag:frag:SAMP_FRAG samplod-un.frag:frag:SAMPLOD_FRAG \
         samp-sn.frag:frag:SAMP_FRAG+SCALE=0.5+BIAS=$SN_BIAS samplod-sn.frag:frag:SAMPLOD_FRAG+SCALE=0.5+BIAS=$SN_BIAS \
         samp-f.frag:frag:SAMP_FRAG+SCALE=0.0625+BIAS=$SN_BIAS samplod-f.frag:frag:SAMPLOD_FRAG+SCALE=0.0625+BIAS=$SN_BIAS \
         samp-uf.frag:frag:SAMP_FRAG+SCALE=0.0625 samplod-uf.frag:frag:SAMPLOD_FRAG+SCALE=0.0625 \
         usamp-0.frag:frag:USAMP_FRAG usamp-8.frag:frag:USAMP_FRAG+SHIFT=8 usamp-16.frag:frag:USAMP_FRAG+SHIFT=16 \
         usamp-24.frag:frag:USAMP_FRAG+SHIFT=24 isamp-0.frag:frag:ISAMP_FRAG isamp-8.frag:frag:ISAMP_FRAG+SHIFT=8 \
         isamp-16.frag:frag:ISAMP_FRAG+SHIFT=16 isamp-24.frag:frag:ISAMP_FRAG+SHIFT=24 ufetch.frag:frag:UFETCH_FRAG \
         'solid.frag:frag:SOLID_FRAG+COLOR=vec4(0.9375,0.1875,0.125,0.25)' \
         'solid-b.frag:frag:SOLID_FRAG+COLOR=vec4(0.9375,0.1875,0.75,0.25)'; do
    name=${s%%:*} rest=${s#*:}
    stage=${rest%%:*} defs=${rest#*:}
    var=$(echo "$name" | tr .- __)
    dflags=$(echo "$defs" | tr + '\n' | sed 's/^/-D/' | tr '\n' ' ')
    # shellcheck disable=SC2086
    glslangValidator -S "$stage" $dflags -V --vn "spv_$var" -o "$out/$name.h" "$src/shaders.glsl" >/dev/null
    cat "$out/$name.h" >>"$gen.tmp"
    rm -f "$out/$name.h"
done
mv "$gen.tmp" "$gen"
${CC:-cc} -std=gnu11 -O1 -Wall -Werror -I"$out" -o "$out/formats" "$src/formats.c" -lvulkan -lm
echo "built $out/formats"
