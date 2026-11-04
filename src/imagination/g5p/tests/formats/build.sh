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
         lay2d-0.frag:frag:LAY2D_FRAG+LOD=0.0 lay2d-1.frag:frag:LAY2D_FRAG+LOD=1.0 \
         lay2d-2.frag:frag:LAY2D_FRAG+LOD=2.0 lay2d-3.frag:frag:LAY2D_FRAG+LOD=3.0 \
         layarr-0.frag:frag:LAYARR_FRAG+LOD=0.0 layarr-1.frag:frag:LAYARR_FRAG+LOD=1.0 \
         layarr-2.frag:frag:LAYARR_FRAG+LOD=2.0 \
         lay3d-1-1.frag:frag:LAY3D_FRAG+DEPTH=1.0+LOD=1.0 lay3d-2-0.frag:frag:LAY3D_FRAG+DEPTH=2.0+LOD=0.0 \
         lay3d-2-2.frag:frag:LAY3D_FRAG+DEPTH=2.0+LOD=2.0 lay3d-3-0.frag:frag:LAY3D_FRAG+DEPTH=3.0+LOD=0.0 \
         lay3d-4-1.frag:frag:LAY3D_FRAG+DEPTH=4.0+LOD=1.0 lay3d-8-0.frag:frag:LAY3D_FRAG+DEPTH=8.0+LOD=0.0 \
         lay3d-16-0.frag:frag:LAY3D_FRAG+DEPTH=16.0+LOD=0.0 \
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
