#!/bin/sh
# SPDX-License-Identifier: MIT
# Build vertex-input: compile the shaders with glslangValidator into C arrays
# (generated here, never committed), then the program against the system's
# Vulkan loader.
#   ./build.sh [OUT_DIR]     (default: the current directory)
# Needs glslangValidator, a C compiler and the Vulkan headers/loader.
set -eu
src=$(cd "$(dirname "$0")" && pwd)
out=${1:-.}
mkdir -p "$out"
gen="$out/vertex-input-spirv.h"
: >"$gen.tmp"
for s in color.vert:COLOR_VERT:vert instance.vert:INSTANCE_VERT:vert vertex_index.vert:VERTEX_INDEX_VERT:vert \
         attrib16.vert:ATTRIB16_VERT:vert color.frag:COLOR_FRAG:frag \
         int_imin.vert:INTOP_VERT,OP_IMIN:vert int_imax.vert:INTOP_VERT,OP_IMAX:vert \
         int_umin.vert:INTOP_VERT,OP_UMIN:vert int_umax.vert:INTOP_VERT,OP_UMAX:vert \
         int_isub.vert:INTOP_VERT,OP_ISUB:vert int_ineg.vert:INTOP_VERT,OP_INEG:vert \
         fmt.vert:FMT_VERT:vert fmt_int.vert:FMT_INT_VERT:vert \
         inst_attr.vert:INST_ATTR_VERT:vert inst_div.vert:INST_DIV_VERT:vert; do
    name=${s%%:*} rest=${s#*:}
    def=${rest%%:*} stage=${rest#*:}
    var=$(echo "$name" | tr . _)
    defs=$(echo "$def" | sed 's/^/-D/; s/,/ -D/g')
    # shellcheck disable=SC2086
    glslangValidator -S "$stage" $defs -V --vn "spv_$var" -o "$out/$name.h" "$src/shaders.glsl" >/dev/null
    cat "$out/$name.h" >>"$gen.tmp"
    rm -f "$out/$name.h"
done
mv "$gen.tmp" "$gen"
${CC:-cc} -std=gnu11 -O1 -Wall -Werror -I"$out" -o "$out/vertex-input" "$src/vertex-input.c" -lvulkan -lm
echo "built $out/vertex-input"
