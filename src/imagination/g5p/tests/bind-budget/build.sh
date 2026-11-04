#!/bin/sh
# SPDX-License-Identifier: MIT
# Build bind-budget against the system's Vulkan loader.
#   ./build.sh [OUT_DIR]     (default: the current directory)
set -eu
src=$(cd "$(dirname "$0")" && pwd)
out=${1:-.}
mkdir -p "$out"
${CC:-cc} -std=gnu11 -O1 -Wall -Werror -o "$out/bind-budget" "$src/bind-budget.c" -lvulkan
echo "built $out/bind-budget"
