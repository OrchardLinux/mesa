# G9P textures (AI-written)

Dependency: `es3-mrt (shared graphics include and ISA encoders)`. Enable `-Dimagination-g9p-es3=true` in Mesa; default is off.
This branch has not run on A10. A9 evidence is evidence for the source implementation, not this G9P port.

## Build and run on A10

Needs a C compiler, Vulkan headers/loader, and glslangValidator (except limits).
Use an A10 render node and explicitly select the ICD installed from this branch:

```sh
cd src/imagination/g9p/tests/textures
make
export VK_DRIVER_FILES=/path/to/powervr_mesa_icd.aarch64.json
build/tt-vk --case one,arr,arr-layer,cube,npot,npot-lod,arr-mip
```

Expected: every case exact, final `result=PASS`. Exit status must be zero; do not use `--offline` for hardware proof.
No A9 kernel interface, device scripts, Apple programs, generated SPIR-V, or version override is required.

## Evidence and A10 checks

A9 source: textypes rerun exact (docs/gpu/experiments/textypes.md). Shared-driver 2026-11-04: `TEX-BLEND RESULT PASS 22 of 22` (docs/gpu/orchard-port-tex.md). This does not prove all NPOT mip-chain cases on G9P.

Check cube opcode 087d (not among compared ISA forms), dimension nibble, 64-byte/16-KiB layer stride and NPOT mip twiddling. 3D and cube arrays are not implemented; array mip sampling uses the view base level.
