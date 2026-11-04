# G9P mrt (AI-written)

Dependency: `es3-limits`. Enable `-Dimagination-g9p-es3=true` in Mesa; default is off.
This branch has not run on A10. A9 evidence is evidence for the source implementation, not this G9P port.

## Build and run on A10

Needs a C compiler, Vulkan headers/loader, and glslangValidator (except limits).
Use an A10 render node and explicitly select the ICD installed from this branch:

```sh
cd src/imagination/g9p/tests/mrt
make
export VK_DRIVER_FILES=/path/to/powervr_mesa_icd.aarch64.json
build/mrt-vk --case one,mrt2,mrt4,mrt4-vary
```

Expected: every job `result=EXACT`, every guard zero, final `result=PASS`. Exit status must be zero; do not use `--offline` for hardware proof.
No A9 kernel interface, device scripts, Apple programs, generated SPIR-V, or version override is required.

## Evidence and A10 checks

A9 ICD: 2/4 targets exact (docs/gpu/experiments/mrt.md). Shared-driver rerun: `G5P-1B-MRT RESULT PASS one mrt2 mrt4 mrt4-vary: 12 jobs exact`, 2026-11-04 (docs/gpu/orchard-port-mrt-xfb.md).

Check 4-target cluster word 0xe3 and register 0x4070 value 0xfa; A9 W3c8/W6b0 equivalents are absent from the A10 stream. Three targets, mixed formats and multi-target blending are refused.
