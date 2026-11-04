# G9P depthstencil (AI-written)

Dependency: `es3-mrt (shared draw/clear helpers)`. Enable `-Dimagination-g9p-es3=true` in Mesa; default is off.
This branch has not run on A10. A9 evidence is evidence for the source implementation, not this G9P port.

## Build and run on A10

Needs a C compiler, Vulkan headers/loader, and glslangValidator (except limits).
Use an A10 render node and explicitly select the ICD installed from this branch:

```sh
cd src/imagination/g9p/tests/depthstencil
make
export VK_DRIVER_FILES=/path/to/powervr_mesa_icd.aarch64.json
build/depth-vk --case one,depth,reload,reload16,stencil,cube16,clip,bias,size1
```

Expected: each job `result=EXACT bad=0 guard=0`, final `result=PASS`. Exit status must be zero; do not use `--offline` for hardware proof.
No A9 kernel interface, device scripts, Apple programs, generated SPIR-V, or version override is required.

## Evidence and A10 checks

A9 ICD: four attachment formats exact (docs/gpu/experiments/depth-msaa.md). G5P shared-driver run 2026-11-04: `G5P-1B-DSMS RESULT PASS depth cube16 clip bias size1` (docs/gpu/orchard-port-ds-msaa.md §7b). G9P port untested.

Depth/stencil sampling and D24S8 are NOT implemented. Check stencil base versus depth load, D32F_S8 separate plane layout, D16 background clear. Only attachment usage is advertised.
