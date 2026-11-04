# G9P msaa (AI-written)

Dependency: `es3-depthstencil (MRT device programs and depth/ISP setup)`. Enable `-Dimagination-g9p-es3=true` in Mesa; default is off.
This branch has not run on A10. A9 evidence is evidence for the source implementation, not this G9P port.

## Build and run on A10

Needs a C compiler, Vulkan headers/loader, and glslangValidator (except limits).
Use an A10 render node and explicitly select the ICD installed from this branch:

```sh
cd src/imagination/g9p/tests/msaa
make
export VK_DRIVER_FILES=/path/to/powervr_mesa_icd.aarch64.json
build/msaa-vk --case one,resolve,depth,store,resolvecmd
```

Expected: each supported case `result=EXACT bad=0`, final `result=PASS`. Exit status must be zero; do not use `--offline` for hardware proof.
No A9 kernel interface, device scripts, Apple programs, generated SPIR-V, or version override is required.

## Evidence and A10 checks

A9 ICD: eight cases exact (docs/gpu/experiments/msaa.md). G5P shared-driver 2026-11-04: `G5P-1B-DSMS RESULT PASS depth cube16 clip bias size1 one; 4x resolve depth` (docs/gpu/orchard-port-ds-msaa.md §7b). Stored-sample/CPU resolve proof belongs to the ICD, not that shared-driver run.

Untested A10 assumptions: cluster word 0xd3, missing W3c4/W3c8 equivalents, DOUTU rate 2, PBE multisample layout 4/emit 1c09/resolve bit, CPU sample layout. Verify the kernel 4-sample HWRT dataset and Apple sample positions; a sample-pattern override may be needed. No sample masks, alpha-to-coverage, multisample LOAD, stored depth or multiple colour targets. Not full ES3 MSAA support.
