# G9P limits (AI-written)

Dependency: `OrchardLinux/mesa main @ 73a49abe247`. Enable `-Dimagination-g9p-es3=true` in Mesa; default is off.
This branch has not run on A10. A9 evidence is evidence for the source implementation, not this G9P port.

## Build and run on A10

Needs a C compiler, Vulkan headers/loader, and glslangValidator (except limits).
Use an A10 render node and explicitly select the ICD installed from this branch:

```sh
cd src/imagination/g9p/tests/limits
make
export VK_DRIVER_FILES=/path/to/powervr_mesa_icd.aarch64.json
build/limits
```

Expected: `G9P-LIMITS RESULT PASS`. Exit status must be zero; do not use `--offline` for hardware proof.
No A9 kernel interface, device scripts, Apple programs, generated SPIR-V, or version override is required.

## Evidence and A10 checks

No A9 hardware claim applies to A10 physical-device limits. This commit constrains existing G9P reporting; later feature branches raise their own limits.

Verify the physical-device reports on A10. This is a query test, not a rendering or ES3 conformance claim.
