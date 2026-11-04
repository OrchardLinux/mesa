# G9P xfb (AI-written)

Dependency: `es3-mrt (shared graphics helpers and ISA encoders)`. Enable `-Dimagination-g9p-es3=true` in Mesa; default is off.
This branch has not run on A10. A9 evidence is evidence for the source implementation, not this G9P port.

## Build and run on A10

Needs a C compiler, Vulkan headers/loader, and glslangValidator (except limits).
Use an A10 render node and explicitly select the ICD installed from this branch:

```sh
cd src/imagination/g9p/tests/xfb
make
export VK_DRIVER_FILES=/path/to/powervr_mesa_icd.aarch64.json
build/xfb-test --case=attr
build/xfb-test --case=attr16
build/xfb-test --case=first3
build/xfb-test --dup=3000
```

Expected: `result=PASS`; unsupported query/byte-count subchecks may print skipped. Exit status must be zero; do not use `--offline` for hardware proof.
No A9 kernel interface, device scripts, Apple programs, generated SPIR-V, or version override is required.

## Evidence and A10 checks

A9 source: two buffers / 3000 vertices exact (docs/gpu/experiments/xfb.md); attr/attr16 ran on hardware. first3 and the general record/counter suite were fake-node only; no claim of hardware proof for those.

Verify r0 includes firstVertex (`first3`), all four 4-byte store lanes, second-buffer stride and instance/counter handling. Queries and DrawIndirectByteCount are not implemented. The limits-only query test inherited from the base is intentionally superseded after features are enabled.
