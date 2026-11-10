# tex-blend: a hardware test of textures, samplers and blending

`tex-blend` is a plain Vulkan 1.0 program. It doesn't depend on the driver, so the same binary checks the G5P (A9)
and G9P (A10) paths of the PowerVR driver, or any other Vulkan device.

```sh
./build.sh /tmp/tex-blend          # glslangValidator, cc, the Vulkan headers and loader
/tmp/tex-blend/tex-blend           # every rung; --list names them, RUNG... runs some
```

Each rung clears a 64×64 linear `B8G8R8A8_UNORM` target, draws once, reads the target back and compares every
pixel with a CPU reference:

| Rung | What it draws | Compared |
|---|---|---|
| `tex-nearest` | a 4×4 `R8G8B8A8_UNORM` texture over the whole target, nearest, clamp to edge | exact |
| `tex-npot` | a 3×5 texture, the same | exact |
| `tex-swizzle` | the 4×4 texture through a view swizzled (B, G, R, A) | exact |
| `tex-premult` | a premultiplied 4×4 texture blended `ONE`, `ONE_MINUS_SRC_ALPHA` (a compositor's window) | ±1 |
| `blend-premult` | a constant colour over the left half, `ONE`, `ONE_MINUS_SRC_ALPHA` | ±1, right half exact |
| `blend-straight-over` | `SRC_ALPHA`, `ONE_MINUS_SRC_ALPHA`; alpha `ONE`, `ONE_MINUS_SRC_ALPHA` | ±1 |
| `blend-straight` | `SRC_ALPHA`, `ONE_MINUS_SRC_ALPHA` on colour and alpha | ±1 |
| `blend-dst-color` | `DST_COLOR`, `ZERO` | ±1 |
| `blend-subtract` | `SRC_ALPHA`, `ONE_MINUS_SRC_ALPHA` with `SUBTRACT`; alpha `ONE`, `ONE`, `ADD` | ±1 |
| `blend-constant` | `CONSTANT_COLOR`, `ONE_MINUS_SRC_COLOR` with dynamic blend constants | ±1 |
| `blend-sa-one` | `SRC_ALPHA`, `ONE` on colour and alpha (glmark2 pulsar, terrain) | ±1 |
| `blend-sa-omsa-z1` | `SRC_ALPHA`, `ONE_MINUS_SRC_ALPHA`; alpha `ZERO`, `ONE` (glmark2 desktop) | ±1 |
| `blend-sa-omsa-10` | `SRC_ALPHA`, `ONE_MINUS_SRC_ALPHA`; alpha `ONE`, `ZERO` (vkmark desktop) | ±1 |

Textures are linear and written by the host, so the test doesn't use the transfer path. The ±1 tolerance covers
the blend arithmetic's rounding, which belongs to the hardware.

The output is one line per rung, `TEX-BLEND <rung> PASS|FAIL ...` (a failure names its first bad pixel, got and
expected), and then `TEX-BLEND RESULT PASS|FAIL <passed> of <run>`. The exit status is 0 on PASS. `--no-verify` runs
every rung without comparing, for a fake device node. `--device N` picks another physical device.

Reference run on lavapipe (llvmpipe, LLVM 22.1.8): `TEX-BLEND RESULT PASS 13 of 13`. The largest difference was 0 on
the texture rungs and 1 on the blend rungs.
