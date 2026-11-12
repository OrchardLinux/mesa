# triangle: a hardware test of the G5P draw path

`triangle` is a plain Vulkan program with no driver internals, so the same binary checks the G5P (A9) and G9P
(A10) paths of the PowerVR driver, or any other Vulkan device.

```sh
./build.sh /tmp/triangle                     # glslangValidator, cc, the Vulkan headers and loader
/tmp/triangle/triangle --triangle --verify   # one cleared 64×64 target with one red triangle, checked
```

- `--triangle --verify`:
  1. It clears a 64×64 linear `B8G8R8A8_UNORM` image (pre-filled with `0x5a`).
  2. It draws one red triangle from a vertex buffer, with a negative-height viewport, covering (8,56) (56,56) (32,8).
  3. It reads the image back and checks every pixel at least 1 px from an edge: red inside, the clear colour outside.
  4. The edge band is reported apart and may hold only those two colours.

  Verdict: `G5P-PVR-TRIANGLE PASS|FAIL`.
- `--render --verify`: one clear-only pass, every pixel compared (`G5P-PVR-RENDER`).
- `--load --verify`: two passes on one image. The second pass has `loadOp = LOAD` and draws a green triangle beside
  the first pass's red one; only a real load keeps the red one (`G5P-PVR-LOAD`). This belongs to the follow-up LOAD
  PR.
- `--cycles N`: creates and destroys the device N times in one instance.
- `--submit`: also two empty submissions chained by a semaphore.
- `--bos PATH`: reads our A9 kernel's BO accounting from debugfs. Leave it out on other devices.

The shaders are GLSL (`shaders.glsl`), compiled by `build.sh` with glslangValidator. No SPIR-V binary is committed.
Not built by meson.
