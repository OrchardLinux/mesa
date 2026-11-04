# formats: a hardware test of the colour formats a device advertises

`formats` is a plain Vulkan 1.0 program. It doesn't depend on the driver, so the same binary checks the G5P (A9)
and G9P (A10) paths of the PowerVR driver, or any other Vulkan device.

```sh
./build.sh /tmp/formats            # glslangValidator, cc, the Vulkan headers and loader
/tmp/formats/formats               # every rung the device advertises; --list names them
/tmp/formats/formats 'r-*' s-R8_UNORM    # some rungs (a trailing * is a prefix)
/tmp/formats/formats --coverage    # also fail on any advertised format/feature no rung checks
```

For each of 61 colour formats (8-bit RGBA and its sRGB forms, R/RG 8, snorm, 16-bit unorm/snorm, half and single
float, 10_10_10_2, 11_11_10 and 9_9_9_5 float, 565 and 4444, every integer format, the ten ETC2/EAC formats), it runs
the rungs that the device's optimal-tiling features ask for, and skips the rest as "not advertised":

| Rung | Feature | What it does |
|---|---|---|
| `s-<fmt>` | `SAMPLED_IMAGE` | an 8×8 texture uploaded by `vkCmdCopyBufferToImage`, sampled nearest over a 64×64 `B8G8R8A8_UNORM` target. Integer formats: `texture()` of an integer sampler, one rung per byte of the channel width (`-8`, `-16`, `-24` suffixes) |
| `f-<fmt>` | `SAMPLED_IMAGE_FILTER_LINEAR` | the same texture sampled halfway between two texels (weights ½, ½) |
| `m-<fmt>` | `SAMPLED_IMAGE` | a two-level texture, level 1 by `textureLod()` (R8, RG8, RGBA8 sRGB, RGBA16F, RGBA32F, RG11B10F, ETC2 RGB8: each texel size) |
| `x-<fmt>` | `SAMPLED_IMAGE` | `texelFetch()` of an integer texture (R32UI, RGBA8UI) |
| `r-<fmt>` | `COLOR_ATTACHMENT` | a clear, then a constant colour over the left half, read back by `vkCmdCopyImageToBuffer` |
| `b-<fmt>` | `COLOR_ATTACHMENT_BLEND` | the same, blended `SRC_ALPHA`, `ONE_MINUS_SRC_ALPHA` (alpha `ONE`, `ONE_MINUS_SRC_ALPHA`) |
| `k-<fmt>` | `BLIT_SRC` (ETC2/EAC) | an 8×8 compressed image blitted 1:1, nearest, into `B8G8R8A8_UNORM` (`R16G16B16A16_SFLOAT` for signed EAC), read back |

Render rungs exist for the formats a four-f16 fragment output can fill: 8-bit RGBA and sRGB, 10_10_10_2, R8, RG8,
half float and 11_11_10 float. A device that advertises a colour attachment of any other format (integers, for
example) fails `--coverage`, because nothing here checks it. `--coverage` also requires a `k-` rung for every ETC2/EAC
format that advertises `BLIT_SRC` (Vulkan's `textureCompressionETC2` requires it); blits of uncompressed formats are
the transfer path's, not a format's, and are not checked here.

Every pixel is compared **exactly** with a CPU reference (the decoders are in `formats.c`, ETC2/EAC included: the
test writes ETC2 blocks in individual and differential mode, opaque, and EAC blocks, and decodes them itself). The texel values are
drawn from a fixed seed and kept only where the expected 8-bit output is at least a quarter step from a rounding
boundary, because an implementation may write 8-bit UNORM through f16 rounded toward zero (the A9 does, up to an
eighth of a step low). sRGB channels need 0.4 of a step, and linearly filtered sRGB pairs are close enough that
filtering before or after the decode gives the same byte, because implementations decode sRGB less precisely than
the exact curve. Render and blend values are exact in f16 and in the 11/10-bit floats, and blend results too
(source alpha ¼, destination 0 or 1).

The output is one line per format with what the device advertises (`S`ampled, `F`iltered, `C`olour attachment,
`B`lend), then one line per rung, `FORMATS <rung> PASS|FAIL|SKIP ...` (a failure names its first bad pixel, got and
expected bytes), and then `FORMATS RESULT PASS|FAIL <passed> of <run> (<n> not advertised)`. The exit status is 0
on PASS. `--no-verify` runs every rung without comparing, for a fake device node. `--device N` picks another
physical device.

Reference run on lavapipe (llvmpipe, LLVM 22.1.8): `FORMATS RESULT PASS 173 of 173 (0 not advertised)`. Signed EAC
blits keep their values a quarter f16 ulp from rounding, because implementations decode signed R11 with different
precision.
