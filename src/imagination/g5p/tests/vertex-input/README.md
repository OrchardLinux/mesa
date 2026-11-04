# vertex-input: a hardware test of vertex input

`vertex-input` is a plain Vulkan 1.1 program (it also uses `VK_KHR_index_type_uint8` when the device has it). It
doesn't depend on the driver, so the same binary checks the G5P (A9) and G9P (A10) paths of the PowerVR driver, or any
other Vulkan device.

```sh
./build.sh /tmp/vertex-input          # glslangValidator, cc, the Vulkan headers and loader
/tmp/vertex-input/vertex-input        # every rung; --list names them, RUNG... runs some
```

Each rung clears a 64×64 linear `B8G8R8A8_UNORM` target, draws pixel-aligned quads or lines whose colour is constant
over each primitive, reads the target back and compares every pixel with a CPU picture, exactly. Lines leave the pixels
around their ends (and a line strip its vertical joins) out of the comparison: those belong to the rasterization rule.

| Rung | What it draws |
|---|---|
| `list` | 8 quads as a triangle list, position (`R32G32_SFLOAT`) and colour (`R32G32B32A32_SFLOAT`) interleaved in one binding |
| `bindings-2` | the same, the position in binding 0 and the colour in binding 1 |
| `bind-offsets` | binding 0 bound 16 bytes into its buffer; the colour at attribute offset 16 of a 32-byte stride |
| `indexed-16` | 8 quads from 16-bit indices |
| `indexed-32` | 32-bit indices, from index 6, with a vertex offset of 4 |
| `indexed-8` | 8-bit indices (skipped without the extension) |
| `instanced` | one quad, 4 instances: `gl_InstanceIndex` moves each 16 pixels and sets its green |
| `instanced-indexed` | the same through 16-bit indices with a vertex offset of 4 |
| `vertex-index` | `gl_VertexIndex >> 2` sets each quad's blue; 4 strip draws from vertex 8 |
| `vertex-index-indexed` | the same through 32-bit indices with a vertex offset of 8 |
| `attribs-16` | 16 attributes over 3 bindings: a position and 15 `R32_SFLOAT`, summed five by five into red, green and blue |
| `strip` | one triangle strip of 3 quads |
| `fan` | two triangle fans of one quad each |
| `lines` | 6 horizontal lines (a line list) |
| `line-strip` | a zigzag line strip |
| `int-imin` `int-imax` `int-umin` `int-umax` `int-isub` `int-ineg` | one integer operation per vertex shader over an 8×8 grid of quads (quad index `gl_VertexIndex >> 2` × `gl_InstanceIndex`, 8 instances, 16-bit indices); each channel is one form of the operation (register pair, small literal, push constant, long literal; `int-ineg` also `abs`), compared with the same arithmetic on the CPU |

The output is one line per rung, `VERTEX-INPUT <rung> PASS|FAIL|SKIP ...` (a failure names its first bad pixel, got and
expected), and then `VERTEX-INPUT RESULT PASS|FAIL <passed> of <run>`; a skipped rung counts as neither. The exit
status is 0 on PASS. `--no-verify` runs every rung without comparing, for a fake device node. `--device N` picks
another physical device.

| `fmt-unorm8` `fmt-bgra8` `fmt-snorm8` `fmt-uscaled8` `fmt-sscaled8` | 8 quads whose colour is `R8G8B8A8_UNORM`, `B8G8R8A8_UNORM`, `_SNORM`, `_USCALED`, `_SSCALED` |
| `fmt-unorm16` `fmt-snorm16` `fmt-uscaled16` `fmt-sscaled16` | the same with `R16G16B16A16_UNORM`, `R16G16_SNORM` (blue and alpha the (0,0,0,1) fill), `R16G16B16A16_USCALED`, `_SSCALED` |
| `fmt-half` | position `R16G16_SFLOAT`, colour `R16G16B16A16_SFLOAT` |
| `fmt-offsets8` `fmt-offsets16` | `R8G8B8A8_UNORM` at byte 9 (across two dwords), `R16G16_UNORM` at byte 14 |
| `fmt-int32` | `R32G32_UINT` and `R32G32_SINT` attributes |
| `inst-first` `inst-first-indexed` | the instanced quad from `firstInstance` 2 (2 instances) and 1 (3 instances, indexed, vertex offset 4): `gl_InstanceIndex` includes the first instance |
| `inst-rate` | a per-instance colour and offset (divisor 1), 4 instances from `firstInstance` 1: instance i reads element 1 + i |
| `inst-divisor0` | the offset per instance, the colour in a binding of divisor 0, from `firstInstance` 2: every instance has element 2's colour (skipped without the zero divisor) |
| `inst-div3` `inst-div-mixed` | 8 instances (offset divisor 1), the colour's red/green and blue/alpha from bindings of divisors 3 and 3 (`firstInstance` 0) and 2 and 5 (`firstInstance` 3): instance i reads element first + i / d |
| `restart-16` `restart-32` `restart-8` | 8 quads as one indexed triangle strip whose quads are cut apart by primitive restart (`0xffff`, `0xffffffff`, `0xff`; the 8-bit one skipped without the extension) |

The `fmt-*` rungs draw colour × scale + bias (push constants) and compare with the format's conversion on the CPU; each
value is chosen a quarter step away from 8-bit rounding (Vulkan lets a colour round to either neighbour; the A9 writes
8-bit targets through f16 rounded toward zero), the format's edges first. A format the device doesn't give
`VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT` skips its rung.

Reference run on lavapipe (llvmpipe, LLVM 22.1.8): `VERTEX-INPUT RESULT PASS 43 of 43`.
