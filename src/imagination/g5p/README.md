# G5P: the Apple A9 shader backend

`src/imagination/g5p` compiles NIR to the shader ISA of the Apple A9's GPU
(S8000/S8003, iPad 5): a PowerVR Rogue core (BVNC 9.15.6.2) run by Apple's
own firmware, with Apple's own USC instruction set, the generation before the
A10's G9P. It is not conformant.

Both Apple generations run on one Mesa driver, the upstream PowerVR Vulkan
driver with a backend per generation: the G5P compiler sits beside
`src/imagination/g9p`, and the A9 has a device-info entry
(`common/device_info/apple-a9.h`) with the feature bit `apple_g5p`. The
kernel side is the `powervr` driver's Apple backend for both: it reports the
A9's BVNC, which selects the entry, and takes the A9's jobs as Rogue command
streams with the A9's stream extensions (`vulkan/pvr_arch_g5p.c`,
"command streams"). The A9's paths of the driver are the `pvr_arch_g5p_*.c`
and `pvr_g5p_*` files in `vulkan/`, reached from the shared files through
`PVR_HAS_FEATURE(dev_info, apple_g5p)` and the hooks of
`vulkan/pvr_g5p_hooks.h`. Compute shaders are compiled by the G9P compiler in
its A9 mode (`g9p/g9p_a9*.h`).

None of this is proposed upstream. It was written by AI agents (every commit
carries `Generated-by: LLM`).

## The A8

The Apple A8's GPU (T7000, BVNC 3.10.4.5, `common/device_info/apple-a8.h`)
takes these paths too: its entry sets `apple_g5p` and `apple_g4p`. The
evidence for sharing them:

- Apple's LLVM backend for these GPUs (`libLLVM.dylib`, iOS 16) has one
  processor table with `g4p-b1` (A8), `g5p-b0` (A9) and `g9p-a0`/`g9p-b0`
  (A10), and the features "Enable G4 instructions", "Enable G5
  instructions" and "Enable G9 instructions". The A8 is the base set; the
  A9 adds to it. The A8 has its own scheduling model, the A9 and A10 share
  one.
- On an A8 (iPadOS 15) Metal's compute kernels end the A10 way (`0x8000` at
  a 64-byte boundary after `0xc000` padding), its empty kernel is the A10's,
  and its CDM control stream has the seven-word records this driver writes
  for the A9 (`g5p_cdm.h`).

Where the A8 is known to differ, `apple_g4p` selects its form:

- PDS programs encode DOUT the Rogue way, not in Apple's extended form
  (`PVR_G5P_PDS_FEATURES()`, `vulkan/pvr_g5p.h`): Metal's compute PDS
  programs on an A8 decode to the A10's operands in Rogue's field layout.
- The end of tile's EVENT_PIXEL_PDS_INFO counts the USC shared size in
  Rogue's unit: 0x202 for one target on the A8, 0x402 on the A9
  (`g5p_mrt_eot_info_g4p()`).

- Compute: Apple's A8 residency limits (`apple-a8.h`), the PDS in Rogue's
  layout (`G5P_PDS_DOUT()`), the kernel launch and KERNEL8 max_instances
  as AGXMetalA8 builds them (`pvr_arch_g5p_compute.c`, `g5p_cdm.h`).
- USC (`g5p_usc_g4p`, from Apple's A8 compiler output): the long emitpix,
  the A8 varying iterator, the A8 end of tile (`1e010070`, fed the target's
  address); the background program is the A9's.

Where the A8 differs and this driver has no A8 form yet, it refuses by
name: texture samples, several colour targets, integer targets, 4x, LOAD.

## Origin

The A9 driver is Berkay Kösebay's: the shader backend, the state packing, the
PDS programs and the device work behind them, brought up on an iPad 5 (J71t)
against his own kernel driver and UAPI (`apple-g5p`), OrchardLinux/mesa
PR #7. It was then moved onto the powervr kernel interface; what was not
carried over is listed in that change's description.

## Files

- `g5p_usc.[ch]`: the field encoders, one function per instruction form. Each
  form was measured from Apple's A9 compiler output (`applegpu_g5p`) and
  checked against an independent Python reference.
- `pco_g5p_general.[ch]`: the backend, NIR to G5P: instruction selection,
  register allocation, the vertex and fragment register conventions,
  varyings, textures, multiple render targets, transform feedback stores,
  control flow and registers above r31.
- `g5p_blend.c`, `g5p_recipes.c`: the blend epilogues (built from Apple's
  blend forms) and fixed lowerings.
- `g5p_mrt_out.h`: the output-register plan for one to four colour targets.
- `tests/`: the encoders and lowerings against their references
  (`meson test`, with `-Dbuild-tests=true`).

## Parts ported from G9P

These came from Henrik Klee's G9P compiler and encoders
(`src/imagination/g9p`), adapted to the A9's encodings. The ISA words of the
two generations are identical for every form compared so far.

| Part | G9P source commits | Where here |
|---|---|---|
| Discards in blending programs become a select | `7013b551398`, `7a64e1d1bf4` | `pco_g5p_general.c` |
| Multiply-add fusion | `28cdfceab8e` | `pco_g5p_general.c` |
| If/else and loops on the execution-mask stack; their encoders | `d91427625ff` | `pco_g5p_general.c`, `g5p_usc.c` |
| Registers above r31 (homes) and their moves | `d1ab45daed3` | `pco_g5p_general.c`, `g5p_usc.c` |

## The notes these comments cite

Comments cite Berkay's notes by path, like `docs/gpu/experiments/xfb.md` or
`isa-g5p.md`, and test tools by path, like `tools/gpu-userspace/g5p_usc.py`.
The published ones are in [brkosebay/a9-gpu](https://github.com/brkosebay/a9-gpu)
under the same paths. Some, like the full ISA notes, are not published yet.

## Status

Run on A9 hardware with the apple-g5p kernel driver: the desktop (labwc,
Hyprland) on the GPU, GPU clients through linux-dmabuf, up to four colour
targets, depth/stencil, transform feedback with 4-byte stores, and OpenGL
ES 3.0 through Zink; PR #7 also carries 4x multisampling and compute. On the
powervr kernel interface nothing has run on hardware yet.
