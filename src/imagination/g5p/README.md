# G5P: the Apple A9 shader backend

`src/imagination/g5p` compiles NIR to the shader ISA of the Apple A9's GPU
(S8000/S8003, iPad 5): a PowerVR Rogue core (BVNC 9.15.6.2) run by Apple's
own firmware, with Apple's own USC instruction set, the generation before the
A10's G9P. It is built only with `-Dimagination-g5p-experimental=true` and is
not conformant.

This is step 1a of the plan to run both Apple generations on one Mesa driver
built on the upstream PowerVR Vulkan driver, with a backend per generation:
the G5P compiler sits beside `src/imagination/g9p`, and the A9 has a
device-info entry (`common/device_info/apple-a9.h`) with the feature bit
`apple_g5p`. Nothing of G9P is changed. Nothing in the PowerVR driver uses
this backend yet; step 1b moves the A9 onto the `pvr_*` driver behind
`PVR_HAS_FEATURE(dev_info, apple_g5p)`.

None of this is proposed upstream. It was written by AI agents (every commit
carries `Generated-by: LLM`).

## Files

- `g5p_usc.[ch]`: the field encoders, one function per instruction form. Each
  form was measured from Apple's A9 compiler output (`applegpu_g5p`) and
  checked against an independent Python reference.
- `pco_g5p_general.[ch]`: the general backend, NIR to G5P: instruction
  selection, register allocation, the vertex and fragment register
  conventions, varyings, textures, multiple render targets, transform
  feedback stores, control flow and registers above r31.
- `g5p_blend.c`, `g5p_recipes.c`: the blend epilogues (built from Apple's
  blend forms) and fixed lowerings.
- `g5p_mrt_out.h`: the output-register plan for one to four colour targets.
- `pco_g5p.[ch]`: the first, captured-shape compiler, which reads PCO IR
  before Rogue scheduling. It is kept for the shapes it matches.

## Parts ported from G9P

These came from Henrik Klee's G9P compiler and encoders
(`src/imagination/g9p`), adapted to the A9's encodings. The ISA words of the
two generations are identical for every form compared so far.

| Part | G9P source commits | Where here |
|---|---|---|
| Discards in blending programs become a select | `7013b551398`, `7a64e1d1bf4` | `pco_g5p_general.c` |
| Multiply-add fusion (behind `PVR_G5P_FUSE=ffma`) | `28cdfceab8e` | `pco_g5p_general.c` |
| If/else and loops on the execution-mask stack; their encoders | `d91427625ff` | `pco_g5p_general.c`, `g5p_usc.c` |
| Registers above r31 (homes) and their moves | `d1ab45daed3` | `pco_g5p_general.c`, `g5p_usc.c` |

## The notes these comments cite

Comments cite our notes by path, like `docs/gpu/experiments/xfb.md` or
`isa-g5p.md`, and test tools by path, like `tools/gpu-userspace/g5p_usc.py`.
The published ones are in [brkosebay/a9-gpu](https://github.com/brkosebay/a9-gpu)
under the same paths. Some, like the full ISA notes, are not published yet.

## Status

Proven on A9 hardware with our own ICD and kernel interface (not in this
tree yet): the desktop (labwc, Hyprland) on the GPU, GPU clients through
linux-dmabuf, up to four colour targets, depth/stencil, transform feedback
with 4-byte stores, and OpenGL ES 3.0 through Zink. Control flow ran on the
hardware in a short ladder; its desktop use, registers above r31 and FMA
fusion are checked offline only. Not done: compute.
