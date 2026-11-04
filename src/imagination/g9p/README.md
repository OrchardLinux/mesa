# G9P: the Apple A10 GPU

`src/imagination/g9p` and the `g9p` paths of `src/imagination/vulkan` drive
the Apple A10's GPU (T8010, iPad 6): a PowerVR Rogue core run by Apple's
own firmware, with Apple's own shader ISA. The PDS programs, control
streams, render passes and descriptors are upstream PowerVR code; the
shader compiler (`g9p_compile.c`, NIR to G9P), the ISA encoders
(`g9p_isa.h`) and the job layouts are G9P's own. Shared code that G9P must
bypass is gated on `PVR_HAS_FEATURE(dev_info, apple_g9p)`.

This is OrchardLinux's fork of Mesa; none of it is proposed upstream.

## The notes these comments cite

Comments cite the A10 notes by file and section, like `ISA.md "Register
allocation"`. They are in
[OrchardLinux/gpu](https://github.com/OrchardLinux/gpu):

- `ISA.md`, `RENDER.md`, `COMPUTE.md`, `MESA.md`, `PROVENANCE.md` and the
  others: `a10/`, with the bring-up guide in `a10/bringup/` (its `mesa.md`
  explains this driver in topic order);
- `tools/…` (for example `tools/pvr-tri.c`, `tools/g9p-cdm.h`,
  `tools/isa/g9p.py`) and `probes/…`: under `tools/a10/` there, the probes
  in `tools/a10/tools/isa/probes/`.

The kernel driver is in [orchard](https://github.com/OrchardLinux/orchard)
(`kernel/patches/drivers/gpu`).

## The OpenGL ES 3.0 enablers (`-Dimagination-g9p-es3`)

An experimental build option, off by default, for the pieces Zink needs to
offer OpenGL ES 3.0 honestly. They are ported from the Apple A9 (G5P)
driver of the iPad 5 port, where each one runs on hardware; the two ISAs
match on every instruction form compared. **None of it has run on an
A10.** With the option off the build is unchanged, byte for byte. The
code is in `g9p_es3_isa.h`, `../vulkan/pvr_g9p_es3.[ch]` and
`../vulkan/pvr_arch_g9p_es3_gfx.h`, called from the G9P paths behind
`PVR_G9P_ES3`; the A9 notes are in
[brkosebay/a9-gpu](https://github.com/brkosebay/a9-gpu)
(`docs/gpu/experiments/`: `mrt.md`, `xfb.md`, `depth-msaa.md`, `es3.md`).

| Piece | On the A9 | What to check first on the A10 |
|---|---|---|
| The physical device reports what the G9P paths do (1, 2 or 4 colour targets, 1 or 4 samples, XFB, the depth formats below) | — | Zink's version and limits (`eglinfo`, `GL_MAX_DRAW_BUFFERS`) |
| 2 and 4 colour targets (8-bit RGBA, all alike, no blending) | exact | 2 targets need no new register value; 4 targets set 0xe3 in cluster 1's words and 0xfa in 0x4070 (the A9's 8-register values); the A9 also changes two words the A10 stream does not carry (A9 W3c8, W6b0) |
| Transform feedback: the vertex shader stores its records, 4-byte stores, up to 4 buffers | exact (2 buffers, 3000 vertices) | that r0 is the vertex index with the first vertex included; counters are resolved while recording (right for Zink's in-order command buffers) |
| D16, D32F, S8, D32F_S8 attachments and stencil tests | exact (D16, D32F, S8, D32F_S8) | the stencil base (your notes: it broke the depth load), the separate stencil plane of D32F_S8 (upstream Rogue interleaves it, F64Z), D16's 16-bit background depth |
| 2D arrays and cube maps | exact | the sample word's dimension nibble and the cube op `087d` (not in the compared forms), the layer stride rule (64 bytes, 16 KiB past 16 KiB), NPOT mip twiddling in the CPU copies |
| 4× MSAA, one colour target: stored samples, PBE resolve, CPU resolve | exact (8 cases) | the per-cluster word 0xd3 (the A9 rule at width 4), the A9's W3c8 (not in the A10 stream), the TA's multisample word (the kernel's dataset code), DOUTU sample rate 2, the PBE's multisample layout 4 and emit `1c09` |

Not built: sampling depth (the A10 keeps depth in a layout of its own; this
is the one ES 3.0 term the G9P paths cannot honour yet), XFB queries,
3D textures, cube arrays, blending with several targets, and, at 4×,
several targets, sample shading, alpha to coverage, sample masks, LOADs and
stored depth.

**4× MSAA on the A10.** On the A9 it is exact in 8 cases (store, resolve,
depth on chip, alpha-to-coverage, sample masks). The render-target half is
upstream's: a dataset with 4 samples, whose TE_AA your kernel already sets
as iOS does (5 for 4 samples, the A9's value too). The rest is userspace
and in the option (`pvr_arch_g9p_es3_gfx.h` "4x MSAA"): ISP_AA mode 4X,
ISP_CTL's low nibble 7, the standard sample positions in 0xfa8, the
per-cluster words for 4 samples, the PBE's multisample layout and emit
`1c09` for stored samples (dword 5 bit 10 for a PBE resolve), DOUTU sample
rate 2 for the background and fragment kicks, ISPCTL upass 7 in the draws,
and a CPU resolve for transfer resolves (`pvr_arch_g9p_es3_transfer.h`).
The 4× surface (16 bytes per pixel, sample s at +4s, twiddled) is the
A9's layout.
