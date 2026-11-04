# G9P ES 3.0 rungs

Small hardware tests for the `-Dimagination-g9p-es3=true` feature branches
in the OrchardLinux Mesa fork. Each one checks a feature against a
CPU reference and prints PASS/FAIL. They come from the Apple A9 (G5P) port
of the iPad 5. Hardware proof is stated per case below, with fake-node-only cases
labelled. **None of them has run on an A10.**

They use plain Vulkan, or EGL/GLES through Zink. There is no DRM ioctl, no `/dev`
path, no GPU dump and no environment hook that changes the driver's behaviour.
Only the GLSL is shipped. The Makefile turns it into SPIR-V headers.

**Packaging:** each feature branch carries its own primary Vulkan program under
`src/imagination/g9p/tests/<feature>/`. This combined kit also keeps the GLES
and depth-sampling diagnostics for work spanning several features.

## Build (native aarch64 Linux)

Dependencies: a C compiler, GNU make, `glslangValidator` (glslang), the Vulkan
headers and loader (`libvulkan`), and EGL + GLESv2 headers and libraries (Mesa's).
No Python is needed: every program checks its own results.

```sh
make            # all eight programs into build/; make vk / make gles builds half
make clean      # removes build/ (the generated SPIR-V headers too)
```

## Local feature branches (2026-11-04)

The primary per-feature programs are copied into Mesa at
`src/imagination/g9p/tests/<feature>/`, with a feature-only Makefile and README.
This combined kit remains a development superset, not a promise that every
program passes every branch. All eight programs build on aarch64 Linux.

| Branch | Dependency | Primary program |
|---|---|---|
| `es3-limits` | upstream main `73a49abe247` | Mesa's `tests/limits` query probe (separate from these eight rendering programs) |
| `es3-mrt` | `es3-limits` | `mrt-vk` |
| `es3-xfb` | `es3-mrt` shared graphics/ISA helpers | `xfb-test` |
| `es3-depthstencil` | `es3-mrt` shared draw/clear helpers | `depth-vk` |
| `es3-textures` | `es3-mrt` shared graphics/ISA helpers | `tt-vk` |
| `es3-msaa` | `es3-depthstencil` depth/ISP and MRT device programs | `msaa-vk` |

`ds-sample-vk` is the eighth program. It renders D32F / D32F_S8, samples the
depth plane raw and with comparison, and samples stencil. Build: `make build/ds-sample-vk`;
run: `build/ds-sample-vk`; inspect its `result=EXACT bad=0` lines and final PASS.
It verifies the later A9 sampling port (lane c, `docs/gpu/orchard-port-ds-msaa.md`
§G10). **None of these G9P branches implements depth/stencil sampling or D24S8**,
so this is a diagnostic for future G9P work, not an expected PASS on these branches.
Do not use an ES version override or `--offline` as hardware proof.

The generic `es3-smoke` and `ds-ladder` span multiple features: run MRT/XFB
subtests only after the corresponding branches are combined, and attachment
rungs after the depth/stencil branch. `zs` still requires the missing sampling path.

## Run

The Vulkan programs need the option-on ICD. A run of a program is one device; run the
programs one at a time.

```sh
export VK_ICD_FILENAMES=/path/to/install/share/vulkan/icd.d/powervr_mesa_icd.aarch64.json   # or VK_DRIVER_FILES
build/mrt-vk --case one,mrt2,mrt4
build/mrt-vk --case one,mrt4-vary
build/xfb-test                       # the suite
build/xfb-test --run3
build/xfb-test --ctl; build/xfb-test --dup=3,sep,nocap; build/xfb-test --dup=3000   # the dup ladder, one job each
build/depth-vk
build/msaa-vk --case one,resolve,depth,store,resolvecmd # es3-msaa only; other branches use --expect-no-msaa
build/tt-vk
```

The GLES programs are **surfaceless**. They use `EGL_MESA_platform_surfaceless`
with no config and draw into FBOs, so they need no Wayland display. Run them with
Zink on the same ICD:

```sh
export MESA_LOADER_DRIVER_OVERRIDE=zink GALLIUM_DRIVER=zink VK_ICD_FILENAMES=...
export MESA_GLES_VERSION_OVERRIDE=3.0   # see below
build/es3-smoke --only mrt,xfb          # zs samples depth: see below
build/ds-ladder --list | while read r; do build/ds-ladder "$r"; done
```

`MESA_GLES_VERSION_OVERRIDE=3.0` is needed because Mesa's ES 3.0 check requires
`ARB_depth_buffer_float` and `OES_depth_texture_cube_map`. The series does not sample
depth, so Zink may offer only ES 2.0 and the 3.0 context request fails (`no ES 3.0
context`). Run `eglinfo` or check `GL_VERSION` first: if Zink already offers 3.0,
leave the override out. The override is a Mesa debug variable. It changes the version
the context reports, not what the driver does.

Exit codes: 0 pass, 1 mismatch, 2 a guard byte changed (`mrt-vk`, `depth-vk`), 3 setup
error, 4 a job refused or failed. The printed VkResult tells which: -8 refused, -4
device lost, 2 fence timeout. `--offline` on every program is for a driver that
executes nothing: only acceptance is judged.

## The rungs

Each entry lists what the rung tests, what PASS looks like, what the A9 printed and
what to check first on the A10. "A9: hardware" means the A9 ran it on the GPU.

### `mrt-vk`: 1, 2 and 4 colour targets (A9: hardware; A10: untested)

Each case runs 3 jobs on 64x64 linear BGRA8 targets. Every target sits in its own
allocation with 16 KiB guards on both sides. The model is exact per pixel: the quad
gets its output, everything else gets the target's clear.

PASS: one `job=` line per job ending `rtN:bad=0,guard=0 ... result=EXACT`, then
`result=PASS`. On the A9:

```
device=Apple A9 G5P ... maxColorAttachments=4 maxFragmentOutputAttachments=4
case=mrt4-vary targets=4 jobs=3 row_pitch=256 inferred=register-form-outputs
  job=0 quad=16,24-48,56 ms=18.3 rt0:bad=0,guard=0 rt1:bad=0,guard=0 rt2:bad=0,guard=0 rt3:bad=0,guard=0 result=EXACT
result=PASS
```

On the A10, check first:
- The header line shows `maxColorAttachments=4`. With the option off, run `--expect-max 1`: every several-target case must be refused before submit.
- `mrt2` passes but `mrt4` does not: check the 4-target words your README lists (0xe3 in cluster 1, 0xfa in 0x4070).
- `foreign=` counts pixels that hold another target's clear or output: the per-target PBE/output-half offset is wrong. `sentinel=` counts pixels never written: the target's store did not happen.
- `mrt4-vary` alone fails: the register-form output writes (halves 4..15).

### `xfb-test`: transform feedback (A9: see each mode; A10: untested)

Every capture buffer is filled with 0xa5 and bound 32 bytes in, so a stray byte
anywhere is caught. Records are `(id, ~id, float(id), 1.0)`.

- **`--run3`** (A9: hardware): record, overflow (48-byte binding: 1 triangle fits), discard and 3000-vertex dup, using the attribute-less cover VS. A9 lines:
  ```
  r3=record pixels=4096/4096 records=6 counter=96/96 lanes=0/0/0/0 outside=0 query=2/2 want=2/2 result=PASS
  r3=overflow pixels=4096/4096 records=3 counter=48/48 lanes=0/0/0/0 outside=0 query=1/2 want=1/2 result=PASS
  r3=discard pixels=0/0 records=6 counter=96/96 lanes=0/0/0/0 outside=0 query=2/2 want=2/2 result=PASS
  r3-result record=PASS overflow=PASS discard=PASS dup=PASS result=PASS
  ```
  The G9P option has no XFB queries, so it prints `query=skipped` in place of `query=...`.
- **`--ctl`, `--dup=N[,sep][,nocap]`** (A9: hardware, every rung below): one job per process. Run them in this order: `--ctl`, `--dup=3,sep,nocap`, `--dup=3,sep`, `--dup=3`, `--dup=3000,sep,nocap`, `--dup=300,sep`, `--dup=1020,sep`, `--dup=3000,sep`, `--dup=3000`. After any FAIL, run `--ctl`. A9 lines (the A9 copy also printed `splits=`):
  ```
  dl=ctl result=PASS
  dl=3000 vertices=3000 splits=6 vb=alias cap=1 counter=48000/48000 exact=3000 wrong=0 first_wrong=-1 tail_bad=0 once=3000 twice=0 more=0 result=PASS
  ```
- **Suite** (no option; `--case=NAME` for one): `rec`, `first3`, `overflow`, `instanced`, `attr`, `discard`, `resume`, `dup`, then `result=PASS`. `attr16` runs only with `--case=attr16`. On the A9, `attr` and `attr16` ran on hardware (`case=attr records=6 counters=96/192 result=PASS`, `case=attr16 records=6 counters=96/96 result=PASS`). The rest of the suite ran only on a fake device there.
  - Without `transformFeedbackQueries`, `overflow` and `instanced` print `query=skipped(...)`.
  - Without `transformFeedbackDraw`, `resume` drops its third pass (`vkCmdDrawIndirectByteCountEXT`) and prints `bytecount=skipped(...)`. Both are expected with this series.

On the A10, check first:
- **`first3`**: it answers your README's first XFB question, whether r0 is the vertex index with the first vertex included. It wants records 3..8. A FAIL with `first_bad=32` holding 0 means the index started at 0.
- `lanes=a/b/c/d` counts bad bytes per dword of the records, so one lane per 4-byte store. A single bad lane means one store offset is wrong. All four lanes bad, with `outside=0`, means the stores never landed.
- `attr` checks the second buffer (stride 32) and `attr16` the same at stride 16. On the A9, records were exact while positions all landed in slot 0: the stride-32 multiply gave 0.
- `discard pixels_with>0`: rasterizer discard drew pixels. Records missing only under discard: the stores commit only when something is rasterized.
- A wedge or fence timeout on `dup` (3000 vertices) but not on `--dup=3`: vertex-data splits or the vertex PDS. On the A9 it was the PDS's LAST flag.

### `depth-vk`: D16, D32F, S8, D32F_S8 (A9: hardware, all 18 jobs; A10: untested)

The colour target is checked against a model of the depth/stencil result. Depth is
never read back or sampled, which matches the series.

| Case | What it draws and expects |
|---|---|
| `depth` | one job: D32F, LESS, nearest wins |
| `reload`, `reload16` | 3 jobs with LOAD between them: D32F and D16 |
| `stencil` | D32F_S8: Replace, then Equal, then NotEqual, across passes |
| `bias` | constant depth bias of -64 |
| `size` | 100x75: NPOT, not 32-aligned |
| `cube16` | vkcube's depth: D16, LessEqual, store DONT_CARE |
| `clip` | D32F_S8 in one job (no LOAD): Replace in a rect, then Equal and NotEqual (a stencil clip). Shared G5P driver PASS on A9, 2026-11-04; G9P untested |
| `size1` | 100x75 in one job (no LOAD). Shared G5P driver PASS on A9, 2026-11-04; G9P untested |

PASS: every `case=... job=N result=EXACT bad=0 guard=0`, then `result=EXACT`. A9
lines include `depth formats=4/4 D24S8=0`, `case=reload16 job=1 result=EXACT bad=0
guard=0 footprint=8192 size=16384` and `case=size job=0 result=EXACT bad=0 guard=0
footprint=61440 size=65536`. `footprint=` is only information: it is the highest
byte the store wrote.

On the A10, check first:
- `reload` fails on job 1 (red lost): the depth load. Your notes say a stencil base breaks it, so compare with `stencil`.
- `stencil` job 1 or 2 is wrong: the separate stencil plane of D32F_S8 or the stencil base. The A9 layout is a plane of its own in the second half; Rogue interleaves it as F64Z.
- `reload16` fails while `reload` passes: D16's 16-bit background depth (1.0 = 0xffff).
- `guard>0`: the store wrote outside the image.

### `msaa-vk`: 4x MSAA (A9 source: eight hardware cases; G9P port: untested)

Branch `es3-msaa` includes code, behind `-Dimagination-g9p-es3=true`, for one
colour target with 1 or 4 samples, multisample store, PBE resolve, CPU transfer
resolve and depth kept on chip. Build its own Mesa branch; run:

```sh
build/msaa-vk --case one,resolve,depth,store,resolvecmd
```

Expected: sample-count mask `0x5`, each supported case `result=EXACT bad=0`, and
final `result=PASS`. `raw_bad` examines the A9 sample layout directly and is
additional diagnostic information on A10. Every result here is still untested on A10.
The default full suite also includes `a2c`, `smask` and `smaskshader`; those are
not implemented by this G9P port and must not be described as passing.

Other feature branches report one sample. On those branches use
`build/msaa-vk --expect-no-msaa`: 1x exact, seven 4x cases skipped, final PASS.

A9 evidence: the standalone ICD passed all eight cases pixel-exactly
(`docs/gpu/experiments/msaa.md`). The shared G5P driver passed `one,resolve,depth`
on 2026-11-04 (`docs/gpu/orchard-port-ds-msaa.md` §7b). This is not evidence
that the G9P implementation works.

A10 first checks: HWRT dataset TE_AA / multisample positions / region sizes;
cluster word 0xd3 and missing W3c4/W3c8 equivalents; DOUTU sample rate 2;
PBE layout 4, emit 1c09 and resolve bit; CPU sample layout. The existing UAPI
can carry the fields, but the kernel's sample-pattern override may need work.
Multisample LOAD, stored depth, alpha-to-coverage, sample masks and several
colour targets at 4x are refused.

### `tt-vk`: 2D arrays and cube maps (A9: hardware; A10: untested)

Cases: `one`, `arr`, `arr-layer`, `cube`, `npot`, `npot-lod`, `arr-mip`. Every pixel is
one exact texel (nearest, mip-nearest). PASS: `case=NAME result=EXACT bad=0 maxdiff=0`
for each case, then `result=PASS`. The A9 printed exactly that for all seven. `cube`,
`npot-lod` and `arr-mip` passed only after the fixes below.

On the A10, check first:
- `cube` BAD with blue 0 at row 0: the direction reached the sampler as 2D coordinates. The cube op (`087d`) is missing. Blue = 40 x face read (want 120, face 3 = -Y).
- `arr-mip` BAD while `arr` passes: the layer stride of a mipmapped array. On the A9 it is the level chain rounded to 64 bytes, or to 16 KiB once it reaches 16 KiB.
- `npot-lod` BAD: NPOT mip twiddling in the CPU copies. On the A9, each level is twiddled over its own extent padded to powers of two.
- `arr-layer` BAD: the layer from the coordinate (float to integer, round-to-nearest).

### `es3-smoke`: ES 3.0 through Zink (A9: hardware; A10: untested)

Features: `mrt` (4 RGBA8 draw buffers), `xfb` (one vec4 per vertex under rasterizer
discard), `zs` (D24S8 texture as the depth attachment, then sampled plain and with
LEQUAL compare). The A9 printed:

```
SMOKE ctx  PASS OpenGL ES 3.0 Mesa 27.0.0-devel glError=0x0
SMOKE mrt  PASS status=0x8cd5 program=1 t0=4080bfff t1=ff0000ff t2=00ff00ff t3=0000ffff (want 40{80|7f}bfff ff0000ff 00ff00ff 0000ffff) glError=0x0
SMOKE xfb  PASS program=4 mapped=1 v0=(0.5,1.5,2.5,3.5) v2=(8.5,9.5,10.5,11.5) exact=1 (want v0=(0.5,1.5,2.5,3.5) v2=(8.5,9.5,10.5,11.5)) glError=0x0
SMOKE zs   PASS status=0x8cd5/0x8cd5 plain in=80 out=ff shadow in=00 out=ff (want plain 7f|80/ff, shadow 00/ff) glError=0x0
RESULT PASS 0 failure(s)
```

On the A10, check first:
- **`zs` is expected to FAIL with this series**, because sampling depth is not built. Run `--only mrt,xfb`.
- `SMOKE ctx FAIL`: Zink did not offer ES 3.0 (see the override above).
- `mrt` fails with a GL error or a `status` other than 0x8cd5 (complete): Zink offers fewer than 4 draw buffers. Check `GL_MAX_DRAW_BUFFERS` and `GL_MAX_COLOR_ATTACHMENTS`. If only `t1..t3` are wrong, compare with `mrt-vk`.
- `xfb` with `mapped=1 exact=0` and every value `-7`: no record was stored. This rung is Zink's path to the same stores `xfb-test` checks.

### `ds-ladder`: depth/stencil under blending and discard (A9: hardware except `ds_clear_rect`; A10: untested)

These are the `ds_*` rungs of the A9 blend ladder, plus `ctl`. Each runs on a 256x128
RGBA8 target with a `GL_DEPTH24_STENCIL8` renderbuffer. The series has no D24S8, so
Zink renders it as D32F_S8. Blending is only on the single target. Run one rung per
process. PASS: `RUNG <name> PASS bad_pixels=0 max_error<=1 glError=0x0`. On the A9:

```
RUNG ds_premult PASS bad_pixels=0 max_error=1 glError=0x0 err_hist(0/1/2/3+) r=29998/2770/0/0 g=31290/1478/0/0 b=31268/1500/0/0 a=30002/2766/0/0
```

`ctl`, the five Qt shapes (`ds_premult`, `ds_premult_load`, `ds_qs_text`,
`ds_qs_text_load`, `ds_clip_premult`) and all ten `ds_discard*` rungs passed on A9
hardware. `ds_clear_rect` ran only on a fake device there.

On the A10, check first:
- `ctl` FAIL: the harness (context, FBO, texelFetch) is broken, so ignore the rest.
- `*_load` fails while its no-load twin passes: depth/stencil LOAD between passes. The same question as `depth-vk reload`.
- `ds_clip_premult` fails: a stencil write with colour mask 0, then stencil EQUAL.
- `ds_discard_depth_write_order` fails with the probe colour hidden: a discarded fragment still wrote depth. Check how your discarding (punch-through) programs and ZLS interact.
- `ds_discard_stencil_faces` fails on one half: the per-face stencil state (ISPA/ISPB per face).

## Files

`Makefile`; `shaders/common/` (pass, solid, color, uv, tex); `mrt/`, `xfb/`, `depth/`,
`msaa/`, `textures/` (Vulkan); `gles/` (EGL/GLES). The sources are MIT. `.gitignore`
keeps `build/`, with its generated SPIR-V headers, out of git.
