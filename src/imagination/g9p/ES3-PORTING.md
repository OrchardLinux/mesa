# ES3 follow-through from the A9 port

Generated-by: LLM. Adapted driver structure follows the G9P port.

This is an implementation starting point and a test package for G9P, not an A10
hardware result. The series preserves all of the older ten-commit G9P adaptation:
limits, MRT compiler outputs and device programs, XFB vertex stores/bindings,
depth/stencil attachments, array/cube sampling and image layouts. It also retains
the later partial MSAA implementation. Nothing here has run on an A10.

The new tests use public Vulkan only. They contain GLSL and generate their own
SPIR-V at build time. No A9 kernel ABI, device scripts, Apple program binaries or
captured arrays are included. Select the candidate ICD explicitly before running.

## A9 evidence and the first A10 checks

Pass counts below describe A9 results. They do not qualify these G9P paths.
Run names are retained so a missing result is not replaced with an inference.

| Piece | A9 evidence | Check first on A10 | G9P status in this series |
|---|---|---|---|
| MRT | `mrt-3331b2b41c2`: one/mrt2/mrt4 exact; `mrt-e77c2137632`: same; `mrt-3331b2b41c2-r2`: 4 rungs including mrt4-vary | MRT2 first, then MRT4 output width and per-cluster words; A9's W3c8/W6b0 have no direct A10 stream slot | adapted implementation and primary test retained |
| XFB queries + discard | `xfb-eb1aa2ce790`, run `xfb-20261104T143005Z`: rec/two/overflow/resume/query/discard, **6/6 exact** | Capture/counters before queries; whole-primitive overflow; resubmission and query availability; discard must suppress colour/depth while keeping vertex stores | existing XFB retained; new six-case `tests/xfb-stream` probe; query/discard implementation remains follow-up |
| Sampled D32F/D32F_S8 + compare | `dsms-6dec648c9d4`, run `dsamp-20261104T144703Z`: depth control plus raw/cmp/raw8/cmp8/stencil8, **6/6** | Stored depth layout and plane base first; raw sampling before compare operand order | attachment implementation retained; typed sampling recipe below and `tests/depthstencil/build/ds-sample-vk` |
| Vertex input | `vtxin-71fa6eb573e`, `device-run-13`: **45/45** | Fetch byte/norm conversion, firstInstance, divisor 0/1/>1, restart sentinels, point-size output ordering | recipe and API-only `tests/vertex-input`; no untested vertex-format/divisor caps added |
| Formats slice 1 + ETC2/EAC | `formats-aaabb7a16da/device-run`: **118/118**, then **39/39 probes**, every rung 4096 pixels exact | G9P texture/PBE format interpretation; integer sampler result bits; compressed block order and mip padding | exact slice-1 test source in `tests/formats`; no G5P descriptor table or cap promotions copied |
| 4x MSAA | `dsms-1aff46dae07/device-run-ms`: **8/8**, one/resolve/depth alongside depth controls; old standalone implementation has additional cases | Kernel render-target dataset/sample positions before userspace PBE/EOT sample-rate state | partial G9P implementation retained; test its supported subset first, not full sample-mask/a2c support |
| CPU transfer then vkMapMemory | G5P fix `0eed6eda1c3` (publication `327241fcbd1`), encountered after CPU texture uploads | Copy to previously unmapped memory, then map/read exact bytes; buffer and image | normal-map fix adapted, `tests/memory` regression added; no new A10 run |
| Duplicate GEM import | G5P `c8b91b29e97`, fixes the compositor's first-frame duplicate-handle refusal | Export a BO and re-import it; same DRM file must acquire the existing BO, not a second wrapper/binding | already implemented in G9P's DRM winsys; no duplicate patch added |

The map fix and duplicate-import fix are traced to concrete code changes and
observed failures. No separate A9 numerical pass count is invented for them.
The depth/stencil test's stencil8 result is extra evidence, not a claim that
sampled stencil is a GLES3 version requirement.

## 1. XFB stream queries and rasterizer discard

A9 source point: `eb1aa2ce790`. The reusable part is the bookkeeping algorithm,
not the A9 winsys or its job submission records. The current G9P implementation
resolves counters while recording. That does not supply a general query model:
it cannot by itself handle repeated submissions, query availability, draw-time
conditions or GPU-produced counters. Do not enable transformFeedbackQueries just
because the capture rungs pass.

For an A10 implementation:

1. Record ordered Begin/Draw/End, ResetQuery/BeginQuery/EndQuery/CopyQuery events.
   Preserve them with the command buffer; reset execution cursors per submission.
   Do not resolve an application's counter/query values at command-record time.
2. At submission, resolve buffer bindings, offsets, sizes and the counter resume
   offset. Capture complete primitives only. The smallest remaining buffer
   capacity decides how many whole primitives fit. Track **written** and
   **needed** separately; overflow is not a larger written count.
3. Run the vertex program's stores, wait for the corresponding job completion
   before exposing CPU results, then update application counters. Give each query
   a pair of 64-bit counters plus availability; implement 32/64-bit query copy,
   stride and WITH_AVAILABILITY consistently.
4. Preserve outside-render-pass reset/copy event ordering, including events after
   the final render. Test replay, multiple Begin/End regions and overflow before
   advertising queries. The six-case probe checks an overflow query with written
   3, needed 4, and copied availability 1.

For rasterizer discard, keep vertex stores/capture active. Suppress raster
colour/depth/stencil work; a missing viewport in a discard pipeline must not be
dereferenced. The A9 path uses tag-write-disable and its colour-only ISP form;
use the named G9P/Rogue ISP fields rather than transplanting an A9 word. Verify
no colour or depth side effect and exact capture bytes on A10. G9P's pass builder,
shader key, job model and geometry-only submissions must agree before lifting
this limitation. This is why the A9 submit-side query/discard code was not copied
into G9P as an allegedly finished feature.

Build/run after implementation:

```
cd src/imagination/g9p/tests/xfb-stream
make
build/xfb-vk --case rec,two,overflow,resume,query,discard
```

Each case must say result=EXACT. `--offline` only records commands and is not
hardware proof. The older `tests/xfb` package is retained unchanged for its
additional firstVertex, instance and overflow diagnostics.

## 2. Sampled depth: recipe to start the A10 layout work

A9 source point: `6dec648c9d4`. The A9 stores D32F depth in a padded Morton plane
and samples that exact plane after the render. Its texture key is the R32F format
code 0x221, with Vulkan depth swizzle (D,0,0,1). D32F_S8 has a separate stencil
plane; the stencil-only probe reads R8_UINT (code 0x100). These are A9 decoded
format identities, not an asserted A10 descriptor layout.

The working A9 subset is deliberately narrow: one sample, square power-of-two
extent, one mip and one layer. Both ZLS and the texture descriptor use the same
padded dimensions and plane base. D32F and D32F_S8 advertise sampled-image and
depth-comparison only after raw and comparison pixels pass; no linear filtering
is assumed for 32-bit float depth. Compose the application's image-view swizzle
on (D,0,0,1), not the GL-style RRR1 used in an early capture.

On A10, first determine the ZLS stored byte order/stride/padding and D32F_S8 plane
offset. G9P already has depth-specific storage behavior; its attachment code is
not sufficient evidence that the A9 layout can be sampled unchanged. Build a
sampled descriptor over the actual stored plane, prove raw D32F and raw D32F_S8,
then add comparator input to the G9P sample encoding. Vulkan compare means
reference OP texel: for LESS with reference 0.4, stored 0.201 is false and 0.601 is
true. Do not flip the operands just to match an all-zero/all-one image.

The portable ds-sample-vk test renders those depth values, then samples them into
colour: raw wants red 51 inside and 153 outside; compare wants 0 inside and 255
outside. It checks every channel and every pixel. Cases raw8/cmp8 repeat this for
D32F_S8. Stencil8 expects 9 inside, 3 outside. Unsupported sampled formats return
NOT-OFFERED, not PASS. Depth sampling remains unimplemented on G9P in this series.
No GLES version override is proof of this missing term.

## 3. Vertex fetch, divisors, primitive restart and points

A9 source point: `71fa6eb573e`, plus the PDS divide forms validated in
`vtxin-afd1dd67e32` (mixed-divisor rerun 3/3). The API-only 45-rung client is
included; the G5P PDS builder/compiler is not substituted for G9P's.

The reusable implementation split is:

- Fetch records describe a binding stride, attribute byte offset, hardware fetch
  width and post-fetch conversion. For 8/16-bit norm/scaled/half formats, fetch
  dwords and convert/extract at the shader boundary; do not silently claim tight
  byte strides or packed formats which the fetch path cannot address.
- firstInstance participates in per-instance addressing; gl_InstanceIndex must
  match Vulkan independently of the fetch divisor. Divisor zero broadcasts the
  binding's selected record. For divisors above one, divide the instance index
  with the PDS unsigned reciprocal/shift recipe, then apply stride. The A9 recipe
  uses the AGXMetalA9-derived mad/divide fields and was pixel-tested for divisor 3
  and mixed divisors. Check the corresponding G9P PDS fields before reusing it;
  an inferred constant is not a proven A10 instruction form.
- Primitive restart uses the all-ones value of the actual index width, not a
  single 32-bit sentinel. Apply baseVertex to ordinary indices, not the restart
  marker. Test strips/lines at both segment boundaries.
- For points, split gl_PerVertex IO so gl_PointSize is not dropped. Put point
  size in the last vertex output word, after linked varyings, with the matching
  output-selection/PDS state. A default size must not be emitted twice.

The A9 45-rung result does not mean every Vulkan vertex format is implemented:
8/16-bit integer, 2_10_10_10 and some tight strides were still unadvertised/refused
at that checkpoint. G9P must advertise only its implemented forms.

## 4. Format slice 1 and ETC2/EAC

A9 source point: `aaabb7a16da`; exact portable test source is included. It checks
advertised capabilities rather than a private format table. The 118 baseline
rungs and 39 extra probes qualified A9 rows; they do not justify switching on
all G9P formats. Start with sample-only R/RG, normalized and integer formats,
then narrow/half render targets and blending, then compressed sampling/mips.

For a format, derive one texture/PBE key, channel selectors, byte/block size and
separate sampled/filter/render/blend flags. Shader output packing and tile width
must match the PBE interpretation; a valid texture key is not proof of a valid
render target. Verify integer samples a byte at a time so high bits cannot be
hidden by a float conversion. Check sRGB at both texture and output boundaries.

ETC2/EAC needs block-aware dimensions, row pitch and mip/layer placement. The A9
probe uses small deterministic compressed blocks and CPU texel references. The
same tests are included for G9P; any CPU copy/blit decompression must use the
format utility's decoder rather than treating compressed blocks as texels. Do
not copy the A9's image descriptor words or Morton conventions without checking
G9P's view and transfer code. This series adds tests/recipes, not unproven format
cap promotions.

## 5. Memory fixes shared with the A10

`pvr_arch_g9p_cpu_addr()` in the existing G9P CPU transfer path leaves BOs mapped.
The application's first normal vkMapMemory must reuse that mapping. Calling
pvr_drm_winsys_buffer_map again would assert because that winsys requires an
unmapped BO. The new option-gated normal-map hook returns the mapped base plus
the requested offset; placed-address behavior is unchanged. Run both cases of
`tests/memory/build/map-after-transfer` on A10.

Duplicate-GEM-handle import is already handled by
`winsys/powervr/pvr_drm_bo.c:pvr_drm_winsys_buffer_create_from_fd`: the dmabuf lock
covers PRIME import and lookup, and a live handle acquires the existing wrapper.
Its destruction path shares that lock. This is the behavior added separately
by G5P commit c8b91b29e97; G9P should retain its existing implementation. Verify
export/re-import identity and balanced references, including concurrent imports,
but do not add a second deduplication table.

## Deliberately not enabled

Depth sampling, XFB queries and the newer A9 vertex/format capabilities above
remain G9P follow-through work. D24S8 stays disabled: the A10 X8_D24 path had
crashed the firmware in the original notes. D32F_S8 is the tested substitute on
A9, not an A10 proof. MSAA is partial: the existing one/resolve/depth/store/
resolvecmd subset is code for qualification, not alpha-to-coverage or arbitrary
sample-mask support. No new A10 correctness or GLES3 conformance result is claimed.
