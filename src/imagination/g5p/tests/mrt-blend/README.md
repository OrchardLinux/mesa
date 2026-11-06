# MRT blend pixel controls

Build mrt-blend.c against EGL/GLES3. It runs15 scenes (two/three/four
RGBA8 targets, full writes, independent masks, target1/3 blending,
mixed masks+blend, and all-zero masks), comparing every byte exactly.
Backgrounds use distinct exact0/1 channels; sources use their complement,
alpha.5. Normalized output is0/255,128 for.5 and191 for.75.

Each scene clears all attachments and draws in the same render. N-target
LOAD is still refused by the driver and is outside this topic. An early
control with glFinish between clear/draw correctly hit that refusal; no
LOAD support is claimed. Host fake runs must end FAIL (it computes no pixels),
with15 comparisons and no driver refusal. Device runs must end PASS15/bad0.
There is no tolerance, recording or no-verify switch.
