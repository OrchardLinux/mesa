# WR3 immutable texture controls

Generated-by: LLM. Four independent CPU per-texel arithmetic expectations
(two shaders, two offsets). Input channel c of texel i is
`((3*i+5*c)%29+1)/32`. Run the numeric harness with the exact sources and
`expected.json`. Fake readback must fail.

These admitted controls exercise ordinary texture moves/retention and
arithmetic. They do not prove that the live-range-splitting retry ran or
reproduce the full YUV/WebRender scene. Dense correlation graphs tested by
the workspace lane still refuse and are retained as separate boundaries.
Actual G5P pixels and the real YUV scene must be checked by the runner.
