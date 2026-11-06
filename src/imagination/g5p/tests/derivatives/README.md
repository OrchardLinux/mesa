# Fragment derivatives under uniform control flow

The unchanged A9 derivative encoder is witnessed under Apple's execution-mask
stack by `fw_l7_*` shaders in linux-a9 `tools/ios-witness-r1/witness.metal`.
The four encoder tests match actual straight-line/uniform-loop instructions.

`uniform_*` and `straight` GLSL fixtures must compile through the real frontend;
`varying_*` must retain the named divergent-control-flow refusal. In particular,
a varying break or continue must not be treated as safe simply because the
remaining active lanes reconverge before the derivative. Track loop exits too.
Divergence metadata must survive out-of-SSA register-web construction.

Exact26 historical refusals are frozen in `cts-targets.txt`. Their host pipeline
admission is not numerical pixel evidence. The120 MSAA framebuffer-completeness
failures are separate and are not fixed by admitting derivatives in control flow.
