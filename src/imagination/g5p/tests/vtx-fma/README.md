# Vertex FMA shared-source staging

The four non-control shaders reproduce R32 compiler refusals, through the real
SPIR-V frontend. `check.py COMPILER OUT --baseline` checks those exact refusals;
without `--baseline` it requires all five shaders to compile. The register-only
control must retain identical USC before and after.

`abs_uniform`, `neg_uniform`, and `neg_register_uniform` require moving the shared
operand into a register before the witnessed modifier FMA. `live_addend` requires
the register FMA's repeat-read group rather than an unwitnessed shared tuple.
All remain fused operations. No pixel verdict is inferred from compilation.

Decode the resulting USC with the linux-a9 `vs2bank/usc-decode.py`, then apply its
native bank and tuple auditors. Numerical qualification still requires the
focused CTS flight, paired with exact R32 and unchanged stock Zink.
