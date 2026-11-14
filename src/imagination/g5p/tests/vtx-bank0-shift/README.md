# Vertex bank0 literal shifts

Generated-by: LLM. R36 base14d75cb56e9. Native G5P register IADD
repeatedly doubles a bank0 source for literal shifts1..4; existing bank1
shift packets remain unchanged. Integer overflow is modulo2^32. No new
register allocation, high destinations, launch fields or memory forms.

The A10 developer branch was checked: its g9p_shift packet is not an A9
witness. This topic reuses our existing native G5P IADD encoder instead.

Host evidence in linux-a9 build/gpu/l7-vtx-next: corpus270 byte identity,
8 Meson gates, 20 real Zink-linked native repro programs with all8 exports,
native bank/control checks, 10,360 decoded-chain arithmetic executions and
20 bank-bit corruption negatives. Numeric verifier rejects fake pixels.
Native five-mode device packet is required before numerical qualification.

cts-dependencies.txt lists40 XFB cases whose first shift refusal clears.
All still stop at the transform-feedback bank0 address-add guard; zero
new CTS admissions/fixes. The21 sibling shared-integer cases depend on
clean9fcec4d4918 port860bf76c741 and reach the same next guard. L10 owns
address/store work. SERIAL0, HIGH0 and unchanged allocator bounds.
