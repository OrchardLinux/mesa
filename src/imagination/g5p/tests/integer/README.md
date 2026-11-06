# Graphics integer builtins

`discovery-candidates.txt` freezes the current328 named CTS refusals from the
2026-11-06 GLES3.1 discovery. A `_fragment` test can be blocked by its vertex
harness; suffix is not the stage of the refusal. These are exact case names,
not a predicted pass count. Host fake verdicts never establish pixels.

The graphics frontend lowers bitfield extraction/insertion with NIR's
zero/full-width guards. Vertex popcount/reverse use NIR's SWAR sequences;
fragment native popcount/reverse stay unchanged. Compute options stay separate.
Native arithmetic controls, all real pipeline admissions and numerical GPU
results must be qualified before default promotion.
