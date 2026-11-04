# vertex-input

Portable vertex-input, divisor, restart and points probe. A9 device-run-13 (vtxin-71fa6eb573e) passed 45/45; no A10 run. Unsupported formats are skipped explicitly. This test does not add driver capabilities. See ../../ES3-PORTING.md.

Build: `make`. Generated headers and binaries stay in `build/`. Select the candidate ICD with VK_DRIVER_FILES and compare real pixels; recording-only modes do not qualify hardware.
