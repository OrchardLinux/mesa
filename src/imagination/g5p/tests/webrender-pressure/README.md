# WebRender register-pressure numeric probes

Generated-by: LLM. Host reference validation does not execute G5P USC.

Compile wr2-numeric.c with -lEGL -lGLESv2 -lm. Each manifest row supplies
vertex/fragment filename (key), uOffset and four CPU-computed expected colour
channels. Invoke with those seven arguments followed by --reference on
llvmpipe, or --device on Zink/G5P. The oracle checks all 4096 channels within
one RGBA8 unit, and checks GL errors and renderer identity. Two offsets keep
texture values distinct and exercise nonconstant address arithmetic.

The vertex probes retain and mix texture results across eight varying vec4s.
The fragment probes multiply neighbouring texel scalars and independently
reduce them, exercising pressure/scheduling and the existing high-home moves.
Run PVR_G5P_VS_BANK0_FIRST=1 as a second device arm, shader cache disabled.
Retain PVR_G5P_PRESSURE_LOG=1 and driver refusal logs: a pixel pass alone does
not prove which retry was selected. These probes do not qualify private memory
spills; none is enabled. The full SVG branch retry still needs device pixels
from actual WebRender content.
