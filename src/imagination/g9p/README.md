# G9P: the Apple A10 GPU

`src/imagination/g9p` and the `g9p` paths of `src/imagination/vulkan` drive
the Apple A10's GPU (T8010, iPad 6): a PowerVR Rogue core run by Apple's
own firmware, with Apple's own shader ISA. The PDS programs, control
streams, render passes and descriptors are upstream PowerVR code; the
shader compiler (`g9p_compile.c`, NIR to G9P), the ISA encoders
(`g9p_isa.h`) and the job layouts are G9P's own. Shared code that G9P must
bypass is gated on `PVR_HAS_FEATURE(dev_info, apple_g9p)`.

This is OrchardLinux's fork of Mesa; none of it is proposed upstream.

## The notes these comments cite

Comments cite the A10 notes by file and section, like `ISA.md "Register
allocation"`. They are in
[OrchardLinux/gpu](https://github.com/OrchardLinux/gpu):

- `ISA.md`, `RENDER.md`, `COMPUTE.md`, `MESA.md`, `PROVENANCE.md` and the
  others: `a10/`, with the bring-up guide in `a10/bringup/` (its `mesa.md`
  explains this driver in topic order);
- `tools/…` (for example `tools/pvr-tri.c`, `tools/g9p-cdm.h`,
  `tools/isa/g9p.py`) and `probes/…`: under `tools/a10/` there, the probes
  in `tools/a10/tools/isa/probes/`.

The kernel driver is in [orchard](https://github.com/OrchardLinux/orchard)
(`kernel/patches/drivers/gpu`).
