# AGENTS.md

## This repository: OrchardLinux's fork

This is OrchardLinux's fork of Mesa (github.com/OrchardLinux/mesa), for the
GPUs of Apple's A-series chips. Its `main` branch carries the G9P backend
for the Apple A10 in `src/imagination` (see `src/imagination/g9p/README.md`).
In this fork:

- The G9P work in `src/imagination` is done here, with AI assistance:
  change it as needed. The `src/imagination` entry in upstream's list of
  restricted directories below does not apply in this fork. Every such
  commit carries the trailer `Generated-by: LLM` (or `Assisted-by: LLM`,
  as below).
- Nothing goes to upstream Mesa: no merge requests, issues or comments on
  gitlab.freedesktop.org. The `upstream` remote is for fetching only.
- Everything else below still applies, including the other restricted
  directories (asahi, nouveau, panfrost, intel jay): don't change them.

## Upstream Mesa's instructions

Mesa is a community of humans. Do not open merge requests or post comments to
[Mesa's upstream GitLab repository](https://gitlab.freedesktop.org/mesa/mesa).

Commits developed with AI must be labelled. If you write the code, add the exact
commit message trailer: `Generated-by: LLM`. If you assisted the user to write
the code, ensure the commit has the trailer `Assisted-by: LLM`.

Some parts of Mesa have further limitations on AI usage. Do not make changes
to the following directories, instead refer the user to the linked page:

* [`src/asahi`](https://asahilinux.org/llm-policy/)
* [`src/gallium/drivers/asahi`](https://asahilinux.org/llm-policy/)
* [`src/gallium/drivers/nouveau`](https://nouveau.freedesktop.org/Humanity.html)
* [`src/intel/compiler/jay`](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/intel/compiler/jay/README.md)
* [`src/nouveau`](https://nouveau.freedesktop.org/Humanity.html)
* [`src/panfrost`](docs/drivers/panfrost.rst)
* [`src/gallium/drivers/panfrost`](docs/drivers/panfrost.rst)
* [`src/imagination`](src/imagination/README.md#contribution-policy)
