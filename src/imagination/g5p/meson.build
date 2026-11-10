# SPDX-License-Identifier: MIT

# Apple A9 (G5P) shader backend: NIR to Apple's A9 USC ISA, with its field
# encoders. Built only with -Dimagination-g5p-experimental=true. It reads
# PCO IR (pco_g5p.c) for its captured-shape matcher, so it builds against
# pco's generated headers, and links nothing into the PowerVR driver yet.
inc_powervr_g5p = include_directories(['.'])

libpowervr_g5p = static_library(
  'powervr_g5p',
  files(
    'g5p_blend.c',
    'g5p_recipes.c',
    'g5p_usc.c',
    'pco_g5p.c',
    'pco_g5p_general.c',
  ),
  include_directories : [inc_imagination, inc_powervr_g5p, inc_powervr_compiler,
                         inc_include, inc_compiler, inc_src],
  c_args : [imagination_c_args, no_override_init_args, '-Wno-psabi'],
  dependencies : [dep_csbgen, idep_mesautil, idep_nir, idep_pco_pygen],
  gnu_symbol_visibility : 'hidden',
  install : false,
)

idep_powervr_g5p = declare_dependency(
  link_with : libpowervr_g5p,
  include_directories : inc_powervr_g5p,
)
