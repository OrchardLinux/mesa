# SPDX-License-Identifier: MIT

# Apple A7 (G3) compute: NIR to the A7's Rogue USC forms through pco's field
# encoders (g3_compute.h).
inc_powervr_g3 = include_directories(['.'])

libpowervr_g3 = static_library(
  'powervr_g3',
  files('g3_compute.c', 'g3_pco.c'),
  include_directories : [inc_imagination, inc_powervr_g3, inc_powervr_compiler,
                         inc_include, inc_compiler, inc_src],
  c_args : [imagination_c_args, no_override_init_args, '-Wno-psabi'],
  dependencies : [idep_mesautil, idep_nir, idep_pco_pygen],
  gnu_symbol_visibility : 'hidden',
  install : false,
)

idep_powervr_g3 = declare_dependency(
  link_with : libpowervr_g3,
  include_directories : inc_powervr_g3,
)

# g3_selftest: the encoders against the programs that ran on the A7 (cdm/r6.h)
g3_selftest = executable(
  'g3_selftest',
  files('g3_selftest.c'),
  include_directories : [inc_imagination, inc_powervr_g3, inc_include, inc_src],
  link_with : [libpowervr_g3, libpowervr_compiler],
  dependencies : [idep_mesautil, idep_nir],
  install : false,
)
