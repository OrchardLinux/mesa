# SPDX-License-Identifier: MIT

# Apple A10 (G9P) shader compiler and encoders.
inc_powervr_g9p = include_directories(['.'])

libpowervr_g9p = static_library(
  'powervr_g9p',
  files('g9p_compile.c'),
  include_directories : [inc_imagination, inc_powervr_g9p, inc_include, inc_src],
  c_args : [imagination_c_args, no_override_init_args],
  dependencies : [idep_mesautil, idep_nir],
  gnu_symbol_visibility : 'hidden',
  install : false,
)

idep_powervr_g9p = declare_dependency(
  link_with : libpowervr_g9p,
  include_directories : inc_powervr_g9p,
)
