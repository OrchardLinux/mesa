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
    '../vulkan/pvr_arch_g5p_dgl.c',
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

g5p_blend_test = executable(
  'g5p-blend-test',
  files('tests/blend.c', 'g5p_usc.c'),
  include_directories : [inc_powervr_g5p],
  dependencies : [dep_m],
  install : false,
)
test('g5p-blend-equations', g5p_blend_test, timeout : 120)

g5p_vs_trunc_test = executable(
  'g5p-vs-trunc-test',
  files('tests/vs-trunc.c', 'g5p_usc.c', 'g5p_recipes.c'),
  include_directories : [inc_powervr_g5p],
  dependencies : [dep_m],
  install : false,
)
test('g5p-vs-trunc', g5p_vs_trunc_test, timeout : 60)

g5p_vsops_test = executable(
  'g5p-vsops-test',
  files('tests/vsops.c', 'g5p_usc.c'),
  include_directories : [inc_powervr_g5p],
  dependencies : [dep_m],
  install : false,
)
test('g5p-vsops-encodings', g5p_vsops_test, timeout : 60)

g5p_vsalu_test = executable(
  'g5p-vsalu-test',
  files('tests/vsalu.c', 'g5p_usc.c', 'g5p_recipes.c'),
  include_directories : [inc_powervr_g5p],
  dependencies : [dep_m],
  install : false,
)
test('g5p-vsalu-encodings', g5p_vsalu_test, timeout : 60)

g5p_dynamic_shift_test = executable(
  'g5p-dynamic-shift-test',
  files('tests/dynamic-shift.c', 'g5p_usc.c'),
  include_directories : [inc_powervr_g5p],
  dependencies : [dep_m],
  install : false,
)
test('g5p-dynamic-shifts', g5p_dynamic_shift_test, timeout : 60)

# Evaluate the actual integer lowering graph against independent references.
g5p_integer_lowering_test = executable(
  'g5p-integer-lowering-test',
  files('tests/integer/lowering.c'),
  include_directories : [inc_powervr_g5p, inc_src],
  dependencies : [idep_nir],
  install : false,
)
test('g5p-integer-lowering', g5p_integer_lowering_test, timeout : 120)

g5p_integer_encodings_test = executable(
  'g5p-integer-encodings-test',
  files('tests/integer/encodings.c', 'g5p_usc.c'),
  include_directories : [inc_powervr_g5p],
  install : false,
)
test('g5p-integer-encodings', g5p_integer_encodings_test, timeout : 60)
