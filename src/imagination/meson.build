# Copyright © 2022 Imagination Technologies Ltd.
# SPDX-License-Identifier: MIT

imagination_c_args = cc.get_supported_arguments(
  '-Werror=pointer-arith',
)

# Apple A10 (G9P): the OpenGL ES 3.0 enablers, opt-in only (untested on the A10)
if get_option('imagination-g9p-es3')
  imagination_c_args += ['-DPVR_G9P_ES3']
endif

inc_imagination = include_directories([
  '.',
  'common',
  'include',
])

subdir('csbgen')

if with_imagination_vk or with_tools.contains('imagination')
  subdir('common')
  subdir('pco')
  subdir('ds')
  subdir('pdsc')
  subdir('g9p')
endif

if with_imagination_vk
  subdir('vulkan')

  if with_tools.contains('drm-shim')
    subdir('drm-shim')
  endif
endif
