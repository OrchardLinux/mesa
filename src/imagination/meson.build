# Copyright © 2022 Imagination Technologies Ltd.
# SPDX-License-Identifier: MIT

imagination_c_args = cc.get_supported_arguments(
  '-Werror=pointer-arith',
)

# Apple A9 (G5P): the device-info entry and src/imagination/g5p, opt-in only
with_imagination_g5p = get_option('imagination-g5p-experimental')
if with_imagination_g5p
  imagination_c_args += ['-DPVR_APPLE_G5P']
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
  # G5P first: the G9P compiler calls G5P's A9 encoders when the option is on.
  if with_imagination_g5p
    subdir('g5p')
  endif
  subdir('g9p')
endif

if with_imagination_vk
  subdir('vulkan')

  if with_tools.contains('drm-shim')
    subdir('drm-shim')
  endif
endif
