#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate per-device error-result wrappers from the Vulkan registry."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vulkan' / 'util'))
from vk_entrypoints import get_entrypoints_from_xml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    entries = [e for e in get_entrypoints_from_xml([args.xml], beta=False)
               if e.is_device_entrypoint() and e.return_type == 'VkResult'
               and e.alias is None]
    out = ['/* SPDX-License-Identifier: MIT */',
           '/* Generated from vk.xml; do not edit. */', '']
    for e in entries:
        if e.guard:
            out += [f'#ifdef {e.guard}']
        p = e.params[0]
        device = {'VkDevice': f'vk_device_from_handle({p.name})',
                  'VkQueue': f'vk_queue_from_handle({p.name})->base.device',
                  'VkCommandBuffer': f'vk_command_buffer_from_handle({p.name})->base.device'}[p.type]
        out += ['static VKAPI_ATTR VkResult VKAPI_CALL',
                f'pvr_errlog_{e.name}({e.decl_params()})', '{',
                f'   struct vk_device *pvr_errlog_vk = {device};',
                '   struct pvr_device *pvr_errlog_device = container_of(pvr_errlog_vk, struct pvr_device, vk);',
                f'   VkResult r = pvr_errlog_device->errlog_orig->{e.name}({e.call_params()});',
                '   if (r < 0)', f'      pvr_errlog_result("vk{e.name}", r);',
                '   return r;', '}']
        if e.guard:
            out += ['#endif']
        out += ['']
    out += ['static void', 'pvr_errlog_patch(struct vk_device_dispatch_table *t,',
            '                 struct vk_device_dispatch_table *orig)', '{', '   *orig = *t;']
    for e in entries:
        if e.guard:
            out += [f'#ifdef {e.guard}']
        out += [f'   if (t->{e.name})', f'      t->{e.name} = pvr_errlog_{e.name};']
        if e.guard:
            out += ['#endif']
    out += ['}', '']
    Path(args.out).write_text('\n'.join(out))


if __name__ == '__main__':
    main()
