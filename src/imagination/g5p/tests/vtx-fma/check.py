#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Generated-by: LLM
"""Compile FMA regression shaders with the real standalone G5P compiler.
Usage: check.py COMPILER OUT [--baseline]
Requires glslangValidator. This checks compiler acceptance, never GPU pixels.
"""
import argparse
from pathlib import Path
import subprocess
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('compiler'); p.add_argument('out', type=Path)
p.add_argument('--baseline', action='store_true')
a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
for f in sorted(Path(__file__).parent.glob('*.vert')):
    spv = a.out / (f.name + '.spv')
    subprocess.run(['glslangValidator', '-V', str(f), '-o', str(spv)], check=True)
    r = subprocess.run([a.compiler, 'vert', str(spv), str(a.out / (f.name + '.usc')), '0', '0'], capture_output=True, text=True)
    (a.out / (f.name + '.log')).write_text(r.stdout + r.stderr)
    expected = 77 if a.baseline and f.stem != 'control' else 0
    assert r.returncode == expected, (f.name, r.returncode, r.stderr)
    if expected:
        assert ('ffma with a source modifier and a uniform' in r.stderr or
                'register addend that is read again' in r.stderr), r.stderr
    print(f.name, 'expected refusal' if expected else 'compiled')
