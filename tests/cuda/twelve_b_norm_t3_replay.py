"""Exact actual three-row norm/residual/quantization replay and graph screen."""
from pathlib import Path
import argparse
import ctypes as C
import json
import random
import struct
from twelve_b_device_lab import Device, P, U, F, read, finite

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('fixtures', type=Path)
p.add_argument('library', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--timing', action='store_true')
a = p.parse_args()
m = json.loads((a.fixtures / 'captures.json').read_text())
assert m['tokens'] == 3 and m['width'] == 3840 and m['groups'] == 507
lib = C.CDLL(str(a.library.resolve()))
fn = lib.gem16_12b_norm_t3
fn.argtypes = [U, P, P, P, P, P, P, P, P, F, P]
fn.restype = C.c_int
d = Device()
cases = []
checks = 0
try:
    projection = d.buffer(46080); residual = d.buffer(46080)
    postweight = d.buffer(7680); preweight = d.buffer(7680)
    post = d.buffer(46080); hidden = d.buffer(46080)
    packed = d.buffer(5760); scales = d.buffer(720)
    for i, case in enumerate(m['cases']):
        slot = case['slot']
        postweight.put(read(a.fixtures / f'post-{slot}.bin', 7680))
        preweight.put(read(a.fixtures / f'pre-{slot}.bin', 7680))
        pv = read(a.fixtures / f'part-{i}-0.bin', 46080)
        rv = read(a.fixtures / f'part-{i}-49152.bin', 46080)
        expected = [read(a.fixtures / f'part-{i}-{off}.bin', n)
                    for off, n in ((98304, 46080), (147456, 5760), (155648, 720))]
        def launch(v, null_post=False, alias=False):
            assert fn(v, projection.ptr, postweight.ptr, residual.ptr,
                      P() if null_post else post.ptr, residual.ptr if alias else hidden.ptr,
                      preweight.ptr, packed.ptr, scales.ptr, F(case['divisor']), d.stream) == 0
        def run(v, x=pv, y=rv, null_post=False, alias=False):
            projection.put(x); residual.put(y)
            launch(v, null_post, alias); d.synchronize()
            h = residual.get() if alias else hidden.get()
            finite(h)
            if not null_post: finite(post.get())
            return [h, packed.get(), scales.get(), projection.get()] + ([] if null_post else [post.get()])
        for null_post in (False, True):
            for alias in (False, True):
                parent = run(0, null_post=null_post, alias=alias)
                candidate = run(1, null_post=null_post, alias=alias)
                assert parent == candidate and candidate[:3] == expected, (i, null_post, alias)
                checks += 1
        if not a.timing:
            for edge in range(6):
                rng = random.Random(171117 + edge + i * 10)
                vals = [0.0 if edge == 0 else ((-1 if j % 2 else 1) * 1e-30 if edge == 1 else rng.uniform(-100, 100) if edge == 2 else rng.choice([0, .99609375, 1.00390625, -1.00390625, 3.14159265]) * rng.choice([.0001, 1, 10000])) for j in range(11520)]
                x = struct.pack('11520f', *vals)
                y = struct.pack('11520f', *([0.0] * 11520 if edge < 3 else [-v for v in vals]))
                assert run(0, x, y) == run(1, x, y), (i, edge)
                checks += 1
        projection.put(pv); residual.put(rv)
        samples = d.paired(lambda v: launch(v, null_post=True), samples=15) if a.timing else []
        d.guards()
        cases.append({**case, 'bit_equal': True, 'samples': samples})
        print('PASS', i, flush=True)
    with a.output.open('x') as f:
        json.dump({'diagnostic_only': True, 'checks': checks, 'cases': cases,
                   'actual_hidden_packed_scales_exact': True,
                   'projection_postnorm_intermediates_and_alias_exact': True,
                   'guards_and_finite_passed': True}, f, indent=2)
finally:
    d.close()
