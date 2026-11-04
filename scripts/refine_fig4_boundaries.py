# -*- coding: utf-8 -*-
"""Two-stage refinement of the Figure 4 boundaries: coarse bracket, then a 10 um local rescan.

WHY. `gen_fig4_panels.py` locates the anoxic-onset and sGC-loss diameters by interpolating a
50 um scan. That is adequate where the two are hundreds of microns apart and NOT adequate at the
transition, which is the one place the panel has to be trusted:

        f = 0.04    delta_d = +2.3 um   on the 50 um grid
        f = 0.04    delta_d = +7.5 um   on a 10 um grid, reproduced EXACTLY at 2 um

A threefold difference in the pivotal number, purely from interpolation across a coarse cell. Both
crossings are smooth, so refinement converges quickly -- but a panel whose points come from
different grid resolutions is not defensible, and the temptation is to quietly use the fine value at
4% and the coarse values elsewhere. Every point on panel B must come from the same procedure.

SO: bracket on the coarse scan, then rescan +/- one coarse step at 10 um and interpolate there.
Uniform treatment, every fraction, whatever the size of delta_d.

Anoxic onset is [O2]_centre = 0.1 uM, the CONTINUOUS event -- never the first non-zero core
radius, which jumps 0 -> ~28 um and pins the onset to the step.

Out: data/fig4_A_boundaries.csv and data/fig4_B_delta.csv, overwritten with refined values.
"""
from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import spheroid_field as S                      # noqa: E402
from spheroid_crossover import interp_cross     # noqa: E402

DATA = os.path.join(ROOT, 'data')
A_CSV = os.path.join(DATA, 'fig4_A_boundaries.csv')
B_CSV = os.path.join(DATA, 'fig4_B_delta.csv')
COARSE_STEP = 50
FINE_STEP = 10
VERIFIED = (200, 1600)
SGC_NM = 1.0
ANOXIC_UM = S.ANOXIC_THRESHOLD * 1e6


def refine(f, d_guess, kind):
    """Rescan +/- one coarse step around d_guess at FINE_STEP and interpolate the crossing."""
    lo = max(VERIFIED[0], int(d_guess) - COARSE_STEP)
    hi = min(VERIFIED[1], int(d_guess) + COARSE_STEP)
    ds, vals, prev = [], [], None
    for d in range(lo, hi + 1, FINE_STEP):
        try:
            info, prof = S.describe(d / 2.0, f_nos2=f, _guess=prev)
        except RuntimeError as exc:
            print('    f=%.3f d=%d FAILED: %s' % (f, d, exc))
            prev = None
            continue
        prev = prof
        ds.append(d)
        vals.append(info['o2_centre_uM'] if kind == 'anoxia' else info['no_centre_nM'])
    target = ANOXIC_UM if kind == 'anoxia' else SGC_NM
    return interp_cross(ds, vals, target, rising=False)


def main():
    rows = list(csv.DictReader(open(A_CSV, encoding='utf-8')))
    out_a, out_b = [], []
    print('REFINED BOUNDARIES  (coarse bracket -> %d um rescan, uniform for every fraction)'
          % FINE_STEP)
    print('   f_NOS2   d_anoxia um      d_sGC um     delta_d um    (coarse delta_d)')
    for r in rows:
        f = float(r['f_nos2'])
        ca = float(r['d_anoxia_um']) if r['d_anoxia_um'] else None
        cs = float(r['d_sgc_um']) if r['d_sgc_um'] else None
        if ca is None or cs is None:
            out_a.append(r)
            continue
        fa = refine(f, ca, 'anoxia')
        fs = refine(f, cs, 'sgc')
        if fa is None or fs is None:
            print('   %6.3f   refinement did not bracket; coarse values retained' % f)
            out_a.append(r)
            continue
        delta, coarse = fs - fa, cs - ca
        print('   %6.3f   %8.1f (%+.1f)  %8.1f (%+.1f)  %+9.1f    %+.1f'
              % (f, fa, fa - ca, fs, fs - cs, delta, coarse))
        r = dict(r)
        r['d_anoxia_um'] = round(fa, 1)
        r['d_sgc_um'] = round(fs, 1)
        out_a.append(r)
        out_b.append({'f_nos2': f, 'delta_d_um': round(delta, 1),
                      'sign': 'sGC_first' if delta < 0 else 'anoxia_first'})

    sign_change = None
    for i in range(1, len(out_b)):
        if out_b[i - 1]['delta_d_um'] < 0 <= out_b[i]['delta_d_um']:
            sign_change = (out_b[i - 1]['f_nos2'], out_b[i]['f_nos2'])
    print('')
    print('  the ordering changes sign between f = %.3f and f = %.3f' % sign_change
          if sign_change else '  no sign change')

    with open(A_CSV, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(out_a)
    with open(B_CSV, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['f_nos2', 'delta_d_um', 'sign'])
        w.writeheader()
        w.writerows(out_b)
    print('  rewrote fig4_A_boundaries.csv and fig4_B_delta.csv at uniform %d um resolution'
          % FINE_STEP)
    return 0


if __name__ == '__main__':
    sys.exit(main())
