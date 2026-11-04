# -*- coding: utf-8 -*-
"""Where does the hypoxia / NO-signalling ordering change sign?

THE QUANTITY:

        delta_d(f) = d(sGC signalling lost at the centre) - d(anoxic core first appears)

    delta_d < 0  -> NO signalling is lost BEFORE anoxia exists   (the 2% regime)
    delta_d > 0  -> anoxia forms while NO still reaches the core (the 15%+ regime)

    The regime boundary is simply delta_d = 0.

WHY THIS RUN EXISTS. The six-fraction sweep put the sign change between f = 0.02 and f = 0.05, but
the 5% margin was only ~20 um on a 690 um diameter -- under 3%, and not resolvable at that grid. The
claim is load-bearing for the manuscript, so the boundary gets located rather than interpolated by
eye from six points.

Both crossings must lie inside the independently verified 200-1600 um band (scipy vs Wolfram agree
to 4-5 significant figures there and diverge beyond). Anything outside is reported as out-of-band.

Out: data/spheroid_crossover.csv
"""
from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import spheroid_field as S   # noqa: E402

FRACTIONS = (0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15)
D_LO, D_HI, D_STEP = 520, 960, 10
VERIFIED = (200, 1600)
SGC_NM = 1.0

OUT = os.path.join(ROOT, 'data', 'spheroid_crossover.csv')


def interp_cross(ds, vals, target, rising):
    """Linear interpolation of the first crossing of `target`. None if it never crosses."""
    for i in range(1, len(ds)):
        a, b = vals[i - 1], vals[i]
        hit = (a <= target < b) if rising else (a >= target > b)
        if hit and b != a:
            return ds[i - 1] + (target - a) * (ds[i] - ds[i - 1]) / (b - a)
    return None


def main():
    rows = []
    print('CROSSOVER: delta_d = d(sGC lost) - d(anoxic onset)')
    print('  diameter grid %d-%d um step %d   |  verified band %d-%d um\n'
          % (D_LO, D_HI, D_STEP, *VERIFIED))
    print('   f_NOS2   d_anoxia um   d_sGC um   delta_d um   regime')
    for f in FRACTIONS:
        ds, cores, noc, o2c, prev = [], [], [], [], None
        for d in range(D_LO, D_HI + 1, D_STEP):
            try:
                info, prof = S.describe(d / 2.0, f_nos2=f, _guess=prev)
            except RuntimeError as exc:
                print('   f=%.3f d=%d SOLVE FAILED: %s' % (f, d, exc))
                prev = None
                continue
            prev = prof
            ds.append(d)
            cores.append(info['r_core_um'])
            noc.append(info['no_centre_nM'])
            o2c.append(info['o2_centre_uM'])

        # ANOXIC ONSET IS INTERPOLATED ON CENTRE [O2], NOT ON CORE RADIUS. The core radius
        # jumps from 0 to ~28 um the moment it exists, so interpolating it just returns the lower
        # grid point and pins the onset to the 10 um step -- which was the same order as the
        # |delta_d| being measured at the bracketing fractions. Centre [O2] crosses the 0.1 uM
        # anoxic definition CONTINUOUSLY, so it locates the same event to well under a step.
        d_anox = interp_cross(ds, o2c, S.ANOXIC_THRESHOLD * 1e6, rising=False)
        d_sgc = interp_cross(ds, noc, SGC_NM, rising=False)
        if d_anox is None or d_sgc is None:
            print('   %6.3f   %s' % (f, 'crossing not inside the scanned window'))
            rows.append({'f_nos2': f, 'd_anoxia_um': d_anox, 'd_sgc_um': d_sgc,
                         'delta_d_um': None, 'regime': 'not bracketed'})
            continue
        delta = d_sgc - d_anox
        regime = ('sGC lost FIRST (inverted)' if delta < 0 else
                  'anoxia first (ordering holds)')
        flag = '' if VERIFIED[0] <= min(d_anox, d_sgc) and max(d_anox, d_sgc) <= VERIFIED[1] \
               else '  OUT OF VERIFIED BAND'
        print('   %6.3f   %11.1f   %8.1f   %+10.1f   %s%s'
              % (f, d_anox, d_sgc, delta, regime, flag))
        rows.append({'f_nos2': f, 'd_anoxia_um': round(d_anox, 1),
                     'd_sgc_um': round(d_sgc, 1), 'delta_d_um': round(delta, 1),
                     'regime': regime})

    # where does delta_d change sign?
    good = [r for r in rows if r['delta_d_um'] is not None]
    lo = hi = None
    for i in range(1, len(good)):
        if good[i - 1]['delta_d_um'] < 0 <= good[i]['delta_d_um']:
            lo, hi = good[i - 1]['f_nos2'], good[i]['f_nos2']
    print('')
    if lo is None:
        print('  no sign change inside the scanned fractions')
    else:
        print('  delta_d changes sign between f = %.3f and f = %.3f' % (lo, hi))
        near = [r for r in good if abs(r['delta_d_um']) <= D_STEP]
        if near:
            print('  |delta_d| <= one diameter step (%d um) at f = %s -- these are NOT resolved '
                  'and the boundary must be reported as a BAND'
                  % (D_STEP, ', '.join('%.3f' % r['f_nos2'] for r in near)))

    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['f_nos2', 'd_anoxia_um', 'd_sgc_um',
                                           'delta_d_um', 'regime'])
        w.writeheader()
        w.writerows(rows)
    print('\nwrote %s' % OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
