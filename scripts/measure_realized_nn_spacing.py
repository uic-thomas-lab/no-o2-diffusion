# -*- coding: utf-8 -*-
"""What nearest-neighbour spacings does the tissue density series ACTUALLY realise?

WHY THIS EXISTS. The fixed-N sweep was nearly re-anchored to 13 / 18 / 26 / 35 um, taken from
the continuum estimate 0.554*(V/n)^(1/3) for the 2/5/15/40% conditions. That formula ignores the
hard-core exclusion, and **13 um is physically impossible here: the cell diameter is 15.84 um**.
Characteristic spacings inferred from a density are not realised nearest-neighbour distances.

So the biological reference for the fixed-N geometry sweep must be the DISTRIBUTIONS the
hard-core density ensembles actually produce, measured on the same locked seeds, with the same
definition of nearest neighbour used for the fixed-N series.

DEFINITION, applied identically to both series: for every cell, the distance to its closest
other cell. That is a per-cell quantity, N values per realisation. (The bridging analysis uses a
different object -- MUTUAL nearest-neighbour PAIRS -- and the two must not be conflated.)

Placement only; no field is solved, so this is cheap and exact.
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import tornado_v2 as T                                            # noqa: E402

RHO = 2.02e8
R_FOCUS = 50.0
FRACTIONS = (0.02, 0.05, 0.15, 0.40)
DENS_SEEDS, DENS_OFFSET = 50, 1000          # locked s8

N_FIXED = 10
R_SCATTER = (28.0, 40.0, 55.0, 70.0, 85.0, 100.0)
SPACING_SEEDS, SPACING_OFFSET = 20, 5000


def hard_core(n, rng, dmin, radius, tries=200000):
    pts = []
    for _ in range(tries):
        if len(pts) >= n:
            break
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        q = v * radius * rng.random() ** (1.0 / 3.0)
        if all(np.linalg.norm(q - w) >= dmin for w in pts):
            pts.append(q)
    return pts


def per_cell_nn(pts):
    a = np.array(pts)
    d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def summarise(label, allnn, extra=''):
    v = np.concatenate(allnn)
    q = np.percentile(v, [5, 25, 50, 75, 95])
    print('  %-22s n=%5d  min %5.1f   5%% %5.1f  IQR %5.1f-%5.1f  median %5.1f  95%% %5.1f  '
          'mean %5.1f %s' % (label, v.size, v.min(), q[0], q[1], q[3], q[2], q[4], v.mean(), extra))
    return q[0], q[2], q[4]


def main():
    a0 = T.A0_UM
    dmin = 2 * a0
    vfoc = 4.0 / 3.0 * np.pi * (R_FOCUS * 1e-4) ** 3
    print('REALISED nearest-neighbour spacing, hard-core placement, dmin = %.2f um\n' % dmin)

    print('TISSUE DENSITY SERIES — the biological reference (50 seeds each)')
    bands = {}
    for frac in FRACTIONS:
        n = int(round(frac * RHO * vfoc))
        if n < 2:
            print('  %4.0f%%  only %d cell — no neighbour' % (frac * 100, n))
            continue
        allnn = []
        for s in range(DENS_SEEDS):
            pts = hard_core(n, np.random.default_rng(DENS_OFFSET + s), dmin, R_FOCUS)
            if len(pts) == n:
                allnn.append(per_cell_nn(pts))
        cont = 0.554 * (4 / 3 * np.pi * R_FOCUS ** 3 / n) ** (1 / 3.)
        bands[frac] = summarise('%4.0f%% (%2d cells)' % (frac * 100, n), allnn,
                                '   [continuum estimate said %.0f um]' % cont)

    print('\nFIXED-N SWEEP — what was actually sampled (10 cells, 20 seeds each)')
    sweep = []
    for R in R_SCATTER:
        allnn = []
        for s in range(SPACING_SEEDS):
            pts = hard_core(N_FIXED, np.random.default_rng(SPACING_OFFSET + int(R) * 100 + s),
                            dmin, R)
            if len(pts) == N_FIXED:
                allnn.append(per_cell_nn(pts))
        q = summarise('R = %5.0f um' % R, allnn)
        sweep.append(q[1])

    lo_all = min(b[0] for b in bands.values())
    hi_all = max(b[2] for b in bands.values())
    print('\n  tissue realises %.0f to %.0f um NN spacing across 2-40%% NOS2+ '
          '(5th to 95th percentile)' % (lo_all, hi_all))
    print('  the fixed-N sweep spans medians %.0f to %.0f um' % (min(sweep), max(sweep)))
    covered = min(sweep) <= lo_all + 2 and max(sweep) >= hi_all - 2
    print('  %s' % ('  -> the existing sweep COVERS the realisable tissue range; no rerun needed'
                    if covered else
                    '  -> the sweep does NOT cover the realisable range end to end'))
    print('\n  13 um (a continuum formula) ignores hard-core')
    print('     exclusion and lies below the %.2f um cell diameter.' % dmin)


if __name__ == '__main__':
    main()
