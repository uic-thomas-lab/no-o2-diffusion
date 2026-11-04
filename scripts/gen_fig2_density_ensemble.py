# -*- coding: utf-8 -*-
"""Figure 2, density row: 50-seed ensemble on hard-core placement.

PRE-REGISTRATION — fixed before the run, before any result is seen.

    N_SEEDS   = 50        at EVERY expressing fraction, not only at 2%.
                          The ensemble size must not be outcome-dependent.
    FRACTIONS = 2, 5, 15, 40 %
    O2        = 10 uM     the reference condition shared with the oxygen row
    PLACEMENT = hard-core Poisson in a 50 um focus, min separation 2*a0
    REPRESENTATIVE = seed whose peak [NO] is closest to the ensemble MEDIAN,
                     computed per fraction. INDEPENDENT OF THE FRAGMENTATION
                     OUTCOME -- never choose a realisation because it looks
                     connected, or because it is visually attractive.

## Why 50 and not 20, and why not 100

Fragmentation at 2% is going to be reported as a NUMBER, and 20 realisations is too thin for a
proportion near 0.38: each realisation moves the estimate 5 percentage points, so "38%" would
look more precise than it is. At 50 each realisation is 2 points. 50 supports a statement like
"roughly one-third to one-half of 2% realisations fragment"; it does NOT distinguish 35% from
45%, and must not be used to. 100 would only be justified if the fragmentation probability
itself became a headline endpoint, which it is not.

## Reporting rule, frozen with the seed count

  * CONTINUOUS quantities (peak [NO], coverage): ensemble mean +/- SD.
  * BINARY quantities (fragmented, all-cells-connected): report **n/N and the percentage with a
    binomial Wilson 95% CI** -- NEVER mean +/- SD. A proportion's SD is not its uncertainty, and
    "1.4 +/- 0.5 components" concealed that 38% of realisations fragmented.

Writes `data/fig2_density_ensemble.csv` (per seed x per fraction).
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import tornado_v2 as T                                            # noqa: E402

N_SEEDS = 50
FRACTIONS = (0.02, 0.05, 0.15, 0.40)
O2_UM = 10.0
RHO = 2.02e8
R_FOCUS = 50.0
THRESH_NM = (1.0, 10.0, 30.0)

L, N = 140.0, 141
_ax = np.linspace(-L, L, N)
DX = _ax[1] - _ax[0]
GX, GY, GZ = np.meshgrid(_ax, _ax, _ax, indexing='ij')
IN_FOCUS = (GX ** 2 + GY ** 2 + GZ ** 2) <= R_FOCUS ** 2
N_FOCUS = int(IN_FOCUS.sum())


def wilson(x, n, z=1.96):
    """Binomial Wilson score interval. Use this for proportions, never mean +/- SD."""
    if n == 0:
        return (float('nan'),) * 3
    ph = x / n
    d = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z / d * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return 100 * ph, 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def hard_core(n, rng, dmin, tries=20000):
    pts = []
    for _ in range(tries):
        if len(pts) >= n:
            break
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        q = v * R_FOCUS * rng.random() ** (1.0 / 3.0)
        if all(np.linalg.norm(q - w) >= dmin for w in pts):
            pts.append(q)
    return pts


def main():
    p = T.nominal_params()
    coeff = T.cell_coeffs(p, O2_UM * 1e-6)
    a0 = T.A0_UM
    vfoc = 4.0 / 3.0 * np.pi * (R_FOCUS * 1e-4) ** 3

    print('Figure 2 density row — %d-seed ensemble' % N_SEEDS)
    print('  PRE-REGISTERED: %d seeds at EVERY fraction, [O2] = %.0f uM, dx = %.2f um\n'
          % (N_SEEDS, O2_UM, DX))

    rows = []
    for frac in FRACTIONS:
        n_cells = int(round(frac * RHO * vfoc))
        for s in range(N_SEEDS):
            pts = hard_core(n_cells, np.random.default_rng(1000 + s), 2 * a0)
            F = np.zeros_like(GX)
            for q in pts:
                F += T.sphere_field(
                    np.sqrt((GX - q[0]) ** 2 + (GY - q[1]) ** 2 + (GZ - q[2]) ** 2), coeff)
            F *= 1e9
            r = dict(frac=frac, seed=s, n_cells=len(pts), peak_nM=float(F.max()))
            for c in THRESH_NM:
                M = F > c
                r['cov_focus_pct_%g' % c] = 100.0 * float((M & IN_FOCUS).sum()) / N_FOCUS
            M1 = F > 1.0
            lab, nl = ndimage.label(M1)
            r['components_1nM'] = int(nl)
            ids = {lab[tuple(int(round((x + L) / DX)) for x in q)] for q in pts}
            r['all_cells_one_component'] = int(len(ids) == 1 and 0 not in ids)
            r['fragmented'] = int(nl > 1)
            rows.append(r)
        print('    f_NOS2 = %4.0f%%  (%2d cells)  done' % (frac * 100, n_cells))

    print('\n  CONTINUOUS quantities — ensemble mean +/- SD')
    print('  frac  cells    peak nM        >1nM coverage of focus     >10nM')
    for frac in FRACTIONS:
        g = [r for r in rows if r['frac'] == frac]
        f = lambda k: (np.mean([r[k] for r in g]), np.std([r[k] for r in g]))
        pk, c1, c10 = f('peak_nM'), f('cov_focus_pct_1'), f('cov_focus_pct_10')
        print('  %4.0f%%  %4d   %6.2f±%.2f  %16.1f±%.1f%%  %7.2f±%.2f%%'
              % (frac * 100, g[0]['n_cells'], pk[0], pk[1], c1[0], c1[1], c10[0], c10[1]))

    print('\n  BINARY quantities — n/N with binomial Wilson 95% CI  (never mean±SD)')
    print('  frac   fragmented (>1 component)        all cells in ONE component')
    for frac in FRACTIONS:
        g = [r for r in rows if r['frac'] == frac]
        xf = sum(r['fragmented'] for r in g)
        xc = sum(r['all_cells_one_component'] for r in g)
        pf, lf, uf = wilson(xf, len(g))
        pc, lc, uc = wilson(xc, len(g))
        print('  %4.0f%%   %2d/%d = %5.1f%%  [%4.1f–%4.1f]   %2d/%d = %5.1f%%  [%4.1f–%4.1f]'
              % (frac * 100, xf, len(g), pf, lf, uf, xc, len(g), pc, lc, uc))

    print('\n  REPRESENTATIVE geometries (median-peak rule, independent of fragmentation)')
    for frac in FRACTIONS:
        g = [r for r in rows if r['frac'] == frac]
        med = float(np.median([r['peak_nM'] for r in g]))
        rep = min(g, key=lambda r: abs(r['peak_nM'] - med))
        print('    %4.0f%%  seed %2d   peak %.2f nM   (ensemble median %.2f)   fragmented=%d'
              % (frac * 100, rep['seed'], rep['peak_nM'], med, rep['fragmented']))

    out = os.path.join(ROOT, 'data', 'fig2_density_ensemble.csv')
    keys = list(rows[0].keys())
    with open(out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join('%g' % r[k] if isinstance(r[k], float) else str(r[k])
                              for k in keys) + '\n')
    print('\n  wrote %s  (%d rows)' % (os.path.relpath(out, ROOT), len(rows)))


if __name__ == '__main__':
    main()
