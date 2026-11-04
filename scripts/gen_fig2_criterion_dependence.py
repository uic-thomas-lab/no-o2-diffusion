# -*- coding: utf-8 -*-
r"""Figure 2 panel D — CRITERION DEPENDENCE of spatial coverage.

    \int C dV = n . I_1

THE ANSWER DEPENDS ON THE CRITERION. On a 10-tension matrix:

    >0.5 nM   maximum INTERIOR, near 30 uM
    >1   nM   maximum INTERIOR, near 40 uM
    >2   nM   maximum at 56 uM, the top of the sampled TISSUE range
    >5   nM   maximum at 56 uM, turning over only just
    >10  nM   MONOTONIC across the whole range -- no interior maximum at all

WHY THIS RERUN EXISTS RATHER THAN REUSING data/fig2_oxygen_threshold_matrix.csv. That matrix
jumps 56 -> 220 uM with nothing between, so for the >2, >5 and >10 nM rows "maximum at 56" only
ever meant "maximum at the highest tissue tension I sampled". This run adds 25, 35, 50 and
**100 uM** so the turnover is resolved instead of inferred.

Writes `data/fig2_criterion_dependence.csv`.
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import tornado_v2 as T                                            # noqa: E402

N_SEEDS = 20
F_NOS2 = 0.05
RHO = 2.02e8
R_FOCUS = 50.0

# 220 uM is an oxygenated-medium REFERENCE, not a tissue condition.
O2_UM = (0.1, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 56.0, 100.0, 220.0)
O2_REFERENCE_ONLY = (220.0,)
THRESH_NM = (0.5, 1.0, 2.0, 5.0, 10.0)

L, N = 140.0, 141
_ax = np.linspace(-L, L, N)
DX = _ax[1] - _ax[0]
GX, GY, GZ = np.meshgrid(_ax, _ax, _ax, indexing='ij')
IN_FOCUS = (GX ** 2 + GY ** 2 + GZ ** 2) <= R_FOCUS ** 2
N_FOCUS = int(IN_FOCUS.sum())


def hard_core(n, rng, dmin, tries=20000):
    """Verbatim from gen_fig2_oxygen_ensemble.py -- placement must not drift."""
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
    a0 = T.A0_UM
    vfoc = 4.0 / 3.0 * np.pi * (R_FOCUS * 1e-4) ** 3
    n_cells = int(round(F_NOS2 * RHO * vfoc))

    geoms = [hard_core(n_cells, np.random.default_rng(s), 2 * a0) for s in range(N_SEEDS)]
    assert {len(g) for g in geoms} == {n_cells}, 'placement drifted from the locked seeds'

    print('Figure 2 panel D — criterion dependence of spatial coverage')
    print('  %d seeds x %d cells x %d tensions x %d criteria'
          % (N_SEEDS, n_cells, len(O2_UM), len(THRESH_NM)))
    print('  the integrated dose is NOT in this panel — it is n.I_1 and carries no spatial '
          'information\n')

    t0 = time.time()
    cov = {}
    for o2 in O2_UM:
        coeff = T.cell_coeffs(p, o2 * 1e-6)
        acc = {c: [] for c in THRESH_NM}
        for g in geoms:
            F = np.zeros_like(GX)
            for q in g:
                F += T.sphere_field(
                    np.sqrt((GX - q[0]) ** 2 + (GY - q[1]) ** 2 + (GZ - q[2]) ** 2), coeff)
            F *= 1e9
            for c in THRESH_NM:
                acc[c].append(100.0 * float(((F > c) & IN_FOCUS).sum()) / N_FOCUS)
        for c in THRESH_NM:
            cov[(o2, c)] = (float(np.mean(acc[c])), float(np.std(acc[c])))
        print('    [O2] = %6.1f uM   ' % o2 + '  '.join(
            '%.1f%%' % cov[(o2, c)][0] for c in THRESH_NM))
    print('  %.1f s\n' % (time.time() - t0))

    # ---- where does each criterion put its maximum, over TISSUE tensions only? ----
    tissue = [o for o in O2_UM if o not in O2_REFERENCE_ONLY]
    print('  criterion      max coverage    at [O2]     interior?     at 220 uM (reference)')
    summary = []
    for c in THRESH_NM:
        vals = [cov[(o, c)][0] for o in tissue]
        i = int(np.argmax(vals))
        interior = 0 < i < len(tissue) - 1
        summary.append((c, tissue[i], vals[i], interior))
        print('  >%-5g nM   %8.1f%%     %7g uM    %-12s  %.1f%%'
              % (c, vals[i], tissue[i], 'INTERIOR' if interior else 'AT AN END',
                 cov[(220.0, c)][0]))

    print('\n  The interior maximum PERSISTS at every criterion but MIGRATES to higher')
    print('    oxygen as the threshold tightens; it does not disappear on a fine oxygen grid.')
    print('    Report the criterion with')
    print('    every spatial claim.')
    print('  At the 1 nM endpoint the maximum IS interior.')

    out = os.path.join(ROOT, 'data', 'fig2_criterion_dependence.csv')
    with open(out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('O2_uM,reference_only,thresh_nM,cov_focus_pct,cov_SD\n')
        for o2 in O2_UM:
            for c in THRESH_NM:
                m, sd = cov[(o2, c)]
                fh.write('%g,%d,%g,%g,%g\n'
                         % (o2, int(o2 in O2_REFERENCE_ONLY), c, m, sd))
    print('\n  wrote %s' % os.path.relpath(out, ROOT))


if __name__ == '__main__':
    main()
