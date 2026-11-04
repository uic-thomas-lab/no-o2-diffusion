# -*- coding: utf-8 -*-
"""Figure 2 — oxygen x threshold matrix over TUMOUR-RELEVANT oxygenation.

Purpose: decide whether coverage is greatest at 56 uM or between 10 and 56 uM, which a coarser
oxygen sampling cannot resolve.

PRE-REGISTRATION -- fixed before the run.

    O2_UM      = 1, 2, 5, 10, 15, 20, 30, 40, 56    tumour-relevant sweep
                 + 220 retained ONLY as an external reference, NOT a tissue condition
    THRESH_NM  = 0.5, 1, 2, 5, 10
                 0.5 and 1 nM bracket the sensitive sGC range; higher values describe
                 increasingly demanding NO exposure. 1 nM is NOT treated as a universal
                 hard threshold -- that assumption is what this matrix tests.
    N_SEEDS    = 20, the SAME geometries as gen_fig2_oxygen_ensemble.py (rng(seed), seed 0..19),
                 so the paired design holds and the two runs are directly comparable.
    F_NOS2     = 0.05

WHY 0.5 nM MATTERS. At 10 uM an isolated cell reaches only 0.615 nM at the 23.2 um half-spacing,
so at a 1 nM criterion the isolated cell FAILS and the population SUCCEEDS, while at 0.5 nM the
isolated cell may already succeed on its own. Collective threshold bridging is therefore itself a
function of the signalling criterion, which is a stronger and more honest statement than declaring
one threshold universal.

Per (seed, O2, threshold): coverage of the 50 um focus, connected components, whether all cells lie
in one component, and the fraction of nearest-neighbour midpoints above threshold. Per (O2): peak,
and the ANALYTIC isolated-cell concentration at the mean half-spacing.

Writes `data/fig2_oxygen_threshold_matrix.csv`.
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import itertools
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import tornado_v2 as T                                            # noqa: E402

N_SEEDS = 20
O2_UM = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 56.0, 220.0)
O2_REFERENCE_ONLY = (220.0,)          # not a tissue condition
THRESH_NM = (0.5, 1.0, 2.0, 5.0, 10.0)
F_NOS2 = 0.05
RHO = 2.02e8
R_FOCUS = 50.0

L, N = 140.0, 141
_ax = np.linspace(-L, L, N)
DX = _ax[1] - _ax[0]
GX, GY, GZ = np.meshgrid(_ax, _ax, _ax, indexing='ij')
IN_FOCUS = (GX ** 2 + GY ** 2 + GZ ** 2) <= R_FOCUS ** 2
N_FOCUS = int(IN_FOCUS.sum())


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


def nn_pairs(pts):
    """Nearest-neighbour pairs, each cell to its closest partner."""
    out = set()
    for i, a in enumerate(pts):
        j = min((k for k in range(len(pts)) if k != i),
                key=lambda k: np.linalg.norm(np.array(a) - np.array(pts[k])))
        out.add(tuple(sorted((i, j))))
    return sorted(out)


def main():
    p = T.nominal_params()
    a0 = T.A0_UM
    vfoc = 4.0 / 3.0 * np.pi * (R_FOCUS * 1e-4) ** 3
    n_cells = int(round(F_NOS2 * RHO * vfoc))

    print('Figure 2 — oxygen x threshold matrix')
    print('  PRE-REGISTERED: %d seeds, %d O2 values, %d thresholds, %d cells/focus'
          % (N_SEEDS, len(O2_UM), len(THRESH_NM), n_cells))
    print('  220 uM is an external reference, NOT a tissue condition\n')

    geoms = [hard_core(n_cells, np.random.default_rng(s), 2 * a0) for s in range(N_SEEDS)]
    assert {len(g) for g in geoms} == {n_cells}
    half = float(np.mean([np.linalg.norm(np.array(g[i]) - np.array(g[j]))
                          for g in geoms for i, j in nn_pairs(g)])) / 2.0
    print('  mean nearest-neighbour half-spacing = %.1f um\n' % half)

    rows = []
    for o2 in O2_UM:
        coeff = T.cell_coeffs(p, o2 * 1e-6)
        solo_mid = float(T.sphere_field(np.array([half]), coeff)[0]) * 1e9
        for s, pts in enumerate(geoms):
            F = np.zeros_like(GX)
            for q in pts:
                F += T.sphere_field(
                    np.sqrt((GX - q[0]) ** 2 + (GY - q[1]) ** 2 + (GZ - q[2]) ** 2), coeff)
            F *= 1e9
            pairs = nn_pairs(pts)
            for ct in THRESH_NM:
                M = F > ct
                lab, nl = ndimage.label(M)
                ids = {lab[tuple(int(round((x + L) / DX)) for x in q)] for q in pts}
                mids = [F[tuple(int(round((x + L) / DX))
                                for x in (np.array(pts[i]) + np.array(pts[j])) / 2)] > ct
                        for i, j in pairs]
                rows.append(dict(
                    o2_uM=o2, reference_only=int(o2 in O2_REFERENCE_ONLY), seed=s, thresh_nM=ct,
                    peak_nM=float(F.max()),
                    cov_focus_pct=100.0 * float((M & IN_FOCUS).sum()) / N_FOCUS,
                    components=int(nl),
                    all_cells_one=int(len(ids) == 1 and 0 not in ids),
                    nn_bridged_pct=100.0 * float(np.mean(mids)) if mids else float('nan'),
                    solo_midpoint_nM=solo_mid,
                    solo_reaches=int(solo_mid > ct)))
        print('    [O2] = %6.1f uM  done   (isolated cell at the midpoint: %.3f nM)'
              % (o2, solo_mid))

    out = os.path.join(ROOT, 'data', 'fig2_oxygen_threshold_matrix.csv')
    keys = list(rows[0].keys())
    with open(out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join('%g' % r[k] if isinstance(r[k], float) else str(r[k])
                              for k in keys) + '\n')
    print('\n  wrote %s  (%d rows)' % (os.path.relpath(out, ROOT), len(rows)))


if __name__ == '__main__':
    main()
