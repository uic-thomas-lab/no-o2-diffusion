# -*- coding: utf-8 -*-
"""Figure 2, oxygen row: paired-ensemble regeneration on hard-core placement.

PRE-REGISTRATION — fixed before the run, and before any spread is interpreted.

    N_SEEDS        = 20          predefined; NOT chosen after seeing the spread
    O2_CONDITIONS  = 220, 56, 10, 1, 0.1 uM      as shipped
    REFERENCE_O2   = 10 uM       the condition the density row is built at
    F_NOS2         = 0.05
    PLACEMENT      = hard-core Poisson in a 50 um focus, min separation 2*a0
    REPRESENTATIVE = the seed whose peak [NO] at REFERENCE_O2 is closest to the
                     ensemble MEDIAN. Selected by this rule, never by appearance.

PAIRED DESIGN, which is the point of this file. The N_SEEDS geometries are generated ONCE
and the IDENTICAL geometry is run at every oxygen condition. Oxygen is therefore the only
variable changing within a seed, and the oxygen effect is separated from the placement effect
rather than confounded with it. Regenerating positions per oxygen condition would mix the two
and is the failure this design exists to prevent.

## Why this replaces the shipped oxygen row

Outputs `data/fig2_oxygen_ensemble.csv` (per seed x per O2) and a summary table.
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

# ---- pre-registered constants -------------------------------------------------------------
N_SEEDS = 20
O2_CONDITIONS_UM = (220.0, 56.0, 10.0, 1.0, 0.1)
REFERENCE_O2_UM = 10.0
F_NOS2 = 0.05
R_FOCUS = 50.0
THRESH_NM = (1.0, 10.0, 30.0)

# ---- grid ---------------------------------------------------------------------------------
L, N = 140.0, 141
_ax = np.linspace(-L, L, N)
DX = _ax[1] - _ax[0]
GX, GY, GZ = np.meshgrid(_ax, _ax, _ax, indexing='ij')
IN_FOCUS = (GX ** 2 + GY ** 2 + GZ ** 2) <= R_FOCUS ** 2
N_FOCUS = int(IN_FOCUS.sum())


def hard_core(n, rng, dmin, tries=8000):
    """Hard-core Poisson: uniform in the sphere, rejecting closer than dmin."""
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


def field_nM(pts, coeff):
    F = np.zeros_like(GX)
    for q in pts:
        F += T.sphere_field(np.sqrt((GX - q[0]) ** 2 + (GY - q[1]) ** 2 + (GZ - q[2]) ** 2), coeff)
    return F * 1e9


def main():
    p = T.nominal_params()
    a0 = T.A0_UM
    rho = 2.02e8
    n_cells = int(round(F_NOS2 * rho * (4.0 / 3.0 * np.pi * (R_FOCUS * 1e-4) ** 3)))

    print('Figure 2 oxygen row — paired ensemble')
    print('  PRE-REGISTERED: %d seeds, reference %.0f uM, f_NOS2 %.0f%%, %d cells/focus'
          % (N_SEEDS, REFERENCE_O2_UM, F_NOS2 * 100, n_cells))
    print('  grid %d^3 at dx = %.2f um; focus = %d voxels\n' % (N, DX, N_FOCUS))

    # geometries generated ONCE, reused at every oxygen condition
    geoms = [hard_core(n_cells, np.random.default_rng(s), 2 * a0) for s in range(N_SEEDS)]
    got = {len(g) for g in geoms}
    assert got == {n_cells}, 'placement failed to realise %d cells: got %s' % (n_cells, got)
    print('  %d geometries generated once and held fixed across all oxygen conditions' % N_SEEDS)

    rows = []
    for o2 in O2_CONDITIONS_UM:
        coeff = T.cell_coeffs(p, o2 * 1e-6)
        for s, pts in enumerate(geoms):
            F = field_nM(pts, coeff)
            r = dict(o2_uM=o2, seed=s, peak_nM=float(F.max()))
            for c in THRESH_NM:
                M = F > c
                r['cov_focus_pct_%g' % c] = 100.0 * float((M & IN_FOCUS).sum()) / N_FOCUS
                r['vol_um3_%g' % c] = float(M.sum()) * DX ** 3
            M1 = F > 1.0
            r['components_1nM'] = int(ndimage.label(M1)[1])
            lab = ndimage.label(M1)[0]
            ids = {lab[tuple(int(round((x + L) / DX)) for x in q)] for q in pts}
            r['all_cells_one_component'] = int(len(ids) == 1 and 0 not in ids)
            rows.append(r)
        print('    [O2] = %6.1f uM   done' % o2)

    # ---- representative seed, by the pre-registered rule ----------------------------------
    ref = [r for r in rows if r['o2_uM'] == REFERENCE_O2_UM]
    med = float(np.median([r['peak_nM'] for r in ref]))
    rep = min(ref, key=lambda r: abs(r['peak_nM'] - med))['seed']

    print('\n  ENSEMBLE, mean +/- SD across %d seeds' % N_SEEDS)
    print('  [O2] uM     peak nM        >1nM cov of focus     >10nM cov     components')
    for o2 in O2_CONDITIONS_UM:
        g = [r for r in rows if r['o2_uM'] == o2]
        f = lambda k: (np.mean([r[k] for r in g]), np.std([r[k] for r in g]))
        pk, cv, c10, cp = f('peak_nM'), f('cov_focus_pct_1'), f('cov_focus_pct_10'), f('components_1nM')
        print('  %7.1f  %6.2f±%.2f  %13.1f±%.1f%%  %8.2f±%.2f%%  %6.2f±%.2f'
              % (o2, pk[0], pk[1], cv[0], cv[1], c10[0], c10[1], cp[0], cp[1]))

    print('\n  representative geometry = seed %d (peak %.2f nM at %.0f uM, ensemble median %.2f)'
          % (rep, [r for r in ref if r['seed'] == rep][0]['peak_nM'], REFERENCE_O2_UM, med))
    print('  Selected by the pre-registered median rule, not by appearance.')

    out = os.path.join(ROOT, 'data', 'fig2_oxygen_ensemble.csv')
    keys = list(rows[0].keys())
    with open(out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join('%g' % r[k] if isinstance(r[k], float) else str(r[k])
                              for k in keys) + '\n')
    print('\n  wrote %s  (%d rows)' % (os.path.relpath(out, ROOT), len(rows)))
    print('  representative seed recorded as %d' % rep)


if __name__ == '__main__':
    main()
