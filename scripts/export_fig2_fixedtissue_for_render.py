# -*- coding: utf-8 -*-
"""Export representative fixed-tissue fraction fields for the Wolfram renderer.

SAME TISSUE, SAME LABEL RANKING, IN ALL FOUR PANELS. Only the fraction of cells designated NOS2+
changes; every cell POSITION is identical and the subsets are NESTED. The legend must say so, or a
reader will assume the geometry was redrawn.

THIS SERIES NEEDS ITS OWN BOX. At 40% the full-field V(>0.5 nM) is ~3.1e7 um3, a roughly 195 um
equivalent radius -- far beyond the 99 um oxygen box or the 135 um old-density box. The extent is
MEASURED here and reported so the renderer's box is sized rather than guessed.

SELECTION IS A PREDECLARED MULTIVARIATE RULE: the realisation minimising its standardized
distance from the ensemble means of peak [NO] and V(>1 nM) across ALL FOUR fractions. An earlier
version used median-peak at 40% alone, which lets ONE panel drive the pick -- it chose realisation
11, which ranks 12th of 20 under this rule. Never chosen for appearance.

Writes `data/fig2_fixedtissue_render/<stem>.f32` + `.json`.
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import tornado_v2 as T                                            # noqa: E402
from fft_field import Solver                                      # noqa: E402
import tissue as TIS                                              # noqa: E402

O2_UM = 10.0
R_EXPR = 90.0
R_CORE = 50.0
L_DOMAIN = 250.0
DX = 2.0
R_PAD = 120.0
FRACTIONS = (0.02, 0.05, 0.15, 0.40)
SEED0 = 20000                       # must match gen_fig2_fixedtissue_fraction.py
OUT = os.path.join(ROOT, 'data', 'fig2_fixedtissue_render')
CSV = os.path.join(ROOT, 'data', 'fig2_fixedtissue_fraction.csv')


def main():
    rows = list(csv.DictReader(open(CSV, encoding='utf-8')))
    byf = defaultdict(list)
    for r in rows:
        byf[float(r['frac_pct'])].append(r)

    coeff = T.cell_coeffs(T.nominal_params(), O2_UM * 1e-6)
    a0 = T.A0_UM
    dmin = 2 * a0
    os.makedirs(OUT, exist_ok=True)

    # ONE realisation for ALL four panels: the nested labelling means the 2% cells are a
    # subset of the 5%, and so on, so a single tissue lets the reader watch the SAME cells light
    # up as the fraction rises. Choosing a different realisation per fraction would redraw the
    # geometry between panels and destroy the property the experiment was built to have.
    #
    # PREDECLARED MULTIVARIATE CRITERION: the realisation minimising its
    # standardized distance from the ensemble means of peak [NO] and V(>1 nM) across ALL FOUR
    # fractions, so no single panel drives the pick.
    by_r = defaultdict(dict)
    for r in rows:
        by_r[int(r['realisation'])][float(r['frac_pct'])] = r
    reals = sorted(by_r)
    keys = [(f * 100, k) for f in FRACTIONS for k in ('peak_nM', 'fullV_1')]
    cols = {k: np.array([float(by_r[r][k[0]][k[1]]) for r in reals]) for k in keys}
    d2 = np.zeros(len(reals))
    for k in keys:
        d2 += ((cols[k] - cols[k].mean()) / cols[k].std(ddof=1)) ** 2
    r_sel = reals[int(np.argmin(d2))]
    print('Fixed-tissue fraction renders')
    print('  ONE realisation for all four panels, multivariate pick: #%d '
          '(standardized distance %.3f over %d quantities)'
          % (r_sel, float(np.sqrt(d2.min())), len(keys)))

    rng = np.random.default_rng(SEED0 + r_sel)
    pts, att, jam = TIS.rsa_ball(R_EXPR, dmin, rng)
    ip, _ = TIS.interior_packing(pts, R_EXPR, a0)
    labels = TIS.nested_labels(len(pts), FRACTIONS, rng)
    print('  tissue: %d cells in R_expr = %g um, interior packing %.4f' % (len(pts), R_EXPR, ip))

    s_ = Solver(L_DOMAIN, DX, coeff, len(labels[0.40]), 5e-4, r_pad=R_PAD)
    ax = s_.ax
    n = s_.n
    print('  %s\n' % s_.describe())

    biggest = 0.0
    for f in sorted(FRACTIONS):
        idx = labels[f]
        F = s_.field(pts[idx]).astype(np.float32)
        ext = {}
        for c in (0.5, 1.0):
            m = F > c
            if m.any():
                ix, iy, iz = np.where(m)
                ext['%g' % c] = max(float(np.abs(ax[i]).max()) for i in (ix, iy, iz))
            else:
                ext['%g' % c] = 0.0
        biggest = max(biggest, ext['0.5'])
        stem = 'fixedtissue_%gpct' % (f * 100)
        F.tofile(os.path.join(OUT, stem + '.f32'))
        json.dump({
            'role': 'fixed-tissue nested-fraction series: ONE tissue, ONE ranking, '
                    'nested labels -- only the FRACTION EXPRESSING changes',
            'frac_pct': f * 100, 'realisation': r_sel,
            'n_total_cells': len(pts), 'n_src': len(idx),
            'R_expr_um': R_EXPR, 'R_core_um': R_CORE, 'interior_packing': ip,
            'O2_uM': O2_UM, 'field_units': 'nM',
            'shape': [n, n, n], 'order': 'C', 'dtype': 'float32_le',
            'grid_bounds_um': [-L_DOMAIN, L_DOMAIN], 'grid_spacing_um': DX,
            'cell_radius_um': float(a0),
            'cell_centres_um': [[float(v) for v in q] for q in pts[idx]],
            'peak_nM': float(F.max()), 'max_extent_um': ext,
            'selection_rule': 'one realisation for all fractions, chosen as the minimiser of '
                              'standardized distance from the ensemble means of peak and V(>1 nM) '
                              'across all four fractions (predeclared, multivariate)',
        }, open(os.path.join(OUT, stem + '.json'), 'w', encoding='utf-8'), indent=2)
        print('  %-24s %4d src   peak %7.3f nM   0.5 nM reaches %5.1f um   1 nM %5.1f um'
              % (stem, len(idx), F.max(), ext['0.5'], ext['1']))
        del F

    print('\n  largest 0.5 nM extent = %.1f um  ->  this series needs a box of at least %d um'
          % (biggest, int(biggest + 8)))
    print('  far beyond the 99 um oxygen box and the 135 um old-density box: its own box and')
    print('    its own crop, not comparable by eye to the other series (the 50 um grid is the ruler).')
    print('  grid is %d^3 / +/-%g um and must be passed to panelFull explicitly.' % (n, L_DOMAIN))


if __name__ == '__main__':
    main()
