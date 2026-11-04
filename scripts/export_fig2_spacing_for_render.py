# -*- coding: utf-8 -*-
"""Export representative fixed-N spacing fields as raw volumes for the Wolfram renderer.

WHY THESE PANELS EARN THEIR SPACE. The spacing curve shows a broad interior maximum, but a
curve cannot show WHY. These renders can: at tight packing the ten fields merge into a single
blob with a high centre and little reach; at wide packing they separate into ten nearly
independent shells. The tradeoff between summation and unique territory is visible directly.

SAME TEN CELLS AND THE SAME TOTAL NO SOURCE IN EVERY PANEL. Only the arrangement differs.
That is the whole point of the experiment and the legend must say it, or a reader will assume
more cells were added.

REPRESENTATIVE SELECTION is the project rule, unchanged: the realisation whose peak [NO] is
closest to the ensemble MEDIAN at that spacing. Never chosen for connectivity or appearance.

GRID. The spacing sweep ran on +/-200 um at dx = 2 um (201^3) because the widest scatter
needed the room. These exports keep that grid, so the Wolfram side must be told 201 and 200 --
it is hardwired to 141 and 140 for the oxygen and density series.

Writes `data/fig2_spacing_render/<stem>.f32` + `.json`.
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import tornado_v2 as T                                            # noqa: E402

OUT = os.path.join(ROOT, 'data', 'fig2_spacing_render')
CSV = os.path.join(ROOT, 'data', 'fig2_fixedN_spacing.csv')

N_CELLS = 10
O2_UM = 10.0
SEED_OFFSET = 5000                       # must match gen_fig2_fixedN_spacing.py
R_RENDER = (28.0, 40.0, 55.0, 70.0, 100.0)

L, N = 200.0, 201
_ax = np.linspace(-L, L, N)
DX = _ax[1] - _ax[0]
GX, GY, GZ = np.meshgrid(_ax, _ax, _ax, indexing='ij')


def hard_core(n, rng, dmin, radius, tries=200000):
    """Verbatim from gen_fig2_fixedN_spacing.py -- placement must not drift."""
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


def main():
    import csv
    from collections import defaultdict
    rows = list(csv.DictReader(open(CSV, encoding='utf-8')))
    byR = defaultdict(list)
    for r in rows:
        byR[float(r['R_scatter_um'])].append(r)

    p = T.nominal_params()
    a0 = T.A0_UM
    coeff = T.cell_coeffs(p, O2_UM * 1e-6)
    os.makedirs(OUT, exist_ok=True)

    print('Fixed-N spacing renders — %d cells, Q_total constant, %g uM uniform' % (N_CELLS, O2_UM))
    print('  grid +/-%g um, %d^3, dx = %g um\n' % (L, N, DX))

    for R in R_RENDER:
        g = byR[R]
        med = float(np.median([float(r['peak_nM']) for r in g]))
        rep = min(g, key=lambda r: abs(float(r['peak_nM']) - med))
        seed = int(rep['seed'])
        nn = float(rep['nn_mean_um'])

        rng = np.random.default_rng(SEED_OFFSET + int(R) * 100 + seed)
        pts = hard_core(N_CELLS, rng, 2 * a0, R)
        assert len(pts) == N_CELLS
        assert abs(float(rep['peak_nM'])) > 0

        F = np.zeros_like(GX)
        for q in pts:
            F += T.sphere_field(
                np.sqrt((GX - q[0]) ** 2 + (GY - q[1]) ** 2 + (GZ - q[2]) ** 2), coeff)
        F = (F * 1e9).astype(np.float32)

        # extent of the OUTER contour, so the renderer's box can be sized without guessing
        ext = {}
        for c in (0.5, 1.0):
            M = F > c
            if M.any():
                ix, iy, iz = np.where(M)
                ext['%g' % c] = max(float(np.abs(_ax[i]).max()) for i in (ix, iy, iz))
            else:
                ext['%g' % c] = 0.0

        stem = 'spacing_R%g' % R
        F.tofile(os.path.join(OUT, stem + '.f32'))
        json.dump({
            'role': 'fixed-N spacing series: SAME 10 cells, SAME total NO source, '
                    'different arrangement',
            'R_scatter_um': R, 'nn_mean_um': nn, 'seed': seed,
            'n_cells': N_CELLS, 'O2_uM': O2_UM,
            'field_units': 'nM', 'shape': [N, N, N], 'order': 'C', 'dtype': 'float32_le',
            'grid_bounds_um': [-L, L], 'grid_spacing_um': DX,
            'cell_radius_um': float(T.A0_UM),
            'cell_centres_um': [[float(x) for x in q] for q in pts],
            'peak_nM': float(F.max()),
            'ensemble_median_peak_nM': med,
            'max_extent_um': ext,
            'selection_rule': 'realisation whose peak [NO] is closest to the ensemble median',
        }, open(os.path.join(OUT, stem + '.json'), 'w', encoding='utf-8'), indent=2)

        print('  %-16s NN %5.1f um  seed %2d  peak %6.2f nM (median %6.2f)  '
              'outer 0.5 nM reaches %5.1f um'
              % (stem, nn, seed, F.max(), med, ext['0.5']))

    biggest = 0.0
    for R in R_RENDER:
        m = json.load(open(os.path.join(OUT, 'spacing_R%g.json' % R), encoding='utf-8'))
        biggest = max(biggest, m['max_extent_um']['0.5'])
    print('\n  largest 0.5 nM extent across the series = %.1f um' % biggest)
    print('     -> the Wolfram box for this series must be at least %d um' % int(biggest + 6))
    print('  the renderer is hardwired to 141^3 / +/-140 for the other two series;')
    print('     this one is %d^3 / +/-%g and must be passed explicitly.' % (N, L))


if __name__ == '__main__':
    main()
