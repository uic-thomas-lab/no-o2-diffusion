# -*- coding: utf-8 -*-
"""FIXED-N SPACING SERIES: same sources, arranged differently.

## The question this exists to answer

The existing density series varies the NOS2+ fraction in a FIXED volume, so

    rho_NOS2 = N / V

and raising it necessarily does two inseparable things: it raises total NO-producing capacity
AND shortens typical intercellular spacing. That is not a flaw in the biological experiment --
it is what "more cells express NOS2" actually means. The limitation is that the series cannot say
WHICH of those two consequences produces the increase in signalling volume. Going from 5% to 40%
multiplies total production 8.4x; coverage could rise for that reason alone, with no contribution
from organisation whatsoever, and the density panel would look identical either way.

This series holds the sources constant and moves only the geometry:

    N = 10 cells, so Q_total = 10 * Q_cell is CONSTANT
    the radius over which those ten are scattered is swept

so it asks: how much does spatial organisation alone alter the NO field when the amount of
NO-producing machinery is held fixed?

    "more sources"   vs   "the same sources arranged differently"

## THE INDEPENDENT VARIABLE IS SPACING, NOT DENSITY

The computational domain does not change, so the global density N/V_domain is CONSTANT here.
What changes is local packing. Do not call this a density series. The declared independent
variable is **mean nearest-neighbour spacing**, measured per realisation and reported; the
scatter radius is only the knob that sets it.

## PRE-REGISTRATION -- fixed before the run, before any result is seen

    N_CELLS    = 10        fixed. Not 2: two cells answer a pair-overlap question and do not
                           represent a population -- no meaningful connectivity or fragmentation.
    R_SCATTER  = 28, 40, 55, 70, 85, 100 um      six packings
    N_SEEDS    = 20        per spacing, matching the oxygen ensemble
    O2         = 10 uM     uniform, the Figure 2 anchor. A uniform field is what makes the
                           isolation clean: no vessel, no gradient, no boundary asymmetry.
    PLACEMENT  = hard-core Poisson, minimum centre-to-centre separation 2*a0, as everywhere else

    PRIMARY endpoint    absolute V([NO] > C_T), C_T = 0.5, 1, 2 nM
                        NOT focus-normalised: changing the occupied radius makes any fixed
                        denominator artificial.
    SECONDARY           peak [NO]; connected components at 1 nM; fraction of neighbour midpoints
                        above threshold; convex-hull volume of the cell centres, as an
                        occupied-volume normalisation.

## THE PREDICTION IS A PREDICTION, NOT AN ASSUMPTION

Two effects pull opposite ways at fixed total source:

    tighter spacing -> greater field summation      -> higher concentration
    tighter spacing -> more redundant overlap       -> less unique territory per cell

so an interior optimum in total signalling volume is physically plausible. It is to be TESTED.
If V is monotonic in spacing, that is the result and it gets reported as such.

## "COLLECTIVE" DOES NOT MEAN COOPERATIVE

With feedback off the NO equation is LINEAR and fields superimpose exactly. So even if tighter
packing raises signalling volume, this establishes a **geometric population effect of source
spacing** -- collective field overlap at fixed total NO production -- and NOT biochemical
cooperation or synergy. The threshold-bridging metric is "collective" only in the intuitive
sense that several sources together cross a threshold one cannot; it too arises entirely from
linear superposition. Do not write "cooperative", "synergy" or "nonlinear" anywhere near this.

## Confound guard

Changing the scatter radius must not change anything else that matters -- in particular the
distance from the sources to the computational boundary. The grid is enlarged to +/-200 um for
this series and the maximum concentration on the grid face is recorded per realisation; if it
approaches C_T the absolute volume is truncated and the point is not usable.

Writes `data/fig2_fixedN_spacing.csv`.
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
from scipy import ndimage
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import tornado_v2 as T                                            # noqa: E402

N_CELLS = 10
R_SCATTER = (28.0, 40.0, 55.0, 70.0, 85.0, 100.0)
N_SEEDS = 20
O2_UM = 10.0
THRESH_NM = (0.5, 1.0, 2.0)
SEED_OFFSET = 5000          # distinct from the oxygen (0..) and density (1000+) ensembles

L, N = 200.0, 201
_ax = np.linspace(-L, L, N)
DX = _ax[1] - _ax[0]
GX, GY, GZ = np.meshgrid(_ax, _ax, _ax, indexing='ij')
DV = DX ** 3


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


def nn_pairs(pts):
    out = set()
    for i, a in enumerate(pts):
        j = min((k for k in range(len(pts)) if k != i),
                key=lambda k: np.linalg.norm(np.array(a) - np.array(pts[k])))
        out.add(tuple(sorted((i, j))))
    return sorted(out)


def main():
    p = T.nominal_params()
    a0 = T.A0_UM
    coeff = T.cell_coeffs(p, O2_UM * 1e-6)

    print('FIXED-N SPACING SERIES — same sources, arranged differently')
    print('  N = %d cells (Q_total constant), %d seeds, [O2] = %g uM uniform'
          % (N_CELLS, N_SEEDS, O2_UM))
    print('  independent variable is MEAN NEAREST-NEIGHBOUR SPACING, not density')
    print('  prediction (interior optimum) is TO BE TESTED, not assumed\n')

    rows = []
    t0 = time.time()
    for R in R_SCATTER:
        acc = []
        for s in range(N_SEEDS):
            rng = np.random.default_rng(SEED_OFFSET + int(R) * 100 + s)
            pts = hard_core(N_CELLS, rng, 2 * a0, R)
            assert len(pts) == N_CELLS, \
                'R = %g um could not seat %d cells at seed %d (got %d)' % (R, N_CELLS, s, len(pts))

            F = np.zeros_like(GX)
            for q in pts:
                F += T.sphere_field(
                    np.sqrt((GX - q[0]) ** 2 + (GY - q[1]) ** 2 + (GZ - q[2]) ** 2), coeff)
            F *= 1e9

            pairs = nn_pairs(pts)
            nn = float(np.mean([np.linalg.norm(np.array(pts[i]) - np.array(pts[j]))
                                for i, j in pairs]))
            hull = float(ConvexHull(np.array(pts)).volume)
            face = float(max(F[0].max(), F[-1].max(), F[:, 0].max(),
                             F[:, -1].max(), F[:, :, 0].max(), F[:, :, -1].max()))

            r = dict(R_scatter_um=R, seed=s, nn_mean_um=nn, hull_um3=hull,
                     peak_nM=float(F.max()), max_face_nM=face)
            for c in THRESH_NM:
                M = F > c
                r['vol_%g_um3' % c] = float(M.sum()) * DV
            lab, nl = ndimage.label(F > 1.0)
            r['components_1nM'] = int(nl)
            mids = [F[tuple(int(round((x + L) / DX))
                            for x in (np.array(pts[i]) + np.array(pts[j])) / 2)] > 1.0
                    for i, j in pairs]
            r['nn_mid_above_1nM_pct'] = 100.0 * float(np.mean(mids))
            acc.append(r)
            rows.append(r)

        m = lambda k: float(np.mean([a[k] for a in acc]))
        sd = lambda k: float(np.std([a[k] for a in acc]))
        print('  R = %5.0f um   NN spacing %5.1f um   peak %6.2f nM   '
              'V>1nM %9.3g +/- %.2g um3   hull %8.3g um3   face %.3f nM'
              % (R, m('nn_mean_um'), m('peak_nM'), m('vol_1_um3'), sd('vol_1_um3'),
                 m('hull_um3'), max(a['max_face_nM'] for a in acc)))
    print('\n  %.1f s' % (time.time() - t0))

    worst = max(r['max_face_nM'] for r in rows)
    print('  CONFOUND GUARD: highest [NO] on the grid face = %.4f nM (criterion 0.5 nM)'
          % worst)
    print('     %s' % ('contained — absolute volumes are measurements' if worst < 0.5 * 0.2
                       else 'TOO CLOSE TO THE BOUNDARY — volumes may be truncated'))

    # ---- is V(>1 nM) monotonic in spacing, or does it turn over? ----
    print('\n  V(>1 nM) against spacing — the prediction under test:')
    means = []
    for R in R_SCATTER:
        g = [r for r in rows if r['R_scatter_um'] == R]
        means.append((float(np.mean([r['nn_mean_um'] for r in g])),
                      float(np.mean([r['vol_1_um3'] for r in g]))))
    for nn, v in means:
        print('    NN %5.1f um   V>1nM %10.4g um3' % (nn, v))
    i = int(np.argmax([v for _, v in means]))
    interior = 0 < i < len(means) - 1
    print('  maximum at NN = %.1f um   -> %s'
          % (means[i][0], 'INTERIOR OPTIMUM' if interior else 'AT AN END — monotonic over the '
             'sampled range'))

    out = os.path.join(ROOT, 'data', 'fig2_fixedN_spacing.csv')
    keys = list(rows[0].keys())
    with open(out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join('%g' % r[k] if isinstance(r[k], float) else str(r[k])
                              for k in keys) + '\n')
    print('\n  wrote %s  (%d rows)' % (os.path.relpath(out, ROOT), len(rows)))


if __name__ == '__main__':
    main()
