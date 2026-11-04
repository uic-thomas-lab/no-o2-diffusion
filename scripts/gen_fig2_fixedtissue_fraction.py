# -*- coding: utf-8 -*-
"""PRODUCTION: the fixed-tissue nested-fraction density experiment.

THE EXPERIMENT. A NOS2-expressing region of radius R_expr contains a fixed cellular population;
only the LABEL on those cells changes. Nested subsets 2% subset 5% subset 15% subset 40% come from
ONE random ranking per realisation, so raising the fraction CONVERTS cells while every position
stays put and the comparison across fractions is PAIRED.

R_expr IS A BIOLOGICAL PARAMETER, NOT A BUFFER. Extending it adds sources and changes the
system: core peak [NO] rose 30.3 -> 40.6 nM from R = 90 to 150 um at 40% and was still moving 5.6%.
That is physical -- for a screened kernel the shell contribution goes as r*exp(-r/L), rising to a
maximum near r = L. Outside R_expr is ordinary non-expressing tissue: the sink in this model is
already a uniform continuum, so excluding sources IS "same sink, no sources".

FROZEN CONFIGURATION, pre-registered before this run:

    R_expr  90 um    smallest region satisfying the >=10-source requirement at 2%
    R_core  50 um    standardised measurement region, fixed BEFORE the saturation result
    L      250 um    numerical domain. NOT 200: its face value at 40% is 0.475 nM, marginal
                     against the 0.5 nM endpoint, so V(>0.5) could be clipped. 250 gives 0.113.
    dx       2 um    converged (gate 2)
    pad    120 um    gate A showed 80-240 um indistinguishable
    deposition       trilinear, frozen on peak convergence (gate 2)

## WHAT IS REPORTED, AND WHY

PRIMARY: peak [NO] and absolute full-field V(>C_T). Both remain unsaturated across the series --
amplitude and territory.

SECONDARY: core coverage, which SATURATES (100% at every threshold by 15%). That is a RESULT,
not a measurement failure: beyond ~15% NOS2+ the extra source strength stops expanding
low-nanomolar territory inside the core and appears instead as amplitude and outward expansion.
Do NOT raise the threshold selectively to keep coverage responsive -- that changes the question
and breaks comparability across fractions. Do NOT shrink the core.

Writes `data/fig2_fixedtissue_fraction.csv`.
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
from fft_field import Solver                                      # noqa: E402
import tissue as TIS                                              # noqa: E402

O2_UM = 10.0
R_EXPR = 90.0
R_CORE = 50.0
L_DOMAIN = 250.0
DX = 2.0
R_PAD = 120.0
FRACTIONS = (0.02, 0.05, 0.15, 0.40)
THRESH = (0.5, 1.0, 2.0)
N_REAL = 20
SEED0 = 20000


def per_cell_nn(pts):
    a = np.asarray(pts)
    d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def main():
    coeff = T.cell_coeffs(T.nominal_params(), O2_UM * 1e-6)
    a0 = T.A0_UM
    dmin = 2 * a0

    print('PRODUCTION — fixed tissue, nested NOS2+ fractions')
    print('  R_expr %g | R_core %g | L %g | dx %g | pad %g | trilinear | %d realisations'
          % (R_EXPR, R_CORE, L_DOMAIN, DX, R_PAD, N_REAL))

    # one solver for the whole ensemble: the kernel transform depends only on grid and [O2]
    probe = TIS.rsa_ball(R_EXPR, dmin, np.random.default_rng(SEED0))[0]
    n_max = int(round(max(FRACTIONS) * len(probe))) + 20
    s_ = Solver(L_DOMAIN, DX, coeff, n_max, 5e-4, r_pad=R_PAD)
    ax = s_.ax
    GX, GY, GZ = np.meshgrid(ax, ax, ax, indexing='ij')
    core = (GX ** 2 + GY ** 2 + GZ ** 2) <= R_CORE ** 2
    ncore = int(core.sum())
    del GX, GY, GZ
    dv = DX ** 3
    print('  %s | core %d voxels\n' % (s_.describe(), ncore))

    rows = []
    t_all = time.time()
    for r in range(N_REAL):
        rng = np.random.default_rng(SEED0 + r)
        pts, att, jam = TIS.rsa_ball(R_EXPR, dmin, rng)
        ip, _ = TIS.interior_packing(pts, R_EXPR, a0)
        labels = TIS.nested_labels(len(pts), FRACTIONS, rng)
        for f in sorted(FRACTIONS):
            idx = labels[f]
            src = pts[idx]
            C = s_.field(src)
            row = dict(realisation=r, frac_pct=f * 100, n_total=len(pts),
                       n_src=len(idx), interior_packing=ip,
                       peak_nM=float(C.max()), core_peak_nM=float(C[core].max()),
                       nn_median_um=float(np.median(per_cell_nn(src))) if len(src) > 1 else 0.0)
            for c in THRESH:
                m = C > c
                row['fullV_%g' % c] = float(m.sum()) * dv
                row['coreV_%g' % c] = float((m & core).sum()) * dv
                row['coreCov_%g' % c] = 100.0 * float((m & core).sum()) / ncore
            row['face_nM'] = float(max(C[0].max(), C[-1].max(), C[:, 0].max(),
                                       C[:, -1].max(), C[:, :, 0].max(), C[:, :, -1].max()))
            rows.append(row)
            del C
        if r == 0 or (r + 1) % 5 == 0:
            print('  realisation %2d/%d  %d cells, packing %.4f  (%.0f s elapsed)'
                  % (r + 1, N_REAL, len(pts), ip, time.time() - t_all))

    print('\n  ensemble means (mean +/- SD over %d realisations)' % N_REAL)
    print('  %6s %8s %12s %14s %14s %10s %10s'
          % ('f', 'n_src', 'peak nM', 'fullV>1 um3', 'fullV>0.5', 'core>1 %', 'core>2 %'))
    for f in sorted(FRACTIONS):
        g = [x for x in rows if abs(x['frac_pct'] - f * 100) < 1e-9]
        m = lambda k: float(np.mean([x[k] for x in g]))
        sd = lambda k: float(np.std([x[k] for x in g]))
        print('  %5.1f%% %8.1f %7.3f+/-%-5.3f %13.5g %14.5g %9.2f %10.2f'
              % (f * 100, m('n_src'), m('peak_nM'), sd('peak_nM'), m('fullV_1'),
                 m('fullV_0.5'), m('coreCov_1'), m('coreCov_2')))

    worst_face = max(x['face_nM'] for x in rows)
    print('\n  containment: worst face value across the ensemble = %.4f nM  %s'
          % (worst_face, 'ok — below the 0.5 nM endpoint' if worst_face < 0.5
             else 'TRUNCATED'))

    out = os.path.join(ROOT, 'data', 'fig2_fixedtissue_fraction.csv')
    keys = list(rows[0].keys())
    with open(out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(','.join(keys) + '\n')
        for x in rows:
            fh.write(','.join('%g' % x[k] if isinstance(x[k], float) else str(x[k])
                              for k in keys) + '\n')
    print('  wrote %s  (%d rows, %.0f s total)'
          % (os.path.relpath(out, ROOT), len(rows), time.time() - t_all))


if __name__ == '__main__':
    main()
