# -*- coding: utf-8 -*-
r"""Figure 2 — the two readouts folded into the existing summary panels.

NO NEW PANELS. Both quantities go INTO panels D and H, not beside them.

    panel D (oxygen)   third readout: THRESHOLD-FREE integrated dose  \int C dV
    panel H (density)  second readout: ABSOLUTE volume above 1 nM, in um^3

## Why each one exists

**Integrated dose vs oxygen.** The figure's central claim -- coverage is non-monotonic in oxygen --
is stated at a chosen criterion, so it invites the question of whether the non-monotonicity is an
artefact of choosing 0.5 or 1 nM. \int C dV answers that with NO threshold at all.

ITS OPTIMUM IS ANALYTIC. For steady state with linear removal, integrating the governing
equation over all space gives \int C dV = Q / k(O2). With Michaelis-Menten synthesis and
k = k_obs.[O2] + k_anoxia,

    \int C dV  proportional to  x / [(Km + x)(a x + b)],    a = k_obs,  b = k_anoxia

and d/dx of that vanishes at

    [O2]_opt = sqrt(b.Km/a) = sqrt(Km . [O2]*) = sqrt(42 x 24.83) = 32.3 uM

**PLOT THE DOSE NORMALISED, NOT ABSOLUTE.** Unlike the >1 nM volume, the dose integral IS
truncated by the grid: C is positive everywhere and 0.337 nM on the face is not negligible tail.
The LOCATION and TOPOLOGY of the optimum are validated; the absolute value of \int C dV is not,
because neither the finite-domain truncation nor the Q/k normalisation has been verified. Until
they are, this is normalised to its own maximum.

    oxygen   rng(seed),        seed 0..19    5 cells, 5% expressing fraction
    density  rng(1000 + seed), seed 0..49    2 / 5 / 15 / 40%

Any drift in placement here would silently decouple these curves from the coverage curves they
are drawn beside. The hard_core() routine is copied verbatim from the two generators.

Writes `data/fig2_panelD_oxygen_v2.csv` and `data/fig2_panelH_density_v2.csv`, and leaves the
existing v1 CSVs untouched.
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
import constants as _C                                            # noqa: E402
import parameters as _P                                           # noqa: E402

N_SEEDS_O2 = 20
N_SEEDS_DENS = 50
DENSITY_OFFSET = 1000
F_NOS2 = 0.05
O2_DENSITY_UM = 10.0
FRACTIONS = (0.02, 0.05, 0.15, 0.40)
RHO = 2.02e8
R_FOCUS = 50.0

# Tumour-relevant sweep, refined either side of the analytic stationary point so the arch is
# resolved rather than inferred. 220 uM is retained as the oxygenated-medium REFERENCE only.
O2_UM = (0.1, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 56.0, 100.0, 220.0)

L, N = 140.0, 141
_ax = np.linspace(-L, L, N)
DX = _ax[1] - _ax[0]
GX, GY, GZ = np.meshgrid(_ax, _ax, _ax, indexing='ij')
IN_FOCUS = (GX ** 2 + GY ** 2 + GZ ** 2) <= R_FOCUS ** 2
N_FOCUS = int(IN_FOCUS.sum())
DV = DX ** 3


def hard_core(n, rng, dmin, tries=20000):
    """Verbatim from gen_fig2_oxygen_ensemble.py / gen_fig2_density_ensemble.py."""
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


def field(pts, coeff):
    F = np.zeros_like(GX)
    for q in pts:
        F += T.sphere_field(
            np.sqrt((GX - q[0]) ** 2 + (GY - q[1]) ** 2 + (GZ - q[2]) ** 2), coeff)
    return F * 1e9


def analytic_dose(x_uM):
    """x / [(Km + x)(k_obs.x + k_anoxia)], up to a constant. Shape only -- see the docstring."""
    km = _P.KM * 1e6
    a, b = _C.K_OBS_PUB, _C.K_ANOXIA_PUB
    return x_uM / ((km + x_uM) * (a * x_uM * 1e-6 + b))


def write(path, keys, rows):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join(('%g' % r[k]) if isinstance(r[k], float) else str(r[k])
                              for k in keys) + '\n')
    print('  wrote %s  (%d rows)' % (os.path.relpath(path, ROOT), len(rows)))


def main():
    p = T.nominal_params()
    a0 = T.A0_UM
    vfoc = 4.0 / 3.0 * np.pi * (R_FOCUS * 1e-4) ** 3
    km = _P.KM * 1e6
    o2star = _C.O2_STAR_UM

    print('Figure 2 — integrated dose (panel D) and absolute volume (panel H)')
    print('  Km = %.4g uM   [O2]* = %.4g uM   sqrt(Km.[O2]*) = %.4g uM  <-- ANALYTIC optimum'
          % (km, o2star, np.sqrt(km * o2star)))
    print('  analytic stationary point, NOT a sampled argmax\n')

    # ---------------- panel D: oxygen ----------------
    n5 = int(round(F_NOS2 * RHO * vfoc))
    geoms = [hard_core(n5, np.random.default_rng(s), 2 * a0) for s in range(N_SEEDS_O2)]
    assert {len(g) for g in geoms} == {n5}, 'oxygen placement drifted from the locked seeds'
    print('  oxygen: %d seeds x %d cells x %d tensions' % (N_SEEDS_O2, n5, len(O2_UM)))

    t0 = time.time()
    drows = []
    for o2 in O2_UM:
        coeff = T.cell_coeffs(p, o2 * 1e-6)
        pk, c1, v1, ds = [], [], [], []
        for g in geoms:
            F = field(g, coeff)
            pk.append(float(F.max()))
            c1.append(100.0 * float(((F > 1.0) & IN_FOCUS).sum()) / N_FOCUS)
            v1.append(float((F > 1.0).sum()) * DV)
            ds.append(float(F.sum()) * DV)
        drows.append(dict(
            O2_uM=o2,
            peak_nM=float(np.mean(pk)), peak_SD=float(np.std(pk)),
            cov1_pct=float(np.mean(c1)), cov1_SD=float(np.std(c1)),
            vol1_um3=float(np.mean(v1)), vol1_SD=float(np.std(v1)),
            dose_nM_um3=float(np.mean(ds)), dose_SD=float(np.std(ds))))
        print('    [O2] = %6.1f uM   peak %6.2f nM   cov %5.1f%%   dose %.4g'
              % (o2, drows[-1]['peak_nM'], drows[-1]['cov1_pct'], drows[-1]['dose_nM_um3']))
    print('  oxygen pass: %.1f s' % (time.time() - t0))

    dmax = max(r['dose_nM_um3'] for r in drows)
    cmax = max(r['cov1_pct'] for r in drows)
    for r in drows:
        r['dose_pct_of_max'] = 100.0 * r['dose_nM_um3'] / dmax
        r['dose_SD_pct'] = 100.0 * r['dose_SD'] / dmax
        r['analytic_pct_of_max'] = 100.0 * analytic_dose(r['O2_uM']) / analytic_dose(
            np.sqrt(km * o2star))
    print('\n  numeric dose argmax over the SAMPLED set = %g uM   (analytic %.4g uM)'
          % (max(drows, key=lambda r: r['dose_nM_um3'])['O2_uM'], np.sqrt(km * o2star)))
    print('  coverage argmax over the sampled set      = %g uM   (max %.1f%% of focus)'
          % (max(drows, key=lambda r: r['cov1_pct'])['O2_uM'], cmax))

    write(os.path.join(ROOT, 'data', 'fig2_panelD_oxygen_v2.csv'),
          ['O2_uM', 'peak_nM', 'peak_SD', 'cov1_pct', 'cov1_SD', 'vol1_um3', 'vol1_SD',
           'dose_nM_um3', 'dose_SD', 'dose_pct_of_max', 'dose_SD_pct', 'analytic_pct_of_max'],
          drows)

    # ---------------- panel H: density ----------------
    coeff10 = T.cell_coeffs(p, O2_DENSITY_UM * 1e-6)
    print('\n  density: %d seeds x %d fractions at %g uM' % (N_SEEDS_DENS, len(FRACTIONS),
                                                            O2_DENSITY_UM))
    t0 = time.time()
    hrows = []
    for frac in FRACTIONS:
        n = int(round(frac * RHO * vfoc))
        pk, c1, c10, v1, v10, face = [], [], [], [], [], []
        for s in range(N_SEEDS_DENS):
            g = hard_core(n, np.random.default_rng(DENSITY_OFFSET + s), 2 * a0)
            assert len(g) == n, 'density placement drifted at frac %g seed %d' % (frac, s)
            F = field(g, coeff10)
            pk.append(float(F.max()))
            c1.append(100.0 * float(((F > 1.0) & IN_FOCUS).sum()) / N_FOCUS)
            c10.append(100.0 * float(((F > 10.0) & IN_FOCUS).sum()) / N_FOCUS)
            v1.append(float((F > 1.0).sum()) * DV)
            v10.append(float((F > 10.0).sum()) * DV)
            face.append(max(F[0].max(), F[-1].max(), F[:, 0].max(), F[:, -1].max(),
                            F[:, :, 0].max(), F[:, :, -1].max()))
        hrows.append(dict(
            frac_pct=frac * 100, n_cells=n,
            cov1_pct=float(np.mean(c1)), cov1_SD=float(np.std(c1)),
            cov10_pct=float(np.mean(c10)), cov10_SD=float(np.std(c10)),
            vol1_um3=float(np.mean(v1)), vol1_SD=float(np.std(v1)),
            vol10_um3=float(np.mean(v10)), vol10_SD=float(np.std(v10)),
            peak_nM=float(np.mean(pk)), peak_SD=float(np.std(pk)),
            max_face_nM=float(np.max(face))))
        r = hrows[-1]
        print('    f = %4.0f%%  (%2d cells)  cov %5.1f%%   V>1nM %.4g +/- %.2g um3   '
              'max face %.3f nM' % (frac * 100, n, r['cov1_pct'], r['vol1_um3'],
                                    r['vol1_SD'], r['max_face_nM']))
    print('  density pass: %.1f s' % (time.time() - t0))

    worst = max(r['max_face_nM'] for r in hrows)
    print('\n  DOMAIN CHECK: highest [NO] on the grid face across all density realisations = '
          '%.3f nM' % worst)
    if worst >= 1.0:
        print('  FAIL: the >1 nM region REACHES the grid boundary. Absolute volume is a LOWER '
              'BOUND, not a measurement, and MUST NOT be plotted as an unbounded quantity.')
    else:
        print('  PASS: %.0f%% below the 1 nM criterion, so the >1 nM region is fully contained '
              'and the absolute volume is a measurement.' % (100 * (1 - worst)))

    write(os.path.join(ROOT, 'data', 'fig2_panelH_density_v2.csv'),
          ['frac_pct', 'n_cells', 'cov1_pct', 'cov1_SD', 'cov10_pct', 'cov10_SD',
           'vol1_um3', 'vol1_SD', 'vol10_um3', 'vol10_SD', 'peak_nM', 'peak_SD',
           'max_face_nM'], hrows)

    print('\n  Panel D plots the dose NORMALISED (dose_pct_of_max). The absolute integral is')
    print('     truncated by the finite grid and its Q/k normalisation is unverified.')


if __name__ == '__main__':
    main()
