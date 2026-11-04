# -*- coding: utf-8 -*-
"""Export the six rendered Figure 2 NO fields as raw scalar volumes plus metadata.

## Output, per condition

  <stem>.f32   raw little-endian float32, C order, shape (N,N,N), units nM
  <stem>.json  cell coordinates and radius, grid bounds and spacing, units, [O2],
               expressing fraction, seed, and the parameter-set identity

Read in Mathematica with BinaryReadList[..., "Real32"] and ArrayReshape.

  oxygen ensemble   rng(seed)          -> seed 7 at 220 / 56 / 10 uM
  density ensemble  rng(1000 + seed)   -> seed 8 @2%, seed 4 @15%, seed 36 @40%

The 5% / 10 uM anchor is the OXYGEN seed 7 field, not density seed 48. Seed 48 remains the
density ensemble's pre-specified representative and is retained in the audit record, but is not
rendered: rendering it would make one biological condition appear twice with two different
geometries and destroy the intersection of the two axes.
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
import parameters as _P                                           # noqa: E402
import constants as _C                                            # noqa: E402

OUT = os.path.join(ROOT, 'data', 'fig2_render')
R_FOCUS, RHO = 50.0, 2.02e8
L, N = 140.0, 141
_ax = np.linspace(-L, L, N)
DX = _ax[1] - _ax[0]
GX, GY, GZ = np.meshgrid(_ax, _ax, _ax, indexing='ij')

OXYGEN_SEED = 7
DENSITY_SEEDS = {0.02: 8, 0.15: 4, 0.40: 36}
DENSITY_OFFSET = 1000


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


def write(stem, pts, o2_uM, frac, seed, role):
    p = T.nominal_params()
    F = np.zeros_like(GX)
    for q in pts:
        F += T.sphere_field(
            np.sqrt((GX - q[0]) ** 2 + (GY - q[1]) ** 2 + (GZ - q[2]) ** 2),
            T.cell_coeffs(p, o2_uM * 1e-6))
    F = (F * 1e9).astype(np.float32)

    os.makedirs(OUT, exist_ok=True)
    F.tofile(os.path.join(OUT, stem + '.f32'))
    meta = {
        'role': role,
        'field_units': 'nM',
        'shape': [N, N, N],
        'order': 'C',
        'dtype': 'float32_le',
        'grid_bounds_um': [-L, L],
        'grid_spacing_um': DX,
        'axis_um': 'x, y, z all span grid_bounds_um with grid_spacing_um',
        'focus_radius_um': R_FOCUS,
        'cell_radius_um': float(T.A0_UM),
        'cell_centres_um': [[float(x) for x in q] for q in pts],
        'n_cells': len(pts),
        'O2_uM': o2_uM,
        'expressing_fraction': frac,
        'rho_cells_per_mL': RHO,
        'seed': seed,
        'thresholds_nM': [1.0, 10.0, 30.0],
        'peak_nM': float(F.max()),
        'parameter_set': {
            'K_m_NOS2_M': _P.KM, 'V_max_NOS2_M_s': _P.VMAX,
            'RHO_TUMOUR': _C.RHO_TUMOUR, 'PHI_TUMOUR': _C.PHI_TUMOUR,
            'CELL_VOLUME_MODEL_UM3': _C.CELL_VOLUME_MODEL_UM3,
            'K_OBS_PUB': _C.K_OBS_PUB, 'K_ANOXIA_PUB': _C.K_ANOXIA_PUB,
            'O2_STAR_uM': _C.O2_STAR_UM,
            'decisions': 'parameter set of src/parameters.py'},
    }
    with open(os.path.join(OUT, stem + '.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)
    print('  %-34s %2d cells  peak %6.2f nM   %s' % (stem, len(pts), F.max(), role))
    return meta


def main():
    a0 = T.A0_UM
    vfoc = 4.0 / 3.0 * np.pi * (R_FOCUS * 1e-4) ** 3
    print('Figure 2 render export — seeds locked\n')

    # oxygen series: ONE geometry, three tensions. 10 uM is also the shared anchor.
    n5 = int(round(0.05 * RHO * vfoc))
    pts = hard_core(n5, np.random.default_rng(OXYGEN_SEED), 2 * a0)
    assert len(pts) == n5, 'oxygen seed %d did not realise %d cells' % (OXYGEN_SEED, n5)
    for o2 in (220.0, 56.0, 40.0, 30.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.1):
        role = ('SHARED ANCHOR 5%/10uM — oxygen and density axes intersect here' if o2 == 10.0
                else 'oxygenated-medium REFERENCE, not a tissue condition' if o2 == 220.0
                else 'oxygen series')
        write('oxygen_seed%d_%guM' % (OXYGEN_SEED, o2), pts, o2, 0.05, OXYGEN_SEED, role)

    write('density_seed%d_5pct' % OXYGEN_SEED, pts, 10.0, 0.05, OXYGEN_SEED,
          'density series, 5% -- SAME geometry and condition as the 10 uM oxygen panel')
    for frac, sd in sorted(DENSITY_SEEDS.items()):
        n = int(round(frac * RHO * vfoc))
        q = hard_core(n, np.random.default_rng(DENSITY_OFFSET + sd), 2 * a0)
        assert len(q) == n, 'density seed %d did not realise %d cells' % (sd, n)
        write('density_seed%d_%gpct' % (sd, frac * 100), q, 10.0, frac, sd, 'density series')

    print('\n  wrote %s' % os.path.relpath(OUT, ROOT))
    print('  density seed 48 (5%% ensemble representative) retained in the record, NOT rendered.')


if __name__ == '__main__':
    main()
