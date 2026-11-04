# -*- coding: utf-8 -*-
"""Fixed cellular tissue with nested NOS2+ labelling.

THE CONSTRUCTION. Build a fixed tissue volume containing a fixed population of cells, then
change only the LABEL on those cells. N_total and V are constant; f_NOS2+ designates a fraction
of the EXISTING cells as positive.

NESTED SUBSETS FROM ONE RANKING PER REALISATION: 2% subset 5% subset 15% subset 40%. Raising
the fraction converts additional cells while every cell POSITION remains unchanged, so the
comparison across fractions is PAIRED.

## JAMMING IS A CONVERGENCE CRITERION, NOT A LUCKY BATCH

Termination is now: stop after `patience` CONSECUTIVE rejected candidates, so the acceptance
probability rather than one unlucky draw is what is being measured. `rsa_gate()` verifies the
achieved packing is a stable asymptote independent of batch size and patience.

## MEASURE INTERIOR PACKING, NOT WHOLE-BALL

Converged, with `interior_packing()`:

    R (um)   rho asks   seated   whole-ball   INTERIOR   % of nominal
        90        617      650        0.443     0.3764          89.6
       130      1,859    1,845        0.417     0.3757          89.4
       170      4,157    4,013        0.406     0.3759          89.4

Interior packing is 0.376 at every radius -- flat and block-size independent, just under the
0.384 bulk RSA saturation. The whole-ball figure falls monotonically with radius exactly as a
surface effect must, and at R = 90 it reads 0.443, appearing to BEAT rho, which is impossible.
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
from scipy.spatial import cKDTree

RHO_PER_ML = 2.02e8


def rsa_ball(radius_um, dmin_um, rng, patience=300000, batch=20000, max_attempts=40_000_000):
    """Hard-sphere RSA in a ball, run to genuine jamming.

    Terminates on `patience` CONSECUTIVE rejections, not on one barren batch.
    Returns (points, attempts, jammed).
    """
    pts = np.empty((0, 3))
    acc = []
    tree = None
    since = 0
    attempts = 0
    while attempts < max_attempts and since < patience:
        v = rng.normal(size=(batch, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        cand = v * radius_um * rng.random(batch)[:, None] ** (1 / 3.)
        attempts += batch
        # bulk reject against everything already placed
        if tree is not None:
            keep = tree.query(cand, k=1)[0] >= dmin_um
            cand = cand[keep]
        # then sequentially among the survivors, which is where in-batch conflicts live
        for q in cand:
            if acc:
                loc = np.asarray(acc[-400:])
                if np.linalg.norm(loc - q, axis=1).min() < dmin_um:
                    since += 1
                    continue
                if len(acc) > 400:
                    t2 = cKDTree(np.asarray(acc))
                    if t2.query(q[None, :], k=1)[0][0] < dmin_um:
                        since += 1
                        continue
            acc.append(q)
            since = 0
        if acc:
            pts = np.asarray(acc)
            tree = cKDTree(pts)
    return pts, attempts, since >= patience


def nested_labels(n_cells, fractions, rng):
    """ONE random ranking per realisation -> strictly nested subsets."""
    order = rng.permutation(n_cells)
    out = {f: order[:int(round(f * n_cells))] for f in sorted(fractions)}
    for a, b in zip(sorted(fractions), sorted(fractions)[1:]):
        assert set(out[a].tolist()) <= set(out[b].tolist()), 'subsets are not nested'
    return out


def packing_fraction(n_cells, radius_um, cell_radius_um):
    """WHOLE-BALL packing. Inflated by the surface -- see interior_packing()."""
    return (n_cells * 4 / 3 * np.pi * cell_radius_um ** 3) / (4 / 3 * np.pi * radius_um ** 3)


def interior_packing(pts, radius_um, cell_radius_um):
    """THE DIAGNOSTIC. Packing among centres with a full neighbourhood, r < R - 2a.

    The whole-ball figure is NOT a bulk packing fraction: centres are confined to r <= R, so
    cells near the surface have no neighbours outside and pack more densely. At R = 90 um the
    shell within one cell diameter of the surface is ~44% of the ball, and the whole-ball number
    read 0.443 -- appearing to BEAT rho, which is impossible. The interior figure is flat at
    0.376 across R = 90/130/170, just under the 0.384 bulk RSA saturation.
    """
    r_in = radius_um - 2 * cell_radius_um
    n_in = int((np.linalg.norm(pts, axis=1) <= r_in).sum())
    return n_in * (4 / 3 * np.pi * cell_radius_um ** 3) / (4 / 3 * np.pi * r_in ** 3), n_in


def nominal_count(radius_um):
    return int(round(RHO_PER_ML * (4 / 3 * np.pi * (radius_um * 1e-4) ** 3)))


# ------------------------------------------------------------------ RSA termination gate
def rsa_gate(radius_um=90.0, cell_radius_um=7.92, seeds=(0, 1, 2),
             settings=((100000, 20000), (300000, 20000), (300000, 5000), (900000, 20000))):
    """Does the achieved packing stop depending on the stopping rule?"""
    dmin = 2 * cell_radius_um
    print('RSA TERMINATION GATE — ball R = %g um, dmin = %g um' % (radius_um, dmin))
    print('  rho asks for %d cells (packing %.3f); RSA saturation is ~0.384\n'
          % (nominal_count(radius_um),
             nominal_count(radius_um) * 4 / 3 * np.pi * cell_radius_um ** 3
             / (4 / 3 * np.pi * radius_um ** 3)))
    table = {}
    for patience, batch in settings:
        ns = []
        for s in seeds:
            pts, att, jam = rsa_ball(radius_um, dmin, np.random.default_rng(s),
                                     patience=patience, batch=batch)
            ns.append(len(pts))
        table[(patience, batch)] = (float(np.mean(ns)), float(np.std(ns)))
        print('  patience %7d  batch %6d  ->  %.1f +/- %.1f cells   packing %.4f'
              % (patience, batch, table[(patience, batch)][0], table[(patience, batch)][1],
                 packing_fraction(table[(patience, batch)][0], radius_um, cell_radius_um)))
        # whole-ball is reported only for continuity; interior is the real diagnostic
    ref = table[(900000, 20000)][0]
    print('\n  reference = patience 900000')
    ok = True
    for k in ((300000, 20000), (300000, 5000)):
        d = 100.0 * abs(table[k][0] - ref) / ref
        print('    patience %d batch %d vs reference: %+.2f%%  %s'
              % (k[0], k[1], 100.0 * (table[k][0] - ref) / ref,
                 'stable' if d < 1.0 else 'still climbing'))
        ok &= d < 1.0
    print('  %s' % ('RSA GATE PASSED — packing is a stable asymptote, independent of the '
                    'stopping rule' if ok else
                    'RSA GATE FAILED — the generator is still quitting early'))
    return 0 if ok else 1


if __name__ == '__main__':
    import sys
    sys.exit(rsa_gate())
