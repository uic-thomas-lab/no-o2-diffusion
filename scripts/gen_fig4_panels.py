# -*- coding: utf-8 -*-
"""Figure 4 panel data: the diameter x NOS2-fraction regime map and its supporting panels.

THE HEADLINE THIS FIGURE MAKES:

    Tumour growth drives an oxygen-pinned NO signalling shell, while NOS2 abundance determines
    whether anoxia develops before or after central NO/sGC signalling is lost.

PANELS
  A  regime map: d_shell, d_anoxia, d_sGC against NOS2 fraction. The centrepiece. The biological
     boundary is where d_anoxia and d_sGC CROSS, between 3 and 4%.
  B  delta_d = d_sGC - d_anoxia against fraction, with a zero line and the 3-4% transition band.
  C  [O2] at the migrating NO maximum, with the analytic sqrt(Km*[O2]*) = 32.30 uM reference.
  D  radial profiles at fixed fraction, three diameters: centre-enriched -> shell -> centre-depleted.
  E  radial profiles at fixed diameter, three fractions: why the ordering flips.

DO NOT FIT A CURVE TO PANEL B AND QUOTE AN INTERPOLATED CRITICAL FRACTION. Connect measured
points; mark 3-4% as a resolved BAND. The fraction grid is one percentage point wide there. Closed
questions #9 and #17 of the Figure 3 record bind: no argmax off a coarse sweep.

VERIFIED BAND 200-1600 um: scipy collocation and Wolfram NDSolve agree to 4-5 significant figures
there and diverge beyond, where NDSolve reports boundary-residual failures. Nothing outside it is
plotted, and a boundary that leaves it is recorded as out-of-band rather than extrapolated.

f = 1.00 is an UPPER-BOUND REFERENCE STATE, not a nominal tumour. Mark it as such in the panel.

Out: data/fig4_A_boundaries.csv, fig4_B_delta.csv, fig4_C_shell_oxygen.csv,
     fig4_D_profiles.csv, fig4_E_profiles.csv
"""
from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import spheroid_field as S            # noqa: E402
from spheroid_crossover import interp_cross   # noqa: E402

DATA = os.path.join(ROOT, 'data')
VERIFIED = (200, 1600)
SGC_NM = 1.0
ANOXIC_UM = S.ANOXIC_THRESHOLD * 1e6          # 0.1 uM

FRACTIONS = (0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.25, 0.40, 0.70, 1.00)
D_SCAN = tuple(range(200, 1601, 50))

# panel D: one fraction, three diameters spanning the three topologies
D_PANEL_F = 0.15
# panel E: one diameter near the transition, three fractions spanning the sign change
E_PANEL_D = 670


def scan(f):
    ds, o2c, noc, rmax, o2max, cor, prev = [], [], [], [], [], [], None
    for d in D_SCAN:
        try:
            info, prof = S.describe(d / 2.0, f_nos2=f, _guess=prev)
        except RuntimeError as exc:
            print('   f=%.3f d=%d SOLVE FAILED: %s' % (f, d, exc))
            prev = None
            continue
        prev = prof
        ds.append(d)
        o2c.append(info['o2_centre_uM'])
        noc.append(info['no_centre_nM'])
        rmax.append(info['r_at_max_um'] if info['interior_max'] else 0.0)
        o2max.append(info['o2_at_max_uM'] if info['interior_max'] else None)
        cor.append(info['centre_over_rim'])
    return ds, o2c, noc, rmax, o2max, cor


def write(name, cols, rows):
    p = os.path.join(DATA, name)
    with open(p, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print('  wrote %s  (%d rows)' % (name, len(rows)))


def main():
    a_rows, b_rows, c_rows = [], [], []
    print('Figure 4 panel data.  analytic optimum sqrt(Km*[O2]*) = %.2f uM'
          % ((S.KM_UM * S.C.O2_STAR_UM) ** 0.5))
    for f in FRACTIONS:
        ds, o2c, noc, rmax, o2max, cor = scan(f)
        d_anox = interp_cross(ds, o2c, ANOXIC_UM, rising=False)
        d_sgc = interp_cross(ds, noc, SGC_NM, rising=False)
        # shell separation is DISCRETE at this grid -- r_max jumps rather than growing from zero,
        # so it is recorded as a bracket, never interpolated like the other two.
        d_shell_lo = d_shell_hi = None
        for i in range(1, len(ds)):
            if rmax[i - 1] <= 1.0 < rmax[i]:
                d_shell_lo, d_shell_hi = ds[i - 1], ds[i]
                break
        oob = [n for n, v in (('anoxia', d_anox), ('sGC', d_sgc))
               if v is not None and not (VERIFIED[0] <= v <= VERIFIED[1])]
        a_rows.append({'f_nos2': f,
                       'd_shell_lo_um': d_shell_lo, 'd_shell_hi_um': d_shell_hi,
                       'd_anoxia_um': None if d_anox is None else round(d_anox, 1),
                       'd_sgc_um': None if d_sgc is None else round(d_sgc, 1),
                       'out_of_verified_band': ','.join(oob) or ''})
        if d_anox is not None and d_sgc is not None:
            b_rows.append({'f_nos2': f, 'delta_d_um': round(d_sgc - d_anox, 1),
                           'sign': 'sGC_first' if d_sgc < d_anox else 'anoxia_first'})
        for d, r, o in zip(ds, rmax, o2max):
            if o is not None:
                c_rows.append({'f_nos2': f, 'd_um': d, 'r_max_um': round(r, 1),
                               'o2_at_max_uM': round(o, 3)})
        print('  f=%.3f  shell %s-%s  anoxia %s  sGC %s'
              % (f, d_shell_lo, d_shell_hi,
                 None if d_anox is None else round(d_anox),
                 None if d_sgc is None else round(d_sgc)))

    write('fig4_A_boundaries.csv',
          ['f_nos2', 'd_shell_lo_um', 'd_shell_hi_um', 'd_anoxia_um', 'd_sgc_um',
           'out_of_verified_band'], a_rows)
    write('fig4_B_delta.csv', ['f_nos2', 'delta_d_um', 'sign'], b_rows)
    write('fig4_C_shell_oxygen.csv', ['f_nos2', 'd_um', 'r_max_um', 'o2_at_max_uM'], c_rows)

    # ---- panel D: three topologies at one fraction -----------------------------------
    prof_rows = []
    for d in (400, 800, 1400):
        r, o2, no = S.solve(d / 2.0, f_nos2=D_PANEL_F)
        for i in range(0, len(r), 10):
            prof_rows.append({'f_nos2': D_PANEL_F, 'd_um': d, 'r_um': round(r[i], 2),
                              'o2_uM': round(o2[i], 4), 'no_nM': round(no[i], 4)})
    write('fig4_D_profiles.csv', ['f_nos2', 'd_um', 'r_um', 'o2_uM', 'no_nM'], prof_rows)

    # ---- panel E: three fractions at one diameter, spanning the sign change -----------
    prof_rows = []
    for f in (0.02, 0.04, 0.15):
        r, o2, no = S.solve(E_PANEL_D / 2.0, f_nos2=f)
        for i in range(0, len(r), 10):
            prof_rows.append({'f_nos2': f, 'd_um': E_PANEL_D, 'r_um': round(r[i], 2),
                              'o2_uM': round(o2[i], 4), 'no_nM': round(no[i], 4)})
    write('fig4_E_profiles.csv', ['f_nos2', 'd_um', 'r_um', 'o2_uM', 'no_nM'], prof_rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
