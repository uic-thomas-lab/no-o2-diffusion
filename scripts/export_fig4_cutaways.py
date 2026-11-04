# -*- coding: utf-8 -*-
"""Export the radial fields for Figure 4's cutaway matrix, plus the oxygen reference key.

THE FIGURE. A 2 x 3 matrix of NO cutaways is the dominant visual;
the plots below only prove what the reader has already seen. Rows are NOS2 fraction 2% and 15%;
columns are three diameters chosen from the SOLVED FIELDS for biological contrast, not for round
numbers or even spacing:

    340 um   PRE-SHELL: the LARGEST diameter at which neither row has a free interior maximum.
             Scanned to find it: at 360 um the 2% row has already formed a shell (r_max 53 um),
             so 340 is the honest ceiling. The column looks small because the tumour IS small;
             that is a consequence of drawing all six at true relative scale.
    500 um   oxygen-pinned shell clearly separated (r_max 198 / 167 um), no anoxia, full signalling
    850 um   BOTH rows anoxic, and only the higher fraction still signals into the core:
               2%   core 174 um, centre 0.08 nM, signalling front 240 um -> OUTSIDE the core
               15%  core  88 um, centre 1.58 nM, signalling front   0 um -> core fully competent

    900 um was rejected: at 15% the centre has just dropped to 0.96 nM, which blurs the contrast
    the column exists to make.

EACH NO CUTAWAY CARRIES THE OXYGEN INFORMATION AS TWO THIN CONTOURS rather than being paired with
its own oxygen sphere: the 0.1 uM anoxic boundary and the 32.3 uM permissive shell,
sqrt(K_m*[O2]*), which is where the NO maximum should sit if it is oxygen-pinned. One larger oxygen
cutaway serves as the key for all six.

RENDERING RULES, and they are the point of the panel:
  1. TRUE RELATIVE SIZE across columns. Do NOT normalise each sphere to the same drawn radius; the
     reader must see the tumour growing. One shared scale bar.
  2. ONE NO COLOUR MAPPING for all six panels. Do NOT rescale per panel -- the amplitude difference
     between 2% and 15% is part of the biology.
  3. The two oxygen contours read quietly; neither may dominate the NO field.
  4. Almost no text inside the panels. Row labels left, diameters above, topology named ONCE below.

Out: data/fig4_cutaway_profiles.csv  (long form: f_nos2, d_um, r_um, o2_uM, no_nM)
     data/fig4_cutaway_markers.csv   (per panel: anoxic radius, 32.3 uM radius, r_max, fronts)
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import spheroid_field as S   # noqa: E402

DATA = os.path.join(ROOT, 'data')
ROWS = (0.02, 0.15)
COLS = (340, 600, 850)
KEY_D = 850                     # the oxygen reference cutaway
O2_SHELL_UM = (S.KM_UM * S.C.O2_STAR_UM) ** 0.5      # 32.30
O2_ANOXIC_UM = S.ANOXIC_THRESHOLD * 1e6              # 0.1
SGC_NM = 1.0


def crossing(r, y, target, from_centre=True):
    """Outermost radius at which y crosses target (None if it never does)."""
    idx = np.where(np.diff(np.sign(y - target)) != 0)[0]
    if not idx.size:
        return None
    i = idx[-1] if from_centre else idx[0]
    a, b = y[i], y[i + 1]
    return float(r[i] + (target - a) * (r[i + 1] - r[i]) / (b - a)) if b != a else float(r[i])


def main():
    prof, mark = [], []
    for f in ROWS:
        prev = None
        for d in COLS:
            r, o2, no = S.solve(d / 2.0, f_nos2=f, _guess=prev)
            prev = (r, o2, no)
            for i in range(0, len(r), 4):
                prof.append({'f_nos2': f, 'd_um': d, 'r_um': round(float(r[i]), 2),
                             'o2_uM': round(float(o2[i]), 5),
                             'no_nM': round(float(no[i]), 5)})
            i_max = int(np.argmax(no))
            m = max(3, len(r) // 200)
            interior = bool(m <= i_max <= len(r) - 1 - m
                            and no[i_max] > no[0] and no[i_max] > no[-1])
            anox = r[o2 < O2_ANOXIC_UM]
            mark.append({
                'f_nos2': f, 'd_um': d, 'R_um': round(d / 2.0, 1),
                'r_anoxic_um': round(float(anox.max()), 1) if anox.size else 0.0,
                'r_o2shell_um': (lambda v: round(v, 1) if v else '')(
                    crossing(r, o2, O2_SHELL_UM)),
                'r_max_um': round(float(r[i_max]), 1) if interior else '',
                'o2_at_max_uM': round(float(o2[i_max]), 2) if interior else '',
                'no_max_nM': round(float(no[i_max]), 3),
                'no_centre_nM': round(float(no[0]), 3),
                'no_rim_nM': round(float(no[-1]), 3),
                'r_sgc_front_um': (lambda v: round(v, 1) if v else 0.0)(
                    crossing(r, no, SGC_NM)) if no[0] < SGC_NM else 0.0,
            })
            print('  f=%.2f d=%4d  anoxic %6s  32.3uM shell %6s  r_max %6s  NOmax %7.3f'
                  % (f, d, mark[-1]['r_anoxic_um'], mark[-1]['r_o2shell_um'],
                     mark[-1]['r_max_um'], mark[-1]['no_max_nM']))

    # ---- the oxygen reference key: one larger cutaway, no NO on it ----------------
    r, o2, no = S.solve(KEY_D / 2.0, f_nos2=0.15)
    for i in range(0, len(r), 4):
        prof.append({'f_nos2': 'KEY', 'd_um': KEY_D, 'r_um': round(float(r[i]), 2),
                     'o2_uM': round(float(o2[i]), 5), 'no_nM': round(float(no[i]), 5)})

    with open(os.path.join(DATA, 'fig4_cutaway_profiles.csv'), 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['f_nos2', 'd_um', 'r_um', 'o2_uM', 'no_nM'])
        w.writeheader()
        w.writerows(prof)
    with open(os.path.join(DATA, 'fig4_cutaway_markers.csv'), 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(mark[0].keys()))
        w.writeheader()
        w.writerows(mark)
    print('\n  NO colour scale must span 0 to %.3f nM across ALL six panels'
          % max(m['no_max_nM'] for m in mark))
    print('  wrote fig4_cutaway_profiles.csv (%d rows) and fig4_cutaway_markers.csv' % len(prof))
    return 0


if __name__ == '__main__':
    sys.exit(main())
