# -*- coding: utf-8 -*-
"""Paired within-realisation analysis of the fixed-tissue fraction series.

WHY PAIRED. Nested labelling means every fraction within a realisation shares the SAME cellular
tissue and the SAME random ranking -- raising the fraction CONVERTS cells rather than redrawing
them. So fraction-to-fraction changes are paired, and the marginal SDs understate how cleanly
fraction itself moves the field.

NOT for significance stars. The point is that the paired effect distribution gives a much
tighter interval than the marginal spread implies, which matters for the Results text and for
answering a reviewer who reads the marginal error bars as the uncertainty on the change.

Fold changes are summarised as the GEOMETRIC mean with a CI from the log-scale SD, because these
are ratios and their distribution is multiplicative.

Writes `figures/final/Fig2/data/fig2_panelH_fixedtissue.csv` for the Origin build.
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import csv
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, 'data', 'fig2_fixedtissue_fraction.csv')
OUT = os.path.join(ROOT, 'figures', 'final', 'Fig2', 'data',
                   'fig2_panelH_fixedtissue.csv')

FRACS = (2.0, 5.0, 15.0, 40.0)
STEPS = ((2.0, 5.0), (5.0, 15.0), (15.0, 40.0))


def main():
    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    by = defaultdict(dict)
    for r in rows:
        by[int(r['realisation'])][float(r['frac_pct'])] = r
    reals = sorted(by)
    print('fixed-tissue fraction series: %d realisations x %d fractions\n'
          % (len(reals), len(FRACS)))

    # ---------------- marginal ----------------
    print('MARGINAL (mean +/- SD across realisations)')
    print('  %6s %8s %16s %18s %12s' % ('f', 'n_src', 'peak nM', 'V>1nM um3', 'core>1 %'))
    agg = {}
    for f in FRACS:
        g = [by[r][f] for r in reals]
        m = lambda k: float(np.mean([float(x[k]) for x in g]))
        sd = lambda k: float(np.std([float(x[k]) for x in g], ddof=1))
        agg[f] = {k: (m(k), sd(k)) for k in
                  ('n_src', 'peak_nM', 'fullV_1', 'fullV_0.5', 'fullV_2',
                   'coreCov_1', 'coreCov_2', 'nn_median_um')}
        print('  %5.1f%% %8.1f %8.3f +/- %-5.3f %11.4g +/- %-9.3g %9.2f'
              % (f, agg[f]['n_src'][0], agg[f]['peak_nM'][0], agg[f]['peak_nM'][1],
                 agg[f]['fullV_1'][0], agg[f]['fullV_1'][1], agg[f]['coreCov_1'][0]))

    # ---------------- paired ----------------
    print('\nPAIRED within-realisation fold change (geometric mean, 95% CI from log SD)')
    print('  %-12s %-28s %-28s' % ('step', 'peak [NO]', 'V(>1 nM)'))
    for a, b in STEPS:
        line = '  %4.0f -> %-4.0f%%  ' % (a, b)
        for key in ('peak_nM', 'fullV_1'):
            fold = np.array([float(by[r][b][key]) / float(by[r][a][key]) for r in reals])
            lg = np.log(fold)
            gm = float(np.exp(lg.mean()))
            sem = float(lg.std(ddof=1) / np.sqrt(len(lg)))
            lo, hi = np.exp(lg.mean() - 1.96 * sem), np.exp(lg.mean() + 1.96 * sem)
            cv = 100.0 * float(fold.std(ddof=1) / fold.mean())
            line += '%5.3fx [%5.3f-%5.3f] CV %4.1f%%   ' % (gm, lo, hi, cv)
        print(line)

    # the comparison that makes the point
    print('\n  why pairing matters: ratio of MARGINAL to PAIRED uncertainty on the change')
    for a, b in STEPS:
        for key, lab in (('peak_nM', 'peak'), ('fullV_1', 'V>1nM')):
            ma, mb = agg[a][key], agg[b][key]
            # naive unpaired propagation of the ratio
            rel_unpaired = np.sqrt((ma[1] / ma[0]) ** 2 + (mb[1] / mb[0]) ** 2)
            fold = np.array([float(by[r][b][key]) / float(by[r][a][key]) for r in reals])
            rel_paired = float(fold.std(ddof=1) / fold.mean())
            print('    %4.0f->%-4.0f%%  %-6s  unpaired %5.2f%%   paired %5.2f%%   '
                  '%.1fx tighter' % (a, b, lab, 100 * rel_unpaired, 100 * rel_paired,
                                     rel_unpaired / rel_paired if rel_paired else float('nan')))

    # ---------------- sublinearity ----------------
    print('\n  scaling from 15%% to 40%%:')
    r15, r40 = agg[15.0], agg[40.0]
    print('    sources %5.1f -> %5.1f  = %.2fx' % (r15['n_src'][0], r40['n_src'][0],
                                                   r40['n_src'][0] / r15['n_src'][0]))
    print('    peak    %5.2f -> %5.2f  = %.2fx' % (r15['peak_nM'][0], r40['peak_nM'][0],
                                                   r40['peak_nM'][0] / r15['peak_nM'][0]))
    print('    V>1 nM  %.3g -> %.3g = %.2fx' % (r15['fullV_1'][0], r40['fullV_1'][0],
                                                r40['fullV_1'][0] / r15['fullV_1'][0]))
    print('  territory grows SUBLINEARLY in source number -- consistent with increasing spatial')
    print('    redundancy among overlapping fields. The fixed-N experiment is what ISOLATES')
    print('    geometry and earns the mechanistic attribution; this shows the same signature in')
    print('    the biologically coupled series.')

    # ---------------- CSV for Origin ----------------
    with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('frac_pct,n_src,peak_nM,peak_SD,V1_e6,V1_e6_SD,'
                 'V05_e6,V05_e6_SD,V2_e6,V2_e6_SD,coreCov1,coreCov2,nn_median_um\n')
        for f in FRACS:
            a = agg[f]
            fh.write('%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g\n'
                     % (f, a['n_src'][0], a['peak_nM'][0], a['peak_nM'][1],
                        a['fullV_1'][0] / 1e6, a['fullV_1'][1] / 1e6,
                        a['fullV_0.5'][0] / 1e6, a['fullV_0.5'][1] / 1e6,
                        a['fullV_2'][0] / 1e6, a['fullV_2'][1] / 1e6,
                        a['coreCov_1'][0], a['coreCov_2'][0], a['nn_median_um'][0]))
    print('\n  wrote %s' % os.path.relpath(OUT, ROOT))


if __name__ == '__main__':
    sys.exit(main())
