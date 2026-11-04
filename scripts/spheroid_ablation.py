# -*- coding: utf-8 -*-
"""The 2 x 2 mechanism ablation: which link makes the shell, and which makes the ordering?

THE QUESTION. The migrating shell does NOT by itself demonstrate a relay. It
can arise simply because the tumour's oxygen profile sweeps through a fixed ~32 uM optimum as the
sphere grows. Distinguishing that from a genuine spatial handoff requires cutting the links one at
a time:

    | model                    | O2 -> NOS2 | NO -| respiration |
    | full                     | yes        | yes              |
    | no respiratory feedback  | yes        | no               |
    | no O2 gating of NOS2     | no         | yes              |
    | decoupled                | no         | no               |

REMOVAL STAYS OXYGEN-DEPENDENT IN ALL FOUR. Only the SOURCE's oxygen dependence and the
NO -> respiration link are ablated; k_deg keeps its k_obs*[O2]*rho + k_anoxia*rho form throughout.

"RELAY" IS NOT EARNED BY THIS RUN ALONE. It needs the spatial causal chain demonstrated:
  NO(r1) -> dO2(r2) > 0 -> dJ_NOS2(r2) > 0 -> dNO(r2) > 0 for r2 > r1. The radial diagnostic at the
  end computes exactly those differences between full and feedback-off, so the chain can be
  inspected rather than asserted.

Out: data/spheroid_ablation.csv          boundaries per model and fraction
     data/spheroid_ablation_radial.csv   the radial diagnostic at one condition
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import spheroid_field as S                      # noqa: E402
from spheroid_crossover import interp_cross     # noqa: E402

DATA = os.path.join(ROOT, 'data')
MODELS = (('full', True, True),
          ('no_resp_feedback', True, False),
          ('no_o2_gating', False, True),
          ('decoupled', False, False))
FRACTIONS = (0.02, 0.15, 0.40)
D_SCAN = tuple(range(200, 1601, 50))
SGC_NM = 1.0
ANOXIC_UM = S.ANOXIC_THRESHOLD * 1e6
RADIAL_AT = (0.15, 850)


def scan(f, gating, feedback):
    ds, o2c, noc, rmax, o2max, prev = [], [], [], [], [], None
    for d in D_SCAN:
        try:
            info, prof = S.describe(d / 2.0, f_nos2=f, o2_gating=gating,
                                    respiratory_inhibition=feedback, _guess=prev)
        except RuntimeError as exc:
            print('   %s f=%.2f d=%d FAILED: %s' % ('gate' if gating else 'flat', f, d, exc))
            prev = None
            continue
        prev = prof
        ds.append(d)
        o2c.append(info['o2_centre_uM'])
        noc.append(info['no_centre_nM'])
        rmax.append(info['r_at_max_um'] if info['interior_max'] else 0.0)
        o2max.append(info['o2_at_max_uM'] if info['interior_max'] else None)
    return ds, o2c, noc, rmax, o2max


def main():
    rows = []
    print('MECHANISM ABLATION  (removal stays O2-dependent in all four models)')
    print('  model                f_NOS2   d_anoxia   d_sGC    delta_d   shell?   [O2] at max')
    for name, gating, feedback in MODELS:
        for f in FRACTIONS:
            ds, o2c, noc, rmax, o2max = scan(f, gating, feedback)
            if not ds:
                continue
            da = interp_cross(ds, o2c, ANOXIC_UM, rising=False)
            dg = interp_cross(ds, noc, SGC_NM, rising=False)
            free = [v for v in o2max if v is not None]
            shell = next((d for d, rr in zip(ds, rmax) if rr > 1.0), None)
            delta = (dg - da) if (da is not None and dg is not None) else None
            print('  %-20s %5.2f   %8s  %8s  %9s   %6s   %s'
                  % (name, f,
                     '-' if da is None else '%.0f' % da,
                     '-' if dg is None else '%.0f' % dg,
                     '-' if delta is None else '%+.0f' % delta,
                     '-' if shell is None else '%d' % shell,
                     '-' if not free else '%.1f-%.1f' % (min(free), max(free))))
            rows.append({'model': name, 'o2_gating': gating, 'resp_feedback': feedback,
                         'f_nos2': f,
                         'd_anoxia_um': None if da is None else round(da, 1),
                         'd_sgc_um': None if dg is None else round(dg, 1),
                         'delta_d_um': None if delta is None else round(delta, 1),
                         'd_shell_um': shell,
                         'o2_at_max_lo': None if not free else round(min(free), 2),
                         'o2_at_max_hi': None if not free else round(max(free), 2)})

    with open(os.path.join(DATA, 'spheroid_ablation.csv'), 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- the relay diagnostic: full vs feedback-off, radius by radius --------------
    f, d = RADIAL_AT
    r, o2_full, no_full = S.solve(d / 2.0, f_nos2=f, respiratory_inhibition=True)
    _, o2_off, no_off = S.solve(d / 2.0, f_nos2=f, respiratory_inhibition=False)
    jf = S.prod_u(o2_full, f)
    jo = S.prod_u(o2_off, f)
    out = []
    for i in range(0, len(r), 20):
        out.append({'r_um': round(float(r[i]), 1),
                    'o2_full_uM': round(float(o2_full[i]), 4),
                    'o2_off_uM': round(float(o2_off[i]), 4),
                    'd_o2_uM': round(float(o2_full[i] - o2_off[i]), 4),
                    'J_full_nM_s': round(float(jf[i]), 4),
                    'J_off_nM_s': round(float(jo[i]), 4),
                    'd_J_nM_s': round(float(jf[i] - jo[i]), 4),
                    'no_full_nM': round(float(no_full[i]), 4),
                    'no_off_nM': round(float(no_off[i]), 4),
                    'd_no_nM': round(float(no_full[i] - no_off[i]), 4)})
    with open(os.path.join(DATA, 'spheroid_ablation_radial.csv'), 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    k = int(np.argmax(o2_full - o2_off))
    print('\n  relay diagnostic at f=%.2f d=%d um:' % (f, d))
    print('    largest oxygen gain from feedback  %+.2f uM at r = %.0f um'
          % (o2_full[k] - o2_off[k], r[k]))
    print('    NOS2 output there                  %+.3f nM/s  (%.3f -> %.3f)'
          % (jf[k] - jo[k], jo[k], jf[k]))
    print('    [NO] there                         %+.3f nM   (%.3f -> %.3f)'
          % (no_full[k] - no_off[k], no_off[k], no_full[k]))
    print('  wrote spheroid_ablation.csv and spheroid_ablation_radial.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
