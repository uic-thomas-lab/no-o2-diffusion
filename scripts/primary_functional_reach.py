# -*- coding: utf-8 -*-
"""Functional NO reach as a function of the source's distance from the vessel.

THE HYPOTHESIS
--------------
Does the vascular oxygen gradient create a spatial window in which NOS2-derived NO has
maximal functional reach? Oxygen acts on both sides of the NO balance and they oppose:

    O2 down  =>  NOS2 production DOWN      (substrate limited, Km = 42 uM)
             =>  NO removal      DOWN      (k_deg = rho(k_obs[O2] + k_anoxia))

Near the vessel: plenty of synthesis, rapid removal.
Deep in tissue:  NO persists, but NOS2 cannot make enough of it.
In between:      potentially a maximum.

The location of the source relative to the oxygen source is the central independent
variable, not a nuisance variable. A source at depth d has production set by O2(d):

    Q_NO(d) = Vmax * O2(d) / (Km + O2(d))

but its NO does NOT travel through a uniform medium at O2(d). It travels through

    k_NO(x) = rho[k_obs*O2(x) + k_anoxia]

which varies along the whole propagation path. This analysis resolves that path rather
than collapsing it to a single local O2 value.

WHAT IS MEASURED
----------------
For a single NOS2+ source at depth d, in the real vessel-generated gradient:

    R_T(d) = max{ r : C_NO(r; d) >= C_T }

reported SEPARATELY inward and outward, because the two directions are not equivalent --
inward runs into rising O2 (faster removal) and an absorbing wall; outward runs into falling
O2 (slower removal) but the source itself is weaker.

Also reported: total extent above threshold, and the peak concentration at the source.

OUT: data/primary_functional_reach.csv
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import gate1_pair_cooperation as G1                        # noqa: E402
import gate4_vessel_pair as G4                             # noqa: E402
from kinetics import k_degradation, diffusion_length, r_no  # noqa: E402
from constants import K_OBS_CORR, K_ANOXIA_CORR            # noqa: E402

DEPTHS = list(range(20, 261, 10))
THRESHOLDS = [0.5, 1.0, 2.0]


def main():
    t0, rows = time.time(), []
    print("PRIMARY RESULT -- functional NO reach vs source depth\n")
    print("  single NOS2+ source, real vessel-generated O2 gradient")
    print("  the NO field is solved through the SPATIALLY VARYING sink, not a uniform one\n")

    r = np.arange(0.0, G4.RMAX + G4.DX, G4.DX)
    x, o2 = G4.solve_o2_vessel()

    hdr = ("  depth  [O2]    L_local  prod.rel   peak    " +
           "".join("  in@%.1f out@%.1f tot@%.1f" % (t, t, t) for t in THRESHOLDS))
    print(hdr)
    print("     um    uM         um              nM    " +
          "".join("      um     um      um" for _ in THRESHOLDS))

    for d in DEPTHS:
        if d - G1.A0 < G4.DX:
            continue
        C = G4.solve_no_vessel(x, o2, [float(d)], r)[:, 0] * 1e9
        i = int(np.argmin(np.abs(x - d)))
        o2d = float(o2[i])
        kd = k_degradation(o2d, K_OBS_CORR, G1.RHO, k_anoxia=K_ANOXIA_CORR)
        Lloc = diffusion_length(G1.D_NO * 1e-8, kd) * 1e4
        prel = r_no(o2d, G1.VMAX, G1.KM) / r_no(56.0e-6, G1.VMAX, G1.KM)
        peak = float(C.max())

        rec = dict(depth_um=d, o2_uM=o2d * 1e6, L_local_um=Lloc,
                   production_rel=prel, peak_nM=peak)
        cells = []
        for t in THRESHOLDS:
            m = C >= t
            if not m.any():
                rin = rout = tot = float("nan")
            else:
                xs_ok = x[m]
                # contiguous run containing the source
                lo = xs_ok[xs_ok <= d].max() if (xs_ok <= d).any() else d
                inward = xs_ok[(xs_ok <= d)]
                outward = xs_ok[(xs_ok >= d)]
                rin = float(d - inward.min()) if inward.size else 0.0
                rout = float(outward.max() - d) if outward.size else 0.0
                tot = float(np.trapezoid(m.astype(float), x))
            rec["reach_in_%.1f" % t] = rin
            rec["reach_out_%.1f" % t] = rout
            rec["extent_%.1f" % t] = tot
            cells += ["%7.1f %6.1f %7.1f" % (rin, rout, tot)]
        rows.append(rec)
        print("  %5d %6.2f %9.2f %9.4f %7.3f    %s"
              % (d, o2d * 1e6, Lloc, prel, peak, "".join(cells)))

    print("\n  THE ANSWER")
    for t in THRESHOLDS:
        col = [(rr["depth_um"], rr["extent_%.1f" % t]) for rr in rows
               if rr["extent_%.1f" % t] == rr["extent_%.1f" % t]]
        if not col:
            print("    %.1f nM: never reached at any depth" % t)
            continue
        dmax, vmax = max(col, key=lambda z: z[1])
        first, last = col[0], col[-1]
        shape = ("RISES, PEAKS, FALLS" if first[1] < vmax > last[1] else
                 "monotonic" if (first[1] < vmax == last[1] or first[1] == vmax > last[1])
                 else "flat/other")
        print("    %.1f nM: extent peaks at depth %.0f um (%.1f um of tissue); %s"
              % (t, dmax, vmax, shape))
        print("             at that depth [O2] = %.2f uM"
              % [rr["o2_uM"] for rr in rows if rr["depth_um"] == dmax][0])

    print("\n    An interior maximum means production and removal genuinely compete and the")
    print("    vascular gradient creates a spatial window. A monotonic curve means one term")
    print("    dominates throughout and there is no window at this parameter set.")

    out = os.path.join(REPO, "data", "primary_functional_reach.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keys = sorted({k for rr in rows for k in rr})
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for rr in rows:
            w.writerow({k: rr.get(k, "") for k in keys})
    print("\n  %.1f min -> %s" % ((time.time() - t0) / 60, os.path.relpath(out, REPO)))


if __name__ == "__main__":
    main()
