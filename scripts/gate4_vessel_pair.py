# -*- coding: utf-8 -*-
"""The NOS2+ cell pair placed in the vessel geometry, so no interaction length has to be assumed.

WHY THIS IS NOT "does the effect survive vascular geometry"
-----------------------------------------------------------
Gate 1 established that the nonlinear cross-talk between NOS2+ cells is carried by the
OXYGEN field, not by direct NO overlap: with interpenetrating sources excluded, an
NO-length component fits at A_NO = -0.003 nM (i.e. zero), the offset is b = -0.0001 nM
(zero), and a single exponential whose length is free recovers 69.5 um against the 66.3 um
computed independently from D_O2/lambda.

But lambda is the author's. It was introduced only because a boundary-free domain has no
oxygen source, and it sets BOTH the magnitude (A ~ 1/sqrt(lambda)) and the range
(L_O2 = sqrt(D_O2/lambda)) of the very effect being measured. Gate 1 therefore PASSED on
existence and mechanism and did NOT pass on physiological magnitude and range.

A vessel supplies oxygen physically. With an explicit vessel and no bulk restoring term,
the oxygen relaxation scale is not chosen at all:

    L_O2 = sqrt(D_O2 * Km_resp / V_M) = 21.0 um     in the O2-LIMITED regime

Since A ~ 1/sqrt(lambda), that predicts the cross-talk here at roughly 0.19%, down from
0.396%. That is a PREDICTION made before running, recorded here so it cannot be adjusted
afterwards.

CAVEAT ON THE 21 um FIGURE. It is the deep-tissue limit, where O2 << Km_resp and
respiration is first order. Near the wall at 56 uM respiration is saturated and zeroth
order, so there is no exponential screening there at all and oxygen falls roughly
parabolically. The scale is depth-dependent -- which is precisely why the pair is run at
several depths rather than one.

WHAT IS DIFFERENT FROM GATE 1, AND MUST BE
------------------------------------------
  * oxygen: Dirichlet 56 uM at the wall, respiration only, NO restoring term, zero flux far
  * NO: absorbing C = 0 at the wall, zero flux far
  * NO MIRROR SYMMETRY. The field is not symmetric about the midpoint once a wall exists,
    so C_A and C_B are solved separately -- three coupled solves per spacing, not two.
  * the hard-sphere guard in G1._source is inherited unchanged, and the near cell must also
    clear the wall by at least A0.

OUT: data/gate4_vessel_pair.csv
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
from scipy.optimize import curve_fit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import gate1_pair_cooperation as G1                       # noqa: E402
from kinetics import k_degradation, diffusion_length      # noqa: E402
from constants import K_OBS_CORR, K_ANOXIA_CORR           # noqa: E402

DX = 0.5
XMAX = 500.0
RMAX = 120.0
O2_WALL = 56.0e-6

DEPTHS = [60.0, 120.0, 220.0]
SPACINGS = [16, 20, 24, 30, 38, 47, 58, 72, 88]


def solve_o2_vessel(inhib=None, dx=DX, xmax=XMAX, tol=1e-14, iters=600):
    """Oxygen from an explicit vessel. Dirichlet at the wall, respiration only.

    No restoring term anywhere -- that is the whole point of this gate.
    """
    x = np.arange(0.0, xmax + dx, dx)
    n = len(x)
    o2 = np.linspace(O2_WALL, 0.0, n)
    inhib = np.ones(n) if inhib is None else np.asarray(inhib, float)
    for _ in range(iters):
        k_eff = G1.V_M / (np.maximum(o2, 0.0) + G1.KM_RESP * inhib)
        di = -2.0 * G1.D_O2 / dx ** 2 - k_eff
        lo = np.full(n, G1.D_O2 / dx ** 2)
        up = np.full(n, G1.D_O2 / dx ** 2)
        rhs = np.zeros(n)
        di[0], up[0], rhs[0] = 1.0, 0.0, O2_WALL                     # vessel wall
        di[-1], lo[-1] = -G1.D_O2 / dx ** 2 - k_eff[-1], G1.D_O2 / dx ** 2   # zero flux
        A = np.zeros((3, n))
        A[0, 1:], A[1, :], A[2, :-1] = up[:-1], di, lo[1:]
        new = np.clip(solve_banded((1, 1), A, rhs), 0.0, O2_WALL)
        if np.max(np.abs(new - o2)) < tol:
            return x, new
        o2 = 0.5 * o2 + 0.5 * new
    raise RuntimeError("solve_o2_vessel did not converge")


def _operator_vessel(x, o2, r, dx=DX):
    """NO operator with an ABSORBING vessel wall at x = 0."""
    nx, nr = len(x), len(r)
    k_deg = k_degradation(o2, K_OBS_CORR, G1.RHO, k_anoxia=K_ANOXIA_CORR)
    inv2 = 1.0 / dx ** 2
    rows, cols, vals = [], [], []
    idx = lambda i, j: i * nr + j                                    # noqa: E731
    for i in range(nx):
        for j in range(nr):
            k = idx(i, j)
            if i == 0 or j == nr - 1:            # C = 0 at the wall and at the far radius
                rows.append(k); cols.append(k); vals.append(1.0)
                continue
            diag = -k_deg[i]
            if i == nx - 1:
                rows.append(k); cols.append(idx(nx - 2, j)); vals.append(2.0 * G1.D_NO * inv2)
                diag -= 2.0 * G1.D_NO * inv2
            else:
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(G1.D_NO * inv2)
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(G1.D_NO * inv2)
                diag -= 2.0 * G1.D_NO * inv2
            if j == 0:
                rows.append(k); cols.append(idx(i, 1)); vals.append(4.0 * G1.D_NO * inv2)
                diag -= 4.0 * G1.D_NO * inv2
            else:
                rp, rm = (r[j] + 0.5 * dx) / r[j], (r[j] - 0.5 * dx) / r[j]
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(G1.D_NO * inv2 * rp)
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(G1.D_NO * inv2 * rm)
                diag -= G1.D_NO * inv2 * (rp + rm)
            rows.append(k); cols.append(k); vals.append(diag)
    return sp.csr_matrix((vals, (rows, cols)), shape=(nx * nr, nx * nr)).tocsc()


def solve_no_vessel(x, o2, centres, r):
    A = _operator_vessel(x, o2, r)
    S = G1._source(x, o2, centres, r)          # hard-sphere guard inherited unchanged
    return np.maximum(spsolve(A, -S.ravel()).reshape(len(x), len(r)), 0.0)


def run_vessel(centres, coupled=True, iters=80, tol=1e-7):
    r = np.arange(0.0, RMAX + DX, DX)
    x, o2 = solve_o2_vessel()
    if not centres:
        return x, r, np.zeros((len(x), len(r))), o2
    if not coupled:
        return x, r, solve_no_vessel(x, o2, centres, r), o2
    for _ in range(iters):
        C = solve_no_vessel(x, o2, centres, r)
        inhib = 1.0 + G1._radial_average(C, r) / G1.K_I
        _, o2n = solve_o2_vessel(inhib)
        if np.max(np.abs(o2n - o2)) / O2_WALL < tol:
            o2 = o2n
            break
        o2 = 0.5 * o2 + 0.5 * o2n
    return x, r, solve_no_vessel(x, o2, centres, r), o2


def local_L_O2(o2_local):
    """Oxygen relaxation scale in the O2-LIMITED regime, sqrt(D_O2*Km_resp/V_M).

    Valid only where o2 << Km_resp. Where o2 >> Km_resp respiration is zeroth order and
    there is no exponential screening; the returned number is then not a screening length
    and is flagged by the caller.
    """
    return float(np.sqrt(G1.D_O2 * G1.KM_RESP / G1.V_M))


def main():
    t0 = time.time()
    L_O2_derived = local_L_O2(0.0)
    print("GATE 4 -- pair in the vessel geometry; lambda eliminated, not justified\n")
    print("  vessel wall O2 %.1f uM, absorbing NO wall, domain %.0f um, dx %.2f um"
          % (O2_WALL * 1e6, XMAX, DX))
    print("  NO restoring term anywhere. Derived oxygen relaxation scale in the")
    print("  O2-limited regime: sqrt(D_O2*Km_resp/V_M) = %.2f um  (locked parameters only)"
          % L_O2_derived)
    print("  PREDICTION recorded before running: cross-talk ~0.19%, down from 0.396%,")
    print("  because A ~ 1/sqrt(lambda) and the derived scale corresponds to lambda ~ 3.15 /s.\n")

    x0, o20 = solve_o2_vessel()
    print("  unperturbed oxygen profile (no cells):")
    for d in (0, 20, 60, 120, 220, 350):
        i = int(np.argmin(np.abs(x0 - d)))
        print("     x = %3d um   O2 = %8.4f uM   %s"
              % (d, o20[i] * 1e6,
                 "O2 >> Km_resp: respiration SATURATED, no exponential screening"
                 if o20[i] > 5 * G1.KM_RESP else "O2-limited regime"))
    print()

    rows = []
    for xc in DEPTHS:
        i_c = int(np.argmin(np.abs(x0 - xc)))
        kd = k_degradation(o20[i_c], K_OBS_CORR, G1.RHO, k_anoxia=K_ANOXIA_CORR)
        L_NO_here = diffusion_length(G1.D_NO * 1e-8, kd) * 1e4
        print("  DEPTH x_c = %.0f um   unperturbed O2 = %.4f uM   L_NO here = %.2f um"
              % (xc, o20[i_c] * 1e6, L_NO_here))
        sv, dv, cs = [], [], []
        for s in SPACINGS:
            c1, c2 = xc - s / 2.0, xc + s / 2.0
            if c1 - G1.A0 < DX:
                print("     s %3d um  SKIPPED -- near cell at %.1f um would breach the wall"
                      % (s, c1))
                continue
            x, r, CA, _ = run_vessel([c1])
            _, _, CB, _ = run_vessel([c2])
            _, _, CAB, o2AB = run_vessel([c1, c2])
            iA = int(np.argmin(np.abs(x - c1)))
            a = float(CA[iA, 0])
            d = float(CAB[iA, 0] - CA[iA, 0] - CB[iA, 0])
            sv.append(s); dv.append(d * 1e9); cs.append(a * 1e9)
            print("     s %3d um   C_single %8.4f nM   dC_pair %10.6f nM  (%+.4f%%)"
                  % (s, a * 1e9, d * 1e9, 100 * d / a))
            rows.append(dict(x_c_um=xc, s_um=s, o2_unperturbed_uM=o20[i_c] * 1e6,
                             L_NO_um=L_NO_here, C_single_nM=a * 1e9,
                             dC_pair_nM=d * 1e9, dC_pair_pct=100 * d / a))
        if len(sv) >= 3:
            try:
                p, _ = curve_fit(lambda z, A, L: A * np.exp(-z / L),
                                 np.array(sv, float), np.array(dv, float),
                                 p0=[dv[0], 25.0], maxfev=200000,
                                 bounds=([-1, 1], [1, 500]))
                res = np.array(dv) - p[0] * np.exp(-np.array(sv, float) / p[1])
                r2 = 1 - np.sum(res ** 2) / np.sum((np.array(dv) - np.mean(dv)) ** 2)
                print("     FIT  A = %.6f nM (%.4f%% of single)   L = %.2f um   R2 = %.5f"
                      % (p[0], 100 * p[0] / cs[0], p[1], r2))
                print("          derived O2 scale %.2f um, L_NO here %.2f um"
                      % (L_O2_derived, L_NO_here))
                rows.append(dict(x_c_um=xc, s_um="fit", o2_unperturbed_uM=o20[i_c] * 1e6,
                                 L_NO_um=L_NO_here, C_single_nM=cs[0], dC_pair_nM="",
                                 dC_pair_pct="", A_nM=p[0], A_pct=100 * p[0] / cs[0],
                                 L_fit_um=p[1], R2=r2))
            except Exception as e:                                    # noqa: BLE001
                print("     FIT FAILED:", e)
        print()

    out = os.path.join(REPO, "data", "gate4_vessel_pair.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keys = ["x_c_um", "s_um", "o2_unperturbed_uM", "L_NO_um", "C_single_nM", "dC_pair_nM",
            "dC_pair_pct", "A_nM", "A_pct", "L_fit_um", "R2"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for rr in rows:
            w.writerow({k: rr.get(k, "") for k in keys})
    print("  %.1f min -> %s" % ((time.time() - t0) / 60, os.path.relpath(out, REPO)))
    print("\n  READ IT LIKE THIS")
    print("    A around 0.19% and L near 21 um   -> the prediction holds; the oxygen scale")
    print("        is now DERIVED and Gate 1's second half can pass on its own terms.")
    print("    A much larger, or L near 66 um    -> something still carries the closure's")
    print("        signature and lambda was not the only thing setting the range.")
    print("    A varies strongly with depth      -> the effect is O2-tension dependent, which")
    print("        is a result in itself and is what running three depths is for.")


if __name__ == "__main__":
    main()
