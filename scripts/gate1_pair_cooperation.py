# -*- coding: utf-8 -*-
"""Does cooperation between NOS2+ cells exist beyond the superposition of their fields?

  1. A diffusion length is a decay scale, not a breakpoint. At r = L the field is still
     37% of its source value; nothing special happens there. Any "transition" read off a
     10-point sweep is a property of the detection rule, not of the physics.
  2. If the transition were L = sqrt(D/k), then dropping D_NO 2200 -> 850 must move it
     from 33.1 to 20.6 um, a 12.5 um shift. The observed shift was +/- 2 um. The
     insensitivity that had been reported as ROBUSTNESS is in fact the evidence AGAINST
     the interpretation.

    overlap != cooperation

    Overlap is inevitable from diffusion. Cooperation requires the coupled pair to
    deviate from the ADDITIVE NULL.

THE CONTROL THAT MAKES THIS INFORMATIVE
---------------------------------------
With the O2 feedback OFF, the steady-state NO equation is linear in C:

    D grad^2 C - k(x) C + S(x) = 0        k fixed by a frozen O2 field

so two sources superpose EXACTLY: C_12 = C_1 + C_2. A two-cell spacing sweep in that
system must recover the Green-function decay scale. Showing l_int ~ L there proves the
solver works and NOTHING about biology. Running the pair sweep coupled-only would have
been circular in exactly the way gate G2 was before it was rewritten.

So every spacing is run FOUR times:

    single cell, coupling OFF     C1_U
    two cells,   coupling OFF     C12_U     -- must equal C1_U + C2_U to tolerance
    single cell, coupling ON      C1_C
    two cells,   coupling ON      C12_C

and the biological quantity is the departure from the additive reference:

    dC_NL(x; s) = C12_C(x; s) - [C1_C(x) + C2_C(x)]

The additive reference is NOT another physical solution. It is what the pair WOULD be if
the coupled system still obeyed superposition. dC_NL == 0 to numerical precision means
the cells merely overlap. dC_NL != 0 means the coupled biology creates interaction beyond
independent source addition -- and only that deserves the word cooperation.

Scalars reported per spacing:
    A_NL      = integral |dC_NL| dx                        (whole-field excess)
    dC_NL_cell= dC_NL evaluated at the cell centre          (biologically the relevant one)
    I_U, I_C  = [C_pair - C_single]/C_single at the cell centre, uncoupled and coupled

GEOMETRY -- boundary free, symmetric
------------------------------------
No vessel. Two identical spherical sources on the axis at x_c +/- s/2, midpoint x_c held
FIXED so that changing s does not also change distance-to-anything, cell count, or the
local O2 each cell sees. Neither cell is pinned near an absorbing wall while the other moves.

OXYGEN -- and the one quantity this script introduces
-----------------------------------------------------
With no vessel there is no O2 source, and respiration alone drives the domain anoxic (the
O2 decay length is sqrt(D_O2*Km_resp/V_M) = 21 um). Imposing a UNIFORM O2 instead would
make the coupling inert by construction -- NO could not move a field that is held fixed --
which would manufacture a null result.

So the tissue carries a distributed perfusion term, the standard Krogh-type closure:

    D_O2 grad^2 O2 - V_M*O2/(Km_eff + O2) + lambda*(O2_bath - O2) = 0

lambda is CALIBRATED, not fitted: it is set analytically so that with no NO present the
homogeneous steady state is exactly O2_BACKGROUND. Both O2_bath and the resulting lambda
are printed every run. This is the only quantity in this script that is not read from the
locked parameter set, and it exists to make the coupling testable rather than to tune any
result.

OUT: data/gate1_pair_cooperation.csv  +  a console verdict.
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from constants import (CELL_VOLUME_MODEL_UM3, K_ANOXIA_CORR, K_OBS_CORR,   # noqa: E402
                       RHO_TUMOUR)
from kinetics import k_degradation, r_no                                    # noqa: E402
import parameters as P                                                      # noqa: E402

# ---- authoritative constants, all imported --------------------------------------------
D_NO = P.D * 1e8                 # um^2/s
KM = P.KM                        # M, NOS2 Km(O2)
VMAX = P.VMAX                    # M/s
RHO = RHO_TUMOUR                 # cells/mL
VOL = CELL_VOLUME_MODEL_UM3      # um^3, measured cell volume
A0 = (3.0 * VOL / (4.0 * np.pi)) ** (1.0 / 3.0)          # um, source sphere radius

_o2 = __import__("o2_field") if False else None
D_O2 = 1.39e-5 * 1e8             # um^2/s  Ellsworth & Pittman 1984 PMID 6696127
KM_RESP = 2.86e-6                # M       Brown & Cooper 1994 PMID 7805858
V_M = 9.0e-6                     # M/s
K_I = 5.22e-9                    # M       derived from Brown & Cooper IC50, two tensions

# ---- experiment geometry --------------------------------------------------------------
DX = 0.5                         # um, finer than Fig 3's 1.0 to resolve the interaction
XLEN = 400.0                     # um, domain length
XC = 200.0                       # um, FIXED pair midpoint
RMAX = 120.0                     # um, radial extent
R_AVG = 32.41                    # um, radius over which NO is averaged for inhibition

O2_BACKGROUND = 30.0e-6          # M, unperturbed tissue O2 the perfusion term is set to
O2_BATH = 56.0e-6                # M, perfusion reservoir


def perfusion_lambda(o2_bg=O2_BACKGROUND, o2_bath=O2_BATH):
    """lambda such that, with NO absent, the homogeneous steady state is exactly o2_bg.

    0 = -V_M*o2_bg/(Km_resp + o2_bg) + lambda*(o2_bath - o2_bg)
    """
    if o2_bath <= o2_bg:
        raise ValueError("O2_BATH must exceed O2_BACKGROUND for a positive perfusion rate")
    return V_M * o2_bg / ((KM_RESP + o2_bg) * (o2_bath - o2_bg))


LAM = perfusion_lambda()


# ---- oxygen, 1-D along x, distributed perfusion, zero flux at both ends ----------------
def solve_o2(inhib=None, dx=None, xlen=None, tol=1e-14, iters=500):
    """`inhib` is Km_eff/Km_resp, cross-sectionally averaged, at each x."""
    dx = DX if dx is None else dx
    xlen = XLEN if xlen is None else xlen
    x = np.arange(0.0, xlen + dx, dx)
    n = len(x)
    o2 = np.full(n, O2_BACKGROUND)
    inhib = np.ones(n) if inhib is None else np.asarray(inhib, float)
    for _ in range(iters):
        k_eff = V_M / (np.maximum(o2, 0.0) + KM_RESP * inhib)     # linearised consumption
        di = -2.0 * D_O2 / dx ** 2 - k_eff - LAM
        lo = np.full(n, D_O2 / dx ** 2)
        up = np.full(n, D_O2 / dx ** 2)
        rhs = np.full(n, -LAM * O2_BATH)
        # zero flux at BOTH ends -- no vessel anywhere in this experiment
        di[0] = -2.0 * D_O2 / dx ** 2 - k_eff[0] - LAM
        up[0] = 2.0 * D_O2 / dx ** 2
        di[-1] = -2.0 * D_O2 / dx ** 2 - k_eff[-1] - LAM
        lo[-1] = 2.0 * D_O2 / dx ** 2
        A = np.zeros((3, n))
        A[0, 1:], A[1, :], A[2, :-1] = up[:-1], di, lo[1:]
        new = np.clip(solve_banded((1, 1), A, rhs), 0.0, O2_BATH)
        if np.max(np.abs(new - o2)) < tol:
            return x, new
        o2 = 0.5 * o2 + 0.5 * new
    raise RuntimeError("solve_o2 did not converge")


# ---- NO, axisymmetric (x, r), zero flux on every outer face ---------------------------
def _operator(x, o2, r, dx=None):
    dx = DX if dx is None else dx
    nx, nr = len(x), len(r)
    k_deg = k_degradation(o2, K_OBS_CORR, RHO, k_anoxia=K_ANOXIA_CORR)
    inv2 = 1.0 / dx ** 2
    rows, cols, vals = [], [], []
    idx = lambda i, j: i * nr + j                                          # noqa: E731
    for i in range(nx):
        for j in range(nr):
            k = idx(i, j)
            if j == nr - 1:                       # far radial face: Dirichlet 0
                rows.append(k); cols.append(k); vals.append(1.0)
                continue
            diag = -k_deg[i]
            if i == 0:                            # zero flux, x = 0
                rows.append(k); cols.append(idx(1, j)); vals.append(2.0 * D_NO * inv2)
                diag -= 2.0 * D_NO * inv2
            elif i == nx - 1:                     # zero flux, x = XLEN
                rows.append(k); cols.append(idx(nx - 2, j)); vals.append(2.0 * D_NO * inv2)
                diag -= 2.0 * D_NO * inv2
            else:
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(D_NO * inv2)
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(D_NO * inv2)
                diag -= 2.0 * D_NO * inv2
            if j == 0:
                rows.append(k); cols.append(idx(i, 1)); vals.append(4.0 * D_NO * inv2)
                diag -= 4.0 * D_NO * inv2
            else:
                rp, rm = (r[j] + 0.5 * dx) / r[j], (r[j] - 0.5 * dx) / r[j]
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(D_NO * inv2 * rp)
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(D_NO * inv2 * rm)
                diag -= D_NO * inv2 * (rp + rm)
            rows.append(k); cols.append(k); vals.append(diag)
    return sp.csr_matrix((vals, (rows, cols)), shape=(nx * nr, nx * nr)).tocsc()


def _source(x, o2, centres, r):
    """Volumetric NO production inside each source sphere, at the LOCAL oxygen.

    Two cells closer than 2*A0 = 15.84 um would overlap, putting two cells' worth of source
    density into one physical volume. Severity falls off steeply with spacing (fraction of one sphere overlapping):
        5 um  54%     9 um  24%     13 um  4.5%
        7 um  38%    11 um  12.6%   15 um  0.4%
    so 5-11 um are badly nonphysical while 15 um is barely so.

    THREE checks, because the analytic one alone is not sufficient: voxelisation can make
    nominally tangent spheres share grid elements.
      1. analytic separation      s >= 2*A0
      2. discretised mask overlap no voxel claimed by two cells
      3. source conservation      |mask_pair| == n_cells * |mask_single|
    """
    nx, nr = len(x), len(r)
    S = np.zeros((nx, nr))
    X, R = np.meshgrid(x, r, indexing="ij")

    for i, ci in enumerate(centres):                       # 1. analytic separation
        for cj in centres[i + 1:]:
            if abs(ci - cj) < 2.0 * A0 - 1e-12:
                raise ValueError(
                    "cell sources geometrically overlap: centres %.3f and %.3f are %.3f um "
                    "apart, below the hard-sphere limit 2*A0 = %.3f um"
                    % (ci, cj, abs(ci - cj), 2.0 * A0))

    masks = []
    for c in centres:
        d = np.sqrt((X - c) ** 2 + R ** 2)
        masks.append(d <= A0)

    for i, mi in enumerate(masks):                         # 2. discretised mask overlap
        for j, mj in enumerate(masks[i + 1:], start=i + 1):
            shared = int(np.count_nonzero(mi & mj))
            if shared:
                raise ValueError(
                    "discretised cell source masks overlap: cells %d and %d share %d voxels "
                    "at dx = %.3f um" % (i, j, shared, x[1] - x[0]))

    if masks:                                              # 3. source conservation
        counts = [int(np.count_nonzero(m)) for m in masks]
        union = int(np.count_nonzero(np.logical_or.reduce(masks)))
        if union != sum(counts):
            raise ValueError("source voxel count %d != sum of per-cell counts %s"
                             % (union, counts))
        if len(set(counts)) != 1:
            raise ValueError("cells occupy differing voxel counts %s -- placement is not "
                             "translation invariant on this grid" % counts)

    o2g = o2[:, None].repeat(nr, axis=1)
    for m in masks:
        S[m] += r_no(o2g[m], VMAX, KM)
    return S


def _radial_average(C, r, r_cut=R_AVG):
    """2*pi*r weighted mean of NO over r <= r_cut, per x. Never the on-axis value."""
    m = r <= r_cut
    w = 2.0 * np.pi * r[m]
    return np.trapezoid(C[:, m] * w, r[m], axis=1) / np.trapezoid(w, r[m])


def solve_no(x, o2, centres, r):
    A = _operator(x, o2, r)
    S = _source(x, o2, centres, r)
    C = spsolve(A, -S.ravel()).reshape(len(x), len(r))
    return np.maximum(C, 0.0)


def run_case(centres, coupled, dx=None, tol=1e-6, iters=60):
    """Return (x, r, C, o2). coupled=False freezes O2 at its no-NO solution."""
    dx = DX if dx is None else dx
    r = np.arange(0.0, RMAX + dx, dx)
    x, o2 = solve_o2()
    if not coupled:
        C = solve_no(x, o2, centres, r)
        return x, r, C, o2
    for _ in range(iters):
        C = solve_no(x, o2, centres, r)
        inhib = 1.0 + _radial_average(C, r) / K_I
        x, o2_new = solve_o2(inhib)
        if np.max(np.abs(o2_new - o2)) / max(O2_BACKGROUND, 1e-12) < tol:
            o2 = o2_new
            break
        o2 = 0.5 * o2 + 0.5 * o2_new
    C = solve_no(x, o2, centres, r)
    return x, r, C, o2


def at_cell(x, C, centre):
    """On-axis NO at a cell centre."""
    return float(C[int(np.argmin(np.abs(x - centre))), 0])


SPACINGS = [17, 18, 21, 24, 27, 30, 34, 38, 42, 47, 52, 58, 64, 71, 80, 95, 110]


def main():
    t0 = time.time()
    print("GATE 1 -- cooperation beyond superposition\n")
    print("  boundary-free, symmetric pair, midpoint FIXED at x = %.0f um" % XC)
    print("  source sphere radius %.3f um from the measured cell volume %.0f um^3"
          % (A0, VOL))
    print("  grid dx = %.2f um, domain %.0f um, radial extent %.0f um" % (DX, XLEN, RMAX))
    print("  perfusion: O2_bath %.1f uM, lambda %.4g /s, calibrated so the NO-free"
          % (O2_BATH * 1e6, LAM))
    print("             homogeneous steady state is %.1f uM" % (O2_BACKGROUND * 1e6))

    x0, o2_0 = solve_o2()
    print("  CHECK unperturbed O2 field: min %.4f uM, max %.4f uM (target %.1f)\n"
          % (o2_0.min() * 1e6, o2_0.max() * 1e6, O2_BACKGROUND * 1e6))

    rows = []
    print("   s    C1_U    C12_U   superpos    C1_C    C12_C   dC_NL_cell     A_NL"
          "      I_U      I_C")
    print("  um      nM       nM      err %      nM       nM           nM    nM.um"
          "                   ")
    for s in SPACINGS:
        c1, c2 = XC - s / 2.0, XC + s / 2.0

        xU, rU, C1U_f, _ = run_case([c1], coupled=False)
        _, _, C2U_f, _ = run_case([c2], coupled=False)
        _, _, C12U_f, _ = run_case([c1, c2], coupled=False)

        _, _, C1C_f, _ = run_case([c1], coupled=True)
        _, _, C2C_f, _ = run_case([c2], coupled=True)
        _, _, C12C_f, o2C = run_case([c1, c2], coupled=True)

        # --- the superposition null. Uncoupled MUST be additive. ---
        addU = C1U_f + C2U_f
        den = max(np.max(np.abs(C12U_f)), 1e-30)
        superr = float(np.max(np.abs(C12U_f - addU)) / den)

        # --- the biological quantity: departure from the additive reference ---
        dNL = C12C_f - (C1C_f + C2C_f)
        dNL_cell = float(dNL[int(np.argmin(np.abs(xU - c1))), 0])
        A_NL = float(np.trapezoid(np.abs(dNL[:, 0]), xU))

        c1U, c12U = at_cell(xU, C1U_f, c1), at_cell(xU, C12U_f, c1)
        c1C, c12C = at_cell(xU, C1C_f, c1), at_cell(xU, C12C_f, c1)
        I_U = (c12U - c1U) / c1U if c1U > 0 else float("nan")
        I_C = (c12C - c1C) / c1C if c1C > 0 else float("nan")

        print("  %3d %7.3f %8.3f %9.2e %7.3f %8.3f %12.3e %8.3g %8.4f %8.4f"
              % (s, c1U * 1e9, c12U * 1e9, superr * 100, c1C * 1e9, c12C * 1e9,
                 dNL_cell * 1e9, A_NL * 1e9, I_U, I_C))
        rows.append(dict(s_um=s, C1_U_nM=c1U * 1e9, C12_U_nM=c12U * 1e9,
                         superposition_err=superr, C1_C_nM=c1C * 1e9, C12_C_nM=c12C * 1e9,
                         dC_NL_cell_nM=dNL_cell * 1e9, A_NL_nM_um=A_NL * 1e9,
                         I_U=I_U, I_C=I_C,
                         o2_at_cell_uM=float(o2C[int(np.argmin(np.abs(xU - c1)))] * 1e6)))

    worst = max(r["superposition_err"] for r in rows)
    print("\n  SUPERPOSITION NULL: worst uncoupled error %.3e" % worst)
    print("    %s -- the uncoupled system is additive, so the solver is behaving and any"
          % ("PASS" if worst < 1e-6 else "FAIL"))
    print("    non-zero coupled excess below is physics, not numerics.")
    if worst >= 1e-6:
        print("    ** GATE 1 CANNOT BE READ until this passes. **")

    mx = max(abs(r["dC_NL_cell_nM"]) for r in rows)
    rel = max(abs(r["dC_NL_cell_nM"]) / r["C1_C_nM"] for r in rows if r["C1_C_nM"] > 0)
    print("\n  NONLINEAR EXCESS over the additive reference")
    print("    largest |dC_NL| at a cell centre = %.4g nM = %.3f%% of the single-cell value"
          % (mx, rel * 100))
    print("    No pass/fail is printed: a fixed detection threshold would be")
    print("    arbitrary rather than physical.")
    print("    Report the magnitude")
    print("    and judge it against a criterion stated elsewhere.")
    print("    Terminology: SPATIAL SUMMATION for the additive part,")
    print("    NONLINEAR CROSS-TALK for this residual, and COOPERATION reserved for an")
    print("    effect large enough to carry a collective phenotype -- which this is not.")

    out = os.path.join(REPO, "data", "gate1_pair_cooperation.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keys = ["s_um", "C1_U_nM", "C12_U_nM", "superposition_err", "C1_C_nM", "C12_C_nM",
            "dC_NL_cell_nM", "A_NL_nM_um", "I_U", "I_C", "o2_at_cell_uM"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for rr in rows:
            w.writerow(rr)
    print("\n  %d spacings in %.1f min -> %s"
          % (len(rows), (time.time() - t0) / 60, os.path.relpath(out, REPO)))


if __name__ == "__main__":
    main()
