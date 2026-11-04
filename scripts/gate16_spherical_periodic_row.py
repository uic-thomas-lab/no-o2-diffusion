# -*- coding: utf-8 -*-
"""Maximum spacing s_max_parallel(d, C_T) for true spherical sources, via a 3-D periodic unit cell.

Both are interpretable. Neither reopens Gates 1-13.

WHY 3-D IS FORCED
-----------------------
True spherical sources + a row parallel to the vessel is INHERENTLY 3-D. Gate 15 has rod
inflation precisely because a disk in a 2-D Cartesian calculation IS an infinite cylinder out
of plane; no other 2-D shape fixes that. Nor can the axisymmetric (x, r) solver be rotated
into a row -- revolving a row of spheres about the depth axis makes concentric SHELLS.

THE GEOMETRY -- ONE SPHERE, PERIODICITY MAKES THE ROW
-----------------------------------------------------
    x   depth from the vessel.   Dirichlet C = 0 at x = 0; zero flux at XMAX.
    y   ALONG the row.           Mirror at y = 0 (sphere centre) and at y = s/2 (midpoint
                                 between periodic neighbours). The even extension of those
                                 two mirrors IS the infinite row of period s.
    z   transverse.              Mirror at z = 0 (sphere centre) and at z = ZMAX (artificial).

THE REDUCED DOMAIN HOLDS A QUARTER SPHERE, NOT AN OCTANT. Mirrors at y = 0 and z = 0 halve
the sphere TWICE: 1/2 * 1/2 = 1/4. There is NO mirror at x = d, because k(x) and the oxygen
field are not symmetric about the cell centre. Calling it an octant is exactly how a
factor-of-two source-normalisation error gets in, so gate_source() asserts

    4 * integral over the reduced domain  ==  integral over the full sphere.

z IS NOT PERIODIC. The system is periodic only along the row. The cosine transform in z
needs an outer boundary at ZMAX, whose even extension creates mirror rows at 2*ZMAX. The
solver is EXACT for that finite domain; the intended transversely-isolated row is recovered
only once ZMAX is large enough, which gate_zmax() demonstrates rather than assumes.

THE SPECTRAL REDUCTION -- EXACT, NOT AN APPROXIMATION
------------------------------------------------------
With feedback off the problem is linear and k varies ONLY in x:

    -D grad^2 C + k(x) C = S(x, y, z)

The discrete mirror Laplacians in y and z are diagonalised exactly by the cosine basis
cos(pi*m*(j+1/2)/N) with eigenvalues -(4/h^2) sin^2(pi*m/(2N)). Transforming both leaves, for
every transverse mode (m, n),

    [-D d2/dx2 + k(x) + D(lam_m + lam_n)] Chat_mn(x) = Shat_mn(x)

which is TRIDIAGONAL in x. The spherical source need not be separable; it enters only through
the transformed right-hand side. So this is true spherical-source physics without a
seven-million-unknown direct solve.

FOUR PREFLIGHT GATES -- THE SWEEP DOES NOT START UNLESS ALL FOUR PASS
----------------------------------------------------------------------
  SOURCE  quarter-sphere normalisation + production conserved across depth and spacing
  A       isolated sphere vs the existing axisymmetric G4 solver: C_peak, the inward and
          outward profiles, and R_0.5 / R_1 / R_2 -- more than one scalar, because the
          DIRECTIONAL ASYMMETRY is part of the result
  B       spectral vs a DIRECT 3-D sparse solve of the identical discrete problem, on a
          small domain. A alone is insufficient: two implementations can agree while sharing
          a modelling assumption. B tests the algebra itself -- transform, eigenvalues,
          reconstruction, boundary indexing, normalisation.
  C       transverse-domain convergence: ZMAX -> 1.5x -> 2x must not move s_max, the trough,
          or the peak

INVARIANTS HARD-CODED
---------------------
  * PHYSICAL grid spacing h is FIXED as s changes; ny = s/(2h) varies. Holding ny fixed and
    letting h float would make s_max inherit discretisation changes.
  * FRACTIONAL-VOLUME weighting at the sphere boundary, so voxelisation cannot make source
    volume fluctuate as the sphere moves through x.
  * NOS2 production is evaluated LOCALLY from O2(x) throughout the sphere, never collapsed to
    O2(d). The sphere spans a finite oxygen gradient, most steeply near the vessel.

ENDPOINT, FROZEN BEFORE CODING
------------------------------------
  PRIMARY   s_max = largest s whose MINIMUM [NO] on the intercellular path between
            neighbouring cell SURFACES stays >= C_T. Same functional quantity as Gates 14b
            and 15, so the three geometries remain comparable.
  RECORDED  V(C >= C_T)/V_domain at the accepted s_max. Written to the CSV, NEVER used to
            define s_max. Both come from one solve, so the two can never be silently
            interchanged.

OUT: data/gate16_spherical_periodic_row.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import gate1_pair_cooperation as G1                        # noqa: E402
import gate4_vessel_pair as G4                             # noqa: E402
from kinetics import k_degradation, r_no                   # noqa: E402
from constants import K_OBS_CORR, K_ANOXIA_CORR            # noqa: E402

H = 1.0                               # um, FIXED physical spacing in x, y and z
XMAX = 300.0                          # um, NO domain (O2 comes from the locked 500 um solve)
ZMAX = 150.0                          # um, transverse half-extent; gate C tests this
A0 = G1.A0                            # um, 7.918
S_MIN = 18                            # um, EVEN, and > 2*A0 = 15.836
S_MAX = 400                           # um, scan ceiling -> CENSORED, never a measurement
S_COARSE = 20                         # um (even); refinement bisects to 2 um
DEPTHS = [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0]
THRESHOLDS = [0.5, 1.0, 2.0]
SUB = 4                               # subsamples per axis for fractional sphere volume


# --------------------------------------------------------------------------- geometry
def x_grid(xmax=XMAX, h=H):
    """Node-centred in x: x_0 = 0 is the vessel wall itself."""
    return np.arange(0.0, xmax + h, h)


def oxygen(x):
    """The locked 1-D vessel O2 profile, decimated onto x. Never reinterpolated."""
    xf, o2f = G4.solve_o2_vessel()
    step = int(round((x[1] - x[0]) / G4.DX))
    xd, o2d = xf[::step], o2f[::step]
    if len(xd) < len(x) or not np.allclose(xd[:len(x)], x):
        raise ValueError("O2 decimation does not land on the x grid")
    return o2d[:len(x)]


def sphere_weights(x, ny, nz, d, h=H, sub=SUB, frac=True):
    """Fraction of each control volume inside the sphere of radius A0 centred at (d, 0, 0).

    Control volume of (i, j, k) is [x_i-h/2, x_i+h/2] x [j*h, (j+1)*h] x [k*h, (k+1)*h] --
    node-centred in x, cell-centred in y and z, every cell of volume h^3.

    frac=False reproduces the hard mask used by G1._source, which Validation A needs so that
    the comparison against G4 tests the SOLVER rather than the source discretisation.
    """
    nx = len(x)
    nbj = min(ny, int(np.ceil(A0 / h)) + 1)
    nbk = min(nz, int(np.ceil(A0 / h)) + 1)
    ilo = max(0, int(np.floor((d - A0) / h)) - 1)
    ihi = min(nx - 1, int(np.ceil((d + A0) / h)) + 1)

    w = np.zeros((nx, ny, nz))
    if not frac:
        yj = (np.arange(nbj) + 0.5) * h
        zk = (np.arange(nbk) + 0.5) * h
        dd = ((x[ilo:ihi + 1, None, None] - d) ** 2
              + yj[None, :, None] ** 2 + zk[None, None, :] ** 2)
        w[ilo:ihi + 1, :nbj, :nbk] = (dd <= A0 ** 2).astype(float)
        return w

    off = (np.arange(sub) + 0.5) / sub - 0.5                 # sub-cell centres, in units of h
    ax = (x[ilo:ihi + 1, None] + off[None, :] * h - d)       # (nix, sub)
    ay = ((np.arange(nbj)[:, None] + 0.5 + off[None, :]) * h)
    az = ((np.arange(nbk)[:, None] + 0.5 + off[None, :]) * h)
    fx = ax[:, :, None, None, None, None] ** 2
    fy = ay[None, None, :, :, None, None] ** 2
    fz = az[None, None, None, None, :, :] ** 2
    inside = (fx + fy + fz) <= A0 ** 2                        # (nix,sub,nbj,sub,nbk,sub)
    w[ilo:ihi + 1, :nbj, :nbk] = inside.mean(axis=(1, 3, 5))
    return w


def source(x, o2, w):
    """Volumetric NO production, LOCAL O2 throughout the sphere -- never collapsed to O2(d)."""
    return w * r_no(o2, G1.VMAX, G1.KM)[:, None, None]


# --------------------------------------------------------------------------- spectral core
def cos_basis(n):
    """B[m, j] = cos(pi*m*(j+1/2)/n); eigenvalues of the cell-centred mirror Laplacian."""
    m = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    B = np.cos(np.pi * m * (j + 0.5) / n)
    lam = -(4.0 / H ** 2) * np.sin(np.pi * np.arange(n) / (2.0 * n)) ** 2
    wt = np.full(n, 2.0 / n)
    wt[0] = 1.0 / n
    return B, lam, wt


def solve_spectral(x, o2, S, ny, nz, want_full=False, i_probe=None):
    """Solve -D lap C + k C = S on the reduced domain. Returns (C at i_probe, peak, full?).

    Mode index is m*nz + n, matching lam = (lam_y[:, None] + lam_z[None, :]).ravel().
    """
    nx = len(x)
    D = G1.D_NO
    k = k_degradation(o2, K_OBS_CORR, G1.RHO, k_anoxia=K_ANOXIA_CORR)
    By, lamy, wy = cos_basis(ny)
    Bz, lamz, wz = cos_basis(nz)
    lam = (lamy[:, None] + lamz[None, :]).ravel()
    M = ny * nz

    nzi = np.flatnonzero(S.any(axis=(1, 2)))                  # only the sphere's x rows
    Shat = {int(i): (By @ S[i] @ Bz.T).ravel() for i in nzi}

    inv2 = 1.0 / H ** 2
    base = 2.0 * D * inv2 - D * lam                           # mode-dependent diagonal
    cp = np.zeros((nx, M))
    dp = np.zeros((nx, M))
    # i = 0 : Dirichlet C = 0  ->  b = 1, c = 0, rhs = 0
    for i in range(1, nx):
        a_i = -2.0 * D * inv2 if i == nx - 1 else -D * inv2    # mirror at the far face
        c_i = 0.0 if i == nx - 1 else -D * inv2
        b_i = base + k[i]
        den = b_i - a_i * cp[i - 1]
        cp[i] = c_i / den
        dp[i] = (Shat.get(i, 0.0) - a_i * dp[i - 1]) / den

    full = np.zeros((nx, ny, nz)) if want_full else None
    probe = None
    peak = 0.0
    xi = np.zeros(M)
    for i in range(nx - 1, -1, -1):
        xi = dp[i] - cp[i] * xi
        if want_full or i == i_probe:
            Ci = (By.T * wy) @ (xi.reshape(ny, nz) @ (Bz * wz[:, None]))
            Ci = np.maximum(Ci, 0.0)
            if want_full:
                full[i] = Ci
            if i == i_probe:
                probe = Ci
            peak = max(peak, float(Ci.max()))
    return probe, peak, full


def solve_direct(x, o2, S, ny, nz):
    """The SAME discrete problem, assembled explicitly and solved by sparse LU.

    Only for Validation B on a small domain. Index k = (i*ny + j)*nz + kk.
    """
    nx = len(x)
    D = G1.D_NO
    k = k_degradation(o2, K_OBS_CORR, G1.RHO, k_anoxia=K_ANOXIA_CORR)
    inv2 = 1.0 / H ** 2
    N = nx * ny * nz
    idx = lambda i, j, kk: (i * ny + j) * nz + kk             # noqa: E731
    rows, cols, vals = [], [], []
    rhs = np.zeros(N)
    for i in range(nx):
        for j in range(ny):
            for kk in range(nz):
                p = idx(i, j, kk)
                if i == 0:                                     # Dirichlet vessel wall
                    rows.append(p); cols.append(p); vals.append(1.0)
                    continue
                diag = k[i]
                # x : mirror at i = nx-1
                if i == nx - 1:
                    rows.append(p); cols.append(idx(nx - 2, j, kk))
                    vals.append(-2.0 * D * inv2); diag += 2.0 * D * inv2
                else:
                    rows.append(p); cols.append(idx(i - 1, j, kk)); vals.append(-D * inv2)
                    rows.append(p); cols.append(idx(i + 1, j, kk)); vals.append(-D * inv2)
                    diag += 2.0 * D * inv2
                # y and z : mirror at BOTH faces (cell-centred)
                for lo, hi, n_, mk in ((j - 1, j + 1, ny, "y"), (kk - 1, kk + 1, nz, "z")):
                    for nb in (lo, hi):
                        if 0 <= nb < n_:
                            q = idx(i, nb, kk) if mk == "y" else idx(i, j, nb)
                            rows.append(p); cols.append(q); vals.append(-D * inv2)
                            diag += D * inv2
                rows.append(p); cols.append(p); vals.append(diag)
                rhs[p] = S[i, j, kk]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()
    return np.maximum(spsolve(A, rhs).reshape(nx, ny, nz), 0.0)


# --------------------------------------------------------------------------- endpoint
def trough_and_smax_probe(x, o2, d, s, want_full=False):
    """Minimum [NO] (nM) on the intercellular path between neighbouring cell SURFACES."""
    ny, nz = int(round(s / (2.0 * H))), int(round(ZMAX / H))
    w = sphere_weights(x, ny, nz, d)
    S = source(x, o2, w)
    i_d = int(round(d / H))
    probe, peak, full = solve_spectral(x, o2, S, ny, nz, want_full=want_full, i_probe=i_d)
    yj = (np.arange(ny) + 0.5) * H
    outside = yj > A0                                          # between the SURFACES
    line = probe[outside, 0] * 1e9                             # z-index 0 = the row's plane
    return float(line.min()), peak * 1e9, full, w


# --------------------------------------------------------------------------- gates
def gate_source(x, o2):
    print("\n  GATE SOURCE -- quarter-sphere normalisation and production conservation")
    exact = 4.0 / 3.0 * np.pi * A0 ** 3
    ok = True
    ref = {}
    for d in DEPTHS:
        for s in (S_MIN, 100, 300):
            ny, nz = int(round(s / (2.0 * H))), int(round(ZMAX / H))
            w = sphere_weights(x, ny, nz, d)
            vol4 = 4.0 * w.sum() * H ** 3                      # QUARTER sphere -> x4
            q4 = 4.0 * source(x, o2, w).sum() * H ** 3
            if d not in ref:
                ref[d] = (vol4, q4)
                err = abs(vol4 - exact) / exact
                print("    d=%5.1f  4*V_reduced = %8.2f um3  (exact %8.2f, err %.3f%%)  Q=%.4e"
                      % (d, vol4, exact, 100 * err, q4))
                if err > 1e-2:
                    print("      FAIL: quarter-sphere volume off by >1%"); ok = False
            else:
                if abs(vol4 - ref[d][0]) > 1e-9 or abs(q4 - ref[d][1]) / ref[d][1] > 1e-12:
                    print("      FAIL: d=%.1f production changed with spacing s=%d" % (d, s))
                    ok = False
    print("    %s -- volume within tolerance; production identical across every spacing"
          % ("PASS" if ok else "FAIL"))
    return ok


def gate_A(x, o2):
    print("\n  GATE A -- isolated sphere vs the axisymmetric G4 solver")
    print("    hard mask (frac=False) so this tests the SOLVER, not source discretisation")
    s_iso, z_iso = 300, 200.0
    global ZMAX
    z_save, ZMAX = ZMAX, z_iso
    r = np.arange(0.0, G4.RMAX + G4.DX, G4.DX)
    xf, o2f = G4.solve_o2_vessel()
    ok = True
    print("      d    C_peak_G4  C_peak_3D   diff     R0.5 G4/3D    R1 G4/3D     R2 G4/3D")
    for d in (20.0, 40.0, 80.0):
        Cg = G4.solve_no_vessel(xf, o2f, [d], r)[:, 0] * 1e9
        ny, nz = int(round(s_iso / (2.0 * H))), int(round(z_iso / H))
        w = sphere_weights(x, ny, nz, d, frac=False)
        S = source(x, o2, w)
        _, _, full = solve_spectral(x, o2, S, ny, nz, want_full=True)
        axis = full[:, 0, 0] * 1e9                             # along x at y=z~0
        pk_g, pk_3 = float(Cg.max()), float(axis.max())
        cells = []
        for t in (0.5, 1.0, 2.0):
            mg, m3 = Cg >= t, axis >= t
            eg = float(xf[mg].max() - xf[mg].min()) if mg.any() else float("nan")
            e3 = float(x[m3].max() - x[m3].min()) if m3.any() else float("nan")
            cells.append("%5.1f/%5.1f" % (eg, e3))
        dif = 100.0 * (pk_3 - pk_g) / pk_g
        print("    %5.0f %10.3f %10.3f %7.1f%%   %s" % (d, pk_g, pk_3, dif, "  ".join(cells)))
        if abs(dif) > 15.0:
            print("      FAIL: peak differs by >15%%"); ok = False
    ZMAX = z_save
    print("    %s -- residual difference is grid (0.5 vs %.1f um) plus far-field BC"
          % ("PASS" if ok else "FAIL", H))
    return ok


def gate_B():
    print("\n  GATE B -- spectral vs DIRECT 3-D sparse solve, identical discrete problem")
    xs = x_grid(40.0)
    o2 = oxygen(xs)
    ny, nz = 12, 12
    w = sphere_weights(xs, ny, nz, 20.0)
    S = source(xs, o2, w)
    t = time.time(); Cd = solve_direct(xs, o2, S, ny, nz); td = time.time() - t
    t = time.time(); _, _, Cs = solve_spectral(xs, o2, S, ny, nz, want_full=True)
    ts = time.time() - t
    scale = max(Cd.max(), 1e-30)
    rel = float(np.abs(Cs - Cd).max() / scale)
    print("    domain %dx%dx%d   direct %.2fs   spectral %.2fs" % (len(xs), ny, nz, td, ts))
    print("    max |C_spectral - C_direct| / max(C_direct) = %.3e" % rel)
    ok = rel < 1e-9
    print("    %s -- transform, eigenvalues, reconstruction, indexing and normalisation"
          % ("PASS" if ok else "FAIL"))
    return ok


def gate_C(x, o2):
    print("\n  GATE C -- transverse-domain convergence (the z boundary must be inactive)")
    global ZMAX
    z_save = ZMAX
    ok = True
    print("      d     s    ZMAX    trough_nM     peak_nM")
    for d, s in ((40.0, 120), (80.0, 120)):
        vals = []
        for zf in (1.0, 1.5, 2.0):
            ZMAX = z_save * zf
            tr, pk, _, _ = trough_and_smax_probe(x, o2, d, s)
            vals.append((tr, pk))
            print("    %5.0f %5d %7.0f %12.5f %11.4f" % (d, s, ZMAX, tr, pk))
        dt = max(abs(v[0] - vals[0][0]) for v in vals) / max(vals[0][0], 1e-30)
        dp = max(abs(v[1] - vals[0][1]) for v in vals) / max(vals[0][1], 1e-30)
        print("      trough drift %.3f%%   peak drift %.3f%%" % (100 * dt, 100 * dp))
        if dt > 0.01 or dp > 0.01:
            print("      FAIL: z boundary is still active"); ok = False
    ZMAX = z_save
    print("    %s" % ("PASS" if ok else "FAIL"))
    return ok


# --------------------------------------------------------------------------- sweep
def s_max(x, o2, d, thr, cache):
    def tr(s):
        if s not in cache[d]:
            cache[d][s] = trough_and_smax_probe(x, o2, d, s)[:2]
        return cache[d][s][0]

    coarse = list(range(S_MIN, S_MAX + 1, S_COARSE))
    vals = [tr(s) for s in coarse]
    note = "" if bool((np.diff(vals) <= 1e-9).all()) else "NON-MONOTONIC -- inspect"
    if vals[0] < thr:
        return float("nan"), ("fragments even at s_min=%d; " % S_MIN + note).strip("; ")
    if vals[-1] >= thr:
        return float(S_MAX), ("CENSORED at scan ceiling %d; " % S_MAX + note).strip("; ")
    lo = max(s for s, v in zip(coarse, vals) if v >= thr)
    hi = min(s for s, v in zip(coarse, vals) if v < thr)
    while hi - lo > 2:
        mid = int((lo + hi) // 2) & ~1                          # keep s EVEN
        if mid <= lo:
            break
        if tr(mid) >= thr:
            lo = mid
        else:
            hi = mid
    return float(lo), note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--depths", default=None, help="comma-separated depths, um")
    ap.add_argument("--out", default="gate16_spherical_periodic_row",
                    help="output CSV basename, no extension")
    args = ap.parse_args()
    depths = DEPTHS if args.depths is None else [float(v) for v in args.depths.split(",")]

    t0 = time.time()
    print("GATE 16 -- spherical sources, 3-D PERIODIC UNIT CELL, row parallel to the vessel\n")
    print("  reduced domain holds a QUARTER sphere (mirrors at y=0 and z=0; none at x=d)")
    print("  h = %.1f um FIXED; ny = s/(2h) varies with spacing" % H)
    print("  endpoint FROZEN: trough between neighbouring cell SURFACES")
    print("  production from LOCAL O2(x) throughout the sphere, never collapsed to O2(d)")

    x = x_grid()
    o2 = oxygen(x)

    gates = [("SOURCE", gate_source(x, o2)), ("B", gate_B()),
             ("A", gate_A(x, o2)), ("C", gate_C(x, o2))]
    print("\n  PREFLIGHT: " + "  ".join("%s=%s" % (n, "PASS" if v else "FAIL") for n, v in gates))
    if not all(v for _, v in gates):
        print("\n  SWEEP NOT STARTED -- a preflight gate failed.")
        return
    if args.gates_only:
        print("\n  gates only; sweep suppressed by request.")
        return

    print("\n  all gates pass -- starting the sweep\n")
    rows, cache = [], {d: {} for d in depths}
    for thr in THRESHOLDS:
        print("  ===== threshold %.1f nM =====" % thr)
        print("   depth   [O2]    peak_nM    s_max     V(C>=CT)/V")
        for d in depths:
            sm, note = s_max(x, o2, d, thr, cache)
            frac = float("nan")
            if sm == sm and "CENSORED" not in note:
                _, _, full, _ = trough_and_smax_probe(x, o2, d, int(sm), want_full=True)
                frac = float((full * 1e9 >= thr).mean())
            pk = cache[d][S_MIN][1]
            print("   %5.0f %6.2f %9.3f %8s %13s"
                  % (d, o2[int(round(d / H))] * 1e6, pk,
                     "none" if sm != sm else "%.0f" % sm,
                     "  n/a" if frac != frac else "%.4f" % frac))
            if note:
                print("        !! %s" % note)
            rows.append(dict(threshold_nM=thr, depth_um=d,
                             o2_uM=o2[int(round(d / H))] * 1e6, peak_nM=pk, s_max_um=sm,
                             vol_frac_above_CT=frac,
                             censored=int("CENSORED" in note), note=note))
        print()

    print("  THE ANSWER")
    for thr in THRESHOLDS:
        pts = [(r["depth_um"], r["s_max_um"]) for r in rows
               if r["threshold_nM"] == thr and r["s_max_um"] == r["s_max_um"]
               and not r["censored"]]
        if len(pts) < 3:
            print("    %.1f nM: %d uncensored points -- not determinable" % (thr, len(pts)))
            continue
        dm, vm = max(pts, key=lambda z: z[1])
        interior = pts[0][1] < vm > pts[-1][1]
        print("    %.1f nM: s_max peaks at depth %.0f um (%.0f um); %s"
              % (thr, dm, vm, "RISES, PEAKS, FALLS -- tissue translation ESTABLISHED"
                 if interior else
                 "MONOTONIC -- the tissue-scale spacing optimum is FALSIFIED"))

    out = os.path.join(REPO, "data", args.out + ".csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print("\n  %.1f min -> %s" % ((time.time() - t0) / 60, os.path.relpath(out, REPO)))


if __name__ == "__main__":
    main()
