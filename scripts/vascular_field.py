# -*- coding: utf-8 -*-
"""Coupled O2/NO in an idealised vascularised tumour cross-section (Figure 5).

GEOMETRY, frozen. A circular tumour cross-section, effectively infinite along z;
parallel capillaries run along z and appear as circular lumens in the modelled plane; NOS2 is a
uniform volumetric source; the outer circumference keeps the tumour-rim O2 boundary; each vessel
supplies O2 internally AND scavenges NO. Vessel spacing is the only independent variable.

WHY 2-D IS ADMISSIBLE HERE AND NOT ELSEWHERE. Closed question #3 rules that 2-D Cartesian models
cannot give quantitative spacings because disks are infinite rods, a ~2.5x inflation. That ruling
was about CELLS. Capillaries genuinely ARE long cylinders, so a cross-section is exact for them --
but ONLY while NOS2 is a uniform volumetric source. Place discrete NOS2+ cells in this plane and
they become rods and #3 bites. Uniform NOS2 is a GEOMETRIC REQUIREMENT here, not a simplification.

THIS IS A CYLINDER, NOT A SPHERE. Figure 4's spheres and these discs are not quantitatively
comparable and the figure must not imply they are. Comparisons live INSIDE this model; the
zero-vessel disc is the control, and it is not a 2-D restatement of the Figure 4 sphere.

THE LUMEN IS PART OF THE DOMAIN, not a boundary condition. It carries O2 at the wall value and a
first-order NO sink k_lumen with no source -- exactly the structure of the closed-form solution in
gate_perivascular_analytic.py (region I). This avoids hand-rolling a Robin condition, and it means
the single-vessel limit can be checked against that closed form directly.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, cg, LinearOperator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import spheroid_field as SF        # noqa: E402  the 1-D model; constants come through it
import parameters as P3           # noqa: E402  NOS3 block
import constants as C             # noqa: E402

# --- everything physical is taken from the 1-D model so the two cannot drift apart -------------
D_NO_UM, D_O2_UM = SF.D_NO_UM, SF.D_O2_UM          # um^2/s
V_M_UM, KM_RESP_UM, K_I_NM = SF.V_M_UM, SF.KM_RESP_UM, SF.K_I_NM
KM_UM, VMAX_NM, PHI = SF.KM_UM, SF.VMAX_NM, SF.PHI
O2_OPT_UM = (KM_UM * SF.C.O2_STAR_UM) ** 0.5       # 32.30, Figure 1's closed form

K_LUMEN_DEFAULT = 2.0e3        # 1/s, mid of the sourced 1.3-3.5e3 window


def lumen_impedance(k_lumen, a_um=None):
    """NO flux the lumen absorbs per unit wall area, per unit wall concentration [um/s].

    THE VESSEL IS A BOUNDARY IMPEDANCE, NOT A RESOLVABLE REGION. At physiological luminal
    uptake the lumen's own diffusion length is sqrt(D/k_lumen) ~ 1.05 um, so NO entering blood is
    consumed within about a micron. Meshing through that needs dx << 1 um, which over an 850 um
    tumour is unaffordable AND pointless -- the lumen interior is never plotted, and its only
    effect on the tissue is the flux it removes.

    The closed form gives the impedance exactly. In the lumen n = A*I0(r/L_l), so
        -D dn/dr|_a  =  -D * A * I1(a/L_l)/L_l   and   n(a) = A*I0(a/L_l)
        => h = D * I1(a/L_l) / (L_l * I0(a/L_l))
    h -> D/L_l as a/L_l grows, i.e. a wide vessel looks like a half-space of blood.
    """
    from scipy.special import i0, i1
    a = VESSEL_RADIUS_UM if a_um is None else a_um
    L_l = np.sqrt(D_NO_UM / k_lumen)
    return D_NO_UM * i1(a / L_l) / (L_l * i0(a / L_l))
VESSEL_RADIUS_UM = 5.0
O2_RIM_UM = 56.0
O2_WALL_UM = 56.0

# --- ENDOTHELIAL NOS3, the vessel's SOURCE arm -----------------------------------------------
# The vessel is a three-function object: it delivers O2, it PRODUCES NO through eNOS, and it
# removes NO through the lumen. Modelling only delivery and removal was a STRUCTURAL omission,
# not a sensitivity issue: NOS2 output scales with tissue area per vessel (~spacing^-2) while
# eNOS output scales with vessel LENGTH, so the omitted source grows in relative importance
# exactly as vessels crowd -- ~2% of NOS2 at 400 um spacing but ~85% at 60 um.
# And the isoforms differ in oxygen affinity: K_m 4 uM for NOS3 against 42 uM for NOS2, so
# eNOS stays near-saturated in hypoxia where NOS2 shuts down.
_V3 = P3.VASCULAR_ENDOTHELIUM
VMAX3_NM = _V3['Vmax'] * 1e9        # nM/s -- an EFFECTIVE amplitude, not a biochemical Vmax
KM3_UM = _V3['Km'] * 1e6            # uM, 4.0 ( selection rule)
SHELL_UM = C.J_NOS3_SHELL_UM        # 0.8 um endothelial source shell

# --- provenance -------------------------------------------------------------------------------
# AN ARTEFACT MUST NAME THE MODEL THAT MADE IT. A figure or CSV produced with
# enos=False and one produced with enos=True have to be distinguishable FROM DISK ALONE, without
# knowing what time they were written or which edit of this file was live. Anything whose eNOS
# state cannot be reconstructed is PROVENANCE-AMBIGUOUS: keep it, but it does not enter the
# manuscript.
_PARAM_SOURCES = ('src/constants.py', 'src/parameters.py', 'scripts/spheroid_field.py',
                  'scripts/vascular_field.py')


def parameter_set_hash():
    """Short digest over every file that can change a number in this model."""
    import hashlib
    h = hashlib.sha256()
    for rel in _PARAM_SOURCES:
        f = os.path.join(ROOT, rel)
        h.update(rel.encode('utf-8'))
        with open(f, 'rb') as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


def provenance(*, enos, k_lumen, f_nos2, dx, R_um, vessels=(), spacing_um=None,
               a_um=VESSEL_RADIUS_UM, **extra):
    """Everything needed to reconstruct a run. Write it as CSV columns or a JSON sidecar.

    enos, k_lumen and f_nos2 are REQUIRED keywords here for the same reason enos is required in
    solve(): each selects biology, and a provenance record that can omit them defeats its purpose.
    """
    rec = {'enos': bool(enos), 'k_lumen_per_s': float(k_lumen), 'f_nos2': float(f_nos2),
           'dx_um': float(dx), 'R_um': float(R_um), 'n_vessels': len(vessels),
           'vessel_radius_um': float(a_um),
           'spacing_um': None if spacing_um is None else float(spacing_um),
           'shell_um': float(SHELL_UM), 'vmax3_nM_per_s': float(VMAX3_NM), 'km3_uM': float(KM3_UM),
           'param_set': parameter_set_hash()}
    rec.update(extra)
    return rec


def enos_area_per_vessel(a_um):
    """Cross-sectional area of the endothelial source annulus, um^2 per um of vessel length.

    The 0.8 um shell is FAR below the 2 um production grid, so it must never be rasterised
    directly -- a sub-grid shell painted onto whole cells makes the source grid-dependent. The
    integrated production per unit vessel length is preserved instead:
        Q/L = S(O2) * pi * [(a + w)^2 - a^2]
    and deposited across the ring of tissue cells touching the lumen, area-weighted.
    """
    return np.pi * ((a_um + SHELL_UM) ** 2 - a_um ** 2)


def kdeg_tissue(o2_um):
    return SF.kdeg_u(o2_um)


def prod_tissue(o2_um, f_nos2):
    return SF.prod_u(o2_um, f_nos2)


def build_grid(R_um, dx):
    n = int(np.ceil(2 * R_um / dx)) + 1
    ax = (np.arange(n) - (n - 1) / 2.0) * dx
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    return X, Y, ax


def masks(X, Y, R_um, vessels, a_um):
    """inside: tissue+lumen; lumen: inside any vessel."""
    rr = np.hypot(X, Y)
    inside = rr <= R_um
    lumen = np.zeros_like(inside)
    for (vx, vy) in vessels:
        lumen |= (np.hypot(X - vx, Y - vy) <= a_um)
    lumen &= inside
    return inside, lumen


def _solve_linear(inside, dirichlet, dval, coefA, src, dx, robin=None, check_sym=False):
    """Solve  lap(u) - coefA*u + src = 0  on `inside`. Zero flux where the domain ends.

    Eliminating them instead moves each known value into the right-hand side of its neighbours'
    equations, leaving only free nodes and preserving the symmetry of the underlying operator. The
    system also gets smaller. `check_sym=True` verifies ||A - A^T|| / ||A|| before trusting CG.
    """
    ny, nx = inside.shape
    free = inside & ~dirichlet
    idx = -np.ones(inside.shape, dtype=np.int64)
    pts = np.argwhere(free)
    for k, (i, j) in enumerate(pts):
        idx[i, j] = k
    N = len(pts)
    if N == 0:
        return np.where(dirichlet, dval, 0.0)
    A = lil_matrix((N, N))
    b = np.zeros(N)
    inv2 = 1.0 / (dx * dx)
    for k, (i, j) in enumerate(pts):
        diag = -coefA[i, j]
        rhs = -src[i, j]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if not (0 <= ii < ny and 0 <= jj < nx) or not inside[ii, jj]:
                continue                      # zero flux: mirror, contributes nothing
            diag -= inv2
            if dirichlet[ii, jj]:
                rhs -= inv2 * dval[ii, jj]    # known value -> right-hand side
            else:
                A[k, idx[ii, jj]] += inv2
        if robin is not None:
            diag -= robin[i, j]
        A[k, k] += diag
        b[k] = rhs
    Acsr = csr_matrix(A)
    if check_sym:
        asym = abs(Acsr - Acsr.T).max() / abs(Acsr).max()
        print('    symmetry ||A - A^T|| / ||A|| = %.3e  (%s)'
              % (asym, 'symmetric' if asym < 1e-12 else 'NOT SYMMETRIC'))
    if N <= 120000:
        u = spsolve(Acsr, b)
    else:
        d = Acsr.diagonal().copy()
        d[d == 0] = 1.0
        M_ = LinearOperator((N, N), matvec=lambda v: v / d)
        u, info = cg(Acsr, b, rtol=1e-10, atol=0.0, maxiter=100000, M=M_)
        if info != 0:
            raise RuntimeError('CG did not converge (info=%d, N=%d)' % (info, N))
    out = np.where(dirichlet, dval, 0.0)
    for k, (i, j) in enumerate(pts):
        out[i, j] = u[k]
    return out


def solve(R_um, vessels=(), f_nos2=0.15, dx=2.0, a_um=VESSEL_RADIUS_UM,
          k_lumen=K_LUMEN_DEFAULT, o2_rim=O2_RIM_UM, o2_wall=O2_WALL_UM,
          respiratory_inhibition=True, iters=40, tol=1e-4, uniform_o2=None,
          wall_impedance=False, o2_const_Q=None, *, enos):
    """Returns (X, Y, o2_uM, no_nM, inside, lumen).

    uniform_o2: if given, the O2 field is held at that value everywhere (the analytic gate's case).

    enos IS KEYWORD-ONLY AND HAS NO DEFAULT, DELIBERATELY. It selects which
    biology is being solved -- tumour NOS2 alone, or NOS2 plus the endothelial NOS3 wall source --
    so it is a PARAMETER, not a convenience option. There is no scientifically meaningful default.

    THIS COST A DAY. It was briefly `enos=True`. Every caller that omitted the argument silently
    changed model the moment eNOS was added to this module, and nothing on disk recorded which
    branch had run. A k_lumen sweep ended up with one row solved WITH eNOS and the rest without,
    which read as a stronger sink producing MORE NO (1.593 -> 1.939 nM) and cost a full
    convergence, path-dependence and multiplicity investigation before the cause was found. The
    solver was never at fault. Passing this explicitly at every call site is what makes an
    artefact reconstructable; see provenance().
    """
    X, Y, _ = build_grid(R_um, dx)
    inside_all, lumen = masks(X, Y, R_um, vessels, a_um)
    rr = np.hypot(X, Y)
    # the lumen is EXCLUDED from the NO domain and re-enters as a wall impedance
    # RESOLVING THE LUMEN BEATS THE IMPEDANCE, MEASURED. Against the closed form, in tissue:
    #        resolved lumen   0.56% (dx 1.0)   0.21% (dx 0.5)
    #        wall impedance   4.06% (dx 1.0)   2.49% (dx 0.5)
    #   The impedance was introduced to escape the lumen's ~1 um decay length, but a Robin condition
    #   on a CURVED boundary in Cartesian cells is first-order at best; rescaling the staircase
    #   perimeter (4/pi = 1.27x too long) fixed the leading geometric error and it is still ~10x
    #   worse at equal cost. KEPT ONLY AS A DIAGNOSTIC. The real constraint stands and sets the
    #   scale of this figure: the model needs dx ~ 1 um near vessels.
    inside = (inside_all & ~lumen) if wall_impedance else inside_all
    h = lumen_impedance(k_lumen, a_um) if wall_impedance else 0.0
    faces = np.zeros(inside.shape)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = np.roll(np.roll(lumen, di, axis=0), dj, axis=1)
        faces += (inside & nb).astype(float)
    # STAIRCASE PERIMETER CORRECTION. A circle rendered on a Cartesian grid has a staircased
    # boundary whose length is ~4/pi = 1.27x the true circumference, and that error does NOT shrink
    # with dx. Applying the wall impedance per counted face therefore absorbs ~27% too much NO at
    # every grid spacing: the first version of this did exactly that and the gate went from 0.56%
    # (resolved lumen) to 8.7%, barely improving from dx = 2 to dx = 1 -- the signature of a
    # geometric, not a resolution, error. Rescale the counted faces to the TRUE wall length.
    true_perimeter = 2.0 * np.pi * a_um * max(len(vessels), 1)
    staircase = faces.sum() * dx
    scale = (true_perimeter / staircase) if staircase > 0 else 1.0
    robin = (faces * scale * h / (D_NO_UM * dx)) if wall_impedance else None
    # the ring of TISSUE cells touching a lumen: where the endothelial source is deposited
    ring = np.zeros_like(inside_all)
    for _di, _dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ring |= np.roll(np.roll(lumen, _di, axis=0), _dj, axis=1)
    ring &= inside_all & ~lumen
    # THE RIM IS THE OUTERMOST LAYER OF CELLS, NOT A BAND. It was `rr > R - 1.5*dx`, a
    # shell up to 1.5 cells thick, placing the effective tumour surface that far INSIDE the
    # true circle. Gate O1 caught it: error first-order in dx, INDEPENDENT of source
    # strength, and LARGEST FAR FROM THE VESSEL -- the signature of a misplaced outer
    # boundary, not a staircased wall. The NO gate could not see it: NO has ZERO FLUX at
    # the rim and is insensitive to where that rim sits; oxygen is Dirichlet there and is not.
    _nb_out = np.zeros_like(inside_all)
    for _di, _dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _nb_out |= ~np.roll(np.roll(inside_all, _di, axis=0), _dj, axis=1)
    rim = inside & _nb_out

    o2 = np.where(inside, o2_rim, 0.0)
    if uniform_o2 is not None:
        o2 = np.where(inside, uniform_o2, 0.0)
    no = np.zeros_like(o2)

    o2_dir = (inside_all & _nb_out) | lumen
    o2_dval = np.where(lumen, o2_wall, o2_rim)

    for _ in range(iters):
        if uniform_o2 is None:
            km_eff = KM_RESP_UM * (1.0 + np.clip(no, 0, None) / K_I_NM) \
                if respiratory_inhibition else KM_RESP_UM * np.ones_like(no)
            # Picard on the Michaelis-Menten sink: R = V_M o2/(o2+km) linearised as (V_M/(o2+km))*o2
            if o2_const_Q is not None:
                # MANUFACTURED CASE for gate O1: constant volumetric consumption has a closed
                # form in this geometry. No clipping -- the test solution may go negative and
                # clipping would hide a sign or boundary error, which is what the gate is for.
                coefA = np.zeros_like(o2)
                src = np.where(inside_all & ~lumen, -o2_const_Q / D_O2_UM, 0.0)
                o2 = _solve_linear(inside_all, o2_dir, o2_dval, coefA, src, dx)
            else:
                coefA = np.where(inside & ~lumen,
                                 V_M_UM / (np.clip(o2, 0, None) + km_eff), 0.0) / D_O2_UM
                o2_new = _solve_linear(inside_all, o2_dir, o2_dval, coefA,
                                       np.zeros_like(o2), dx)
                o2 = np.clip(o2_new, 0.0, None)

        kk = np.where(lumen, k_lumen, kdeg_tissue(np.clip(o2, 0, None)))
        ss = np.where(lumen, 0.0, prod_tissue(np.clip(o2, 0, None), f_nos2))
        if enos and len(vessels):
            # eNOS is evaluated at ENDOTHELIAL oxygen -- the ring cells sit against the wall, so
            # they see near-luminal O2, not bulk tumour O2. With K_m 4 uM that difference is the
            # biology: the endothelium stays active where NOS2 does not.
            o2e = np.clip(o2, 0.0, None)
            f3 = o2e / (o2e + KM3_UM)
            n_ring = max(int(ring.sum()), 1)
            # total shell area for ALL vessels, spread over the ring cells, area-weighted
            shell_area = enos_area_per_vessel(a_um) * len(vessels)
            ss = ss + np.where(ring, VMAX3_NM * f3 * (shell_area / n_ring) / (dx * dx), 0.0)
        no_new = _solve_linear(inside, np.zeros_like(inside, dtype=bool),
                               np.zeros_like(o2), kk / D_NO_UM, ss / D_NO_UM, dx,
                               robin=robin)
        no_new = np.clip(no_new, 0.0, None)
        dev = np.max(np.abs(no_new - no))
        no = no_new
        if uniform_o2 is not None or dev <= tol * max(np.max(no), 1e-12):
            break
    return X, Y, o2, no, inside, lumen


def gate_single_vessel(dx=1.0, R_um=150.0, o2_uniform=56.0, k_lumen=K_LUMEN_DEFAULT,
                       f_nos2=0.15, wall_impedance=False, *, enos=False):
    """THE GATE. One central vessel, UNIFORM tissue oxygen -> the problem is piecewise
    modified-Bessel and has a closed form. Compare the 2-D solve against it.

    At uniform o2 the tissue is:   D n'' + D n'/r - k n + S = 0,  k, S constant
    lumen:                          D n'' + D n'/r - k_l n = 0
    with continuity of n and flux at r = a, and zero flux at r = R.
    """
    from scipy.special import i0, k0, i1, k1
    k_t = kdeg_tissue(o2_uniform)
    S = prod_tissue(o2_uniform, f_nos2)
    a = VESSEL_RADIUS_UM
    L_t = np.sqrt(D_NO_UM / k_t)
    L_l = np.sqrt(D_NO_UM / k_lumen)
    # lumen:  A I0(r/L_l);  tissue: B I0(r/L_t) + C K0(r/L_t) + S/k_t
    # unknowns A, B, C from: continuity at a, flux at a, zero flux at R
    M = np.array([
        [i0(a / L_l), -i0(a / L_t), -k0(a / L_t)],
        [i1(a / L_l) / L_l, -i1(a / L_t) / L_t, k1(a / L_t) / L_t],
        [0.0, i1(R_um / L_t) / L_t, -k1(R_um / L_t) / L_t]])
    rhs = np.array([S / k_t, 0.0, 0.0])
    A, B, Cc = np.linalg.solve(M, rhs)

    def analytic(r):
        r = np.atleast_1d(r).astype(float)
        out = np.where(r <= a, A * i0(np.clip(r, 1e-9, None) / L_l),
                       B * i0(np.clip(r, a, None) / L_t)
                       + Cc * k0(np.clip(r, a, None) / L_t) + S / k_t)
        return out

    X, Y, o2, no, inside, lumen = solve(R_um, vessels=((0.0, 0.0),), f_nos2=f_nos2, dx=dx,
                                        k_lumen=k_lumen, uniform_o2=o2_uniform,
                                        wall_impedance=wall_impedance, enos=enos)
    rr = np.hypot(X, Y)
    # tissue only: the lumen is no longer part of the NO domain, by design
    sel = inside & ~lumen & (rr >= VESSEL_RADIUS_UM + dx) & (rr <= R_um - 3 * dx)
    num, ana = no[sel], analytic(rr[sel])
    rel = np.abs(num - ana) / np.max(ana)
    print('GATE - single vessel against the closed form (uniform [O2] = %.0f uM)' % o2_uniform)
    print('  k_lumen %.3g /s   L_tissue %.1f um   L_lumen %.2f um' % (k_lumen, L_t, L_l))
    print('  tissue plateau S/k = %.3f nM   wall (r=a) analytic %.3f nM'
          % (S / k_t, float(analytic(a)[0])))
    print('  max |num - analytic| = %.4f nM   (%.3f%% of peak)'
          % (np.max(np.abs(num - ana)), 100 * rel.max()))
    print('  mean abs error       = %.4f nM' % np.mean(np.abs(num - ana)))
    ok = rel.max() < 0.01
    print('  %s' % ('PASS (<1% in tissue)' if ok else
                    'FAIL - do not trust any multi-vessel result'))
    return ok



def gate_o2_analytic(dx=2.0, R_um=150.0, Q=1.0, o2_wall=56.0, o2_rim=56.0, *, enos=False):
    """O1 - the OXYGEN operator against a closed form, in the production geometry.

    THE NO GATE CERTIFIES NOTHING ABOUT OXYGEN. It is the constant-k, uniform-[O2] case, so it
    exercises the NO operator and the vessel treatment only. Michaelis-Menten respiration has no
    useful closed form, so the sink is temporarily replaced by a CONSTANT volumetric consumption Q,
    which does:

        D (c'' + c'/r) = Q ,   c(a) = o2_wall ,   c'(R) = 0
        =>  c(r) = Q r^2/(4D) - (Q R^2 / 2D) ln r + B

    This tests the same Laplacian, the same staircased circular wall, the same outer boundary and
    the same flux directions that production uses. Expect the same non-monotone geometric jitter
    with dx as the NO gate showed: the discrete circle CHANGES with the grid, so the standard is
    bounded error, not monotone convergence.
    """
    a = VESSEL_RADIUS_UM
    A = ((o2_rim - o2_wall) - Q * (R_um ** 2 - a ** 2) / (4 * D_O2_UM)) / np.log(R_um / a)
    B = o2_wall - Q * a ** 2 / (4 * D_O2_UM) - A * np.log(a)

    def analytic(r):
        r = np.clip(np.atleast_1d(r).astype(float), VESSEL_RADIUS_UM, None)
        return Q * r ** 2 / (4 * D_O2_UM) + A * np.log(r) + B

    X, Y, o2, no, inside, lumen = solve(R_um, vessels=((0.0, 0.0),), dx=dx,
                                        o2_const_Q=Q, o2_wall=o2_wall,
                                        o2_rim=o2_rim, iters=1, enos=enos)
    rr = np.hypot(X, Y)
    sel = inside & ~lumen & (rr >= VESSEL_RADIUS_UM + 2 * dx) & (rr <= R_um - 3 * dx)
    num, ana = o2[sel], analytic(rr[sel])
    span = ana.max() - ana.min()
    rel = np.max(np.abs(num - ana)) / span
    print('O1 - oxygen operator against the closed form (constant Q = %.3g uM/s)' % Q)
    print('  analytic spans %.3f to %.3f uM across the tissue' % (ana.min(), ana.max()))
    print('  max |num - analytic| = %.4f uM   (%.3f%% of the span)'
          % (np.max(np.abs(num - ana)), 100 * rel))
    ok = rel < 0.01
    print('  %s' % ('PASS (<1%)' if ok else 'FAIL'))
    return ok


def gate_conservation(R_um=425.0, vessels=((0.0, 0.0),), f_nos2=0.15, dx=2.0, *, enos):
    # NO DEFAULT. O3 integrates respiration through km_eff, which depends on [NO], so the eNOS
    # state changes what this gate is balancing. It must be told which model it is certifying.
    """O3 - global oxygen balance, as a genuine BOUNDARY-FLUX check.

    The real invariant is a boundary statement: oxygen entering through the Dirichlet surfaces --
    vessel walls and tumour rim -- must equal what the tissue consumes.

        sum over (interior node i, Dirichlet neighbour d) of D (c_d - c_i)  =  integral of R dA

    That is what the discretisation actually conserves, so it should hold to solver tolerance
    regardless of how faithfully the staircased circle approximates a circle.
    """
    X, Y, o2, no, inside, lumen = solve(R_um, vessels=vessels, f_nos2=f_nos2, dx=dx, enos=enos)
    rr = np.hypot(X, Y)
    _nbo = np.zeros_like(inside)
    for _di, _dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _nbo |= ~np.roll(np.roll(inside, _di, axis=0), _dj, axis=1)
    rim = inside & _nbo          # must MATCH solve()'s rim, or O3 measures the wrong faces
    dirich = rim | lumen
    interior = inside & ~dirich

    km_eff = KM_RESP_UM * (1.0 + np.clip(no, 0, None) / K_I_NM)
    consumed = (np.where(interior, V_M_UM * o2 / (o2 + km_eff), 0.0)).sum() * dx * dx

    supplied = 0.0
    ny, nx = o2.shape
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb_val = np.roll(np.roll(o2, -di, axis=0), -dj, axis=1)
        nb_is_d = np.roll(np.roll(dirich, -di, axis=0), -dj, axis=1)
        edge = np.zeros_like(interior)
        edge[max(0, di):ny + min(0, di), max(0, dj):nx + min(0, dj)] = True
        take = interior & nb_is_d & edge          # edge mask: np.roll WRAPS, this stops it
        supplied += (D_O2_UM * (nb_val - o2) * take).sum()

    err = abs(supplied - consumed) / max(consumed, 1e-30)
    print('O3 - oxygen conservation (boundary flux vs integrated respiration) - eNOS %s'
          % ('ON (production)' if enos else 'OFF (ablation)'))
    print('  consumed by respiration      %.6g uM um^2/s' % consumed)
    print('  supplied through boundaries  %.6g' % supplied)
    print('  imbalance %.4f%%' % (100 * err))
    ok = err < 0.01
    print('  %s' % ('PASS (<1%)' if ok else 'FAIL'))
    return ok


if __name__ == '__main__':
    # EVERY GATE DECLARES ITS MODEL. Two branches now exist and they are not interchangeable:
    #   ablation   NOS2 only                    enos=False
    #   PRODUCTION NOS2 + endothelial NOS3      enos=True
    # Only the production branch licenses a Figure 5 result; the ablation branch is retained
    # because the mechanism decomposition rests on it.
    print('parameter set %s' % parameter_set_hash())
    print()
    # certifies the NO operator + vessel treatment; its closed form has no shell term, so this
    # gate is meaningful ONLY on the ablation branch
    ok = gate_single_vessel(enos=False)
    print()
    ok &= gate_o2_analytic(enos=False)
    print()
    ok &= gate_conservation(enos=False)
    print()
    ok &= gate_conservation(enos=True)
    sys.exit(0 if ok else 1)
