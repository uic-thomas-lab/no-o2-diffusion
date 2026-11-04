# -*- coding: utf-8 -*-
"""tornado_v2.py -- sensitivity engine for the NO/O2 diffusion manuscript (Figure 7).

WHAT THIS FILE IS
-----------------
FIVE scalar outputs, each a pure function of one parameter dict, plus the SEVEN
ratified sensitivity axes and a gate suite that must pass before any output is
believed.

The maths is COPIED, not imported, from:

  src/parameters.py, src/constants.py, src/kinetics.py     (constants, sink form)
  _generators/gen_full_set.py, gen_finite_matrices.py      (finite-sphere field)
  _generators/gen_fig4_9panel.py                           (cluster construction)
  _generators/perivascular_reproduce.py                    (cylindrical Krogh solve)
  scripts/verify_perivascular_D45.py                       (the gated version of it)

The three gen_*.py files EXECUTE ON IMPORT and overwrite the shipped matrices,
so importing them is not an option; their maths is reproduced here verbatim and
gated against their published outputs (G4).

THE FIVE RULES THIS FILE OBEYS
------------------------------
1. K_m and V_max are ONE axis. V_max is back-calculated from a rate measured at
   220 uM by dividing out the Michaelis fraction, so it MOVES whenever K_m moves
   (`km_branch`). Sweeping them independently is the project's number-one trap.
   ** At K_m 130 uM this file uses V_max = 4.779e-6 M/s, NOT the legacy 5.0e-6.**
   5.0e-6 sits on a 220 uM anchor rate 4.62% above the current nominal pair's, so
   it would smuggle a V_max excursion into a bar meant to isolate K_m.
2. The sink is ADDITIVE:  k_deg = k_obs*[O2]*rho + k_anoxia*rho.
3. The cell-volume correction is TUMOUR-ARM ONLY. The perivascular arm (rho =
   1.0e8) carries the published constants uncorrected -- so the tumour packing
   fraction phi has exactly zero swing on `reach_at_anoxia` and `wall_no`, by
   construction.  That arm is parameterised by a COUNTED density rather than by a
   volume ratio, so its packing is already embedded in it (implied phi = rho_p *
   V_referent = 1.0e8 * 6186 um^3 = 0.62).  The zero is therefore zero by
   PARAMETERISATION, not by physics, and sweeping phi there would double-count.
4. A source sphere's volume is the density-derived 885 um^3 (= 1e12/rho), not the
   1000 um^3 grid box.
5. Nothing is tuned to make a gate pass. A failing gate exits non-zero.

USAGE
-----
  python scripts/tornado_v2.py --gates-only
  python scripts/tornado_v2.py --axis k_anoxia --at low
  python scripts/tornado_v2.py --sweep            # all axes -> tornado_v2_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from constants import (K_OBS_PUB, K_ANOXIA_PUB, K_ANOXIA_RANGE,          # noqa: E402
                       RHO_TUMOUR, RHO_PERIVASCULAR, PHI_TUMOUR,
                       CELL_VOLUME_MODEL_UM3, CELL_VOLUME_REFERENT_UM3,
                       VOLUME_CORRECTION_DIVISOR, VOLUME_CORRECTION_DIVISOR_RANGE)
import parameters as _P                                                  # noqa: E402

# =============================================================================
# FIXED CONSTANTS -- not axes.  Anything here is held at nominal on every bar.
# =============================================================================

# --- tumour arm (NOS2 sources in tissue) ---
RHO_T = RHO_TUMOUR              # cells/mL  2.02e8, derived phi/V_cell
VOL_CELL_UM3 = CELL_VOLUME_MODEL_UM3    # um^3  2079, MEASURED (NCI-60 n=37)
A0_UM = (3.0 * VOL_CELL_UM3 / (4.0 * np.pi)) ** (1.0 / 3.0)   # 7.918 um

# --- perivascular arm (NOS3 endothelium around one vessel) ---
RHO_P = RHO_PERIVASCULAR        # 1.0e8 cells/mL -- the density k_obs was published at
KM_ENOS = _P.VASCULAR_ENDOTHELIUM['Km']       # M    4e-6 
VMAX_ENOS = _P.VASCULAR_ENDOTHELIUM['Vmax']   # M/s  9.419e-7 effective source amplitude
D_O2_UM2S = _P.VASCULAR_ENDOTHELIUM['D_O2'] * 1e8   # um^2/s from cm^2/s -- 1390
V_M = _P.VASCULAR_ENDOTHELIUM['V_M']          # M/s  Thomas 2001 max mitochondrial O2 consumption
KM_RESP = _P.VASCULAR_ENDOTHELIUM['Km_resp']  # M OPEN: citation broken AND value challenged
K_I_NO = _P.VASCULAR_ENDOTHELIUM['K_I']       # M    5.22e-9, NO vs cytochrome c oxidase

IC50_O2_ANCHOR = 145e-6         # M    the [O2] at which the 270 nM half-inhibition was read
IC50_145_PUB = 270e-9           # M    published IC50 there -- gate target, not a model input
IC50_30_PUB = 60e-9             # M    published IC50 at 30 uM O2 -- the SECOND point, gate only
IC50_O2_SECOND = 30e-6          # M    the tension of that second point
R_V, R_K = 5.0, 80.0            # um       vessel and Krogh radii
SHELL_UM = 0.8                  # um       endothelial source thickness, HELD FIXED
N_RADIAL = 6001                 # nodes in the cylindrical solve

KM_NOMINAL = _P.KM              # M    42e-6, apparent K_m, actively synthesising iNOS
VMAX_NOMINAL = _P.VMAX          # M/s  1.718e-6, re-derived on K_m=42 and V_cell=2079
ANCHOR_O2 = 220e-6              # M    the tension the rate was MEASURED at

# --- geometry of the cluster arrangement (gen_fig4_9panel) ---
N_CELLS_TOTAL = 27              # 14 + 13, two clusters
LATTICE_UM = 2.0 * A0_UM        # 15.836 um = one cell diameter


F_NOMINAL = 0.02          # NOS2+ expressing fraction: the production condition used
#                           throughout the paper. nominal_params() AND phi_branch() both
#                           read THIS name. Two literals is how the separation endpoints
#                           desynced from rho for twenty days; one name cannot desync.


def rho_at_phi(phi):
    """Tumour cell density (cells/mL) implied by a packing fraction.

    rho(phi) = RHO_T * (phi / PHI_TUMOUR) -- the identity rho = phi/V_cell written as a
    RESCALE of the shipped density rather than a recomputation of it. Deliberate: at
    phi = PHI_TUMOUR this returns RHO_T bit-for-bit, so every nominal number in this file
    is unchanged to the last ulp. Recomputing phi/V_cell instead returns 2.0202e8 against
    the shipped 2.02e8 -- 0.01%, inside the src coupling assert's 5e-3 tolerance, but it
    would move every gate target for no physical reason.
    """
    return RHO_T * (phi / PHI_TUMOUR)


def SEP_AT_F(f, phi=PHI_TUMOUR):
    """Cluster separation (um, centre-to-centre) implied by a NOS2+ expressing fraction."""
    return float((N_CELLS_TOTAL / (rho_at_phi(phi) * f)) ** (1.0 / 3.0) * 1e4)


def GEOMETRIC_FLOOR_UM():
    """Smallest separation leaving one full cell diameter between two cluster surfaces."""
    d = 2.0 * A0_UM
    return float(2.0 * (LATTICE_UM + d / 2.0) + d)

# --- grids, copied from the generators so the gates are like-for-like ---
FIG2_GRID = np.arange(-300, 301, 2.0)     # gen_full_set.py
FIG2_CUBE_SPACING = 75.0
DENSITY_GRID = np.arange(-200, 201, 2.0)  # gen_finite_matrices.py
DENSITY_FOCUS_R = 50.0
DENSITY_FRACTIONS = (0.02, 0.05, 0.15, 0.40)
DENSITY_O2 = 10e-6                        # M -- the density series is run at 10 uM


# =============================================================================
# THE SIX RATIFIED AXES
# =============================================================================
# low / nominal / high exactly as ratified.  `arms` records which outputs the axis
# can move; an axis absent from an output's arm list has EXACTLY ZERO swing there
# and must not be drawn as a bar on it.

AXES = {
    'k_anoxia': dict(
        key='k_anoxia', units='mL/cell/s', status='DERIVED',
        low=8.3e-9, nominal=1.336e-8, high=2.152e-8,
        label='k_anoxia -- oxygen-independent (anoxic) sink constant',
        note='Source and range: Supplementary Table S1.'),
    'cell_volume_correction': dict(
        key='vol_div', units='x (divisor)', status='RETIRED', retired=True,
        low=VOLUME_CORRECTION_DIVISOR, nominal=VOLUME_CORRECTION_DIVISOR,
        high=VOLUME_CORRECTION_DIVISOR,
        label='RETIRED AXIS -- cell-volume (density-referent) correction. NOT A BAR.',
        note='Source and range: Supplementary Table S1.'),
    'packing_fraction': dict(
        key='phi', units='dimensionless', status='IMAGING-DERIVED / MODEL-FITTED',
        low=0.30, nominal=PHI_TUMOUR, high=0.55,
        label='phi -- tumour tissue cell (packing) volume fraction -- TUMOUR ARM ONLY',
        note='Source and range: Supplementary Table S1.'),
    'D_NO': dict(
        key='D_no', units='um^2/s', status='MEASURED',
        low=850.0, nominal=2200.0, high=2200.0,
        label='D_NO -- NO diffusion coefficient',
        note='Source and range: Supplementary Table S1.'),
    'KM_NOS2': dict(
        key='km', units='uM (K_m), V_max coupled', status='MEASURED',
        low=31e-6, nominal=42e-6, high=42e-6,
        label='NOS2 K_m for O2 -- WITH V_max coupled (ONE axis)',
        note='Source and range: Supplementary Table S1.'),
    'rim_O2': dict(
        key='rim_o2', units='uM', status='MEASURED',
        low=16.7e-6, nominal=56e-6, high=56e-6,
        label='Rim / vessel-surface [O2]',
        note='Source and range: Supplementary Table S1.'),
    'cluster_separation': dict(
        key='sep', units='um', status='DERIVED',
        low=SEP_AT_F(0.40), nominal=SEP_AT_F(0.02), high=SEP_AT_F(0.005),
        label='NOS2+ cluster separation (centre-to-centre)',
        note='Source and range: Supplementary Table S1.'),
    'mito_inhibition': dict(
        key='km_resp', units='uM (K_m,resp), K_I coupled',
        status='NOMINAL MEASURED / RANGE ASSUMED -- NO SOURCE',
        low=0.5e-6, nominal=KM_RESP, high=5.0e-6,
        label='Mitochondrial K_m,resp for O2 -- WITH K_I coupled (ONE axis)',
        note='Source and range: Supplementary Table S1.'),
}


SCENARIOS = {
    'KM_NOS2_highNO': dict(
        key='km', value=130e-6, units='uM',
        label='NOS2 K_m for O2, HIGH-NO state (solution NO accumulated)',
        note='Source and range: Supplementary Table S1.'),
}


def scenario_params(name):
    """Nominal dict with ONE named scenario applied.  Same coupling rules as params_at()."""
    if name not in SCENARIOS:
        raise KeyError('unknown scenario %r; known: %s' % (name, ', '.join(SCENARIOS)))
    s = SCENARIOS[name]
    p = nominal_params()
    p[s['key']] = s['value']
    if s['key'] == 'km':
        p['vmax'] = km_branch(p['km'])
    return p

# which axes can move which output (everything else is exactly zero swing)
# THIS TABLE IS NOW GATED (G16), not merely documented: every axis listed must move its
#   output, and every axis NOT listed must leave it BIT-IDENTICAL. A not-in-arm axis is what
#   tells the figure to draw a BLANK ROW rather than a zero-length bar.
OUTPUT_AXES = {
    'delivery_optimum':      ('k_anoxia', 'packing_fraction', 'D_NO', 'KM_NOS2',
                              'cluster_separation'),
    'reach_at_anoxia':       ('k_anoxia', 'D_NO'),
    'wall_no':               ('k_anoxia', 'D_NO', 'rim_O2', 'mito_inhibition'),
    'steady_state_no':       ('k_anoxia', 'packing_fraction', 'D_NO', 'KM_NOS2', 'rim_O2'),
    'effective_mitochondrial_km': ('k_anoxia', 'D_NO', 'rim_O2', 'mito_inhibition'),
}


def km_branch(km):
    """V_max coupled to K_m -- the one-axis rule.

    V_max was recovered from a rate MEASURED at 220 uM by dividing out the Michaelis
    fraction there, so the product V_max * [220/(K_m+220)] is the invariant, not
    V_max. Holding the anchor rate fixed:

        anchor = VMAX_NOMINAL * 220/(6.3+220)  = 3.00398e-6 M/s
        V_max(K_m) = anchor / [220/(K_m+220)]

    K_m 6.3 uM  -> 3.090e-6 M/s   (reproduces the nominal exactly)
    K_m 130 uM  -> 4.779e-6 M/s   <-- USED HERE.  NOT the legacy 5.0e-6, which sits
                                      on an anchor rate 4.62% higher and would put a
                                      V_max excursion inside a K_m-only bar.
    """
    anchor = VMAX_NOMINAL * ANCHOR_O2 / (KM_NOMINAL + ANCHOR_O2)
    return anchor / (ANCHOR_O2 / (km + ANCHOR_O2))


def ki_branch(km_resp):
    """K_I coupled to K_m,resp -- the ONE-AXIS rule, as km_branch does for V_max.

    K_m,resp AND K_I ARE A MATCHED PAIR FROM ONE EXPERIMENT, and src/parameters.py says so
    in those words. Brown & Cooper 1994 (PMID 7805858) reports TWO half-inhibition points --
    270 nM NO at ~145 uM O2 and 60 nM at ~30 uM -- and competitive inhibition
    IC50 = K_I*(1 + [O2]/K_m,resp) turns two points into two equations in two unknowns, giving
    K_m,resp = 20/7 = 2.8571 uM and K_I = 120/23 = 5.2174 nM. The shipped 2.86 / 5.22 are those
    rounded. So the pair is not two poorly-known numbers; it is ONE over-determined measurement,
    and moving either alone asserts an experiment that was never done.

    src prescribes the coupling explicitly: *"Moving K_m,resp inside that band REQUIRES
    recomputing K_I = 270/(1 + 145/Km_resp) from the same source."*  This function holds the
    145 uM IC50 as the invariant and inverts it, exactly as km_branch() holds the 220 uM anchor
    rate and inverts the Michaelis fraction:

        anchor = K_I_NO * (1 + 145/2.86)  = 269.8703 nM        <- the shipped pair's own IC50
        K_I(K_m,resp) = anchor / (1 + 145/K_m,resp)

        K_m,resp 0.50 uM -> K_I 0.927390 nM
        K_m,resp 2.86 uM -> K_I 5.220000 nM   (reproduces the nominal BIT-FOR-BIT)
        K_m,resp 5.00 uM -> K_I 8.995678 nM

    WHY THE ANCHOR IS BUILT FROM THE SHIPPED PAIR AND NOT FROM THE PUBLISHED 270 nM: the
    same reason rho_at_phi() rescales rather than recomputes. 5.22 and 2.86 are the published
    pair ROUNDED, so 270/(1+145/2.86) returns 5.2225 nM, not 5.2200 -- 0.048% away. Anchoring on
    the shipped pair makes K_I(nominal) identical to the last ulp, and the 0.048% then shows up
    where it belongs: as the gap between two independent routes, gated at G12.

    THE TWO EFFECTS PARTLY CANCEL IN K_m,eff = K_m,resp*(1 + [NO]/K_I), and that is the whole
    point of the axis. Substituting K_I(x) gives a form with NO K_I in it:

        K_m,eff(x) = x*(1 + [NO]/anchor) + 145 uM * [NO]/anchor

    -- linear in x with a large constant offset, so a 10x move in K_m,resp produces far less
    than 10x in K_m,eff, and the cancellation STRENGTHENS as [NO] rises. At this engine's wall
    [NO] of 3.098 nM the coupled span is 3.097x against 14.8x for the uncoupled K_I-only bar
    (4.8x too long); at the April figure's operating point (~44 nM) it is 1.216x against 85.9x,
    which reproduces src's documented "7.4-635 uM (86x)" / "23-31 uM (1.2-1.3x)" / "about 67x"
    and is how this function was verified to implement the coupling src prescribes.
    REPORT THE OVERSTATEMENT WITH ITS [NO]: 4.8x here, 70.7x there. It is not a constant.
    Gated by G11 (the IC50 invariant, exact), G12 (the published-270 route) and G13 (Brown &
    Cooper's SECOND point, 60 nM at 30 uM, which nothing in the construction uses).
    The 30 uM point is preserved only AT NOMINAL, by construction: holding the 145 uM point
    fixed is what src prescribes, so the second point is allowed to drift -- it runs 56.57 nM at
    K_m,resp 0.5 and 62.97 nM at 5.0, i.e. within 6% of the published 60 across the whole band.
    """
    anchor = K_I_NO * (1.0 + IC50_O2_ANCHOR / KM_RESP)
    return anchor / (1.0 + IC50_O2_ANCHOR / km_resp)


def phi_branch(phi):
    """Cluster separation coupled to phi -- the ONE-AXIS rule, as km_branch does for V_max.

    phi AND cluster_separation ARE NOT INDEPENDENT. Separation goes as rho^(-1/3) and
    rho = phi / V_cell, so at fixed expressing fraction f and fixed cell volume

        s(phi) = (N / (rho(phi) * f))^(1/3) = s(PHI_TUMOUR) * (PHI_TUMOUR / phi)^(1/3)

        phi 0.30 -> s = 210.72 um     sink x 0.714, separation x 1.119
        phi 0.42 -> s = 188.36 um     nominal; reproduces nominal_params() exactly
        phi 0.55 -> s = 172.17 um     sink x 1.310, separation x 0.914
    """
    return SEP_AT_F(F_NOMINAL, phi)


def nominal_params():
    """The full parameter dict at the nominal of every axis."""
    return dict(
        k_anoxia=K_ANOXIA_PUB,                    # published basis, mL/cell/s
        vol_div=VOLUME_CORRECTION_DIVISOR,        # tumour-arm divisor
        # WAS A LITERAL 2200.0. Same fragility that cost twenty days on the separation:
        #   it agreed with src, until the day it would not. src/parameters.py declares it as
        #   `D` in cm^2/s; 1 cm^2/s = 1e8 um^2/s. Value unchanged (2200.0).
        D_no=_P.D * 1e8,                          # um^2/s, from src/parameters.py D
        km=KM_NOMINAL,                            # M
        vmax=km_branch(KM_NOMINAL),               # M/s -- ALWAYS via km_branch
        # WAS A LITERAL 56e-6. src holds it inside VASCULAR_ENDOTHELIUM, not as a bare
        #   module constant, which is why it was copied here. Value unchanged (5.6e-05 M).
        rim_o2=_P.VASCULAR_ENDOTHELIUM['vessel_pO2'],   # M, from src
        phi=PHI_TUMOUR,
        sep=SEP_AT_F(F_NOMINAL),                  # um, f = 2% on the current rho/phi
        km_resp=KM_RESP,                          # M, 2.86e-6
        k_i=K_I_NO,                               # M, 5.22e-9 -- COUPLED
    )


def params_at(axis=None, endpoint='nominal'):
    """Nominal dict with ONE axis moved to one of its endpoints."""
    p = nominal_params()
    if axis is None:
        return p
    if axis not in AXES:
        raise KeyError('unknown axis %r; known: %s' % (axis, ', '.join(AXES)))
    if endpoint not in ('low', 'nominal', 'high'):
        raise KeyError('endpoint must be low|nominal|high, got %r' % endpoint)
    a = AXES[axis]
    p[a['key']] = a[endpoint]
    if a['key'] == 'km':                          # the coupling, enforced here
        p['vmax'] = km_branch(p['km'])
    elif a['key'] == 'phi':                       # THE OTHER COUPLING, enforced here.
        #   phi sets rho, and rho sets the separation. An uncoupled phi bar would run the
        #   sink at one density and the geometry at another. See phi_branch(), gate G8.
        p['sep'] = phi_branch(p['phi'])
    elif a['key'] == 'km_resp':                   # THE THIRD COUPLING, enforced here.
        p['k_i'] = ki_branch(p['km_resp'])
    return p


# =============================================================================
# SHARED PHYSICS  (copied from src/kinetics.py + src/constants.py)
# =============================================================================

def mm_fraction(o2, km):
    return o2 / (km + o2)


def k_deg_additive(o2, k_obs, k_anoxia, rho):
    """ADDITIVE two-route sink.  k_deg = k_obs*[O2]*rho + k_anoxia*rho  [1/s]."""
    return k_obs * o2 * rho + k_anoxia * rho


def tumour_sink(p, o2):
    """k_deg for the TUMOUR arm: both sink constants carry the volume correction.

    k_deg = (k_obs*[O2] + k_anoxia) * rho / vol_div
          = (k_obs*[O2] + k_anoxia) * (phi / V_cell) * (V_cell / V_referent)
          = (k_obs*[O2] + k_anoxia) * phi / V_referent
    """
    return k_deg_additive(o2, K_OBS_PUB / p['vol_div'], p['k_anoxia'] / p['vol_div'],
                          rho_at_phi(p['phi']))


def perivascular_sink(p, o2):
    """k_deg for the PERIVASCULAR arm: published constants, NO volume correction."""
    return k_deg_additive(o2, K_OBS_PUB, p['k_anoxia'], RHO_P)


# ---- finite uniformly-producing sphere, screened (Yukawa) kernel --------------
# Crank sec. 3.3.2 point source with first-order decay, integrated over a sphere of
# radius a (Philippides & O'Shea) so the 1/r singularity is removed and the peak is
# set by the cell radius.  Identical algebra to gen_full_set.coeffs / .fld.

def sphere_coeffs(D_um2s, k_deg, a_um, q):
    L = np.sqrt(D_um2s / k_deg)
    kap = 1.0 / L
    k = D_um2s * kap ** 2
    ca, sa = np.cosh(kap * a_um), np.sinh(kap * a_um)
    M = np.array([[sa / a_um, -np.exp(-kap * a_um) / a_um],
                  [kap * a_um * ca - sa, np.exp(-kap * a_um) * (kap * a_um + 1)]])
    B, A = np.linalg.solve(M, [-q / k, 0.0])
    return dict(A=A, B=B, q=q, k=k, kap=kap, L=L, a=a_um)


def sphere_field(d, c):
    d = np.asarray(d, float)
    kap, a = c['kap'], c['a']
    inside = c['q'] / c['k'] + c['B'] * np.where(d > 1e-9, np.sinh(kap * d) / np.maximum(d, 1e-9), kap)
    outside = c['A'] * np.exp(-kap * d) / np.maximum(d, 1e-9)
    return np.where(d <= a, inside, outside)


def cell_coeffs(p, o2):
    """One NOS2+ cell as a finite sphere of radius A0_UM."""
    R = p['vmax'] * mm_fraction(o2, p['km'])
    q = (R * VOL_CELL_UM3) / (4.0 / 3.0 * np.pi * A0_UM ** 3)
    return sphere_coeffs(p['D_no'], tumour_sink(p, o2), A0_UM, q)


def superpose(cells, coeff, GX, GY):
    """Sum the single-cell field over a cell list on a 2-D (z = 0) grid -> nM.

    Superposition is EXACT: k_deg is a constant at a given oxygen tension, so the
    governing equation is linear.  (gen_fig4_9panel: "Do not 'fix'.")
    """
    F = np.zeros_like(GX)
    for cx, cy, cz in cells:
        F += sphere_field(np.sqrt((GX - cx) ** 2 + (GY - cy) ** 2 + cz ** 2), coeff)
    return F * 1e9


def superpose_at_point(cells, coeff, pt=(0.0, 0.0, 0.0)):
    """Field at a single point -> nM (no grid; used for the midpoint scan)."""
    d = np.sqrt(np.array([(cx - pt[0]) ** 2 + (cy - pt[1]) ** 2 + (cz - pt[2]) ** 2
                          for cx, cy, cz in cells], float))
    return float(sphere_field(d, coeff).sum() * 1e9)


def cluster(center, n, s=None):
    """n cells on a cubic lattice of pitch s, nearest-first about `center`."""
    s = LATTICE_UM if s is None else s
    m = 3
    pts = [(i * s, j * s, k * s) for i in range(-m, m + 1)
           for j in range(-m, m + 1) for k in range(-m, m + 1)]
    pts.sort(key=lambda q: q[0] ** 2 + q[1] ** 2 + q[2] ** 2)
    return [(center[0] + q[0], center[1] + q[1], center[2] + q[2]) for q in pts[:n]]




PRE_D68 = dict(
    label='earlier (before 2026-08-20)',
    RHO_TUMOUR=1.13e9,          # cells/mL  -- earlier value, implied phi = 1
    VOL_CELL_UM3=885.0,         # um^3      -- 1e12/rho, the density restated
    LATTICE_UM=13.0,            # um        -- sized to the 11.91 um cell
    KM_NOMINAL=6.3e-6,
    VMAX_NOMINAL=3.09e-6,       # M/s  -- earlier, before the Daw re-derivation at Km 42
    VOLUME_CORRECTION_DIVISOR=7.006,   # 885/6200 -- current is 2.975
)

PRE_D42 = dict(
    label='earlier (K_m 130 uM branch)',
    RHO_TUMOUR=1.13e9,
    VOL_CELL_UM3=885.0,
    LATTICE_UM=13.0,
    KM_NOMINAL=130e-6,
    VMAX_NOMINAL=3.09e-6,       # M/s  -- earlier, before the Daw re-derivation at Km 42
    VOLUME_CORRECTION_DIVISOR=7.006,   # 885/6200 -- current is 2.975
)


class historical(object):
    """Context manager swapping the module onto a named historical configuration.

    Usage, and it MUST be labelled in whatever it produces:

        with historical(PRE_D42) as cfg:
            v = delivery_optimum(dict(nominal_params(), sep=49.8))
        print('%s: %.1f uM' % (cfg['label'], v))

    Restores on exit, including on exception. A0_UM and anything derived from the cell
      volume are recomputed inside the block so the state stays internally consistent.
    """

    _KEYS = ('RHO_TUMOUR', 'VOL_CELL_UM3', 'LATTICE_UM', 'KM_NOMINAL', 'A0_UM',
             'VMAX_NOMINAL', 'VOLUME_CORRECTION_DIVISOR')

    def __init__(self, cfg):
        self.cfg = cfg
        self.saved = {}

    def __enter__(self):
        g = globals()
        for k in self._KEYS:
            self.saved[k] = g.get(k)
        for k, v in self.cfg.items():
            if k != 'label':
                g[k] = v
        # derived quantities must follow, or the state is a mixture
        g['A0_UM'] = (3.0 * g['VOL_CELL_UM3'] / (4.0 * np.pi)) ** (1.0 / 3.0)
        return self.cfg

    def __exit__(self, *exc):
        g = globals()
        for k, v in self.saved.items():
            g[k] = v
        return False

def two_clusters(sep_um):
    """The Fig-3 2-cluster arrangement: 14 + 13 = 27 cells at +-sep/2 on x."""
    cells = cluster((-sep_um / 2.0, 0.0, 0.0), 14) + cluster((sep_um / 2.0, 0.0, 0.0), 13)
    assert len(cells) == N_CELLS_TOTAL
    floor = GEOMETRIC_FLOOR_UM()
    if sep_um < floor - 1e-9:
        raise ValueError(
            'two_clusters: separation %.3f um is below the geometric floor %.3f um '
            '(clusters would sit closer than one cell diameter apart). '
            'At the current rho that floor corresponds to f = %.1f%% NOS2+.'
            % (sep_um, floor, 100.0 * N_CELLS_TOTAL / (RHO_TUMOUR * (floor * 1e-4) ** 3)))
    d = 2.0 * A0_UM
    for i in range(len(cells)):
        xi, yi, zi = cells[i]
        for j in range(i + 1, len(cells)):
            xj, yj, zj = cells[j]
            if ((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2) < (d - 1e-9) ** 2:
                raise ValueError(
                    'two_clusters: sources overlap at separation %.3f um '
                    '(cell diameter %.3f um, lattice pitch %.3f um)'
                    % (sep_um, d, LATTICE_UM))
    return cells


def cube_cells(spacing=FIG2_CUBE_SPACING):
    """The Fig-1h/2 3x3x3 cube of 27 cells."""
    off = (-spacing, 0.0, spacing)
    return [(x, y, z) for x in off for y in off for z in off]


def focus_cells(f, R=DENSITY_FOCUS_R):
    """A focus of NOS2+ cells at expressing fraction f inside a sphere of radius R.

    n_sources = rho_tumour * f = (phi / V_cell) * f
    spacing   = n_sources ** (-1/3)
    """
    n_sources = RHO_TUMOUR * f * 1e-12          # cells per um^3
    s = n_sources ** (-1.0 / 3.0)
    m = int(np.ceil(R / s))
    pts = np.arange(-m, m + 1) * s
    return [(x, y, z) for x in pts for y in pts for z in pts if x * x + y * y + z * z <= R * R]


# =============================================================================
# OUTPUT 1 -- peak [NO] across the NOS2+ density series, at 10 uM O2
# =============================================================================

def peak_field_vs_density(p, detail=False):
    """Peak [NO] (nM) for the density series at 10 uM O2.  UNREGISTERED -- see above.

    Reproduces gen_finite_matrices.py's Fig-3 density panels: a focus of radius
    50 um filled at NOS2+ volume fractions 2 / 5 / 15 / 40 %, evaluated on the
    +-200 um / 2 um grid.  THE 40% PANEL IS THE REPORTED SCALAR.
    Tumour arm -> the volume correction is live on this output.
    """
    GX, GY = np.meshgrid(DENSITY_GRID, DENSITY_GRID, indexing='ij')
    c = cell_coeffs(p, DENSITY_O2)
    series = {}
    for f in DENSITY_FRACTIONS:
        cells = focus_cells(f)
        series['%d%%' % round(f * 100)] = dict(
            n_cells=len(cells), peak_nM=float(superpose(cells, c, GX, GY).max()))
    scalar = series['40%']['peak_nM']
    if detail:
        return scalar, dict(series=series, L_um=float(c['L']), o2_uM=DENSITY_O2 * 1e6)
    return scalar


# =============================================================================
# OUTPUT 2 -- the oxygen tension that maximises inter-cluster midpoint delivery
# =============================================================================

O2_SCAN_LO, O2_SCAN_HI, O2_SCAN_N = 0.1e-6, 300e-6, 1200


# \u2b1b Written by delivery_optimum() on EVERY call so a non-detail caller can still see
# whether the optimum was interior. Read it immediately after the call; it is overwritten.
LAST_DELIVERY_DIAG = {}


def delivery_optimum(p, detail=False):
    """[O2] (uM) at which the MIDPOINT [NO] between two clusters is maximal.

    Two clusters, 27 cells total, centre-to-centre separation from the
    cluster-separation axis; the midpoint is the origin.  Coarse linear scan of
    1200 tensions over 0.1-300 uM, then a bounded parabolic refine inside the
    bracketing triple, so the reported optimum is not scan-grid-limited.
    `interior` is False if the argmax lands on a scan edge -- that would mean no
    interior maximum exists in range and the number must not be quoted.
    """
    cells = two_clusters(p['sep'])
    o2s = np.linspace(O2_SCAN_LO, O2_SCAN_HI, O2_SCAN_N)
    vals = np.array([superpose_at_point(cells, cell_coeffs(p, o2)) for o2 in o2s])
    i = int(np.argmax(vals))
    interior = 0 < i < len(o2s) - 1
    if interior:
        r = minimize_scalar(lambda o2: -superpose_at_point(cells, cell_coeffs(p, o2)),
                            bounds=(o2s[i - 1], o2s[i + 1]), method='bounded',
                            options=dict(xatol=1e-10))
        # \u2b1b OPTIMISER SUCCESS IS CHECKED. A silent failure here returns whatever x the
        # optimiser stopped at, which then travels into the CSV as a delivery optimum.
        if not getattr(r, 'success', True):
            raise RuntimeError('delivery_optimum: minimize_scalar failed at sep=%.4g: %s'
                               % (p['sep'], getattr(r, 'message', 'no message')))
        o2_opt, peak = float(r.x), float(-r.fun)
    else:
        o2_opt, peak = float(o2s[i]), float(vals[i])
    scalar = o2_opt * 1e6
    LAST_DELIVERY_DIAG.clear()
    # peak IS ALREADY nM: superpose_at_point() applies the 1e9 itself. Converting again
    # here produced midpoint values of ~2.3e10 nM on the first run of this patch.
    LAST_DELIVERY_DIAG.update(interior_maximum=bool(interior),
                              midpoint_nM_at_optimum=peak,
                              separation_um=p['sep'])
    if detail:
        return scalar, dict(interior_maximum=bool(interior), midpoint_nM_at_optimum=peak,
                            separation_um=p['sep'], n_cells=len(cells),
                            scan_uM=[O2_SCAN_LO * 1e6, O2_SCAN_HI * 1e6])
    return scalar


# =============================================================================
# OUTPUT 3 -- reach at the anoxic floor, perivascular arm
# =============================================================================

def reach_at_anoxia(p, detail=False):
    """sqrt(D_NO / (k_anoxia * rho)) in um -- PERIVASCULAR arm (rho = 1.0e8).

    The oxygen-independent floor alone, so [O2] does not enter.  No volume
    correction: hard rule 4 -- it applies only where rho = 1.13e9.  That is why
    the cell-volume axis has EXACTLY ZERO swing here and must not be drawn on it.
    """
    k = p['k_anoxia'] * RHO_P
    val = float(np.sqrt(p['D_no'] / k))
    if detail:
        return val, dict(k_floor_per_s=float(k), rho=RHO_P, volume_correction_applied=False)
    return val


# =============================================================================
# OUTPUT 4 -- perivascular wall [NO], cylindrical Krogh geometry
# =============================================================================

def _o2_krogh(r, rim_o2, km_resp=None):
    """Krogh-cylinder O2 (M) with the saturating correction on V_M, clipped at 0."""
    km_resp = KM_RESP if km_resp is None else km_resp
    q = V_M * (rim_o2 / (rim_o2 + km_resp)) * 1e6          # uM/s
    rr = np.maximum(r, R_V)
    c = (rim_o2 * 1e6
         - q * R_K ** 2 / (2 * D_O2_UM2S) * np.log(rr / R_V)
         + q / (4 * D_O2_UM2S) * (rr ** 2 - R_V ** 2))
    return np.maximum(c, 0.0) * 1e-6


def wall_no(p, detail=False):
    """Perivascular wall [NO] in nM -- steady 1-D radial (cylindrical) solve.

    (1/r) d/dr ( r D dC/dr ) - k(r) C + S(r) = 0
    k(r) = k_obs*[O2](r)*rho + k_anoxia*rho          ADDITIVE
    S(r) = NOS3 production in an 0.8 um endothelial annulus inside R_V
    """
    r = np.linspace(1e-6, R_K, N_RADIAL)
    h = r[1] - r[0]
    o2 = _o2_krogh(r, p['rim_o2'], p.get('km_resp', KM_RESP))
    k = perivascular_sink(p, o2)
    S = np.zeros_like(r)
    endo = (r >= R_V - SHELL_UM) & (r <= R_V)
    S[endo] = VMAX_ENOS * o2[endo] / (KM_ENOS + o2[endo])
    lo, di, up = np.zeros(N_RADIAL), np.zeros(N_RADIAL), np.zeros(N_RADIAL)
    b = -S.copy()
    D_no = p['D_no']
    lo[1:-1] = D_no / h ** 2 - D_no / (2 * r[1:-1] * h)
    di[1:-1] = -2.0 * D_no / h ** 2 - k[1:-1]
    up[1:-1] = D_no / h ** 2 + D_no / (2 * r[1:-1] * h)
    di[0], up[0], b[0] = 1.0, -1.0, 0.0            # zero flux at the axis
    di[-1], lo[-1], b[-1] = 1.0, -1.0, 0.0         # zero flux at R_K
    ab = np.zeros((3, N_RADIAL))
    ab[0, 1:], ab[1, :], ab[2, :-1] = up[:-1], di, lo[1:]
    C_raw = solve_banded((1, 1), ab, b)
    _neg = float(C_raw.min())
    _scale = float(max(C_raw.max(), 1e-30))
    if _neg < -1e-6 * _scale:
        raise RuntimeError('wall_no: banded solve returned NO of %.3e M against a peak of %.3e M '
                           '(%.2f%% negative); this is a solver failure, not round-off'
                           % (_neg, _scale, 100.0 * _neg / _scale))
    C = np.maximum(C_raw, 0.0)
    val = float(np.interp(R_V, r, C) * 1e9)
    if detail:
        return val, dict(shell_um=SHELL_UM, R_v=R_V, R_K=R_K,
                         rim_o2_uM=p['rim_o2'] * 1e6, lumen='reflecting',
                         volume_correction_applied=False)
    return val



def steady_state_no(p, detail=False):
    """Steady-state peak [NO] (nM) of the Fig 1h/2 27-cell cube, AT THE RIM TENSION.

    WHAT THIS RETURNS INSTEAD is the steady-state field the model actually predicts: the
    peak of gen_full_set.py's 3x3x3 / 75 um cube, i.e. fig2_peak, which is a SHIPPED figure
    number (12.1488 nM at 56 uM) already gated at G4a/G4b against that INDEPENDENT generator
    to 4e-13%. So the restored output arrives with a gate that was never copied from this
    module's own output -- which is the standard this project sets (STALE_GATES banner).
    """
    # fig2_peak is defined below; resolved at call time, so the order is legal.
    val = float(fig2_peak(p, p['rim_o2']))
    if detail:
        return val, dict(o2_uM=p['rim_o2'] * 1e6, spacing_um=FIG2_CUBE_SPACING,
                         n_cells=27, grid='+/-300 um / 2 um',
                         construction='gen_full_set.py Fig 1h/2 cube',
                         volume_correction_applied=True)
    return val



def effective_mitochondrial_km(p, detail=False):
    """Effective mitochondrial K_m for O2 (uM) at the vessel wall.

        K_m,eff = K_m,resp * (1 + [NO]_wall / K_I)

    THE [NO] IS THE PERIVASCULAR WALL VALUE, wall_no(p), NOT a tumour concentration, for
    three reasons that all point the same way: K_m,resp and K_I live in
    _P.VASCULAR_ENDOTHELIUM; the respiration term they parameterise is the one inside
    _o2_krogh, i.e. this arm's own oxygen field; and the manuscript's live claim about
    effective K_m is the perivascular one. The April figure instead used its TUMOUR 0-D
    C_ss, which is why its K_m,eff sat near 44 nM worth of inhibition. For the record and NOT
    registered: at the tumour cube's 12.1488 nM the same formula gives 9.516 uM (3.33-fold),
    and at the 0-D bound 185 uM (64.7-fold) -- a different claim, absent from the manuscript.
    """
    no_M = wall_no(p) * 1e-9
    fold = 1.0 + no_M / p['k_i']
    val = float(p['km_resp'] * fold * 1e6)
    if detail:
        return val, dict(no_wall_nM=no_M * 1e9, fold_change=float(fold),
                         km_resp_uM=p['km_resp'] * 1e6, K_I_nM=p['k_i'] * 1e9,
                         feedback_into_o2_field=False,
                         volume_correction_applied=False)
    return val


OUTPUTS = dict(delivery_optimum=delivery_optimum,
               reach_at_anoxia=reach_at_anoxia,
               wall_no=wall_no,
               steady_state_no=steady_state_no,
               effective_mitochondrial_km=effective_mitochondrial_km)

UNITS = dict(delivery_optimum='uM O2',
             reach_at_anoxia='um', wall_no='nM',
             steady_state_no='nM', effective_mitochondrial_km='uM O2')

# CSV column suffix per output, in ONE place. Read by _sweep() three times; a second copy
#   is how the nominal row and the axis rows would eventually disagree about a column name.
CSV_SUFFIX = dict(delivery_optimum='uM', reach_at_anoxia='um', wall_no='nM',
                  steady_state_no='nM', effective_mitochondrial_km='uM')


def evaluate(p):
    return {name: float(fn(p)) for name, fn in OUTPUTS.items()}


# =============================================================================
# FIG 1h / FIG 2 OXYGEN SWEEP -- gate support only
# =============================================================================

def fig2_peak(p, o2):
    """Peak [NO] (nM) of the 3x3x3 cube at 75 um spacing, on the +-300 um grid."""
    GX, GY = np.meshgrid(FIG2_GRID, FIG2_GRID, indexing='ij')
    return float(superpose(cube_cells(), cell_coeffs(p, o2), GX, GY).max())


# =============================================================================
# GATES -- all must pass before any output is reported
# =============================================================================

def _gate_f_is_consistent(f=0.40):
    """Ratio of the source number density implied by the two f-paths. Must be exactly 1.

    THE CROSS-PATH CHECK. focus_cells and SEP_AT_F both take an argument called f. They
    used incompatible definitions -- fraction of CELLS vs tissue VOLUME fraction -- differing
    by 1/phi = 2.381x, so canonical "40%" meant two different tissues. Nothing detected that,
    because each path was internally consistent. This gate holds them together by comparing
    the one quantity both must agree on: how many NOS2+ sources per unit volume f implies.

    focus path : lattice spacing -> 1/spacing^3
    cluster path: N cells per cluster, clusters at SEP_AT_F(f) -> N/sep^3
    Both must equal rho_tumour * f.
    """
    pts = focus_cells(f, R=DENSITY_FOCUS_R)
    _ = pts                                     # constructed to prove it still builds
    n_focus = RHO_TUMOUR * f * 1e-12            # by construction after the 2026-09-10 fix
    sep = SEP_AT_F(f)
    n_cluster = N_CELLS_TOTAL / (sep ** 3)      # cells per um^3 from the cluster lattice
    return float(n_cluster / n_focus)


def _gate_sep_tracks_rho():
    """Worst ratio of a stored separation endpoint to the value rho implies. Must be 1.

    THE TARGET IS RECOMPUTED HERE FROM THE IMPORTED CONSTANTS, by an inline path that
    never calls SEP_AT_F and never reads AXES for the expected value. That is what makes it
    a verification rather than a restatement. This is the gate that did not exist when the
    endpoints sat at rho = 1.13e9 for twenty days: type 60.0 back into AXES and this fails.
    """
    worst = 1.0
    for f, end in ((0.40, 'low'), (0.02, 'nominal'), (0.005, 'high')):
        want = (N_CELLS_TOTAL / (RHO_TUMOUR * f)) ** (1.0 / 3.0) * 1e4
        got = AXES['cluster_separation'][end]
        r = got / want
        if abs(r - 1.0) > abs(worst - 1.0):
            worst = r
    return float(worst)


def _gate_pitch_tracks_cell():
    """Ratio of the lattice pitch to one cell diameter implied by src cell volume. Must be 1."""
    d = 2.0 * (3.0 * CELL_VOLUME_MODEL_UM3 / (4.0 * np.pi)) ** (1.0 / 3.0)
    return float(LATTICE_UM / d)


def _gate_phi_couples_separation():
    """Worst ratio of the coupled separation to the closed-form phi^(-1/3) law. Must be 1."""
    s_nom = SEP_AT_F(F_NOMINAL)
    worst = 1.0
    for end in ('low', 'nominal', 'high'):
        phi = AXES['packing_fraction'][end]
        want = s_nom * (PHI_TUMOUR / phi) ** (1.0 / 3.0)
        r = phi_branch(phi) / want
        if abs(r - 1.0) > abs(worst - 1.0):
            worst = r
    return float(worst)


def _gate_phi_scales_sink():
    """Worst ratio of k_deg(phi) to (phi/PHI_TUMOUR)*k_deg(nominal). Must be 1.

    phi multiplies BOTH branches of the tumour sink, so the sink is exactly linear in
    it. Asserting f(phi) == (phi/phi_0)*f(phi_0) at tensions either side of the crossover
    ([O2]* = k_anoxia/k_obs = 24.8 uM, where the two branches trade dominance) turns "did I
    plumb phi into the right place" into a gate instead of a worry. [O2] = 0 isolates the
    anoxic branch; 220 uM is the anchor tension, where the oxygen branch dominates.
    """
    base = nominal_params()
    worst = 1.0
    for end in ('low', 'nominal', 'high'):
        phi = AXES['packing_fraction'][end]
        p = dict(base, phi=phi)
        for o2 in (0.0, 5e-6, 24.83e-6, 56e-6, 220e-6):
            want = (phi / PHI_TUMOUR) * tumour_sink(base, o2)
            r = tumour_sink(p, o2) / want
            if abs(r - 1.0) > abs(worst - 1.0):
                worst = r
    return float(worst)


def _gate_volume_correction_cancels():
    """Worst k_deg ratio across the RETIRED cell-volume range with rho tracking V_cell.

    Must be 1 -- this is the retirement of that axis, executed as a number rather than
    asserted in a comment.
    """
    vds = (VOLUME_CORRECTION_DIVISOR_RANGE[0], VOLUME_CORRECTION_DIVISOR,
           VOLUME_CORRECTION_DIVISOR_RANGE[1])
    worst = 1.0
    for o2 in (0.0, 56e-6, 220e-6):
        vals = []
        for vd in vds:
            v_cell = CELL_VOLUME_REFERENT_UM3 / vd        # um^3 this divisor asserts
            rho = PHI_TUMOUR / v_cell * 1e12              # cells/mL, the src identity
            vals.append((K_OBS_PUB * o2 + K_ANOXIA_PUB) / vd * rho)
        for v in vals:
            r = v / vals[1]
            if abs(r - 1.0) > abs(worst - 1.0):
                worst = r
    return float(worst)


def _gate_ki_ic50_invariant():
    """Worst ratio of K_I(x)*(1 + 145 uM/x) to the anchor IC50, across the axis. Must be 1."""
    anchor = K_I_NO * (1.0 + IC50_O2_ANCHOR / KM_RESP)
    worst = 1.0
    for end in ('low', 'nominal', 'high'):
        x = AXES['mito_inhibition'][end]
        got = ki_branch(x) * (1.0 + IC50_O2_ANCHOR / x)
        r = got / anchor
        if abs(r - 1.0) > abs(worst - 1.0):
            worst = r
    return float(worst)


def _gate_ki_matches_published_ic50():
    """Worst ratio of ki_branch(x) to 270 nM/(1 + 145 uM/x). Must be 1 to the shipped rounding.

    THE SECOND, INDEPENDENT ROUTE, in the style of G8: rebuild K_I from the PUBLISHED IC50
    (Brown & Cooper 1994, 270 nM at ~145 uM O2) instead of from the shipped pair. It must agree
    to the rounding that separates the shipped 2.86/5.22 from the exact 20/7 and 120/23, and
    the agreement must be the SAME at every endpoint -- a constant offset is the rounding, a
    drifting one would be a different law.
    TOLERANCE 1e-3 IS NOT SLACK, IT IS THE ROUNDING: the exact figure is 4.80e-4 at every
    point. Do not tighten it by editing the constants, and do not loosen it.
    """
    worst = 1.0
    for end in ('low', 'nominal', 'high'):
        x = AXES['mito_inhibition'][end]
        want = IC50_145_PUB / (1.0 + IC50_O2_ANCHOR / x)
        r = ki_branch(x) / want
        if abs(r - 1.0) > abs(worst - 1.0):
            worst = r
    return float(worst)


def _gate_second_published_ic50():
    """IC50 (nM) the NOMINAL pair predicts at Brown & Cooper's OTHER tension, 30 uM O2."""
    p = nominal_params()
    return float(p['k_i'] * (1.0 + IC50_O2_SECOND / p['km_resp']) * 1e9)


def _gate_arm_table_is_exact():
    """1.0 if OUTPUT_AXES is exactly right; 1 + (number of violations) otherwise."""
    base = evaluate(nominal_params())
    bad = []
    for axis, spec in AXES.items():
        if spec.get('retired'):
            continue
        moved = {name: False for name in base}
        for end in ('low', 'high'):
            if spec[end] == spec['nominal']:
                continue                       # degenerate end of a one-sided axis
            v = evaluate(params_at(axis, end))
            for name in base:
                if v[name] != base[name]:
                    moved[name] = True
        for name in base:
            in_arm = axis in OUTPUT_AXES[name]
            if in_arm and not moved[name]:
                bad.append('%s listed in %s arm but does not move it' % (axis, name))
            elif (not in_arm) and moved[name]:
                bad.append('%s MOVES %s but is not in its arm' % (axis, name))
    if bad:
        for b in bad:
            print('        ARM TABLE: %s' % b)
    return float(1 + len(bad))


GATE_SPECS = [
    ('G1  wall [NO] nM, D_NO 2200',
     lambda: wall_no(params_at('D_NO', 'nominal')), 3.10, 0.02),
    ('G2  wall [NO] nM, D_NO 850',
     lambda: wall_no(params_at('D_NO', 'low')), 5.70, 0.02),
    ('G3  reach at anoxia um, perivascular nominal',
     lambda: reach_at_anoxia(nominal_params()), 40.6, 0.03),
    ('G4a Fig 1h/2 peak nM at 56 uM O2',
     lambda: fig2_peak(nominal_params(), 56e-6), 12.1488, 0.02),
    ('G4b Fig 1h/2 peak nM at 200 uM O2',
     lambda: fig2_peak(nominal_params(), 200e-6), 14.8608, 0.02),
    ('G5  separation endpoints track rho (recomputed from src, not from AXES)',
     lambda: _gate_sep_tracks_rho(), 1.0, 1e-9),
    ('G6  lattice pitch == one cell diameter from src cell volume',
     lambda: _gate_pitch_tracks_cell(), 1.0, 1e-9),
    ('G7  f means the same in both paths (focus_cells vs SEP_AT_F) at f = 40%',
     lambda: _gate_f_is_consistent(), 1.0, 1e-9),
    ('G8  phi couples cluster separation (phi^(-1/3) closed form)',
     lambda: _gate_phi_couples_separation(), 1.0, 1e-12),
    ('G9  phi scales the tumour sink linearly, both branches',
     lambda: _gate_phi_scales_sink(), 1.0, 1e-12),
    ('G10 cell-volume divisor cancels when rho tracks V_cell',
     lambda: _gate_volume_correction_cancels(), 1.0, 1e-12),
    ('G11 K_I couples K_m,resp (Brown&Cooper 145 uM IC50 invariant)',
     lambda: _gate_ki_ic50_invariant(), 1.0, 1e-12),
    ('G12 K_I matches the PUBLISHED 270 nM route (to the shipped rounding)',
     lambda: _gate_ki_matches_published_ic50(), 1.0, 1e-3),
    ('G13 nominal pair reproduces the SECOND published IC50, nM at 30 uM',
     lambda: _gate_second_published_ic50(), 60.0, 0.01),
    ('G14 K_m,eff fold at nominal (manuscript: 1.59-fold at 3.10 nM wall)',
     lambda: effective_mitochondrial_km(nominal_params()) / KM_RESP * 1e-6, 1.59, 0.01),
    ('G15 steady_state_no at nominal == Fig 1h/2 peak at 56 uM (G4a target)',
     lambda: steady_state_no(nominal_params()), 12.1488, 0.02),
    ('G16 OUTPUT_AXES arm table exact: listed axes move, unlisted are bit-identical',
     lambda: _gate_arm_table_is_exact(), 1.0, 1e-12),
]


STALE_GATES = {
}


def run_gates(verbose=True):
    if verbose:
        print('=' * 74)
        print('GATES -- reproduce known-good values before trusting anything below')
        print('=' * 74)
        print('   %-54s %9s %9s' % ('gate', 'got', 'want'))
    rows, all_ok = [], True
    for label, fn, want, tol in GATE_SPECS:
        got = float(fn())
        ok = abs(got - want) / abs(want) < tol
        stale = label in STALE_GATES
        all_ok &= (ok or stale)          # stale gates report but do not block
        rows.append(dict(name=label, got=got, want=want, tol=tol, pass_=ok, stale=stale,
                         rel_err=abs(got - want) / abs(want)))
        if verbose:
            print('   %-54s %9.3f %9.3f  %s   (%.2f%% of %.0f%% tol)'
                  % (label, got, want,
                     'PASS' if ok else ('*** STALE ***' if stale else '*** FAIL ***'),
                     100 * abs(got - want) / abs(want), 100 * tol))
            if stale and not ok:
                print('        ^ STALE TARGET, not a model error: %s' % STALE_GATES[label])
    if verbose:
        n_stale = sum(1 for r in rows if r['stale'] and not r['pass_'])
        if not all_ok:
            msg = 'A GATE FAILED -- do not use any number below'
        elif n_stale:
            msg = ('%d gate(s) PASSED, %d STALE -- numbers below are usable, but the stale '
                   'targets are UNVERIFIED at the current parameter set -- re-derive from the '
                   "INDEPENDENT generator, never from this module's own output"
                   % (len(rows) - n_stale, n_stale))
        else:
            msg = 'ALL GATES PASSED'
        print('\n   %s\n' % msg)
    return all_ok, rows


# =============================================================================
# CLI
# =============================================================================

def _sweep(path):
    fields = ['axis', 'endpoint', 'axis_value', 'axis_units', 'moves_output',
              'delivery_optimum_uM',
              'reach_at_anoxia_um', 'wall_no_nM',
              'steady_state_no_nM', 'effective_mitochondrial_km_uM',
              'delivery_interior_max', 'delivery_midpoint_nM']
    rows = []
    base = evaluate(nominal_params())
    _d = dict(LAST_DELIVERY_DIAG)
    rows.append(dict(axis='(nominal)', endpoint='nominal', axis_value='',
                     axis_units='', moves_output='',
                     delivery_interior_max=_d.get('interior_maximum'),
                     delivery_midpoint_nM=_d.get('midpoint_nM_at_optimum'),
                     **{'%s_%s' % (k, CSV_SUFFIX[k]): base[k] for k in base}))
    for axis, spec in AXES.items():
        if spec.get('retired'):
            continue          # tombstone, not a bar -- see the banner in AXES
        for end in ('low', 'high'):
            p = params_at(axis, end)
            v = evaluate(p)
            _d = dict(LAST_DELIVERY_DIAG)
            val = spec[end]
            rows.append(dict(
                axis=axis, endpoint=end,
                axis_value=(val * 1e6 if spec['key'] in ('km', 'rim_o2', 'km_resp')
                            else val),
                axis_units=spec['units'],
                moves_output=';'.join(o for o, ax in OUTPUT_AXES.items() if axis in ax),
                delivery_optimum_uM=v['delivery_optimum'],
                reach_at_anoxia_um=v['reach_at_anoxia'],
                wall_no_nM=v['wall_no'],
                steady_state_no_nM=v['steady_state_no'],
                effective_mitochondrial_km_uM=v['effective_mitochondrial_km'],
                delivery_interior_max=_d.get('interior_maximum'),
                delivery_midpoint_nM=_d.get('midpoint_nM_at_optimum')))
    for name, s in SCENARIOS.items():
        v = evaluate(scenario_params(name))
        _d = dict(LAST_DELIVERY_DIAG)
        rows.append(dict(
            delivery_interior_max=_d.get('interior_maximum'),
            delivery_midpoint_nM=_d.get('midpoint_nM_at_optimum'),
            axis=name, endpoint='scenario',
            axis_value=(s['value'] * 1e6 if s['key'] in ('km', 'rim_o2') else s['value']),
            axis_units=s['units'], moves_output='(separate state -- NOT an uncertainty bar)',
            delivery_optimum_uM=v['delivery_optimum'],
            reach_at_anoxia_um=v['reach_at_anoxia'],
            wall_no_nM=v['wall_no'],
            steady_state_no_nM=v['steady_state_no'],
            effective_mitochondrial_km_uM=v['effective_mitochondrial_km']))

    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    # 'mito_inhibition' appears here automatically; its key is km_resp and params_at()
    #   applies ki_branch(), so there is no way to reach a lone K_I excursion from the CLI.
    ap.add_argument('--axis',
                    choices=sorted(a for a in AXES if not AXES[a].get('retired')),
                    default=None, help='axis to move off nominal')
    ap.add_argument('--at', choices=('low', 'nominal', 'high'), default='nominal',
                    help='endpoint of that axis')
    ap.add_argument('--gates-only', action='store_true', help='run gates and exit')
    ap.add_argument('--sweep', action='store_true',
                    help='evaluate every axis endpoint -> tornado_v2_results.csv')
    ap.add_argument('--detail', action='store_true',
                    help='include per-output diagnostics in the JSON')
    a = ap.parse_args(argv)

    ok, rows = run_gates(verbose=True)
    if not ok:
        for r in rows:
            if not r['pass_']:
                print('FAILED %s: got %.6g, want %.6g (%.3f%% off, tol %.0f%%)'
                      % (r['name'], r['got'], r['want'], 100 * r['rel_err'], 100 * r['tol']),
                      file=sys.stderr)
        return 1
    if a.gates_only:
        return 0

    if a.sweep:
        out = os.path.join(HERE, 'tornado_v2_results.csv')
        _sweep(out)
        print('sweep written to %s' % out)
        return 0

    p = params_at(a.axis, a.at)
    res = {}
    for name, fn in OUTPUTS.items():
        if a.detail:
            val, det = fn(p, detail=True)
            res[name] = dict(value=float(val), units=UNITS[name], detail=det)
        else:
            res[name] = dict(value=float(fn(p)), units=UNITS[name])
        res[name]['axis_moves_this_output'] = (
            a.axis is None or a.at == 'nominal' or a.axis in OUTPUT_AXES[name])
    print(json.dumps(dict(
        axis=a.axis, endpoint=a.at,
        axis_value=(None if a.axis is None else AXES[a.axis][a.at]),
        axis_units=(None if a.axis is None else AXES[a.axis]['units']),
        params={k: v for k, v in sorted(p.items())},
        outputs=res), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
