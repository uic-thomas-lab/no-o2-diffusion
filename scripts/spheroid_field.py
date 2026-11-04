# -*- coding: utf-8 -*-
"""Coupled O2/NO fields in a spherical tumour, solved at the model's parameters.

So this script asserts nothing from any prior document. It solves the equations and reports what
they give.

    Km, Vmax                       from parameters.KM / parameters.VMAX      (module level, current)
    rho, k_obs, k_anoxia           from constants.RHO_TUMOUR / K_OBS_CORR / K_ANOXIA_CORR
    D_NO, D_O2, V_M, Km_resp, K_I  from SOLID_TUMOR   (transport only -- current, no alternative)

Run:  python scripts/spheroid_field.py            profile at one diameter + diameter sweep
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.integrate import solve_bvp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import constants as C          # noqa: E402
import parameters as P         # noqa: E402

_ST = P.SOLID_TUMOR
assert _ST['Km'] == 6.3e-6, 'SOLID_TUMOR["Km"] changed; re-check which K_m this script should use'
assert _ST['cell_density'] == 1.13e9, 'SOLID_TUMOR["cell_density"] changed; re-check rho source'

KM = P.KM                      # 4.2e-5 M   NOS2 apparent K_m
VMAX = P.VMAX                  # 1.718e-6 M/s              
RHO = C.RHO_TUMOUR             # 2.02e8 cells/mL           
K_OBS = C.K_OBS_CORR           # use with RHO_TUMOUR
K_ANOX = C.K_ANOXIA_CORR       # use with RHO_TUMOUR
D_NO = _ST['D']                # cm2/s
D_O2 = _ST['D_O2']             # cm2/s
V_M = _ST['V_M_tissue']        # M/s
KM_RESP = _ST['Km_resp']       # M
K_I = _ST['K_I']               # M   NO inhibition constant for cytochrome c oxidase

O2_RIM_DEFAULT = 56e-6         # M   project convention; a NORMAL-tissue value
ANOXIC_THRESHOLD = 0.1e-6      # M   the project's anoxic definition
SGC = 1e-9                     # M


def k_deg(o2):
    """First-order NO removal rate, additive two-route sink."""
    return K_OBS * o2 * RHO + K_ANOX * RHO


PHI = C.PHI_TUMOUR             # 0.42  tissue cell volume fraction


def production(o2, expressing_fraction=1.0):
    """Tissue-level NO source [M/s].

    V_max is an INTRACELLULAR volumetric rate, not a tissue one. The per-cell source is
    V_max * f([O2]) * V_cell, which is how the Figure 2 series was built: at 10 uM with ten cells
    it gives 6.87e-18 mol/s, reproducing that experiment's stated total exactly. Tissue-level
    production is therefore V_max * f * phi, and phi must appear here because the SINK already
    carries it -- RHO_TUMOUR is defined as phi / V_cell.

    Omitting phi inflates the source 1/0.42 = 2.38x while leaving the sink untouched, which
    is a silent 2.4x on every concentration in the solve. It was in the first version of this file.
    """
    return VMAX * o2 / (o2 + KM) * PHI * expressing_fraction


def _bvp(rhs, bc, r, guess, tol):
    from scipy.integrate import solve_bvp as _sb
    return _sb(rhs, bc, r, guess, tol=tol, max_nodes=2000000)


# SCALED UNITS: r in um, [O2] in uM, [NO] in nM, D in um^2/s. The first version solved in cm
# and molar, which puts concentrations at 1e-9 and lengths at 1e-4; solve_bvp's residual test is
# relative to the solution scale and the mesh refined without limit above ~450 um radius. Same
# equations, conditioned.
D_NO_UM = D_NO * 1e8          # cm^2/s -> um^2/s
D_O2_UM = D_O2 * 1e8
V_M_UM = V_M * 1e6            # M/s -> uM/s
KM_RESP_UM = KM_RESP * 1e6
K_I_NM = K_I * 1e9
KM_UM = KM * 1e6
VMAX_NM = VMAX * 1e9          # M/s -> nM/s


RHO_CELLS = RHO
def rho_source(f_nos2):
    return f_nos2 * RHO_CELLS


def kdeg_u(o2_um):
    """[1/s] from [O2] in uM. Uses RHO_CELLS -- ALL cells consume NO, expressing or not.
    ADDITIVE; not the max() form."""
    return K_OBS * (o2_um * 1e-6) * RHO_CELLS + K_ANOX * RHO_CELLS


def prod_u(o2_um, f_nos2=1.0):
    """[nM/s] from [O2] in uM. Scales with the SOURCE density only, and carries PHI because the
    sink carries it through rho = phi/V_cell."""
    return VMAX_NM * o2_um / (o2_um + KM_UM) * PHI * f_nos2


def solve(R_um, o2_rim_um=56.0, respiratory_inhibition=True, n=6001, tol=1e-8,
          f_nos2=1.0, o2_gating=True, _guess=None):
    """Steady state on [r0, R] in um. Returns (r_um, o2_uM, no_nM).

    Symmetry condition dC/dr = 0 at r0, NOT the u = r*C substitution with u(r0) = 0. The
    latter is correct only at r = 0 exactly; at a finite inner node it FORCES C(r0) = 0 and the
    centre concentration -- the quantity this exercise is about -- comes back a hard zero.
    """
    r0 = R_um * 1e-3
    r = np.linspace(r0, R_um, n)

    def o2_rhs(rr, y, no_p):
        # COMPETITIVE inhibition (solver.py:734): NO raises the apparent Km of respiration.
        o2 = np.clip(y[0], 0.0, None)
        km_eff = KM_RESP_UM
        if no_p is not None:
            km_eff = KM_RESP_UM * (1.0 + np.clip(np.interp(rr, r, no_p), 0.0, None) / K_I_NM)
        return np.vstack([y[1], V_M_UM * o2 / (o2 + km_eff) / D_O2_UM - 2.0 * y[1] / rr])

    def o2_bc(ya, yb):
        return np.array([ya[1], yb[0] - o2_rim_um])

    def no_rhs(rr, y, o2_p):
        o2 = np.clip(np.interp(rr, r, o2_p), 0.0, None)
        # o2_gating=False removes the SPATIAL gradient in NOS2 output while leaving the amount
        # at the rim untouched: the source is held at its rim-oxygen value everywhere. Setting the
        # Michaelis fraction to 1 instead would raise production 1.75x (56/(56+42) = 0.571 -> 1.0)
        # and confound "gating removed" with "more NO everywhere". Removal stays O2-dependent in
        # every case -- only the SOURCE's oxygen dependence is ablated.
        src = prod_u(o2, f_nos2) if o2_gating else prod_u(o2_rim_um, f_nos2)
        return np.vstack([y[1],
                          (kdeg_u(o2) * y[0] - src) / D_NO_UM - 2.0 * y[1] / rr])

    def no_bc(ya, yb):
        return np.array([ya[1], yb[1]])

    if _guess is None:
        o2_g = np.vstack([np.full_like(r, o2_rim_um), np.zeros_like(r)])
        n0 = prod_u(o2_rim_um, f_nos2) / kdeg_u(o2_rim_um)
        no_g = np.vstack([np.full_like(r, n0), np.zeros_like(r)])
    else:
        gr, go2, gno = _guess
        o2v = np.interp(r, gr, go2)
        nov = np.interp(r, gr, gno)
        o2_g = np.vstack([o2v, np.gradient(o2v, r)])
        no_g = np.vstack([nov, np.gradient(nov, r)])

    no_p = None
    o2_c = no_c = None
    for _ in range(80 if respiratory_inhibition else 1):
        sa = _bvp(lambda rr, y: o2_rhs(rr, y, no_p), o2_bc, r, o2_g, tol)
        if not sa.success:
            raise RuntimeError('O2 solve failed at R=%.0f um: %s' % (R_um, sa.message))
        o2_c = np.clip(sa.sol(r)[0], 0.0, None)
        o2_g = np.vstack([o2_c, sa.sol(r)[1]])

        sb = _bvp(lambda rr, y: no_rhs(rr, y, o2_c), no_bc, r, no_g, tol)
        if not sb.success:
            raise RuntimeError('NO solve failed at R=%.0f um: %s' % (R_um, sb.message))
        no_new = np.clip(sb.sol(r)[0], 0.0, None)
        no_g = np.vstack([no_new, sb.sol(r)[1]])

        if not respiratory_inhibition:
            return r, o2_c, no_new
        if no_p is not None and np.max(np.abs(no_new - no_p)) <= 1e-6 * max(np.max(no_new), 1e-30):
            return r, o2_c, no_new
        no_p, no_c = no_new, no_new
    return r, o2_c, no_c


def describe(R_um, **kw):
    r, o2, no = solve(R_um, **kw)          # r um, o2 uM, no nM
    anox = r[o2 < ANOXIC_THRESHOLD * 1e6]
    r_core = anox.max() if anox.size else 0.0
    i_max = int(np.argmax(no))
    out = {
        'R_um': R_um, 'r_core_um': r_core,
        'no_centre_nM': no[0], 'no_rim_nM': no[-1],
        'no_max_nM': no[i_max], 'r_at_max_um': r[i_max],
        'o2_centre_uM': o2[0],
        'centre_over_rim': no[0] / no[-1],
        'max_over_rim': no[i_max] / no[-1],
    }
    if r_core > 0:
        j = int(np.argmin(np.abs(r - r_core)))
        out['no_coreedge_nM'] = no[j]
        out['centre_over_coreedge'] = no[0] / no[j]
    # is the profile monotonic from rim inward, or is there an interior maximum?
    # The local [O2] at the position of maximum [NO]. If this stays near the analytic
    # source-sink optimum sqrt(K_m * [O2]*) = sqrt(42 * 24.83) = 32.3 uM while r_max migrates
    # outward by hundreds of microns, the maximum is OXYGEN-PINNED, not position-pinned.
    out['o2_at_max_uM'] = o2[i_max]
    out['no_at_max_nM'] = no[i_max]
    out['o2_opt_uM'] = (KM_UM * C.O2_STAR_UM) ** 0.5
    # A maximum sitting ON a boundary is CONSTRAINED, not free, and must be excluded from the
    # oxygen-pinned statistic: in a small sphere the centre is the maximum only because the domain
    # stops there. Require the peak to be strictly interior by a margin, and to be a real peak.
    m = max(3, len(r) // 200)
    out['interior_max'] = bool(m <= i_max <= len(r) - 1 - m
                               and no[i_max] > no[0] and no[i_max] > no[-1])
    return out, (r, o2, no)


def main():
    print('AUTHORITATIVE PARAMETERS ACTUALLY USED')
    print('  K_m NOS2   %.3g M      V_max %.4g M/s' % (KM, VMAX))
    print('  rho        %.4g /mL    k_obs %.6g   k_anoxia %.6g' % (RHO, K_OBS, K_ANOX))
    print('  D_NO %.3g   D_O2 %.3g   V_M %.3g   Km_resp %.3g   K_I %.3g'
          % (D_NO, D_O2, V_M, KM_RESP, K_I))
    print('  sink: ADDITIVE.  rim [O2] %.0f uM.  anoxic def [O2] < %.1f uM'
          % (O2_RIM_DEFAULT * 1e6, ANOXIC_THRESHOLD * 1e6))

    print('\nPROFILE, one sphere')
    for R in (335.0,):
        for inhib in (True, False):
            d, _ = describe(R, respiratory_inhibition=inhib)
            print('  R = %.0f um (d = %.0f um), respiratory inhibition %s'
                  % (R, 2 * R, 'ON' if inhib else 'OFF'))
            print('    anoxic core radius      %8.1f um' % d['r_core_um'])
            print('    [O2] at centre          %8.4f uM' % d['o2_centre_uM'])
            print('    [NO] centre / rim / max %8.3f / %.3f / %.3f nM'
                  % (d['no_centre_nM'], d['no_rim_nM'], d['no_max_nM']))
            print('    location of the maximum %8.1f um   interior maximum: %s'
                  % (d['r_at_max_um'], d['interior_max']))
            print('    centre / rim            %8.3f' % d['centre_over_rim'])
            if 'centre_over_coreedge' in d:
                print('    centre / core edge      %8.3f  (core edge %.3f nM)'
                      % (d['centre_over_coreedge'], d['no_coreedge_nM']))

    print('\nDIAMETER SWEEP (respiratory inhibition ON)')
    print('   d um   core um   O2c uM    NOc nM    NOrim nM    NOmax nM  r_max um  c/rim')
    prev = None
    for D in (200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1400, 1600, 1800, 2000):
        # continuation: the anoxic interior is stiff and a flat initial guess exceeds the mesh
        # limit above ~500 um. Each solve starts from its converged neighbour.
        d, prof = describe(D / 2.0, _guess=prev)
        prev = prof
        print('  %5d  %7.1f  %8.4f  %8.3f  %9.3f  %10.3f  %8.1f  %6.3f'
              % (D, d['r_core_um'], d['o2_centre_uM'], d['no_centre_nM'],
                 d['no_rim_nM'], d['no_max_nM'], d['r_at_max_um'], d['centre_over_rim']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
