# -*- coding: utf-8 -*-
"""Export the production physics to JSON so the Wolfram FEM validator shares CONSTANTS but not CODE.

THIS IS NOT A TRANSLATION LAYER. It exports numbers only -- no matrices, no stencils, no
solution, no discretisation choices. If it ever starts exporting anything the Wolfram side could
use to reproduce the Python DISCRETISATION, the cross-check has been defeated.

Parameter errors are NOT what this cross-check tests. They are covered by the parameter lock and
the citation verifier. This tests whether the 2-D vascular problem has been FORMULATED correctly --
source placement, coupling signs, boundary conditions, compartment definitions.

Out: data/fem_reference_constants.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import vascular_field as V     # noqa: E402
import spheroid_field as SF    # noqa: E402

# The three vascular states the FEM validator certifies. Chosen to bracket the CLAIM, not to be
# convenient: sparse, the apparent maximum, and dense where the turnover occurs. If the two solvers
# agree on the ORDERING mean_NO(sparse) < mean_NO(intermediate) > mean_NO(dense), the intermediate
# optimum cannot easily be blamed on discretisation.
# Six vascularised states. The original three bracketed the claim; 250, 100 and 75 were added so
# the headline curve is drawn entirely in the production formulation rather than mixing a biased
# Cartesian line with three FEM points. Three points cannot distinguish a sharp 150 um maximum from
# a broad 100-150 um plateau, and that distinction decides whether Results says "150 um spacing",
# "a broad intermediate optimum", or "an intermediate vascular density".
STATES = [
    {'name': 'sparse', 'spacing_um': 400.0},
    {'name': 's250', 'spacing_um': 250.0},
    {'name': 'intermediate', 'spacing_um': 150.0},   # apparent mean-NO maximum
    {'name': 's100', 'spacing_um': 100.0},
    {'name': 's75', 'spacing_um': 75.0},
    {'name': 'dense', 'spacing_um': 60.0},           # turnover region
]
R_UM = 425.0
F_NOS2 = 0.02          # production fraction
ENOS = True            # production model
K_LUMEN = V.K_LUMEN_DEFAULT


def lattice(spacing, R):
    n = int(np.ceil(R / spacing)) + 1
    return [[(i + 0.5) * spacing, (j + 0.5) * spacing]
            for i in range(-n, n + 1) for j in range(-n, n + 1)
            if np.hypot((i + 0.5) * spacing, (j + 0.5) * spacing)
            <= R - 2.5 * V.VESSEL_RADIUS_UM]


def main():
    doc = {
        '_README': 'Constants for the independent Wolfram FEM validator. Physics only.',
        'param_set': V.parameter_set_hash(),
        'units': {'length': 'um', 'O2': 'uM', 'NO': 'nM', 'time': 's'},

        # --- transport -------------------------------------------------------------------
        'D_NO_um2_per_s': float(V.D_NO_UM),
        'D_O2_um2_per_s': float(V.D_O2_UM),

        # --- oxygen sink: competitive inhibition of cytochrome c oxidase by NO -----------
        # v_R = V_M * c / (c + Km_resp * (1 + n / K_I))
        'V_M_uM_per_s': float(V.V_M_UM),
        'Km_resp_uM': float(V.KM_RESP_UM),
        'K_I_nM': float(V.K_I_NM),

        # --- NOS2 source (tissue):  S = Vmax * c/(c+Km) * phi * f_nos2  [nM/s] -----------
        'Vmax_NOS2_nM_per_s': float(V.VMAX_NM),
        'Km_NOS2_uM': float(V.KM_UM),
        'phi': float(V.PHI),

        # --- NOS3 source (endothelial shell): Vmax3 * c/(c+Km3) [nM/s] within the shell --
        'Vmax_NOS3_nM_per_s': float(V.VMAX3_NM),
        'Km_NOS3_uM': float(V.KM3_UM),
        'shell_um': float(V.SHELL_UM),

        # --- NO tissue sink, ADDITIVE, NOT the max() form --------------------------
        # k_deg(c) = K_OBS * (c * 1e-6) * RHO_CELLS + K_ANOX * RHO_CELLS   [1/s], c in uM
        'K_OBS': float(SF.K_OBS),
        'K_ANOX': float(SF.K_ANOX),
        'RHO_CELLS': float(SF.RHO_CELLS),
        # echoed at three oxygen tensions so the Wolfram side can gate its own transcription
        'kdeg_check': {('%g' % c): float(SF.kdeg_u(c)) for c in (0.0, 32.3, 56.0)},
        'prod_check': {('%g' % c): float(SF.prod_u(c, F_NOS2)) for c in (0.0, 32.3, 56.0)},

        # --- luminal NO uptake -----------------------------------------------------------
        'k_lumen_per_s': float(K_LUMEN),

        # --- geometry and boundary conditions --------------------------------------------
        'R_um': R_UM,
        'vessel_radius_um': float(V.VESSEL_RADIUS_UM),
        'O2_rim_uM': float(V.O2_RIM_UM),      # Dirichlet, outer tumour circumference
        'O2_wall_uM': float(V.O2_WALL_UM),    # Dirichlet, every vessel wall
        # NO has ZERO FLUX at the outer rim and the lumen is PART of the NO domain
        # (k = k_lumen, no source) -- it is NOT a boundary condition.
        'NO_outer_bc': 'zero_flux',
        'NO_domain_includes_lumen': True,

        # --- what is being certified ------------------------------------------------------
        'f_nos2': F_NOS2,
        'enos': ENOS,
        'O2_opt_uM': float(V.O2_OPT_UM),
        'sGC_threshold_nM': 1.0,
        'anoxic_threshold_uM': 0.1,
        'states': [dict(s, vessels=lattice(s['spacing_um'], R_UM)) for s in STATES],
    }
    for s in doc['states']:
        s['n_vessels'] = len(s['vessels'])
    out = os.path.join(ROOT, 'data', 'fem_reference_constants.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1)
    print('wrote %s' % out)
    print('  param set %s   f_NOS2 %.2f   eNOS %s   k_lumen %.3g /s'
          % (doc['param_set'], F_NOS2, ENOS, K_LUMEN))
    for s in doc['states']:
        print('  %-13s spacing %6.0f um   %4d vessels' % (s['name'], s['spacing_um'],
                                                          s['n_vessels']))
    print('  kdeg(0/32.3/56 uM) = %s' % doc['kdeg_check'])
    print('  prod(0/32.3/56 uM) = %s' % doc['prod_check'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
