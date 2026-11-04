"""
parameters.py — NO Diffusion Model: All Physical Constants and Configuration

All values are literature-backed. Full documentation with PMIDs, methods, and
sensitivity analysis ranges is in data/PARAMETER_LITERATURE_REFERENCE.md.
Read that file before changing any value here.

Units convention:
  - Concentrations : M  (moles/liter)
  - Distances      : cm
  - Time           : s
  - Rates          : M/s
  - k_obs          : mL·cell⁻¹·M⁻¹·s⁻¹  (see CRITICAL NOTE below)
"""

import numpy as np


D = 2.2e-5  # cm²/s

from constants import (K_OBS_PUB as K_OBS, K_ANOXIA_PUB as K_ANOXIA,
                       K_OBS_CORR, K_ANOXIA_CORR, K_ANOXIA_RANGE,
                       VOLUME_CORRECTION, VOLUME_CORRECTION_DIVISOR, O2_STAR,
                       RHO_TUMOUR, PHI_TUMOUR, CELL_VOLUME_MODEL_UM3)

KM = 42e-6  # M  (42 µM, apparent Km, actively synthesising iNOS at low bulk NO)

VMAX = 1.718e-6  # M/s  (re-derived at KM = 42 µM and V_cell = 2079 µm³; Daw anchor exact)

CELL_DENSITY = RHO_TUMOUR  # cells/cm³ = cells/mL  -- single source of truth


DR = 0.001      # cm  (10 µm — approximately one cell diameter)
DT = 1.0e-4     # s   (time step; CFL condition for 3D: dt ≤ dr²/(6D) = 7.6e-3 s)
R_MAX = 0.1     # cm  (1 mm — biologically relevant diffusion scale)
T_MAX = 1200.0  # s   (steady state reached much earlier; solver exits early)

GRID_CELL_VOLUME_L = DR**3 * 1e-3  # L  (1e-12 L = 1 pL)


O2_ARRAY_FINE = np.logspace(np.log10(0.1e-6), np.log10(220e-6), 100)  # M

O2_ARRAY_SOLVER = np.array([0.1, 1, 10, 25, 50, 100, 130, 150, 220]) * 1e-6  # M

O2_NORMOXIC_THRESHOLD = 50e-6    # M  (>50 µM = normoxic)
O2_HYPOXIC_THRESHOLD = 1e-6      # M  (1–50 µM = hypoxic)
O2_SEVERE_THRESHOLD = 0.1e-6     # M  (<1 µM = severely hypoxic; <0.1 µM = necrotic)


SENSITIVITY = {
    "Vmax_M_s": (VMAX, 1.0e-7, 1.0e-5),

    "Km_M": (KM, 6.3e-6, 130e-6),

    "D_cm2_s": (2.2e-5, 0.85e-5, 2.2e-5),   # (nominal, low, high=nominal)

    "k_obs_mL_cell_M_s": (5.38e-4, 2.7e-4, 5.38e-4),   # (nominal, low, high=nominal)

    "cell_density_per_mL": (1.13e9, 0.5e9, 2.0e9),   # LEGACY -- superseded

    "k_anoxia_mL_cell_s": (K_ANOXIA,) + K_ANOXIA_RANGE,

    "nos2_fraction": (0.10, 0.05, 0.30),

    "K_I_M": (5.22e-9, 0.2e-9, 28e-9),

    "Km_resp_M": (2.86e-6, 0.5e-6, 5.0e-6),
}

KM_SENSITIVITY = np.array([15, 30, 130, 200, 350, 1000]) * 1e-6  # M
KM_SENSITIVITY_LABELS = [
    "15 µM\n(DNIC cell data)",
    "30 µM",
    "130 µM\n(purified enzyme)",
    "200 µM\n(Dweik in vivo)",
    "350 µM\n(Abu-Soud effective)",
    "1000 µM\n(max effective Km)",
]


SOLID_TUMOR = {
    "name": "Solid tumor (NOS2)",
    "NOS_isoform": "NOS2",
    "Km": 6.3e-6,           # M   Rengasamy & Johns 1996 (PMID: 8558447), intact cells
    "D": 2.2e-5,            # cm²/s  Santos 2011 (PMID: 21672575)
    "cell_density": 1.13e9, # cells/mL  Frieboes 2009 (PMID: 16452218)
    "Vmax": 3.09e-6,        # M/s  Daw 2025 (PMID: 40024431), MM-corrected at 220 uM
    "k_obs": 5.38e-4,       # mL·cell⁻¹·M⁻¹·s⁻¹  Thomas 2001 (PMID: 11134509)
    "V_M_tissue": 9.0e-6,     # M/s  tumor volumetric O₂ consumption (max; V_M_eff=7.8e-6 after MM)
    "D_O2": 1.39e-5,          # cm²/s  O₂ diffusion in tissue (Ellsworth & Pittman 1984, PMID 6696127)
    "Km_resp": 2.86e-6,       # M  (20/7 uM) Brown & Cooper 1994, PMID 7805858
    "K_I": 5.22e-9,           # M  NO competitive inhibition constant (Brown & Cooper 1994, PMID: 7805858)
}

VASCULAR_ENDOTHELIUM = {
    "name": "Vascular endothelium (NOS3)",
    "NOS_isoform": "NOS3",
    "Km": 4e-6,             # M   4 µM nominal (7.7 µM sensitivity)
    "D": 2.2e-5,            # cm²/s  Santos 2011 (PMID: 21672575), brain cortex in vivo -- NOMINAL
    "D_low_sensitivity": 0.85e-5,   # cm²/s  Liu 2008 aortic wall (PMID: 18032554).
    "cell_density": 1.0e8,  # cells/mL  (lower than solid tumor)
    "Vmax": 9.419e-7,       # M/s  effective NOS3 source amplitude (NOT calibrated, NOT measured)
    "vessel_pO2": 56e-6,    # M   perivascular [O₂] = 56 µM = 40 mmHg
    "k_obs": 5.38e-4,       # Assume same as Thomas 2001 (best available)
    "D_O2": 1.39e-5,        # cm²/s  O₂ diffusion coefficient in tissue
    "V_M": 2.5e-5,          # M/s    Maximum mitochondrial O₂ consumption rate
    "V_M_tissue": 5.6e-5,   # M/s    Perivascular tissue O₂ consumption (gives R_K=80 µm)
    "Km_resp": 2.86e-6,     # M      Mitochondrial Km for O₂ (2.86 µM)
    "K_I": 5.22e-9,          # M      NO competitive inhibition constant for cyt oxidase (5.22 nM)
}

NEURON = {
    "name": "Neuron (NOS1)",
    "NOS_isoform": "NOS1",
    "Km": 230e-6,           # M   230 µM nominal (190–350 µM sensitivity)
    "D": 2.2e-5,            # cm²/s  Santos 2011 (brain cortex)
    "cell_density": 1.0e8,  # cells/mL  (lower density than tumor)
    "k_obs": 5.38e-4,
}


PROTEIN_PER_CELL_PG = 200.0  # pg/cell  (range: 100–400 pg/cell)

CFL_DT_MAX_3D = DR**2 / (6.0 * D)  # s  ~7.6e-3 s   # UNUSED - derived diagnostic
CFL_DT_MAX_1D = DR**2 / (2.0 * D)  # s  ~2.3e-2 s   # UNUSED - derived diagnostic

NO_THRESHOLD_CGM_LOW_NM = 1.0     # nM - cGMP-mediated processes, band <1-30 nM; PMID 18439435
NO_THRESHOLD_ERK_NM = 10.0        # nM - ERK onset; PMID 18472020 (NOT 18439435, whose
NO_THRESHOLD_AKT_NM = 30.0        # nM - Akt phosphorylation, band 30-100 nM; PMID 18439435
NO_THRESHOLD_HIF1A_NM = 100.0     # nM - HIF-1α stabilization, band 100-300 nM; PMID 18439435
NO_THRESHOLD_P53_NM = 400.0       # nM — p53 Ser15 phosphorylation, cytostatic
NO_THRESHOLD_CYTOTOXIC_NM = 1000.0  # nM — cytotoxic / apoptotic


if __name__ == "__main__":
    print("=" * 60)
    print("NO Diffusion Model — Parameter Summary")
    print("=" * 60)
    print(f"\nCore physical constants:")
    print(f"  D         = {D:.2e} cm²/s")
    print(f"  k_obs     = {K_OBS:.2e} mL·cell⁻¹·M⁻¹·s⁻¹")
    print(f"  Km (NOS2) = {KM*1e6:.0f} µM")
    print(f"  Vmax      = {VMAX:.2e} M/s")
    print(f"  cell_density = {CELL_DENSITY:.2e} cells/mL")

    print(f"\nGrid parameters:")
    print(f"  dr = {DR*1e4:.0f} µm,  dt = {DT:.1e} s,  r_max = {R_MAX*10:.0f} mm")
    print(f"  Grid cell volume = {GRID_CELL_VOLUME_L:.2e} L ({GRID_CELL_VOLUME_L*1e12:.2f} pL)")

    print(f"\nCFL stability limits:")
    print(f"  1D: dt_max = {CFL_DT_MAX_1D:.2e} s  (current dt is {CFL_DT_MAX_1D/DT:.0f}× smaller)")
    print(f"  3D: dt_max = {CFL_DT_MAX_3D:.2e} s  (current dt is {CFL_DT_MAX_3D/DT:.0f}× smaller)")

    print(f"\nO2 arrays:")
    print(f"  Fine  : {len(O2_ARRAY_FINE)} log-spaced points, "
          f"{O2_ARRAY_FINE[0]*1e6:.2f}–{O2_ARRAY_FINE[-1]*1e6:.0f} µM")
    print(f"  Solver: {len(O2_ARRAY_SOLVER)} points — "
          f"{[f'{v*1e6:.1f}' for v in O2_ARRAY_SOLVER]} µM")

    print(f"\nSensitivity ranges:")
    for key, (central, low, high) in SENSITIVITY.items():
        print(f"  {key:30s}: {low:.2e} – {high:.2e}  (central: {central:.2e})")
