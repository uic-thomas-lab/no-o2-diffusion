"""constants.py: the single definition of the NO-removal sink, its constants and its form.

Units: [O2] in M, rho in cells/mL, k_obs in mL·cell⁻¹·M⁻¹·s⁻¹, k_anoxia in
mL·cell⁻¹·s⁻¹, k_deg in s⁻¹.

    k_obs    -> k_obs    × (2079 / 6186)
    k_anoxia -> k_anoxia × (2079 / 6186)     = × 0.33608, i.e. ÷ 2.9755  ("÷2.98")

TUMOUR ARM ONLY. The correction applies where rho = RHO_TUMOUR = 2.02e8 cells/mL.
The perivascular / NOS3 arm carries rho = 1.0e8 — the density the constants were
published against — and is UNAFFECTED. Check rho per file before applying it.
"""


CELL_VOLUME_MODEL_UM3 = 2079.0      # µm³  measured carcinoma cell, n=37

PHI_TUMOUR = 0.42                   # dimensionless, range 0.30-0.55
CELL_VOLUME_REFERENT_UM3 = 6186.0   # µm³  Drochmans 1975, PMID 167028, Table I, unfixed

VOLUME_CORRECTION = CELL_VOLUME_MODEL_UM3 / CELL_VOLUME_REFERENT_UM3  # 0.3360815...
VOLUME_CORRECTION_DIVISOR = 1.0 / VOLUME_CORRECTION                   # 2.9754690...

VOLUME_CORRECTION_DIVISOR_RANGE = (2.0815, 5.2154)


K_OBS_PUB = 5.38e-4     # mL·cell⁻¹·M⁻¹·s⁻¹  Thomas 2001 PNAS 98:355-360, PMID 11134509
K_ANOXIA_PUB = 1.336e-8  # mL·cell⁻¹·s⁻¹      Thomas 2001 re-analysis

K_ANOXIA_RANGE = (8.3e-9, 2.152e-8)


K_OBS_CORR = K_OBS_PUB * VOLUME_CORRECTION        # 1.808118e-4
K_ANOXIA_CORR = K_ANOXIA_PUB * VOLUME_CORRECTION  # 4.490048e-9


RHO_TUMOUR = 2.02e8        # cells/mL  = PHI_TUMOUR / CELL_VOLUME_MODEL_UM3
RHO_PERIVASCULAR = 1.0e8   # cells/mL  NOS3 arm — constants used UNcorrected here


O2_STAR = K_ANOXIA_PUB / K_OBS_PUB   # M
O2_STAR_UM = O2_STAR * 1e6           # 24.83 µM  (was 22.7 at k_anoxia = 1.22e-8)



_phi_implied = RHO_TUMOUR * CELL_VOLUME_MODEL_UM3 / 1e12
assert abs(_phi_implied - PHI_TUMOUR) / PHI_TUMOUR < 5e-3, (
    f"COUPLING BROKEN: rho x V_cell = {_phi_implied:.4f} but PHI_TUMOUR = {PHI_TUMOUR}. "
    f"These three are not independent (phi = rho x V_cell); changing one requires changing "
    f"another.")
assert _phi_implied <= 1.0, (
    f"PHYSICALLY IMPOSSIBLE: rho x V_cell = {_phi_implied:.4f} > 1, i.e. more than 100% of "
    f"tissue volume is cell.")

_gm = (K_ANOXIA_RANGE[0] * K_ANOXIA_RANGE[1]) ** 0.5
assert abs(K_ANOXIA_PUB - _gm) / _gm < 1e-3, (
    f"COUPLING BROKEN: K_ANOXIA_PUB={K_ANOXIA_PUB:.4e} is no longer the geometric mean "
    f"of K_ANOXIA_RANGE ({_gm:.4e}). The nominal is defined AS that mean; if a "
    f"determination moved, the nominal must move with it.")

TUMOUR = dict(k_obs=K_OBS_CORR, k_anoxia=K_ANOXIA_CORR, cell_density=RHO_TUMOUR)
PERIVASCULAR = dict(k_obs=K_OBS_PUB, k_anoxia=K_ANOXIA_PUB, cell_density=RHO_PERIVASCULAR)


def k_deg(o2, k_obs, cell_density, k_anoxia, _check=True):
    """First-order NO removal rate constant [s⁻¹] — the ADDITIVE two-route sink.

        k_deg = k_obs·[O2]·rho + k_anoxia·rho

    This is the single definition of the sink form. Nothing may reimplement it.
    See the module docstring: additive is a plotting convention, not a
    mechanistic finding.
    """
    if _check:
        corr = abs(k_obs / K_OBS_CORR - 1) < 1e-6
        pub = abs(k_obs / K_OBS_PUB - 1) < 1e-6
        if corr or pub:
            want_rho = RHO_TUMOUR if corr else RHO_PERIVASCULAR
            arm = "CORRECTED/tumour" if corr else "PUBLISHED/perivascular"
            if abs(cell_density / want_rho - 1) > 1e-6:
                raise ValueError(
                    f"MIXED BASIS: {arm} k_obs passed with cell_density={cell_density:.3e}, "
                    f"expected {want_rho:.3e}. This is a ~{RHO_TUMOUR/RHO_PERIVASCULAR:.0f}x "
                    f"error that would still return a plausible value. Use **TUMOUR or "
                    f"**PERIVASCULAR, or pass _check=False if this is deliberate.")
            want_ka = K_ANOXIA_CORR if corr else K_ANOXIA_PUB
            if abs(k_anoxia / want_ka - 1) > 1e-6:
                raise ValueError(
                    f"MIXED BASIS: {arm} k_obs passed with k_anoxia={k_anoxia:.4e}, "
                    f"expected {want_ka:.4e}. Both sink constants carry the same volume "
                    f"correction or neither does. Use **TUMOUR or **PERIVASCULAR.")
    return k_obs * o2 * cell_density + k_anoxia * cell_density


J_NOS3_SHELL_UM = 0.8        # um   the numerical source shell -- implementation choice
