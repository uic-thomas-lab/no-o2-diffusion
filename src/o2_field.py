"""o2_field.py — Spatially Varying O2 Fields for the NO Diffusion Model

Provides O2 concentration arrays on the 3D simulation grid. Used only for
the Krogh cylinder figure (Fig 7), where k_deg and R_NO vary spatially.

For all other figures (Figs 1-6, 8-10), O2 is spatially uniform and this
module is not needed — k_deg and R_NO are scalars passed directly to the solver.

Two O2 field models:

  2. UNIFORM — Constant [O2] everywhere (used to generate the baseline scalar
     fields for comparison and to confirm that uniform-O2 runs match the
     superposition results).

Both return:
  o2_field    : ndarray float, shape grid_shape  [M]
  k_deg_field : ndarray float, shape grid_shape  [s^-1]
  r_no_field  : ndarray float, shape grid_shape  [M/s]  (NOS2+ cells only)

Units:
  O2 concentration : M  (moles/liter)
  Distances        : cm
  Rates            : s^-1 or M/s as labeled
"""

import numpy as np
from scipy.optimize import brentq
from kinetics import r_no as compute_r_no, k_degradation


# =============================================================================
# 0. Krogh radius self-consistency functions
# =============================================================================

def _krogh_cylinder_C(r, R_k, R_v, C_vessel, q, D_O2):
    """
    Analytical Krogh cylinder O₂ profile in cylindrical coordinates.

    Solves the steady-state PDE:  D∇²C = q  (uniform consumption)
    with BCs: C(R_v) = C_vessel, dC/dr(R_k) = 0

    Solution: C(r) = C_v + (q / 4D) × [r² − R_v² − 2R_k² ln(r/R_v)]
    """
    pref = q / (4.0 * D_O2)
    return C_vessel + pref * (r**2 - R_v**2 - 2.0 * R_k**2 * np.log(r / R_v))


def compute_krogh_radius(D_O2, C_vessel, V_M_tissue, R_v=5.0e-4, Km_resp=2.86e-6):
    """
    Compute Krogh radius from tissue O₂ consumption parameters.

    Uses the TRUE cylindrical Krogh solution:
        C(r) = C_v + (q/4D) × [r² − R_v² − 2R_k² ln(r/R_v)]
    where q = V_M_eff (constant consumption rate).
    R_K is defined as the radius where C(R_K) = 0.

    The previous slab-geometry formula R_K = sqrt(R_v² + 4DC/q) overestimates
    by ~2× for thin vessels (R_v << R_K) because it ignores the logarithmic
    term that dominates in cylindrical geometry.

    V_M_eff accounts for Michaelis-Menten saturation using the average O₂
    concentration (C_avg ≈ C_vessel/3) as a linearization point.

    Returns (R_K_cm, R_K_um, V_M_eff).
    """
    C_avg = C_vessel / 3.0
    mm_frac = C_avg / (C_avg + Km_resp) if Km_resp > 0 else 1.0
    V_M_eff = V_M_tissue * mm_frac
    q = V_M_eff

    # Solve C(R_K) = 0 numerically using Brent's method
    # Upper bound: slab-geometry estimate (guaranteed to be larger than true R_K)
    R_K_slab = np.sqrt(R_v**2 + 4.0 * D_O2 * C_vessel / q)

    def C_at_boundary(R_k):
        return _krogh_cylinder_C(R_k, R_k, R_v, C_vessel, q, D_O2)

    # Verify bracket: C should be positive near vessel, negative at slab R_K
    C_near = C_at_boundary(R_v * 1.01)
    C_far = C_at_boundary(R_K_slab)
    if C_near < 0 or C_far > 0:
        # Fallback to slab approximation if bracket fails (shouldn't happen)
        R_K = R_K_slab
    else:
        R_K = brentq(C_at_boundary, R_v * 1.01, R_K_slab, xtol=1e-8)

    return R_K, R_K * 1e4, V_M_eff


def tissue_V_M_from_krogh_radius(D_O2, C_vessel, R_K, R_v=5.0e-4, Km_resp=2.86e-6):
    """
    Reverse: compute tissue V_M implied by a given Krogh radius.

    Uses the cylindrical Krogh solution. Solves for q such that C(R_K) = 0:
        0 = C_v + (q/4D) × [R_K² − R_v² − 2R_K² ln(R_K/R_v)]
        q = -4D × C_v / [R_K² − R_v² − 2R_K² ln(R_K/R_v)]
    Then V_M_tissue = q / mm_frac.
    """
    bracket = R_K**2 - R_v**2 - 2.0 * R_K**2 * np.log(R_K / R_v)
    V_M_eff = -4.0 * D_O2 * C_vessel / bracket
    C_avg = C_vessel / 3.0
    mm_frac = C_avg / (C_avg + Km_resp) if Km_resp > 0 else 1.0
    V_M_tissue = V_M_eff / mm_frac
    return V_M_tissue


# =============================================================================
# 1. Uniform O2 field (baseline / comparison)
# =============================================================================

def uniform_o2_field(grid_shape, o2_conc, source_mask,
                     D, vmax, km, k_obs, cell_density):
    """
    Spatially uniform [O2] field — simplest case.

    Parameters
    ----------
    grid_shape : tuple (Nx, Ny, Nz)
    o2_conc : float  [M]  Uniform O2 concentration.
    source_mask : ndarray bool, shape grid_shape  NOS2+ cells (True = producer).
    D : float  [cm^2/s]  (not used here; included for API consistency with Krogh)
    vmax : float  [M/s]
    km : float  [M]
    k_obs : float  [mL/cell/M/s]
    cell_density : float  [cells/mL]

    Returns
    -------
    o2_field    : ndarray [M]   Constant o2_conc everywhere.
    k_deg_field : ndarray [s^-1]  Constant k_deg everywhere.
    r_no_field  : ndarray [M/s]  R_NO at source cells, 0 elsewhere.
    """
    o2_field = np.full(grid_shape, o2_conc, dtype=np.float64)

    k_deg_scalar = k_degradation(o2_conc, k_obs, cell_density)
    k_deg_field = np.full(grid_shape, k_deg_scalar, dtype=np.float64)

    r_no_scalar = compute_r_no(o2_conc, vmax, km)
    r_no_field = np.zeros(grid_shape, dtype=np.float64)
    r_no_field[source_mask] = r_no_scalar

    return o2_field, k_deg_field, r_no_field


# =============================================================================
# 2. Krogh cylinder O2 field
# =============================================================================

def krogh_cylinder_o2(grid_shape, dr, o2_vessel_M, o2_tissue_min_M=None,
                      vessel_radius_um=5.0, krogh_radius_um=None,
                      V_M_eff=None, D_O2=None):
    """
    Classical Krogh cylinder O2 distribution around a single capillary.

    The Krogh cylinder model (Krogh 1919) assumes:
    - A cylindrical capillary of radius R_v at the center of a tissue cylinder
      of radius R_k (the "Krogh radius" = half the intercapillary distance)
    - Steady-state radial O2 diffusion with uniform consumption
    - O2 concentration is highest at the vessel wall and falls to a minimum
      at R_k

    Analytical solution (cylindrical coordinates, ignoring axial variation):
        C(r) = C_v - (Q * r^2) / (4 * D_O2) + (Q * R_v^2) / (2 * D_O2) * ln(R_k/r)

    where:
        C_v  = O2 at vessel wall [M]
        Q    = O2 consumption rate per unit volume [M/s]
        D_O2 = O2 diffusion coefficient in tissue (~2e-5 cm^2/s)
        r    = radial distance from vessel axis [cm]
        R_v  = vessel radius [cm]
        R_k  = Krogh radius [cm]

    When V_M_eff and D_O2 are provided, the true cylindrical Krogh profile
    is used as the initial condition:
        C(r) = C_v + (q/4D) × [r² − R_v² − 2R_k² ln(r/R_v)]

    This is far more accurate than a linear interpolation and provides an
    initial condition much closer to the PDE steady state, dramatically
    reducing solver convergence time.

    Fallback: if V_M_eff and D_O2 are not provided, uses a linear
    interpolation (legacy behavior, NOT recommended).

    Parameters
    ----------
    grid_shape : tuple (Nx, Ny, Nz)
        Grid dimensions. Vessel runs along the z-axis through the grid center.
    dr : float  [cm]  Grid spacing.
    o2_vessel_M : float  [M]  O2 at the vessel wall (normoxia: 220e-6 M = 220 uM).
    o2_tissue_min_M : float or None  [M]
        O2 beyond the Krogh radius. Default: 1 uM (hypoxic threshold).
        Set to 0 for necrotic core boundary.
    vessel_radius_um : float  Capillary radius [um]. Default: 5 um.
    krogh_radius_um : float or None  Krogh radius [um].
        Default: half the grid width (R_max / 2).
    V_M_eff : float or None  [M/s]
        Effective O₂ consumption rate (after MM correction). If provided
        along with D_O2, the true cylindrical Krogh profile is used.
    D_O2 : float or None  [cm²/s]
        O₂ diffusion coefficient in tissue. Required for cylindrical profile.

    Returns
    -------
    o2_field : ndarray float, shape grid_shape  [M]
        Radially varying O2 concentration. Vessel axis = grid z-axis.
    r_field  : ndarray float, shape grid_shape  [cm]
        Radial distance from vessel axis (for external use/plotting).

    Notes
    -----
    Vessel runs along the z-axis (index dimension 2) through the xy-center.
    The NO simulation domain is (R_max)^3 = (1 mm)^3.
    A typical tumor capillary has R_v = 5 um and R_k = 100-200 um (based on
    intercapillary spacing of 200-400 um in tumor microvasculature).

    Reference: Secomb et al. 2004, Microcirculation 11:427-438, PMID: 15355001
    """
    if o2_tissue_min_M is None:
        o2_tissue_min_M = 1e-6     # 1 uM hypoxic threshold

    Nx, Ny, Nz = grid_shape
    R_v = vessel_radius_um * 1e-4  # um -> cm
    R_k = krogh_radius_um * 1e-4 if krogh_radius_um else (Nx // 2) * dr

    # Radial distance from the vessel axis (z-axis through xy-center)
    x = (np.arange(Nx) - Nx // 2) * dr
    y = (np.arange(Ny) - Ny // 2) * dr
    X, Y = np.meshgrid(x, y, indexing='ij')
    r_2d = np.sqrt(X**2 + Y**2)    # shape (Nx, Ny)

    # Broadcast to 3D (same profile at every z)
    r_field = np.broadcast_to(r_2d[:, :, np.newaxis],
                               grid_shape).copy()  # (Nx, Ny, Nz)

    if V_M_eff is not None and D_O2 is not None:
        # True cylindrical Krogh profile (analytical solution)
        q = V_M_eff
        # Avoid log(0) at r=0 by clamping r to R_v minimum
        r_safe = np.maximum(r_2d, R_v)
        pref = q / (4.0 * D_O2)
        o2_2d = o2_vessel_M + pref * (
            r_safe**2 - R_v**2 - 2.0 * R_k**2 * np.log(r_safe / R_v)
        )
        # Inside vessel: constant at vessel O2
        o2_2d = np.where(r_2d <= R_v, o2_vessel_M, o2_2d)
        # Beyond Krogh radius: clamp to minimum
        o2_2d = np.where(r_2d > R_k, o2_tissue_min_M, o2_2d)
    else:
        # Legacy linear interpolation (NOT recommended — poor initial condition)
        o2_2d = np.where(
            r_2d <= R_v,
            o2_vessel_M,
            np.where(
                r_2d <= R_k,
                o2_vessel_M * (R_k - r_2d) / (R_k - R_v)
                + o2_tissue_min_M * (r_2d - R_v) / (R_k - R_v),
                o2_tissue_min_M
            )
        )

    # Clamp: O2 cannot go below minimum or above vessel concentration
    o2_2d = np.clip(o2_2d, o2_tissue_min_M, o2_vessel_M)

    # Broadcast to 3D
    o2_field = np.broadcast_to(o2_2d[:, :, np.newaxis], grid_shape).copy()

    return o2_field, r_field


def krogh_cylinder_fields(grid_shape, dr, source_mask,
                          o2_vessel_M, o2_tissue_min_M,
                          vessel_radius_um, krogh_radius_um,
                          vmax, km, k_obs, cell_density):
    """
    Full spatially-varying field setup for the Krogh cylinder NO simulation.

    Combines krogh_cylinder_o2() with kinetics to produce the three arrays
    required by solve_3d_fd().

    Parameters
    ----------
    grid_shape, dr, source_mask : see source_distributions.py
    o2_vessel_M, o2_tissue_min_M, vessel_radius_um, krogh_radius_um :
        see krogh_cylinder_o2()
    vmax, km, k_obs, cell_density : kinetic parameters from parameters.py

    Returns
    -------
    o2_field    : ndarray [M]
    k_deg_field : ndarray [s^-1]   Spatially varying — high near vessel, low far
    r_no_field  : ndarray [M/s]    Spatially varying at source cells
    r_radial    : ndarray [cm]     Radial distance from vessel axis

    Notes
    -----
    The spatial variation of R_NO at source cells reflects O2-dependent NOS2
    kinetics: cells near the vessel produce more NO (higher O2, higher MM
    fraction) while cells far from the vessel produce less. This is the
    mechanism underlying the Krogh cylinder figure (Fig 7).

    k_deg_field reflects ALL cells consuming NO in an O2-dependent manner.
    """
    o2_field, r_radial = krogh_cylinder_o2(
        grid_shape, dr, o2_vessel_M, o2_tissue_min_M,
        vessel_radius_um, krogh_radius_um
    )

    # Spatially varying k_deg (all cells)
    k_deg_field = k_degradation(o2_field, k_obs, cell_density)

    # Spatially varying R_NO (NOS2+ cells only, O2-dependent)
    r_no_vals = compute_r_no(o2_field, vmax, km)
    r_no_field = np.zeros(grid_shape, dtype=np.float64)
    r_no_field[source_mask] = r_no_vals[source_mask]

    return o2_field, k_deg_field, r_no_field, r_radial


# =============================================================================
# Utility: classify grid cells into O2 zones
# =============================================================================

def classify_o2_zones(o2_field, normoxic_threshold=50e-6,
                      hypoxic_threshold=1e-6, severe_threshold=0.1e-6):
    """
    Classify each grid cell into an O2 zone.

    Parameters
    ----------
    o2_field : ndarray [M]
    normoxic_threshold : float  [M]  > this = normoxic (default 50 uM)
    hypoxic_threshold  : float  [M]  1-50 uM = moderately hypoxic
    severe_threshold   : float  [M]  0.1-1 uM = severely hypoxic; < 0.1 uM = necrotic

    Returns
    -------
    zones : ndarray int, same shape as o2_field
        0 = normoxic    (> 50 uM)
        1 = hypoxic     (1-50 uM)
        2 = severe      (0.1-1 uM)
        3 = necrotic    (< 0.1 uM)
    zone_fractions : dict  Fraction of domain in each zone.
    """
    zones = np.zeros(o2_field.shape, dtype=np.int8)
    zones[o2_field < normoxic_threshold] = 1
    zones[o2_field < hypoxic_threshold] = 2
    zones[o2_field < severe_threshold] = 3

    total = o2_field.size
    zone_fractions = {
        'normoxic':  float(np.sum(zones == 0)) / total,
        'hypoxic':   float(np.sum(zones == 1)) / total,
        'severe':    float(np.sum(zones == 2)) / total,
        'necrotic':  float(np.sum(zones == 3)) / total,
    }
    return zones, zone_fractions


# =============================================================================
# Validation
# =============================================================================

if __name__ == "__main__":
    from parameters import (
        D, DR, R_MAX, VMAX, KM, K_OBS, CELL_DENSITY,
        O2_ARRAY_SOLVER,
        O2_NORMOXIC_THRESHOLD, O2_HYPOXIC_THRESHOLD, O2_SEVERE_THRESHOLD,
    )
    from source_distributions import random_sources

    GRID = (100, 100, 100)

    print("=" * 65)
    print("o2_field.py — Validation")
    print("=" * 65)
    print(f"Grid: {GRID}  DR={DR*1e4:.0f} um  R_MAX={R_MAX*10:.0f} mm")
    print()

    # --- Uniform field test ---
    print("1. Uniform O2 field (50 uM)")
    source_mask, _, _, _ = random_sources(GRID, 0.10, VMAX, seed=42)
    o2_u, k_u, rno_u = uniform_o2_field(
        GRID, 50e-6, source_mask, D, VMAX, KM, K_OBS, CELL_DENSITY
    )
    print(f"   o2  : min={o2_u.min()*1e6:.1f}  max={o2_u.max()*1e6:.1f} uM  "
          f"(should both be 50.0)")
    print(f"   k_deg: min={k_u.min():.3f}  max={k_u.max():.3f} s^-1  "
          f"(should both be {k_u.min():.3f})")
    print(f"   R_NO at sources: {rno_u[source_mask].mean():.3e} M/s  "
          f"(nonzero={np.sum(rno_u > 0):,})")
    print()

    # --- Krogh cylinder field ---
    print("2. Krogh cylinder field")
    print("   Vessel: R_v=5 um at center,  Krogh radius=200 um")
    print("   O2: 220 uM at vessel -> 1 uM at Krogh radius")
    o2_k, r_k = krogh_cylinder_o2(
        GRID, DR,
        o2_vessel_M=220e-6, o2_tissue_min_M=1e-6,
        vessel_radius_um=5.0, krogh_radius_um=200.0
    )
    print(f"   O2 range: {o2_k.min()*1e6:.1f} - {o2_k.max()*1e6:.1f} uM")
    print(f"   At vessel center: {o2_k[50,50,50]*1e6:.1f} uM  (expect 220)")
    print(f"   At r=100um edge:  ", end="")
    # Find cell closest to r=100 um
    idx_100 = int(round(100e-4 / DR))
    print(f"{o2_k[50+idx_100, 50, 50]*1e6:.1f} uM  (expect ~110, midpoint)")
    print(f"   At r=200um edge:  {o2_k[50+int(round(200e-4/DR)), 50, 50]*1e6:.1f} uM  (expect ~1)")

    # Zone classification
    zones, fracs = classify_o2_zones(
        o2_k, O2_NORMOXIC_THRESHOLD, O2_HYPOXIC_THRESHOLD, O2_SEVERE_THRESHOLD
    )
    print(f"   Zone fractions: normoxic={fracs['normoxic']:.1%}  "
          f"hypoxic={fracs['hypoxic']:.1%}  "
          f"severe={fracs['severe']:.1%}  "
          f"necrotic={fracs['necrotic']:.1%}")
    print()

    # --- Full Krogh cylinder field assembly ---
    print("3. Full Krogh cylinder field assembly (for solve_3d_fd)")
    o2_f, k_f, rno_f, r_rad = krogh_cylinder_fields(
        GRID, DR, source_mask,
        o2_vessel_M=220e-6, o2_tissue_min_M=1e-6,
        vessel_radius_um=5.0, krogh_radius_um=200.0,
        vmax=VMAX, km=KM, k_obs=K_OBS, cell_density=CELL_DENSITY
    )
    print(f"   o2    range: {o2_f.min()*1e6:.1f} - {o2_f.max()*1e6:.1f} uM")
    print(f"   k_deg range: {k_f.min():.3f} - {k_f.max():.3f} s^-1")
    print(f"   R_NO  range at sources: "
          f"{rno_f[source_mask].min():.3e} - {rno_f[source_mask].max():.3e} M/s")
    print(f"   k_deg is spatially varying: "
          f"{'YES' if k_f.max() > k_f.min()*1.01 else 'NO (uniform)'}")
    print(f"   R_NO  is spatially varying at sources: "
          f"{'YES' if rno_f[source_mask].std() > 0 else 'NO'}")

    print()
    print("4. O2 gradient across the Krogh cylinder (radial profile)")
    print(f"   {'r (um)':>8}  {'O2 (uM)':>9}  {'k_deg (1/s)':>12}  {'Zone':>10}")
    print(f"   {'-'*45}")
    zone_labels = {0: 'normoxic', 1: 'hypoxic', 2: 'severe', 3: 'necrotic'}
    for r_um in (0, 25, 50, 100, 150, 200, 250, 300):
        r_idx = int(round(r_um * 1e-4 / DR))
        ix = min(50 + r_idx, 99)
        o2_val = o2_f[ix, 50, 50]
        k_val = k_f[ix, 50, 50]
        zone_val = int(zones[ix, 50, 50])
        print(f"   {r_um:>8}  {o2_val*1e6:>9.1f}  {k_val:>12.3f}  "
              f"{zone_labels[zone_val]:>10}")

    print()
    print("[OK] o2_field.py validation complete.")
