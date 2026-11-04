"""source_distributions.py — NOS2+ Cell Placement on the Simulation Grid

Generates the r_no_field array (volumetric NO production rate [M/s]) for
input to solver.py. Nonzero only at NOS2+ source cells; zero everywhere else.

Five distribution modes:
  1. single       — one source cell at the grid center
  2. uniform_grid — NOS2+ cells on a regular lattice (spacing = 1/cbrt(fraction))
  3. random       — NOS2+ cells placed by random sampling (reproducible seed)
  4. clustered    — NOS2+ cells grouped into one or more spatial clusters
  5. shell        — NOS2+ cells confined to a spherical shell (vascular rim model)

All functions return:
  source_mask : ndarray bool, shape grid_shape  True at NOS2+ cells
  r_no_field  : ndarray float, shape grid_shape  R_NO [M/s] at each cell

The NOS2+ fraction is a model input; the values used are stated with each figure.
"""

import numpy as np


# =============================================================================
# Grid coordinate helpers
# =============================================================================

def make_grid_coords(grid_shape, dr):
    """
    Build 3D coordinate arrays for a cubic grid.

    Parameters
    ----------
    grid_shape : tuple (Nx, Ny, Nz)
    dr : float  [cm]  Grid spacing.

    Returns
    -------
    X, Y, Z : ndarray, each shape grid_shape  [cm]
        Coordinates centered at the grid center.
    """
    Nx, Ny, Nz = grid_shape
    x = (np.arange(Nx) - Nx // 2) * dr
    y = (np.arange(Ny) - Ny // 2) * dr
    z = (np.arange(Nz) - Nz // 2) * dr
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z


def grid_center_idx(grid_shape):
    """Return the grid index of the center cell."""
    return tuple(s // 2 for s in grid_shape)


# =============================================================================
# 1. Single source
# =============================================================================

def single_source(grid_shape, r_no_rate):
    """
    One NOS2+ cell at the center of the grid.

    Used for all 1D radial validation runs and single-cell figures.

    Parameters
    ----------
    grid_shape : tuple (Nx, Ny, Nz)
    r_no_rate : float  [M/s]  Volumetric NO production rate.

    Returns
    -------
    source_mask : ndarray bool, shape grid_shape
    r_no_field  : ndarray float, shape grid_shape  [M/s]
    """
    source_mask = np.zeros(grid_shape, dtype=bool)
    cx, cy, cz = grid_center_idx(grid_shape)
    source_mask[cx, cy, cz] = True

    r_no_field = np.zeros(grid_shape, dtype=np.float64)
    r_no_field[source_mask] = r_no_rate

    return source_mask, r_no_field


# =============================================================================
# 2. Uniform grid of NOS2+ cells
# =============================================================================

def uniform_grid_sources(grid_shape, dr, nos2_fraction, r_no_rate, seed=42):
    """
    NOS2+ cells on a regular 3D lattice with spacing chosen to match nos2_fraction.

    The lattice spacing s is chosen so that (1/s)^3 = nos2_fraction / dr^3.
    Since each cell has volume dr^3 and one source per s^3 volume:
        fraction = (dr/s)^3  =>  s = dr / cbrt(fraction)

    Parameters
    ----------
    grid_shape : tuple (Nx, Ny, Nz)
    dr : float  [cm]
    nos2_fraction : float  Fraction of grid cells that are NOS2+ (0-1).
    r_no_rate : float  [M/s]
    seed : int  Random seed (not used here, kept for API consistency).

    Returns
    -------
    source_mask : ndarray bool
    r_no_field  : ndarray float  [M/s]
    n_sources   : int  Actual number of source cells placed.
    actual_fraction : float  Actual NOS2+ fraction achieved.
    """
    Nx, Ny, Nz = grid_shape
    # Lattice step in grid units (cells between sources)
    step = max(1, int(round(nos2_fraction ** (-1.0 / 3.0))))

    source_mask = np.zeros(grid_shape, dtype=bool)
    # Offset by step//2 to center the lattice
    offset = step // 2
    source_mask[offset::step, offset::step, offset::step] = True

    r_no_field = np.zeros(grid_shape, dtype=np.float64)
    r_no_field[source_mask] = r_no_rate

    n_sources = int(np.sum(source_mask))
    actual_fraction = n_sources / float(Nx * Ny * Nz)

    return source_mask, r_no_field, n_sources, actual_fraction


# =============================================================================
# 3. Random (Bernoulli) distribution
# =============================================================================

def random_sources(grid_shape, nos2_fraction, r_no_rate, seed=42):
    """
    NOS2+ cells placed by independent Bernoulli sampling.

    Each grid cell is independently assigned NOS2+ with probability nos2_fraction.
    This models spatially uncorrelated expression across the tumor.

    Parameters
    ----------
    grid_shape : tuple (Nx, Ny, Nz)
    nos2_fraction : float  Expected fraction of NOS2+ cells (0-1).
    r_no_rate : float  [M/s]
    seed : int  NumPy random seed for reproducibility.

    Returns
    -------
    source_mask : ndarray bool
    r_no_field  : ndarray float  [M/s]
    n_sources   : int  Actual number of source cells.
    actual_fraction : float
    """
    rng = np.random.default_rng(seed)
    source_mask = rng.random(grid_shape) < nos2_fraction

    r_no_field = np.zeros(grid_shape, dtype=np.float64)
    r_no_field[source_mask] = r_no_rate

    n_sources = int(np.sum(source_mask))
    actual_fraction = n_sources / float(np.prod(grid_shape))

    return source_mask, r_no_field, n_sources, actual_fraction


# =============================================================================
# 4. Clustered distribution
# =============================================================================

def clustered_sources(grid_shape, dr, n_clusters, nos2_fraction,
                      r_no_rate, cluster_radius_um=50.0, seed=42):
    """
    NOS2+ cells grouped into Gaussian clusters.

    Models tumor regions where NOS2 expression is induced by local
    inflammatory signals (e.g., hypoxic niches, macrophage infiltration).

    Parameters
    ----------
    grid_shape : tuple (Nx, Ny, Nz)
    dr : float  [cm]
    n_clusters : int  Number of cluster centers.
    nos2_fraction : float  Target overall NOS2+ fraction (determines cells per cluster).
    r_no_rate : float  [M/s]
    cluster_radius_um : float  Gaussian sigma of each cluster [um].
    seed : int

    Returns
    -------
    source_mask : ndarray bool
    r_no_field  : ndarray float  [M/s]
    n_sources   : int
    actual_fraction : float
    cluster_centers : ndarray, shape (n_clusters, 3)  Centers in grid index units.
    """
    rng = np.random.default_rng(seed)
    Nx, Ny, Nz = grid_shape
    total_cells = Nx * Ny * Nz
    n_total_sources = int(round(nos2_fraction * total_cells))
    n_per_cluster = max(1, n_total_sources // n_clusters)

    sigma_cells = cluster_radius_um * 1e-4 / dr   # convert um -> grid cells

    # Random cluster centers (avoid edges by sigma margin)
    margin = int(2 * sigma_cells)
    cluster_centers = np.column_stack([
        rng.integers(margin, Nx - margin, n_clusters),
        rng.integers(margin, Ny - margin, n_clusters),
        rng.integers(margin, Nz - margin, n_clusters),
    ])

    source_mask = np.zeros(grid_shape, dtype=bool)

    for cx, cy, cz in cluster_centers:
        # Sample n_per_cluster cells from Gaussian distribution around center
        offsets = rng.normal(0, sigma_cells, (n_per_cluster * 4, 3)).astype(int)
        candidates = np.column_stack([
            cx + offsets[:, 0],
            cy + offsets[:, 1],
            cz + offsets[:, 2],
        ])
        # Keep only in-bounds candidates
        valid = (
            (candidates[:, 0] >= 0) & (candidates[:, 0] < Nx) &
            (candidates[:, 1] >= 0) & (candidates[:, 1] < Ny) &
            (candidates[:, 2] >= 0) & (candidates[:, 2] < Nz)
        )
        candidates = candidates[valid]
        # Deduplicate and take up to n_per_cluster
        seen = set()
        for idx in candidates:
            key = tuple(idx)
            if key not in seen:
                seen.add(key)
                source_mask[idx[0], idx[1], idx[2]] = True
            if len(seen) >= n_per_cluster:
                break

    r_no_field = np.zeros(grid_shape, dtype=np.float64)
    r_no_field[source_mask] = r_no_rate

    n_sources = int(np.sum(source_mask))
    actual_fraction = n_sources / float(total_cells)

    return source_mask, r_no_field, n_sources, actual_fraction, cluster_centers


# =============================================================================
# 5. Shell distribution (vascular rim model)
# =============================================================================

def shell_sources(grid_shape, dr, nos2_fraction, r_no_rate,
                  shell_inner_um=200.0, shell_outer_um=400.0, seed=42):
    """
    NOS2+ cells in a spherical shell (vascular rim model).

    Models the observation that NOS2 expression peaks in the intermediate
    hypoxic zone — not at the vessel (normoxic) and not in the necrotic core,
    but in the 200-400 um shell where O2 is limiting enough to induce NOS2
    but not so low as to kill cells.

    Parameters
    ----------
    grid_shape : tuple (Nx, Ny, Nz)
    dr : float  [cm]
    nos2_fraction : float  Fraction of cells WITHIN the shell that are NOS2+.
    r_no_rate : float  [M/s]
    shell_inner_um : float  Inner radius of shell [um].
    shell_outer_um : float  Outer radius of shell [um].
    seed : int

    Returns
    -------
    source_mask : ndarray bool
    r_no_field  : ndarray float  [M/s]
    n_sources   : int
    actual_fraction : float  (fraction of ALL grid cells that are NOS2+)
    """
    rng = np.random.default_rng(seed)
    Nx, Ny, Nz = grid_shape

    r_inner = shell_inner_um * 1e-4   # um -> cm
    r_outer = shell_outer_um * 1e-4

    X, Y, Z = make_grid_coords(grid_shape, dr)
    R = np.sqrt(X**2 + Y**2 + Z**2)

    # Shell mask: all cells in the shell
    shell_mask = (R >= r_inner) & (R <= r_outer)

    # Randomly assign nos2_fraction of shell cells as NOS2+
    shell_indices = np.argwhere(shell_mask)
    n_shell = len(shell_indices)
    n_nos2 = int(round(nos2_fraction * n_shell))

    chosen = rng.choice(n_shell, size=n_nos2, replace=False)
    source_mask = np.zeros(grid_shape, dtype=bool)
    for idx in shell_indices[chosen]:
        source_mask[tuple(idx)] = True

    r_no_field = np.zeros(grid_shape, dtype=np.float64)
    r_no_field[source_mask] = r_no_rate

    n_sources = int(np.sum(source_mask))
    actual_fraction = n_sources / float(Nx * Ny * Nz)

    return source_mask, r_no_field, n_sources, actual_fraction


# =============================================================================
# Utility: convert 3D source mask to 2D projected list for superposition_2d
# =============================================================================

def sources_to_xy(source_mask, dr, z_slice=None):
    """
    Extract (x, y) coordinates of NOS2+ cells from a 3D source mask.

    For use with solver.superposition_2d(). Projects a single z-slice
    (default: center slice) to get the 2D source positions.

    Parameters
    ----------
    source_mask : ndarray bool, shape (Nx, Ny, Nz)
    dr : float  [cm]
    z_slice : int or None  Z-index to extract. Default: center slice.

    Returns
    -------
    sources_xy : ndarray, shape (N, 2)  [cm]
        (x, y) coordinates of source cells in the selected z-slice.
    """
    Nx, Ny, Nz = source_mask.shape
    if z_slice is None:
        z_slice = Nz // 2

    x_coords = (np.arange(Nx) - Nx // 2) * dr
    y_coords = (np.arange(Ny) - Ny // 2) * dr

    ix, iy = np.where(source_mask[:, :, z_slice])
    sources_xy = np.column_stack([x_coords[ix], y_coords[iy]])
    return sources_xy


# =============================================================================
# Utility: summary stats for a source distribution
# =============================================================================

def source_stats(source_mask, dr):
    """
    Compute summary statistics for a source distribution.

    Returns dict with:
        n_sources       : int    Number of NOS2+ cells
        fraction        : float  NOS2+ fraction
        mean_spacing_um : float  Mean nearest-neighbor distance [um]
        max_spacing_um  : float  Maximum nearest-neighbor distance [um]
    """
    positions = np.argwhere(source_mask).astype(float) * dr * 1e4  # um

    n = len(positions)
    if n < 2:
        return {'n_sources': n, 'fraction': n / float(np.prod(source_mask.shape)),
                'mean_spacing_um': np.nan, 'max_spacing_um': np.nan}

    # Nearest-neighbor distances (subsample if n > 500 for speed)
    if n > 500:
        idx = np.random.choice(n, 500, replace=False)
        pos_sample = positions[idx]
    else:
        pos_sample = positions

    nn_dists = []
    for i, p in enumerate(pos_sample):
        diffs = np.linalg.norm(pos_sample - p, axis=1)
        diffs[i] = np.inf
        nn_dists.append(np.min(diffs))

    return {
        'n_sources':       n,
        'fraction':        n / float(np.prod(source_mask.shape)),
        'mean_spacing_um': float(np.mean(nn_dists)),
        'max_spacing_um':  float(np.max(nn_dists)),
    }


# =============================================================================
# Validation
# =============================================================================

if __name__ == "__main__":
    from parameters import DR, VMAX, KM, K_OBS, CELL_DENSITY
    from kinetics import r_no, k_degradation

    # Use a small grid for fast validation
    GRID = (50, 50, 50)
    O2_TEST = 50e-6    # 50 uM — moderate hypoxia
    RNO = r_no(O2_TEST, VMAX, KM)
    NOS2_FRAC = 0.10   # 10% NOS2+

    print("=" * 65)
    print("source_distributions.py — Validation")
    print("=" * 65)
    print(f"Grid: {GRID}  DR={DR*1e4:.0f} um  O2={O2_TEST*1e6:.0f} uM  "
          f"NOS2_frac={NOS2_FRAC:.0%}  R_NO={RNO:.3e} M/s")
    print()

    # --- 1. Single source ---
    mask, field = single_source(GRID, RNO)
    print(f"1. single:        n={np.sum(mask):5d}  "
          f"fraction={np.sum(mask)/np.prod(GRID):.5f}  "
          f"R_NO_max={np.max(field):.3e} M/s")

    # --- 2. Uniform grid ---
    mask, field, n, frac = uniform_grid_sources(GRID, DR, NOS2_FRAC, RNO)
    print(f"2. uniform_grid:  n={n:5d}  fraction={frac:.5f}  "
          f"(target {NOS2_FRAC:.3f})")

    # --- 3. Random ---
    mask, field, n, frac = random_sources(GRID, NOS2_FRAC, RNO, seed=42)
    stats = source_stats(mask, DR)
    print(f"3. random:        n={n:5d}  fraction={frac:.5f}  "
          f"nn_mean={stats['mean_spacing_um']:.1f} um")

    # --- 4. Clustered ---
    mask, field, n, frac, centers = clustered_sources(
        GRID, DR, n_clusters=3, nos2_fraction=NOS2_FRAC, r_no_rate=RNO,
        cluster_radius_um=40.0, seed=42
    )
    print(f"4. clustered:     n={n:5d}  fraction={frac:.5f}  "
          f"3 clusters @ radius=40 um")

    # --- 5. Shell ---
    mask, field, n, frac = shell_sources(
        GRID, DR, NOS2_FRAC, RNO,
        shell_inner_um=100.0, shell_outer_um=200.0, seed=42
    )
    print(f"5. shell:         n={n:5d}  fraction={frac:.5f}  "
          f"(shell 100-200 um)")

    print()
    print("--- sources_to_xy (center z-slice from random distribution) ---")
    mask_r, _, _, _ = random_sources(GRID, NOS2_FRAC, RNO, seed=42)
    xy = sources_to_xy(mask_r, DR)
    print(f"   Sources in center z-slice: {len(xy)}")
    print(f"   x range: {xy[:,0].min()*1e4:.0f} to {xy[:,0].max()*1e4:.0f} um")
    print(f"   y range: {xy[:,1].min()*1e4:.0f} to {xy[:,1].max()*1e4:.0f} um")

    print()
    print("--- Fraction accuracy across NOS2 fractions (random) ---")
    print(f"{'Target':>10}  {'Actual':>10}  {'Error %':>9}")
    for frac_target in (0.05, 0.10, 0.20, 0.30):
        _, _, n_s, frac_actual = random_sources(GRID, frac_target, RNO, seed=99)
        err = (frac_actual - frac_target) / frac_target * 100
        print(f"{frac_target:>10.3f}  {frac_actual:>10.5f}  {err:>9.2f}")

    print()
    print("[OK] source_distributions.py validation complete.")
