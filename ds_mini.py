"""
DS_mini — Minimal Python implementation of Direct Sampling (DS)
===============================================================
Direct Sampling is a multiple-point geostatistical simulation algorithm.
It reproduces spatial patterns from a Training Image (TI) by scanning it
sequentially and copying values that locally match the already-simulated
neighbourhood.

Reference / original idea:
  Mariethoz, G., Renard, P., Straubhaar, J. (2010).
  The Direct Sampling method to perform multiple‐point geostatistical
  simulations. Water Resources Research, 46(11).
  https://doi.org/10.1029/2008WR007621

This demo closely follows the schematic MATLAB version described at:
  https://wp.unil.ch/gaia/mps/ds-matlab/

Usage
-----
Run directly:
    python ds_mini.py

Or import and call:
    from ds_mini import generate_ti, ds_simulate
    ti  = generate_ti(size=(100, 100))
    sim = ds_simulate(ti, sim_size=(80, 80))
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Training image
# ---------------------------------------------------------------------------

def generate_ti(size=(100, 100), n_objects=50, seed=42):
    """Generate a binary training image with random circular blobs.

    Parameters
    ----------
    size : (int, int)
        Height and width of the training image.
    n_objects : int
        Number of circular objects to place.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ti : ndarray of float, shape ``size``
        Binary image: 1 inside circles, 0 outside.
    """
    rng = np.random.default_rng(seed)
    ny, nx = size
    ti = np.zeros((ny, nx), dtype=float)

    for _ in range(n_objects):
        cx = rng.integers(0, nx)
        cy = rng.integers(0, ny)
        r  = rng.integers(3, max(4, min(ny, nx) // 8))
        ys, xs = np.ogrid[:ny, :nx]
        ti[(ys - cy) ** 2 + (xs - cx) ** 2 <= r ** 2] = 1.0

    return ti


# ---------------------------------------------------------------------------
# Direct Sampling
# ---------------------------------------------------------------------------

def ds_simulate(
    ti,
    sim_size=(80, 80),
    template_half=4,
    max_fraction=0.1,
    threshold=0.05,
    seed=None,
):
    """Run a Direct Sampling unconditional simulation.

    Parameters
    ----------
    ti : ndarray, shape (ny_ti, nx_ti)
        Training image (binary or continuous).
    sim_size : (int, int)
        Height and width of the simulation grid.
    template_half : int
        Half-size of the square template (template side = 2*h+1).
    max_fraction : float in (0, 1]
        Maximum fraction of TI nodes to scan per simulation node.
    threshold : float in [0, 1]
        Distance threshold: scanning stops early when d <= threshold.
        0 means an exact match is required; 1 means any match is accepted.
    seed : int or None
        Random seed.

    Returns
    -------
    sim : ndarray of float, shape ``sim_size``
        Simulated realisation.
    """
    rng   = np.random.default_rng(seed)
    ny_ti, nx_ti = ti.shape
    ny,    nx    = sim_size
    h = template_half

    # Template offsets (all positions in the h×h box)
    dy, dx = np.mgrid[-h:h+1, -h:h+1]
    dy, dx = dy.ravel(), dx.ravel()
    n_template = len(dy)

    # Maximum number of TI nodes to scan per simulation node
    ti_size     = ny_ti * nx_ti
    max_scan    = max(1, int(max_fraction * ti_size))

    # Simulation grid — NaN = not yet simulated
    sim = np.full((ny, nx), np.nan)

    # Valid TI centre positions (keep template inside TI)
    ti_ys = np.arange(h, ny_ti - h)
    ti_xs = np.arange(h, nx_ti - h)
    ti_centres_y, ti_centres_x = np.meshgrid(ti_ys, ti_xs, indexing='ij')
    ti_centres_y = ti_centres_y.ravel()
    ti_centres_x = ti_centres_x.ravel()
    n_ti_centres = len(ti_centres_y)

    # Random simulation path
    sim_indices = rng.permutation(ny * nx)

    for flat_idx in sim_indices:
        si, sj = divmod(int(flat_idx), nx)

        # --- Build data event: informed template nodes ---
        ni = si + dy   # neighbour row indices
        nj = sj + dx   # neighbour col indices

        # Keep nodes inside the simulation grid that are already simulated
        inside   = (ni >= 0) & (ni < ny) & (nj >= 0) & (nj < nx)
        informed = inside & ~np.isnan(sim[ni * inside, nj * inside])
        # (indexing trick: use 0 for out-of-bounds to avoid errors, then mask)
        ni_safe = np.where(inside, ni, 0)
        nj_safe = np.where(inside, nj, 0)
        informed = inside & ~np.isnan(sim[ni_safe, nj_safe])

        if not informed.any():
            # No data event: copy a random TI value
            sim[si, sj] = ti[
                rng.integers(h, ny_ti - h),
                rng.integers(h, nx_ti - h),
            ]
            continue

        data_values  = sim[ni_safe[informed], nj_safe[informed]]
        data_dy      = dy[informed]
        data_dx      = dx[informed]
        n_informed   = informed.sum()

        # --- Scan the TI ---
        scan_order = rng.permutation(n_ti_centres)[:max_scan]

        best_dist = np.inf
        best_val  = np.nan

        for scan_idx in scan_order:
            cy = ti_centres_y[scan_idx]
            cx = ti_centres_x[scan_idx]

            # TI values at the same relative offsets as the data event
            ti_values = ti[cy + data_dy, cx + data_dx]

            # Distance = fraction of mismatches (works for binary; for
            # continuous variables replace with a normalised L1 or L2)
            dist = np.mean(np.abs(ti_values - data_values) > 0.5)

            if dist < best_dist:
                best_dist = dist
                best_val  = ti[cy, cx]

            if dist <= threshold:
                break   # good enough — accept early

        sim[si, sj] = best_val

    return sim


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(ti, sim):
    """Display the training image and the simulation side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(ti,  cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title('Training Image (TI)')
    axes[0].axis('off')

    axes[1].imshow(sim, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title('DS Simulation')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('ds_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Figure saved to ds_result.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Generating training image...')
    ti = generate_ti(size=(100, 100), n_objects=50, seed=42)

    print('Running Direct Sampling simulation...')
    print('  sim_size=80×80, template_half=4, max_fraction=0.10, threshold=0.05')
    sim = ds_simulate(
        ti,
        sim_size=(80, 80),
        template_half=4,
        max_fraction=0.10,
        threshold=0.05,
        seed=0,
    )

    print('Done. Plotting...')
    plot_results(ti, sim)
