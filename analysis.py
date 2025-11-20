"""
analysis.py

Ensemble-averaged scattering analysis for clay disc configurations
produced by simulation.py.

- Reads multiple final_configuration_*.csv from OUTPUT_DIR.
- Each configuration contains:
    x, y, z, nx, ny, nz
  for each particle (center position and disc normal).

- Model:
    * Each clay particle is a thin circular disc (radius RADIUS, thickness THICKNESS).
    * Use analytic disc/cylinder amplitude F_disc(q, alpha), where
        alpha is the angle between q and the disc normal n.
    * Compute for each configuration:
        I(q) = < | sum_i F(q, alpha_i) exp(i q q̂ · r_i) |^2 >_{q̂}
        P(q) = < (1/N) sum_i |F(q, alpha_i)|^2 >_{q̂}
      via Monte Carlo averaging over q̂ directions on the unit sphere.
    * Then ensemble-average over all configs:
        I_avg(q) = mean_config I(q)
        P_avg(q) = mean_config P(q)
        N_avg    = mean_config N_config
        S_eff(q) = I_avg(q) / [ N_avg * P_avg(q) ]

- Also computes ensemble-averaged radial distribution g(r) of centers:
    * For each configuration, accumulate pair-distance histogram with PBC
    * Combine over configs, then normalize by ideal-gas shell volume.

Outputs (all in OUTPUT_DIR):
  - gr_ensemble.csv :
        r, g(r)
  - Iq_disc_ensemble.csv :
        q, I_avg, P_avg, S_eff
  - analysis_disc_ensemble.png :
        2x2 plots of g(r), I_avg(q), P_avg(q), S_eff(q)
"""

import os
import glob
import math

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import j1  # Bessel J1

# =========================
#  Paths / shared parameters
# =========================

# Must match simulation.py
OUTPUT_DIR = "output"

# Geometry (must match simulation.py)
RADIUS = 25.0      # nm
THICKNESS = 1.0    # nm
BOX_SIZE = 200.0   # nm  (cubic box)

# q-range
Q_MIN = 0.01       # nm^-1
Q_MAX = 2        # nm^-1
N_Q   = 200

# MC over q-directions
N_QDIR = 200       # increase for smoother curves (cost ~ N * N_Q * N_QDIR)

# g(r) parameters
R_MAX = BOX_SIZE/2      # nm
N_R   = 200        # bins


# =========================
#  Basic utilities
# =========================

def random_unit_vectors(n_vec):
    """
    Generate n_vec random unit vectors uniformly on the sphere.
    """
    z = 2.0 * np.random.rand(n_vec) - 1.0
    phi = 2.0 * math.pi * np.random.rand(n_vec)
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    x = r_xy * np.cos(phi)
    y = r_xy * np.sin(phi)
    return np.vstack([x, y, z]).T  # (n_vec, 3)


def minimum_image_vec(d, box_size=BOX_SIZE):
    """
    Apply minimum image convention for a vector d under cubic PBC.
    d : (..., 3)
    """
    return d - np.rint(d / box_size) * box_size


# =========================
#  Disc scattering amplitude
# =========================

def F_disc_amplitude(q, cos_alpha):
    """
    Scattering amplitude F(q, alpha) for a thin circular disc
    (approximated by a short cylinder of radius RADIUS, length THICKNESS).

    Standard cylinder amplitude form (up to overall prefactor):

        F(q, alpha) ~ V * sinc(q L cos(alpha) / 2)
                          * 2 J1(q R sin(alpha)) / (q R sin(alpha))

    where:
        V = pi R^2 L
        L = THICKNESS
        R = RADIUS
        alpha = angle between q and disc normal
        cos_alpha = cos(alpha)
        sin_alpha = sqrt(1 - cos_alpha^2)

    We care about relative shape, not absolute scale, so the prefactor V
    simply sets an overall scale.
    """
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    sin_alpha = np.sqrt(1.0 - cos_alpha**2)

    x = q * RADIUS * sin_alpha          # argument for J1
    y = 0.5 * q * THICKNESS * cos_alpha # argument for sinc

    # J1(x)/x with x -> 0 limit
    J1 = j1(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        J1_over_x = np.where(x != 0.0, J1 / x, 0.5)  # limit J1(x)/x ~ 0.5

    # sinc(y) = sin(y)/y with y -> 0 limit 1
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc = np.where(y != 0.0, np.sin(y) / y, 1.0)

    # Volume factor (overall scale)
    V = math.pi * RADIUS**2 * THICKNESS

    F = V * sinc * (2.0 * J1_over_x)
    return F  # real-valued amplitude


# =========================
#  I(q) for a single configuration
# =========================

def compute_Iq_disc_single(positions, normals, q_vals, q_dirs, desc="I(q)"):
    """
    Compute I(q) and P(q) for one configuration of disc-shaped particles
    using the analytic disc amplitude and Monte Carlo orientational averaging.

    positions : (N, 3)
    normals   : (N, 3)  -- unit normals of each disc
    q_vals    : (N_q,)
    q_dirs    : (N_qdir, 3)  -- random unit vectors on sphere (fixed across configs)

    Returns:
      Iq_total : (N_q,) total intensity, orientation-averaged
      Pq       : (N_q,) effective single-particle intensity
    """
    Np = positions.shape[0]
    N_qdir = q_dirs.shape[0]

    Iq_total = np.zeros_like(q_vals, dtype=float)
    Pq = np.zeros_like(q_vals, dtype=float)

    for iq, q in enumerate(tqdm(q_vals, desc=desc, ncols=80, leave=False)):
        I_sum = 0.0
        P_sum = 0.0

        for d in range(N_qdir):
            u = q_dirs[d]  # unit vector, shape (3,)

            # cos(alpha_i) = u · n_i for all particles
            cos_alpha = normals @ u     # (N,)

            # amplitude per particle (real)
            F_i = F_disc_amplitude(q, cos_alpha)   # (N,)

            # phase factor from positions: exp(i q (u · r_i))
            phase_arg = q * (positions @ u)       # (N,)
            phase = np.exp(1j * phase_arg)       # (N,)

            A = np.sum(F_i * phase)              # complex scalar
            I_sum += (A.conjugate() * A).real    # |A|^2

            # single-particle contribution along this direction
            P_dir = np.mean(np.abs(F_i)**2)
            P_sum += P_dir

        Iq_total[iq] = I_sum / N_qdir
        Pq[iq]       = P_sum / N_qdir

    return Iq_total, Pq


# =========================
#  g(r) for a single configuration
# =========================

def compute_gr_hist_single(positions, box_size=BOX_SIZE,
                           r_max=R_MAX, n_bins=N_R):
    """
    Compute pair-distance histogram for one configuration with PBC.

    positions : (N, 3)

    Returns:
      r_centers : (n_bins,)
      hist      : (n_bins,)  -- raw pair counts per shell
      edges     : (n_bins+1,)
    """
    N = positions.shape[0]

    dists = []

    # pair distances with PBC, O(N^2) but N~200 so fine
    for i in range(N):
        for j in range(i + 1, N):
            d = positions[j] - positions[i]
            d = minimum_image_vec(d, box_size)
            r = np.linalg.norm(d)
            if r < r_max:
                dists.append(r)

    dists = np.array(dists, dtype=float)

    if dists.size == 0:
        edges = np.linspace(0.0, r_max, n_bins + 1)
        r_centers = 0.5 * (edges[:-1] + edges[1:])
        hist = np.zeros(n_bins, dtype=float)
        return r_centers, hist, edges

    hist, edges = np.histogram(dists, bins=n_bins, range=(0.0, r_max))
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    return r_centers, hist.astype(float), edges


# =========================
#  Loading configurations
# =========================

def load_configurations(output_dir=OUTPUT_DIR):
    """
    Find and load all final_configuration_*.csv in output_dir.

    Returns:
      configs : list of dicts
        each dict has keys:
           "positions" : (N, 3)
           "normals"   : (N, 3)
           "filename"  : str
    """
    pattern = os.path.join(output_dir, "final_configuration_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[ERROR] No final_configuration_*.csv found in {output_dir}")
        return []

    configs = []
    for path in files:
        try:
            data = np.genfromtxt(path, delimiter=",", names=True)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
            continue

        # Handle case of single row vs multi-row
        if data.ndim == 0:
            data = np.array([data])

        if not all(col in data.dtype.names for col in ("x", "y", "z", "nx", "ny", "nz")):
            print(f"[WARN] File {path} missing required columns; skipping.")
            continue

        pos = np.vstack([data["x"], data["y"], data["z"]]).T
        nvec = np.vstack([data["nx"], data["ny"], data["nz"]]).T

        # normalize normals just in case
        norms = np.linalg.norm(nvec, axis=1, keepdims=True) + 1e-12
        nvec = nvec / norms

        configs.append({
            "positions": pos,
            "normals": nvec,
            "filename": os.path.basename(path),
        })

    print(f"[INFO] Loaded {len(configs)} configurations from {output_dir}")
    return configs


# =========================
#  Main ensemble analysis
# =========================

def main():
    # 1) Load configurations
    configs = load_configurations(OUTPUT_DIR)
    if not configs:
        return

    ncfg = len(configs)

    # 2) Prepare q grid and q-directions (shared across configs)
    q_vals = np.linspace(Q_MIN, Q_MAX, N_Q)
    q_dirs = random_unit_vectors(N_QDIR)

    # 3) Accumulators for ensemble averages
    Iq_accum = np.zeros_like(q_vals, dtype=float)
    Pq_accum = np.zeros_like(q_vals, dtype=float)
    N_accum = 0.0

    hist_sum = np.zeros(N_R, dtype=float)
    r_centers_ref = None
    edges_ref = None

    # 4) Loop over configurations
    for idx, cfg in enumerate(configs):
        positions = cfg["positions"]
        normals = cfg["normals"]
        Np = positions.shape[0]

        print(f"\n[INFO] Config {idx+1}/{ncfg}: {cfg['filename']}, N = {Np}")

        # --- I(q), P(q) for this config ---
        desc = f"I(q) cfg {idx+1}/{ncfg}"
        Iq_single, Pq_single = compute_Iq_disc_single(
            positions, normals, q_vals, q_dirs, desc=desc
        )

        Iq_accum += Iq_single
        Pq_accum += Pq_single
        N_accum += Np

        # --- g(r) histogram for this config ---
        r_centers, hist, edges = compute_gr_hist_single(
            positions, BOX_SIZE, R_MAX, N_R
        )

        if r_centers_ref is None:
            r_centers_ref = r_centers
            edges_ref = edges
        else:
            # sanity check: edges should match
            if not np.allclose(edges_ref, edges):
                print("[WARN] g(r) bin edges differ between configs; "
                      "using first config's binning for ensemble.")
        hist_sum += hist

    # 5) Ensemble averages
    Iq_avg = Iq_accum / ncfg
    Pq_avg = Pq_accum / ncfg
    N_avg = N_accum / ncfg

    # Effective S(q) = I_avg / (N_avg * P_avg)
    denom = N_avg * Pq_avg
    S_eff = np.zeros_like(q_vals, dtype=float)
    mask = np.abs(denom) > 1e-16
    S_eff[mask] = Iq_avg[mask] / denom[mask]

    # 6) Ensemble-averaged g(r)
    if r_centers_ref is None or edges_ref is None:
        print("[WARN] No pair distances found; g(r) will be zero.")
        r_grid = np.linspace(0.0, R_MAX, N_R)
        g_r = np.zeros_like(r_grid)
    else:
        r_grid = r_centers_ref
        edges = edges_ref

        box_volume = BOX_SIZE**3
        rho = N_avg / box_volume
        shell_vol = 4.0 * math.pi * (edges[1:]**3 - edges[:-1]**3) / 3.0

        # hist_sum is total pair counts over all configs
        # divide by number of configs and ideal counts
        ideal = rho * shell_vol
        g_r = (hist_sum / ncfg) / (ideal + 1e-16)

    # 7) Save CSV outputs
    gr_path = os.path.join(OUTPUT_DIR, "gr_ensemble.csv")
    np.savetxt(
        gr_path,
        np.vstack([r_grid, g_r]).T,
        delimiter=",",
        header="r,g(r)",
        comments=""
    )
    print(f"[INFO] g(r) ensemble saved to {gr_path}")

    iq_path = os.path.join(OUTPUT_DIR, "Iq_disc_ensemble.csv")
    out = np.vstack([q_vals, Iq_avg, Pq_avg, S_eff]).T
    header = "q,I_avg,P_avg,S_eff"
    np.savetxt(
        iq_path,
        out,
        delimiter=",",
        header=header,
        comments=""
    )
    print(f"[INFO] I(q) ensemble saved to {iq_path}")

    # 8) 2x2 plot: g(r), I_avg(q), P_avg(q), S_eff(q)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax_gr = axes[0, 0]
    ax_Iq = axes[0, 1]
    ax_Pq = axes[1, 0]
    ax_Sq = axes[1, 1]

    ax_gr.plot(r_grid, g_r, "-k")
    ax_gr.set_xlabel("r (nm)")
    ax_gr.set_ylabel("g(r)")
    ax_gr.set_title("Ensemble-averaged g(r)")

    ax_Iq.loglog(q_vals, Iq_avg, "-b")
    ax_Iq.set_xlabel("q (nm$^{-1}$)")
    ax_Iq.set_ylabel("I(q)")
    ax_Iq.set_title("Ensemble I(q) (disc amplitude)")

    ax_Pq.loglog(q_vals, Pq_avg, "-g")
    ax_Pq.set_xlabel("q (nm$^{-1}$)")
    ax_Pq.set_ylabel("P(q)")
    ax_Pq.set_title("Ensemble P(q) (single disc)")

    ax_Sq.semilogx(q_vals, S_eff, "-r")
    ax_Sq.set_xlabel("q (nm$^{-1}$)")
    ax_Sq.set_ylabel("S_eff(q)")
    ax_Sq.set_title("Effective S(q) = I_avg / (N_avg P_avg)")

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "analysis_disc_ensemble.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[INFO] Ensemble plots saved to {fig_path}")

    print("\n[DONE] analysis.py ensemble analysis finished.")


if __name__ == "__main__":
    main()
