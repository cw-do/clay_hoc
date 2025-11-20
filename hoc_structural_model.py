"""
hoc_structural_model.py

Structural generator for clay-like disc platelets forming "house-of-cards" (HoC) networks.

Goal
----
- Generate 3D configurations of disc-like clay platelets in a cubic box.
- Capture two main structural motifs:
    (1) "Seeds": platelets placed as separated free particles
    (2) "Attached": platelets added near existing ones with orientation rules
        that promote T-configurations (≈90°) and parallel stacks.
- Export center-of-mass positions and orientation vectors for later analysis
  (e.g., SANS I(q) calculation).
- Provide quick 3D snapshots as PNG images.
- Show a **single unified progress bar** across all configurations and all particles.

Key ideas
---------
- Geometry:
    - Each clay platelet is modeled as a thin circular disc:
          radius  RADIUS (nm)
          thickness THICKNESS (nm)
    - The simulation box is a cube of side BOX_SIZE (nm).
- Growth model:
    - Start from a single "seed" platelet at the box center.
    - For each new platelet:
        - With probability PROB_NEW_SEED:
            → Place a new "seed" far from all existing platelets
        - With probability 1 - PROB_NEW_SEED:
            → Attach to a randomly chosen parent platelet with one of:
                 * T-contact-like orientation
                 * parallel-like orientation
                 * random orientation
- Distances:
    - Attached platelets are placed at a parent–child center distance drawn
      from a Gaussian distribution around CENTER_DISTANCE_MEAN with spread
      CENTER_DISTANCE_STD.
    - New "seed" platelets are placed such that their center is at least
      MIN_SEED_DISTANCE away from all existing centers (rejection sampling).
- Output:
    - Coordinates: x,y,z and normal vector components nx,ny,nz in CSV.
    - Structure snapshot: discs drawn as polygons in 3D, saved as PNG.
    - Volume fraction is estimated assuming each disc volume is πR²T.

Dependencies
------------
- numpy
- matplotlib
- numba     (not strictly required here but kept for future extension)
- tqdm      (for the unified progress bar)

Typical usage
-------------
    python hoc_structural_model.py

Then, use your separate analysis script (e.g. analysis_iq_sq.py) to compute
g(r), I(q), S(q), etc. from the generated coordinates.
"""

import os
import math
from collections import deque  # (current version doesn't use it, but kept for possible extension)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from numba import njit, prange  # not essential in this minimal version, but left here for future use
from tqdm import tqdm


# ============================================================
# Global simulation parameters (user-tunable)
# ============================================================

# Number of independent configurations to generate
N_CONFIGS = 20

# Base random seed; actual seed used will be BASE_SEED + config_index
BASE_SEED = 0

# Number of platelets per configuration
#   - Typical range you mentioned: 1000–20000
N_PARTICLES = 2000

# Disc geometry (nm)
#   RADIUS    : clay platelet radius
#   THICKNESS : platelet thickness
RADIUS = 25.0
THICKNESS = 1.0

# Simulation box size (nm)
#   - We assume a cubic box [0, BOX_SIZE]^3
BOX_SIZE = 3000.0

# Output folder where CSV and PNG files will be written
OUTPUT_DIR = "hoc_output_prob0.00"

# Probability that a newly added platelet is a "new seed"
#   - With probability PROB_NEW_SEED:
#         place a new free seed, far away from existing platelets
#   - With probability 1 - PROB_NEW_SEED:
#         attach to an existing platelet with T/parallel/random orientation
PROB_NEW_SEED = 0.0

# Orientation probabilities for attached platelets
#   - When we attach a new platelet to a parent, we choose:
#        PROB_T       : T-contact-like orientation (≈90° between normals)
#        PROB_PARALLEL: parallel orientation (≈0° or 180° between normals)
#        PROB_RANDOM  : fully random orientation
PROB_T = 0.6
PROB_PARALLEL = 0.3
PROB_RANDOM = 0.1

# Angular spread (in degrees) for the orientation rules
#   - ANGLE_SIGMA_T_DEG:
#         Gaussian spread around 90° for T-contact-like orientation
#   - ANGLE_SIGMA_PARALLEL_DEG:
#         Gaussian spread around 0° or 180° for parallel-like orientation
ANGLE_SIGMA_T_DEG = 10.0
ANGLE_SIGMA_PARALLEL_DEG = 10.0

# Parent–child center distance for attached platelets (nm)
#   - Drawn from Gaussian with mean and std below, then floored at 1.5 * RADIUS
CENTER_DISTANCE_MEAN = 2.2 * RADIUS
CENTER_DISTANCE_STD = 0.25 * RADIUS

# Minimum separation for newly seeded (free) platelets (nm)
#   - We attempt to place a new seed so that the distance from all existing
#     platelets is at least MIN_SEED_DISTANCE.
#   - If this fails after MAX_SEED_TRIES attempts, we accept the last candidate
#     (i.e., constraint may be slightly violated in very dense cases).
MIN_SEED_DISTANCE = 2.4 * RADIUS
MAX_SEED_TRIES = 1000

# Maximum number of discs to draw in snapshot (for speed)
MAX_DISCS_TO_DRAW = 2000


# ============================================================
# Helper: logging that doesn't break tqdm progress bar
# ============================================================

def info(msg: str) -> None:
    """
    Print an informational message without breaking the tqdm progress bar.

    We use tqdm.write instead of plain print so the progress bar remains
    on a single line and is re-drawn cleanly after each message.
    """
    tqdm.write(msg)


# ============================================================
# Basic vector utilities
# ============================================================

def random_unit_vector() -> np.ndarray:
    """
    Draw a random unit vector uniformly on the sphere.

    Returns
    -------
    v : (3,) ndarray
        Random vector with |v| = 1.
    """
    z = 2.0 * np.random.rand() - 1.0
    phi = 2.0 * np.pi * np.random.rand()
    r_xy = math.sqrt(max(0.0, 1.0 - z*z))
    x = r_xy * math.cos(phi)
    y = r_xy * math.sin(phi)
    return np.array([x, y, z], dtype=float)


def random_angle(center_deg: float, sigma_deg: float) -> float:
    """
    Sample an angle (in radians) from a Gaussian around center_deg with
    standard deviation sigma_deg.

    e.g. center_deg ≈ 90° for T-contact, 0° or 180° for parallel contact.
    """
    return np.deg2rad(np.random.normal(center_deg, sigma_deg))


# ============================================================
# ClayPlatelet class
# ============================================================

class ClayPlatelet:
    """
    Simple representation of a disc-like clay platelet.

    Attributes
    ----------
    pos : (3,) ndarray
        Center-of-mass position in nm.
    n   : (3,) ndarray
        Unit normal vector of the disc (orientation).
    """

    __slots__ = ("pos", "n")

    def __init__(self, pos, n) -> None:
        pos = np.array(pos, dtype=float)
        n = np.array(n, dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            raise ValueError("Normal vector too small.")
        self.pos = pos
        self.n = n / n_norm

    def basis(self):
        """
        Construct an orthonormal basis (u, v, n) for the disc:

        u, v lie in the disc plane; n is the stored normal.

        Returns
        -------
        u, v, n : tuple of (3,) ndarrays
        """
        n = self.n
        base = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(base, n)) > 0.9:
            base = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, base)
        u /= (np.linalg.norm(u) + 1e-12)
        v = np.cross(n, u)
        v /= (np.linalg.norm(v) + 1e-12)
        return u, v, n


# ============================================================
# Seeding with minimum distance constraint
# ============================================================

def place_new_seed(existing_positions: np.ndarray) -> np.ndarray:
    """
    Place a new "seed" platelet in the box, attempting to keep it
    at least MIN_SEED_DISTANCE away from all existing particle centers.

    Parameters
    ----------
    existing_positions : (M,3) ndarray or None
        Positions of existing platelets.

    Returns
    -------
    pos_new : (3,) ndarray
        New seed position.
    """
    # If this is the very first particle, just draw it anywhere in the box
    if existing_positions is None or existing_positions.size == 0:
        return np.random.rand(3) * BOX_SIZE

    last_candidate = None
    for _ in range(MAX_SEED_TRIES):
        pos = np.random.rand(3) * BOX_SIZE
        dists = np.linalg.norm(existing_positions - pos, axis=1)
        if dists.min() >= MIN_SEED_DISTANCE:
            return pos
        last_candidate = pos

    # Fallback: we failed to find a perfectly separated position; use last candidate
    return last_candidate if last_candidate is not None else (np.random.rand(3) * BOX_SIZE)


# ============================================================
# Orientation rules for attached platelets
# ============================================================

def make_T_normal(parent_n: np.ndarray) -> np.ndarray:
    """
    Generate a new normal vector roughly at 90° (T-contact-like)
    relative to the parent normal.

    That is, if n_parent is the parent normal and n_new is the child normal,
    the angle between them is distributed around 90° with width ANGLE_SIGMA_T_DEG.
    """
    parent_n = parent_n / (np.linalg.norm(parent_n) + 1e-12)

    # Find a vector orthogonal to parent_n
    base = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(base, parent_n)) > 0.9:
        base = np.array([0.0, 1.0, 0.0])
    b = np.cross(parent_n, base)
    b /= (np.linalg.norm(b) + 1e-12)

    angle = random_angle(90.0, ANGLE_SIGMA_T_DEG)
    n_new = math.cos(angle) * parent_n + math.sin(angle) * b
    return n_new / (np.linalg.norm(n_new) + 1e-12)


def make_parallel_normal(parent_n: np.ndarray) -> np.ndarray:
    """
    Generate a new normal vector roughly parallel (0° or 180°) to parent_n.

    Procedure:
    - Choose target angle = 0° or 180° with equal probability.
    - Draw a small deviation from a Gaussian with std ANGLE_SIGMA_PARALLEL_DEG.
    - Rotate parent_n by that small angle around a random axis perpendicular to it.
    """
    parent_n = parent_n / (np.linalg.norm(parent_n) + 1e-12)

    # Choose 0° or 180° as the center
    center_deg = 0.0 if np.random.rand() < 0.5 else 180.0
    angle = random_angle(center_deg, ANGLE_SIGMA_PARALLEL_DEG)

    # Find a rotation axis orthogonal to parent_n
    base = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(base, parent_n)) > 0.9:
        base = np.array([0.0, 1.0, 0.0])
    axis = np.cross(parent_n, base)
    axis /= (np.linalg.norm(axis) + 1e-12)

    # Rodrigues' rotation formula
    K = np.array([
        [0.0,      -axis[2],  axis[1]],
        [axis[2],   0.0,     -axis[0]],
        [-axis[1],  axis[0],  0.0   ],
    ])
    I = np.eye(3)
    R = I + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)

    n_new = R @ parent_n
    return n_new / (np.linalg.norm(n_new) + 1e-12)


# ============================================================
# Structure builder with unified progress bar
# ============================================================

def build_structure_with_progress(n_particles: int, pbar: tqdm) -> list:
    """
    Build a single "house-of-cards-like" configuration of platelets.

    Parameters
    ----------
    n_particles : int
        Number of platelets in this configuration.
    pbar : tqdm instance
        Global progress bar shared across configurations. We call pbar.update(1)
        for each platelet that is added.

    Returns
    -------
    platelets : list of ClayPlatelet
        The complete list of platelets in the configuration.
    """
    platelets = []

    # --- 1) First seed at box center ---
    center = np.array([BOX_SIZE/2.0, BOX_SIZE/2.0, BOX_SIZE/2.0])
    n0 = random_unit_vector()
    platelets.append(ClayPlatelet(center, n0))
    pbar.update(1)

    # --- 2) Grow the remaining platelets ---
    for k in range(1, n_particles):
        # Extract existing positions for seed placement
        existing_positions = np.array([p.pos for p in platelets], dtype=float)

        # Decide: new free seed vs attached platelet
        if np.random.rand() < PROB_NEW_SEED:
            # New seed: far from existing platelets
            pos_new = place_new_seed(existing_positions)
            n_new = random_unit_vector()
            platelets.append(ClayPlatelet(pos_new, n_new))

        else:
            # Attached platelet: choose a parent uniformly at random
            parent_idx = np.random.randint(len(platelets))
            parent = platelets[parent_idx]

            # Choose orientation rule
            r = np.random.rand()
            if r < PROB_T:
                n_new = make_T_normal(parent.n)
            elif r < PROB_T + PROB_PARALLEL:
                n_new = make_parallel_normal(parent.n)
            else:
                n_new = random_unit_vector()

            # Choose parent–child distance
            dist = np.random.normal(CENTER_DISTANCE_MEAN, CENTER_DISTANCE_STD)
            dist = max(dist, 1.5 * RADIUS)

            # Random direction from parent
            d_hat = random_unit_vector()
            pos_new = parent.pos + dist * d_hat

            # Clip to box boundaries
            pos_new = np.clip(pos_new, 0.0, BOX_SIZE)

            platelets.append(ClayPlatelet(pos_new, n_new))

        # Update global progress bar by one platelet
        pbar.update(1)

    return platelets


# ============================================================
# Visualization (3D disc snapshot)
# ============================================================

def disc_polygon(platelet: ClayPlatelet, rad: float = RADIUS, n_segments: int = 30):
    """
    Generate a polygonal approximation of a disc in 3D for plotting.

    Parameters
    ----------
    platelet : ClayPlatelet
        The platelet to draw.
    rad : float, optional
        Radius to use for the disc (should match RADIUS).
    n_segments : int, optional
        Number of segments around the circle.

    Returns
    -------
    verts : list of (3,) ndarrays
        Vertex positions around the disc edge in 3D.
    """
    u, v, n = platelet.basis()
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=True)
    verts = []
    for a in angles:
        verts.append(platelet.pos + rad * np.cos(a) * u + rad * np.sin(a) * v)
    return verts


def plot_structure(platelets: list, filename: str) -> None:
    """
    Plot the structure in 3D as discs and save to a PNG file.

    For large N, we only draw up to MAX_DISCS_TO_DRAW discs to keep
    plotting time manageable.

    Parameters
    ----------
    platelets : list of ClayPlatelet
    filename  : str
        Path of the PNG file to save.
    """
    N = len(platelets)
    if N > MAX_DISCS_TO_DRAW:
        draw_idx = np.random.choice(N, MAX_DISCS_TO_DRAW, replace=False)
    else:
        draw_idx = np.arange(N)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    for idx in draw_idx:
        p = platelets[idx]
        verts = disc_polygon(p)
        poly = Poly3DCollection([verts], alpha=0.6)
        poly.set_edgecolor("k")
        poly.set_facecolor("C0")
        ax.add_collection3d(poly)

    centers = np.array([p.pos for p in platelets])
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
               s=3, c="red", alpha=0.3)

    ax.set_xlim(0.0, BOX_SIZE)
    ax.set_ylim(0.0, BOX_SIZE)
    ax.set_zlim(0.0, BOX_SIZE)
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")
    ax.set_title("House-of-cards structure snapshot")

    ax.set_box_aspect((1.0, 1.0, 1.0))

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    info(f"[INFO] Saved structure snapshot: {filename}")


# ============================================================
# Saving coordinates and simple stats
# ============================================================

def compute_volume_fraction(n_particles: int) -> float:
    """
    Estimate the disc volume fraction φ in the box.

    Assuming each platelet is a solid cylinder with volume:
        V_disc = π R^2 T

    and the box volume is:
        V_box = BOX_SIZE^3

    Parameters
    ----------
    n_particles : int
        Number of platelets.

    Returns
    -------
    phi : float
        Volume fraction φ = N * V_disc / V_box.
    """
    v_disc = math.pi * RADIUS**2 * THICKNESS
    v_box = BOX_SIZE**3
    phi = n_particles * v_disc / v_box
    return phi


def save_coordinates(platelets: list, path: str) -> None:
    """
    Save coordinates to CSV file with columns:
        x, y, z, nx, ny, nz

    Parameters
    ----------
    platelets : list of ClayPlatelet
    path      : str
        Output file path.
    """
    arr = np.array([
        [p.pos[0], p.pos[1], p.pos[2],
         p.n[0],   p.n[1],   p.n[2]]
        for p in platelets
    ])
    header = "x,y,z,nx,ny,nz"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")
    info(f"[INFO] Saved coordinates: {path}")


# ============================================================
# Main driver
# ============================================================

def main() -> None:
    """
    Main function:

    - Create OUTPUT_DIR if it doesn't exist.
    - Create a unified progress bar over all configurations and particles.
    - For each configuration:
        * set random seed
        * build structure (calling build_structure_with_progress)
        * compute volume fraction
        * save coordinates and snapshot.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Total number of "steps" in the progress bar:
    #   each step = adding one platelet
    total_steps = N_CONFIGS * N_PARTICLES
    pbar = tqdm(total=total_steps, desc="Total progress", leave=True)

    for cfg_idx in range(N_CONFIGS):
        seed = BASE_SEED + cfg_idx
        np.random.seed(seed)

        info(f"\n[INFO] Starting config {cfg_idx+1}/{N_CONFIGS}, seed={seed}")

        # Build structure
        platelets = build_structure_with_progress(N_PARTICLES, pbar)

        # Compute simple volume fraction estimate
        phi = compute_volume_fraction(len(platelets))
        info(f"[INFO]   Number of platelets: {len(platelets)}")
        info(f"[INFO]   Estimated volume fraction φ ≈ {phi:.4e}")

        # Build file names
        postfix = f"{cfg_idx:03d}"
        coords_path = os.path.join(OUTPUT_DIR, f"hoc_structure_coords_{postfix}.csv")
        fig_path = os.path.join(OUTPUT_DIR, f"hoc_structure_{postfix}.png")

        # Save results
        save_coordinates(platelets, coords_path)
        plot_structure(platelets, fig_path)

        info(f"[INFO] Finished config {cfg_idx+1}/{N_CONFIGS}")

    pbar.close()
    info("\n[INFO] All configurations complete.")


if __name__ == "__main__":
    main()
