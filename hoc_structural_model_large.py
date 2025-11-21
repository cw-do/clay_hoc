"""
hoc_structural_model_large.py

Large-scale structural generator for clay-like disc platelets
forming "house-of-cards"-like networks.

Goal
----
- Generate up to ~1e6 platelets in a cubic box.
- Keep the *growth rules* similar to hoc_structural_model.py:
    * mixture of "seed" and "attached" platelets
    * attached platelets use T-like orientation rule (or whatever you set by params)
- BUT:
    * skip expensive contact-network analysis
    * skip full plotting (or plot only a small subset)
    * avoid O(N^2) distance checks for seed placement

This script is meant for *large N* configuration generation, e.g.
for volume fraction ~ a few %.
"""

import os
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# Parameters for LARGE SCALE mode
# ============================================================

# Number of independent large configurations
N_CONFIGS = 1



# Random seed base
BASE_SEED = 0

# Disc geometry (nm)
RADIUS = 25.0
THICKNESS = 1.0

# Box size (nm)
#   To get target volume fraction phi ≈ 6 %:
#       phi = N * π R^2 T / L^3
#       L = ( N * π R^2 T / phi )^(1/3)
TARGET_PHI = 0.06
DISC_VOL = math.pi * RADIUS**2 * THICKNESS
#BOX_SIZE = (N_PARTICLES * DISC_VOL / TARGET_PHI) ** (1.0/3.0)
BOX_SIZE = 3000
# Number of platelets per configuration
N_PARTICLES = BOX_SIZE**3 * TARGET_PHI / DISC_VOL
N_PARTICLES = int(N_PARTICLES)



# Growth probabilities
PROB_NEW_SEED = 0.25   # probability to create a "free seed"
PROB_T = 1.0           # you can set 1.0 and use ANGLE_SIGMA_T_DEG large if you like
PROB_PARALLEL = 0.0    # set to 0 for large-scale if you want a single rule
PROB_RANDOM = 0.0

# Angular parameters (deg)
ANGLE_SIGMA_T_DEG = 180.0      # very broad T-rule (covers almost all angles)
ANGLE_SIGMA_PARALLEL_DEG = 10.0

# Parent–child distance (nm)
CENTER_DISTANCE_MEAN = 2.2 * RADIUS
CENTER_DISTANCE_STD = 0.25 * RADIUS

# Output directory
OUTPUT_DIR = "hoc_output_large_v0.06"

# How many particles to draw in optional snapshot
MAX_DRAW = 5000


# ============================================================
# Helper: logging that doesn't break tqdm bar
# ============================================================

def info(msg: str) -> None:
    tqdm.write(msg)


# ============================================================
# Basic random utilities
# ============================================================

def random_unit_vector():
    z = np.random.uniform(-1.0, 1.0)
    phi = np.random.uniform(0.0, 2.0*np.pi)
    r_xy = math.sqrt(max(0.0, 1.0 - z*z))
    x = r_xy * math.cos(phi)
    y = r_xy * math.sin(phi)
    return np.array([x, y, z], dtype=float)


def random_angle(center_deg, sigma_deg):
    return np.deg2rad(np.random.normal(center_deg, sigma_deg))


# ============================================================
# Orientation rules
# ============================================================

def make_T_normal(parent_n):
    parent_n = parent_n / (np.linalg.norm(parent_n) + 1e-12)

    base = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(base, parent_n)) > 0.9:
        base = np.array([0.0, 1.0, 1.0])
    axis = np.cross(parent_n, base)
    axis /= (np.linalg.norm(axis) + 1e-12)

    # note: "T" here is really just "centered at 90°"
    angle = random_angle(90.0, ANGLE_SIGMA_T_DEG)
    # Rodrigues rotation
    K = np.array([
        [0.0,        -axis[2],  axis[1]],
        [axis[2],     0.0,     -axis[0]],
        [-axis[1],    axis[0],  0.0   ],
    ])
    I = np.eye(3)
    R = I + math.sin(angle)*K + (1.0 - math.cos(angle))*(K @ K)

    n_new = R @ parent_n
    return n_new / (np.linalg.norm(n_new) + 1e-12)


def make_parallel_normal(parent_n):
    parent_n = parent_n / (np.linalg.norm(parent_n) + 1e-12)
    center_deg = 0.0 if np.random.rand() < 0.5 else 180.0
    angle = random_angle(center_deg, ANGLE_SIGMA_PARALLEL_DEG)

    base = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(base, parent_n)) > 0.9:
        base = np.array([0.0, 1.0, 0.0])
    axis = np.cross(parent_n, base)
    axis /= (np.linalg.norm(axis) + 1e-12)

    K = np.array([
        [0.0,        -axis[2],  axis[1]],
        [axis[2],     0.0,     -axis[0]],
        [-axis[1],    axis[0],  0.0   ],
    ])
    I = np.eye(3)
    R = I + math.sin(angle)*K + (1.0 - math.cos(angle))*(K @ K)

    n_new = R @ parent_n
    return n_new / (np.linalg.norm(n_new) + 1e-12)


# ============================================================
# Large-scale structure generator
# ============================================================

def generate_large_structure(n_particles: int,
                             box_size: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a large HoC-like structure using numpy arrays only.

    The algorithm:
      - positions: (N,3)
      - normals:   (N,3)
      - i=0: seed at box center
      - i>0: with probability PROB_NEW_SEED → new seed at random position
             else → attach to random parent with orientation rule
    NOTE:
      - We do NOT enforce minimum distance between seeds here:
        at N~1e6 it becomes too expensive to check.
    """
    positions = np.empty((n_particles, 3), dtype=float)
    normals = np.empty((n_particles, 3), dtype=float)

    # First particle: center seed
    center = np.array([box_size/2.0]*3)
    positions[0, :] = center
    normals[0, :] = random_unit_vector()

    for i in range(1, n_particles):
        if np.random.rand() < PROB_NEW_SEED:
            # new seed, just random in box (no global distance check)
            positions[i, :] = np.random.rand(3) * box_size
            normals[i, :] = random_unit_vector()
        else:
            # attach to a random existing platelet
            parent_idx = np.random.randint(0, i)
            parent_pos = positions[parent_idx, :]
            parent_n = normals[parent_idx, :]

            r = np.random.rand()
            if r < PROB_T:
                n_new = make_T_normal(parent_n)
            elif r < PROB_T + PROB_PARALLEL:
                n_new = make_parallel_normal(parent_n)
            else:
                n_new = random_unit_vector()

            dist = np.random.normal(CENTER_DISTANCE_MEAN, CENTER_DISTANCE_STD)
            dist = max(dist, 1.5 * RADIUS)
            d_hat = random_unit_vector()
            pos_new = parent_pos + dist * d_hat
            pos_new = np.clip(pos_new, 0.0, box_size)

            positions[i, :] = pos_new
            normals[i, :] = n_new

    return positions, normals


# ============================================================
# Simple plotting of subset (optional)
# ============================================================

def disc_polygon(pos, n, rad=RADIUS, n_segments=24):
    # build orthonormal basis from normal
    base = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(base, n)) > 0.9:
        base = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, base)
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v /= (np.linalg.norm(v) + 1e-12)

    angles = np.linspace(0.0, 2.0*np.pi, n_segments, endpoint=True)
    verts = []
    for a in angles:
        verts.append(pos + rad*np.cos(a)*u + rad*np.sin(a)*v)
    return verts


def plot_subset(positions, normals, box_size, path_png):
    N = positions.shape[0]
    n_draw = min(N, MAX_DRAW)
    idx = np.random.choice(N, n_draw, replace=False)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i in idx:
        pos = positions[i, :]
        n = normals[i, :]
        verts = disc_polygon(pos, n)
        poly = Poly3DCollection([verts], alpha=0.5)
        poly.set_edgecolor("k")
        poly.set_facecolor("C0")
        ax.add_collection3d(poly)

    ax.scatter(positions[idx, 0], positions[idx, 1], positions[idx, 2],
               s=2, c="red", alpha=0.4)

    ax.set_xlim(0.0, box_size)
    ax.set_ylim(0.0, box_size)
    ax.set_zlim(0.0, box_size)
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_title(f"Subset of large HoC structure (N={N}, drawn={n_draw})")
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.close()
    info(f"[INFO] Saved subset snapshot: {path_png}")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # overall info
    phi_est = N_PARTICLES * DISC_VOL / (BOX_SIZE**3)
    print(f"[INFO] Target phi ≈ {TARGET_PHI:.3f}, "
          f"actual phi ≈ {phi_est:.3f}, BOX_SIZE = {BOX_SIZE:.1f} nm")
    print(f"[INFO] N_PARTICLES = {N_PARTICLES}")

    # progress bar: configs × (just treat each particle as 1 step)
    total_steps = N_CONFIGS * N_PARTICLES
    pbar = tqdm(total=total_steps, desc="Total progress", leave=True)

    for cfg in range(N_CONFIGS):
        seed = BASE_SEED + cfg
        np.random.seed(seed)
        info(f"\n[INFO] Starting LARGE config {cfg+1}/{N_CONFIGS}, seed={seed}")

        # generate structure (in this function there's no pbar; we update here)
        positions, normals = generate_large_structure(N_PARTICLES, BOX_SIZE)
        pbar.update(N_PARTICLES)

        # save coordinates
        postfix = f"{cfg:03d}"
        coords_path = os.path.join(OUTPUT_DIR, f"large_hoc_coords_{postfix}.csv")
        arr = np.hstack([positions, normals])
        np.savetxt(
            coords_path,
            arr,
            delimiter=",",
            header="x,y,z,nx,ny,nz",
            comments=""
        )
        info(f"[INFO] Saved coordinates: {coords_path}")

        # optional snapshot of small subset (for sanity check)
        snapshot_path = os.path.join(OUTPUT_DIR, f"large_hoc_subset_{postfix}.png")
        plot_subset(positions, normals, BOX_SIZE, snapshot_path)

        info(f"[INFO] Finished LARGE config {cfg+1}/{N_CONFIGS}")

    pbar.close()
    info("\n[INFO] All LARGE configurations complete.")


if __name__ == "__main__":
    main()
