"""
hoc_structural_model.py

Structural generator for clay-like disc platelets forming "house-of-cards" (HoC) networks.

Features
--------
- Generate 3D configurations of disc-like clay platelets in a cubic box.
- Two growth modes:
    (1) "Seeds": new platelets placed far from existing ones
    (2) "Attached": new platelets added near existing ones with T/parallel/random
        orientation rules (HoC-like contacts).
- After building a configuration:
    * Build contact network using a fast cell-list + numba neighbour search.
    * Classify contacts as T / parallel / other based on normal–normal angle.
    * Compute degree distribution, contact fractions, cluster sizes.
    * Save:
        - coordinates (x,y,z,nx,ny,nz) → hoc_structure_coords_XXX.csv
        - structural stats → hoc_stats_XXX.txt
        - 3D snapshot with discs coloured by degree + colourbar → hoc_structure_XXX.png
- Single unified tqdm progress bar for all configs × particles.
"""

import os
import math
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from numba import njit, prange
from tqdm import tqdm


# ============================================================
# Global parameters (user-tunable)
# ============================================================

# Number of independent configurations
N_CONFIGS = 10

# Base random seed; actual seed = BASE_SEED + config_index
BASE_SEED = 0

# Number of platelets per configuration (typical: 1e3–2e4)
N_PARTICLES = 50000

# Disc geometry (nm)
RADIUS = 25.0      # platelet radius
THICKNESS = 1.0    # platelet thickness

# Simulation box: cube [0, BOX_SIZE]^3 (nm)
BOX_SIZE = 3000.0

# Output directory
OUTPUT_DIR = "hoc_output_prob0.8_random_n50000"

# Probability for a new platelet to be a "new seed"
#   With probability PROB_NEW_SEED:
#       - place a new free seed far from existing platelets
#   With probability 1 - PROB_NEW_SEED:
#       - attach to a parent platelet with T/parallel/random orientation
PROB_NEW_SEED = 0.8

# Orientation probabilities for ATTACHED platelets
PROB_T = 0        # prefer T-contact (≈90° between normals)
PROB_PARALLEL = 0 # parallel (≈0° or 180°)
PROB_RANDOM = 1   # fully random

# Angular spreads (deg) for orientation rules
ANGLE_SIGMA_T_DEG = 180.0         # spread around 90°
ANGLE_SIGMA_PARALLEL_DEG = 10.0  # spread around 0° or 180°

# Parent–child distance for attached platelets (nm)
CENTER_DISTANCE_MEAN = 2.2 * RADIUS
CENTER_DISTANCE_STD = 0.25 * RADIUS

# Contact detection thresholds
#   - CONTACT_DISTANCE_THRESHOLD : max center distance for "touching"
#   - T_ANGLE_CENTER_DEG, T_ANGLE_WIDTH_DEG : T-contact window
#   - PAR_ANGLE_WIDTH_DEG : parallel window (small angle after folding)
CONTACT_DISTANCE_THRESHOLD = 2.4 * RADIUS  # nm
T_ANGLE_CENTER_DEG = 90.0
T_ANGLE_WIDTH_DEG = 25.0
PAR_ANGLE_WIDTH_DEG = 20.0

# New seed placement: try to keep centers ≥ MIN_SEED_DISTANCE apart
MIN_SEED_DISTANCE = CONTACT_DISTANCE_THRESHOLD
MAX_SEED_TRIES = 1000

# Plotting
MAX_DISCS_TO_DRAW = 2000  # for very large N, draw only a subset


# ============================================================
# Helper: logging that doesn't break tqdm bar
# ============================================================

def info(msg: str) -> None:
    """
    Print informational messages via tqdm.write so that tqdm progress
    bar stays on a single line and isn't corrupted by prints.
    """
    tqdm.write(msg)


# ============================================================
# Basic vector + random utilities
# ============================================================

def random_unit_vector() -> np.ndarray:
    """
    Draw a random unit vector uniformly on the sphere.
    """
    z = 2.0 * np.random.rand() - 1.0
    phi = 2.0 * np.pi * np.random.rand()
    r_xy = math.sqrt(max(0.0, 1.0 - z*z))
    x = r_xy * math.cos(phi)
    y = r_xy * math.sin(phi)
    return np.array([x, y, z], dtype=float)


def random_angle(center_deg: float, sigma_deg: float) -> float:
    """
    Sample an angle (in radians) from N(center_deg, sigma_deg^2) in degrees.
    """
    return np.deg2rad(np.random.normal(center_deg, sigma_deg))


# ============================================================
# ClayPlatelet class
# ============================================================

class ClayPlatelet:
    """
    Disc-like clay platelet.

    Attributes
    ----------
    pos : (3,) ndarray
        Center-of-mass position in nm.
    n   : (3,) ndarray
        Unit normal vector of the disc.
    """

    __slots__ = ("pos", "n")

    def __init__(self, pos, n) -> None:
        pos = np.array(pos, dtype=float)
        n = np.array(n, dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise ValueError("Normal vector too small.")
        self.pos = pos
        self.n = n / norm

    def basis(self):
        """
        Construct orthonormal basis (u, v, n) for the disc frame.

        Returns
        -------
        u, v, n : (3,) ndarrays
            u, v lie in disc plane; n is the stored normal.
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
# Seed placement with minimum distance constraint
# ============================================================

def place_new_seed(existing_positions: np.ndarray) -> np.ndarray:
    """
    Place a new "seed" platelet, trying to keep it at least MIN_SEED_DISTANCE
    away from all existing centers.

    If this fails after MAX_SEED_TRIES, returns the last candidate position.

    Parameters
    ----------
    existing_positions : (M,3) ndarray or None

    Returns
    -------
    pos_new : (3,) ndarray
    """
    if existing_positions is None or existing_positions.size == 0:
        return np.random.rand(3) * BOX_SIZE

    last_candidate = None
    for _ in range(MAX_SEED_TRIES):
        pos = np.random.rand(3) * BOX_SIZE
        dists = np.linalg.norm(existing_positions - pos, axis=1)
        if dists.min() >= MIN_SEED_DISTANCE:
            return pos
        last_candidate = pos

    return last_candidate if last_candidate is not None else (np.random.rand(3) * BOX_SIZE)


# ============================================================
# Orientation rules for attached platelets
# ============================================================

def make_T_normal(parent_n: np.ndarray) -> np.ndarray:
    """
    Generate new normal approximately T-contact-like w.r.t parent_n
    (angle ≈ 90° with Gaussian spread ANGLE_SIGMA_T_DEG).
    """
    parent_n = parent_n / (np.linalg.norm(parent_n) + 1e-12)

    # vector orthogonal to parent_n
    base = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(base, parent_n)) > 0.9:
        base = np.array([0.0, 1.0, 0.0])
    b = np.cross(parent_n, base)
    b /= (np.linalg.norm(b) + 1e-12)

    angle = random_angle(90.0, ANGLE_SIGMA_T_DEG)
    n_new = math.cos(angle)*parent_n + math.sin(angle)*b
    return n_new / (np.linalg.norm(n_new) + 1e-12)


def make_parallel_normal(parent_n: np.ndarray) -> np.ndarray:
    """
    Generate new normal approximately parallel (0° or 180°) to parent_n.
    """
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
# Structure builder (uses unified progress bar)
# ============================================================

def build_structure_with_progress(n_particles: int, pbar: tqdm) -> list:
    """
    Build one HoC-like configuration of n_particles platelets.

    Each added platelet increments the shared progress bar by 1.

    Returns
    -------
    platelets : list[ClayPlatelet]
    """
    platelets = []

    # First seed at box center
    center = np.array([BOX_SIZE/2.0, BOX_SIZE/2.0, BOX_SIZE/2.0])
    n0 = random_unit_vector()
    platelets.append(ClayPlatelet(center, n0))
    pbar.update(1)

    # Grow the rest
    for _ in range(1, n_particles):
        existing_positions = np.array([p.pos for p in platelets], dtype=float)

        if np.random.rand() < PROB_NEW_SEED:
            # New seed, with distance constraint
            pos_new = place_new_seed(existing_positions)
            n_new = random_unit_vector()
            platelets.append(ClayPlatelet(pos_new, n_new))

        else:
            # Attached platelet
            parent_idx = np.random.randint(len(platelets))
            parent = platelets[parent_idx]

            r = np.random.rand()
            if r < PROB_T:
                n_new = make_T_normal(parent.n)
            elif r < PROB_T + PROB_PARALLEL:
                n_new = make_parallel_normal(parent.n)
            else:
                n_new = random_unit_vector()

            dist = np.random.normal(CENTER_DISTANCE_MEAN, CENTER_DISTANCE_STD)
            dist = max(dist, 1.5*RADIUS)
            d_hat = random_unit_vector()
            pos_new = parent.pos + dist*d_hat
            pos_new = np.clip(pos_new, 0.0, BOX_SIZE)

            platelets.append(ClayPlatelet(pos_new, n_new))

        pbar.update(1)

    return platelets


# ============================================================
# Fast neighbour search via cell lists (numba)
# ============================================================

@njit
def _build_cell_list(positions, box_size, cell_size):
    """
    Build a simple 3D cell list for neighbour search.
    """
    N = positions.shape[0]
    ncell = int(box_size // cell_size) + 1

    counts = np.zeros((ncell, ncell, ncell), dtype=np.int64)
    cell_index = np.empty((N, 3), dtype=np.int64)

    for i in range(N):
        x, y, z = positions[i, 0], positions[i, 1], positions[i, 2]
        cx = int(x // cell_size)
        cy = int(y // cell_size)
        cz = int(z // cell_size)
        if cx < 0: cx = 0
        if cy < 0: cy = 0
        if cz < 0: cz = 0
        if cx >= ncell: cx = ncell-1
        if cy >= ncell: cy = ncell-1
        if cz >= ncell: cz = ncell-1
        cell_index[i, 0] = cx
        cell_index[i, 1] = cy
        cell_index[i, 2] = cz
        counts[cx, cy, cz] += 1

    offsets = np.zeros_like(counts)
    acc = 0
    for ix in range(ncell):
        for iy in range(ncell):
            for iz in range(ncell):
                offsets[ix, iy, iz] = acc
                acc += counts[ix, iy, iz]

    cell_particles = np.empty(N, dtype=np.int64)
    temp_counts = np.zeros_like(counts)

    for i in range(N):
        cx, cy, cz = cell_index[i, 0], cell_index[i, 1], cell_index[i, 2]
        idx = offsets[cx, cy, cz] + temp_counts[cx, cy, cz]
        cell_particles[idx] = i
        temp_counts[cx, cy, cz] += 1

    return cell_index, offsets, counts, cell_particles, ncell


@njit(parallel=True)
def _find_contacts_with_cells(positions, normals,
                              box_size, cutoff,
                              cell_size):
    """
    Use cell lists + numba parallel loops to find contacting pairs.

    Returns
    -------
    pair_i, pair_j, pair_ang : 1D arrays
        Contact pairs (i<j) and their normal–normal angles (degrees).
    """
    N = positions.shape[0]
    cutoff2 = cutoff * cutoff

    cell_index, offsets, counts, cell_particles, ncell = \
        _build_cell_list(positions, box_size, cell_size)

    max_pairs = 40 * N  # heuristic upper bound
    pair_i = np.empty(max_pairs, dtype=np.int64)
    pair_j = np.empty(max_pairs, dtype=np.int64)
    pair_ang = np.empty(max_pairs, dtype=np.float64)

    pair_count_arr = np.zeros(1, dtype=np.int64)

    for i in prange(N):
        cx, cy, cz = cell_index[i, 0], cell_index[i, 1], cell_index[i, 2]

        for dx in (-1, 0, 1):
            nx = cx + dx
            if nx < 0 or nx >= ncell:
                continue
            for dy in (-1, 0, 1):
                ny = cy + dy
                if ny < 0 or ny >= ncell:
                    continue
                for dz in (-1, 0, 1):
                    nz = cz + dz
                    if nz < 0 or nz >= ncell:
                        continue

                    start = offsets[nx, ny, nz]
                    cnt = counts[nx, ny, nz]
                    for k in range(cnt):
                        j = cell_particles[start + k]
                        if j <= i:
                            continue

                        dx_ = positions[j, 0] - positions[i, 0]
                        dy_ = positions[j, 1] - positions[i, 1]
                        dz_ = positions[j, 2] - positions[i, 2]
                        r2 = dx_*dx_ + dy_*dy_ + dz_*dz_
                        if r2 > cutoff2:
                            continue

                        # angle between normals
                        n1x, n1y, n1z = normals[i, 0], normals[i, 1], normals[i, 2]
                        n2x, n2y, n2z = normals[j, 0], normals[j, 1], normals[j, 2]
                        dot = n1x*n2x + n1y*n2y + n1z*n2z
                        if dot > 1.0:
                            dot = 1.0
                        if dot < -1.0:
                            dot = -1.0
                        ang_rad = math.acos(dot)
                        ang_deg = ang_rad * 180.0 / math.pi

                        idx_pair = pair_count_arr[0]
                        if idx_pair >= max_pairs:
                            continue
                        pair_i[idx_pair] = i
                        pair_j[idx_pair] = j
                        pair_ang[idx_pair] = ang_deg
                        pair_count_arr[0] = idx_pair + 1

    pair_count = pair_count_arr[0]
    return pair_i[:pair_count], pair_j[:pair_count], pair_ang[:pair_count]


def classify_contact(angle_deg: float) -> str:
    """
    Classify contact type based on normal–normal angle in degrees.

    Returns
    -------
    "T", "parallel", or "other".
    """
    # fold angle into [0, 90] for checking parallel
    if angle_deg > 90.0:
        angle_eff = 180.0 - angle_deg
    else:
        angle_eff = angle_deg

    # T-contact window around T_ANGLE_CENTER_DEG
    if abs(angle_deg - T_ANGLE_CENTER_DEG) < T_ANGLE_WIDTH_DEG:
        return "T"

    # near-parallel if folded angle is small
    if angle_eff < PAR_ANGLE_WIDTH_DEG:
        return "parallel"

    return "other"


def build_contact_network_fast(platelets: list):
    """
    Build contact network for the given platelets.

    Uses a cell list with cutoff = CONTACT_DISTANCE_THRESHOLD.

    Returns
    -------
    adjacency : list[set[int]]
        Neighbour indices for each node.
    contact_types : dict[(int,int) -> str]
        Contact type ("T", "parallel", "other") for each pair (i,j) with i<j.
    angles_deg : (M,) ndarray
        All normal–normal angles (deg) for contacting pairs.
    """
    N = len(platelets)
    positions = np.empty((N, 3), dtype=np.float64)
    normals = np.empty((N, 3), dtype=np.float64)
    for i, p in enumerate(platelets):
        positions[i, :] = p.pos
        normals[i, :] = p.n

    cell_size = CONTACT_DISTANCE_THRESHOLD
    i_idx, j_idx, angles = _find_contacts_with_cells(
        positions, normals, BOX_SIZE,
        CONTACT_DISTANCE_THRESHOLD,
        cell_size
    )

    adjacency = [set() for _ in range(N)]
    contact_types = {}
    angle_list = []

    for i, j, ang in zip(i_idx, j_idx, angles):
        ctype = classify_contact(float(ang))
        adjacency[int(i)].add(int(j))
        adjacency[int(j)].add(int(i))
        contact_types[(int(i), int(j))] = ctype
        angle_list.append(float(ang))

    return adjacency, contact_types, np.array(angle_list, dtype=float)


# ============================================================
# Graph utilities (degree distribution, clusters)
# ============================================================

def degree_distribution(adjacency: list) -> np.ndarray:
    """
    Return degree for each node.
    """
    return np.array([len(neigh) for neigh in adjacency], dtype=int)


def connected_components(adjacency: list) -> list:
    """
    Compute connected components of an undirected graph.
    """
    N = len(adjacency)
    visited = [False]*N
    comps = []

    for i in range(N):
        if visited[i]:
            continue
        queue = deque([i])
        visited[i] = True
        comp = []
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        comps.append(comp)

    return comps


# ============================================================
# Visualization: discs coloured by degree with colourbar
# ============================================================

def disc_polygon(platelet: ClayPlatelet, rad: float = RADIUS, n_segments: int = 32):
    """
    Generate a polygon approximating the disc perimeter in 3D.
    """
    u, v, n = platelet.basis()
    angles = np.linspace(0.0, 2.0*np.pi, n_segments, endpoint=True)
    verts = []
    for a in angles:
        verts.append(platelet.pos + rad*np.cos(a)*u + rad*np.sin(a)*v)
    return verts


def plot_structure_colored_by_degree(platelets: list,
                                     degrees: np.ndarray,
                                     phi: float,
                                     filename: str) -> None:
    """
    Plot 3D structure where each disc is coloured by its degree (number of contacts),
    with a colourbar showing degree.

    Parameters
    ----------
    platelets : list[ClayPlatelet]
    degrees   : (N,) ndarray
        Degree for each platelet.
    phi       : float
        Volume fraction, printed in the title.
    filename  : str
        PNG output path.
    """
    N = len(platelets)

    if N > MAX_DISCS_TO_DRAW:
        draw_idx = np.random.choice(N, MAX_DISCS_TO_DRAW, replace=False)
    else:
        draw_idx = np.arange(N)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    deg_min = degrees.min()
    deg_max = degrees.max()
    if deg_min == deg_max:
        deg_max = deg_min + 1
    norm = Normalize(vmin=deg_min, vmax=deg_max)
    cmap = plt.get_cmap("viridis")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # draw discs
    for idx in draw_idx:
        p = platelets[idx]
        d = degrees[idx]
        color = cmap(norm(d))
        verts = disc_polygon(p)
        poly = Poly3DCollection([verts], alpha=0.85)
        poly.set_edgecolor("k")
        poly.set_facecolor(color)
        ax.add_collection3d(poly)

    centers = np.array([p.pos for p in platelets])
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
               s=3, c="black", alpha=0.3)

    ax.set_xlim(0.0, BOX_SIZE)
    ax.set_ylim(0.0, BOX_SIZE)
    ax.set_zlim(0.0, BOX_SIZE)
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")
    ax.set_box_aspect((1.0, 1.0, 1.0))

    ax.set_title(f"House-of-cards structure\n(color = degree, φ = {phi:.4e})")

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Degree (number of contacts)")

    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()
    info(f"[INFO] Saved structure snapshot: {filename}")


# ============================================================
# Volume fraction + saving coordinates + stats
# ============================================================

def compute_volume_fraction(n_particles: int) -> float:
    """
    Estimate disc volume fraction φ = N * V_disc / V_box.
    """
    v_disc = math.pi * RADIUS**2 * THICKNESS
    v_box = BOX_SIZE**3
    phi = n_particles * v_disc / v_box
    return phi


def save_coordinates(platelets: list, path: str) -> None:
    """
    Save x,y,z,nx,ny,nz for each platelet as CSV.
    """
    arr = np.array([
        [p.pos[0], p.pos[1], p.pos[2],
         p.n[0],   p.n[1],   p.n[2]]
        for p in platelets
    ])
    header = "x,y,z,nx,ny,nz"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")
    info(f"[INFO] Saved coordinates: {path}")


def save_stats(stats: dict, path: str) -> None:
    """
    Save structural statistics to a text file (key: value).
    """
    lines = []
    lines.append("# House-of-cards structural statistics\n\n")
    for k, v in stats.items():
        lines.append(f"{k}: {v}\n")
    with open(path, "w") as f:
        f.writelines(lines)
    info(f"[INFO] Saved stats: {path}")


# ============================================================
# Main driver
# ============================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # unified progress bar over all configs × particles
    total_steps = N_CONFIGS * N_PARTICLES
    pbar = tqdm(total=total_steps, desc="Total progress", leave=True)

    for cfg_idx in range(N_CONFIGS):
        seed = BASE_SEED + cfg_idx
        np.random.seed(seed)

        info(f"\n[INFO] Starting config {cfg_idx+1}/{N_CONFIGS}, seed={seed}")

        # 1) build structure
        platelets = build_structure_with_progress(N_PARTICLES, pbar)
        Np = len(platelets)
        phi = compute_volume_fraction(Np)
        info(f"[INFO]   N_particles = {Np}, volume fraction φ ≈ {phi:.4e}")

        # 2) build contact network
        info("[INFO]   Building contact network (numba + cell list)...")
        adjacency, contact_types, angles_deg = build_contact_network_fast(platelets)

        degrees = degree_distribution(adjacency)
        comps = connected_components(adjacency)
        comp_sizes = [len(c) for c in comps]

        n_contacts = len(contact_types)
        n_T = sum(1 for c in contact_types.values() if c == "T")
        n_parallel = sum(1 for c in contact_types.values() if c == "parallel")
        n_other = n_contacts - n_T - n_parallel

        T_fraction = n_T / n_contacts if n_contacts > 0 else 0.0
        parallel_fraction = n_parallel / n_contacts if n_contacts > 0 else 0.0

        if angles_deg.size > 0:
            angles_eff = np.where(angles_deg > 90.0, 180.0 - angles_deg, angles_deg)
            mean_angle = float(angles_deg.mean())
            std_angle = float(angles_deg.std())
            mean_angle_eff = float(angles_eff.mean())
            std_angle_eff = float(angles_eff.std())
        else:
            mean_angle = float("nan")
            std_angle = float("nan")
            mean_angle_eff = float("nan")
            std_angle_eff = float("nan")

        mean_degree = float(degrees.mean())
        max_degree = int(degrees.max()) if degrees.size > 0 else 0
        largest_cluster = max(comp_sizes) if comp_sizes else 0
        num_clusters = len(comp_sizes)

        # 3) prepare stats dict
        v_disc = math.pi * RADIUS**2 * THICKNESS
        v_box = BOX_SIZE**3

        stats = {
            "config_index": cfg_idx,
            "seed": seed,
            "N_particles": Np,
            "disc_volume_nm3": v_disc,
            "box_volume_nm3": v_box,
            "volume_fraction_phi": phi,
            "PROB_NEW_SEED": PROB_NEW_SEED,
            "CONTACT_DISTANCE_THRESHOLD_nm": CONTACT_DISTANCE_THRESHOLD,
            "N_contacts": n_contacts,
            "N_T_contacts": n_T,
            "N_parallel_contacts": n_parallel,
            "N_other_contacts": n_other,
            "T_contact_fraction": T_fraction,
            "parallel_contact_fraction": parallel_fraction,
            "mean_contact_angle_deg": mean_angle,
            "std_contact_angle_deg": std_angle,
            "mean_contact_angle_folded_deg": mean_angle_eff,
            "std_contact_angle_folded_deg": std_angle_eff,
            "mean_degree": mean_degree,
            "max_degree": max_degree,
            "num_clusters": num_clusters,
            "largest_cluster_size": largest_cluster,
        }

        # 4) save outputs
        postfix = f"{cfg_idx:03d}"
        coords_path = os.path.join(OUTPUT_DIR, f"hoc_structure_coords_{postfix}.csv")
        stats_path = os.path.join(OUTPUT_DIR, f"hoc_stats_{postfix}.txt")
        fig_path = os.path.join(OUTPUT_DIR, f"hoc_structure_{postfix}.png")

        save_coordinates(platelets, coords_path)
        save_stats(stats, stats_path)
        plot_structure_colored_by_degree(platelets, degrees, phi, fig_path)

        info(f"[INFO] Finished config {cfg_idx+1}/{N_CONFIGS}")

    pbar.close()
    info("\n[INFO] All configurations complete.")


if __name__ == "__main__":
    main()
