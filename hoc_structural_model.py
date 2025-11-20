"""
hoc_structural_model.py

Fast structural generator for clay-like platelet "house-of-cards" networks.

Features:
- Purely structural (no electrostatics), with T-contact / parallel / random contact rules.
- Some platelets are "free seeds" (random position + random orientation).
- PROB_NEW_SEED controls how often a new platelet is created as a free seed
  vs attached to an existing platelet.
- New seeds are placed such that they are (attempted to be) at least
  MIN_SEED_DISTANCE away from all existing platelets.
- Multiple configurations with the same parameter set.
- Fast neighbour search using cell lists + numba JIT (O(N) scaling for sparse systems).
- Volume fraction phi is computed and printed, saved in stats, and shown in plot title.
- All outputs are saved under OUTPUT_DIR with configuration index postfix.
- Contact network is ALWAYS computed based on actual distances and normals.
- Optional progress bar via ENABLE_PROGRESS and tqdm (if installed).
"""

import os
import math
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from numba import njit, prange

# =========================
#  Global parameters
# =========================

# Number of configurations to generate
N_CONFIGS = 20
BASE_SEED = 0

# Show progress bar? (requires tqdm; if 없으면 자동으로 조용히 넘어감)
ENABLE_PROGRESS = True

# Number of platelets per configuration (you can set 1000~20000)
N_PARTICLES = 1000

# Geometry
RADIUS = 25.0          # nm
THICKNESS = 1.0        # nm
BOX_SIZE = 2000.0      # nm  (typical large box)

# Output folder
OUTPUT_DIR = "hoc_output_new0.5"

# Probability that a new platelet is a "new seed"
#  - with probability PROB_NEW_SEED: random position + random orientation (but separated)
#  - with probability 1 - PROB_NEW_SEED: attach to an existing platelet
PROB_NEW_SEED = 1

# Contact generation probabilities for ATTACHED platelets
PROB_T = 0.6           # probability to try a T-contact orientation
PROB_PARALLEL = 0.3    # probability to try parallel orientation
PROB_RANDOM = 0.1      # probability to use random orientation

# Angular noise for orientation (in degrees)
ANGLE_SIGMA_T_DEG = 10.0        # spread around 90° for T-contact
ANGLE_SIGMA_PARALLEL_DEG = 10.0 # spread around 0° or 180° for parallel

# Placement distance scale from parent center (for attached platelets)
CENTER_DISTANCE_MEAN = 2.2 * RADIUS   # nm
CENTER_DISTANCE_STD = 0.2 * RADIUS    # nm

# Contact classification thresholds
CONTACT_DISTANCE_THRESHOLD = 2.4 * RADIUS  # nm, max center distance to consider "touching"
T_ANGLE_CENTER_DEG = 90.0
T_ANGLE_WIDTH_DEG = 25.0
PAR_ANGLE_WIDTH_DEG = 20.0

# ---- NEW: seed placement control ----
# 새 seed는 기존 모든 platelet과 이 거리 이상 떨어져 있게 시도
MIN_SEED_DISTANCE = CONTACT_DISTANCE_THRESHOLD   # 예: contact threshold 이상
MAX_SEED_TRIES = 1000   # 너무 빡센 조건에서 무한 루프 방지용

# Plotting control
MAX_DISCS_TO_DRAW = 1000  # if N > this, draw only a random subset of discs

# Try importing tqdm for optional progress bar
try:
    from tqdm import trange
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


# =========================
#  ClayPlatelet class
# =========================

class ClayPlatelet:
    """
    Simple rigid disc-like platelet.

    pos : (3,) center position
    n   : (3,) unit normal vector
    """

    __slots__ = ("pos", "n")

    def __init__(self, pos, normal):
        self.pos = np.array(pos, dtype=float)

        n = np.array(normal, dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise ValueError("Normal vector too small.")
        self.n = n / norm

    def basis_vectors(self):
        """
        Return orthonormal basis (u, v, n) for the disc.
        u, v lie in the disc plane, n is the normal.
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


# =========================
#  Random utilities
# =========================

def random_unit_vector():
    """
    Uniform random unit vector on the sphere.
    """
    z = 2.0 * np.random.rand() - 1.0
    phi = 2.0 * math.pi * np.random.rand()
    r_xy = math.sqrt(max(0.0, 1.0 - z*z))
    x = r_xy * math.cos(phi)
    y = r_xy * math.sin(phi)
    return np.array([x, y, z])


def random_distance():
    """
    Sample center-to-center distance between parent and child platelet.
    Enforce positivity and some lower bound to avoid exact overlap.
    """
    d = np.random.normal(CENTER_DISTANCE_MEAN, CENTER_DISTANCE_STD)
    d = max(d, 1.5 * RADIUS)
    return d


def random_angle_gaussian(center_deg, sigma_deg):
    """
    Sample an angle (in radians) from a Gaussian around center_deg (degrees).
    """
    a_deg = np.random.normal(center_deg, sigma_deg)
    return np.deg2rad(a_deg)


# =========================
#  Seed placement with minimum separation
# =========================

def place_new_seed_separated(existing_positions):
    """
    Place a new seed so that it is (attempted to be) at least MIN_SEED_DISTANCE
    away from all existing platelets (by center distance).

    existing_positions : (M,3) array or None

    - If conditions are satisfied, return that position.
    - If MAX_SEED_TRIES attempts fail (too dense / high phi), return the last candidate
      so that the algorithm does not get stuck (in that case some seeds may be closer
      than MIN_SEED_DISTANCE).
    """
    if existing_positions is None or existing_positions.shape[0] == 0:
        return np.random.rand(3) * BOX_SIZE

    last_pos = None
    for _ in range(MAX_SEED_TRIES):
        pos = np.random.rand(3) * BOX_SIZE
        dists = np.linalg.norm(existing_positions - pos, axis=1)
        if dists.min() >= MIN_SEED_DISTANCE:
            return pos
        last_pos = pos

    # fallback: couldn't find fully separated position; return last candidate
    return last_pos if last_pos is not None else (np.random.rand(3) * BOX_SIZE)


# =========================
#  Orientation generation for new platelets
# =========================

def make_T_orientation(parent_normal):
    """
    Generate a new normal that is approximately perpendicular to parent_normal.
    """
    n_p = parent_normal / (np.linalg.norm(parent_normal) + 1e-12)

    base = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(base, n_p)) > 0.9:
        base = np.array([0.0, 1.0, 0.0])
    b = np.cross(n_p, base)
    b /= (np.linalg.norm(b) + 1e-12)

    angle = random_angle_gaussian(90.0, ANGLE_SIGMA_T_DEG)
    n_new = math.cos(angle) * n_p + math.sin(angle) * b
    n_new /= (np.linalg.norm(n_new) + 1e-12)
    return n_new


def make_parallel_orientation(parent_normal):
    """
    Generate a new normal that is approximately parallel (0° or 180°) to parent_normal.
    Randomly choose 0° or 180°, then add Gaussian noise.
    """
    n_p = parent_normal / (np.linalg.norm(parent_normal) + 1e-12)

    if np.random.rand() < 0.5:
        center_deg = 0.0
    else:
        center_deg = 180.0

    angle = random_angle_gaussian(center_deg, ANGLE_SIGMA_PARALLEL_DEG)

    base = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(base, n_p)) > 0.9:
        base = np.array([0.0, 1.0, 0.0])
    axis = np.cross(n_p, base)
    axis /= (np.linalg.norm(axis) + 1e-12)

    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    I = np.eye(3)
    R = I + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    n_new = R.dot(n_p)
    n_new /= (np.linalg.norm(n_new) + 1e-12)
    return n_new


# =========================
#  Structure generation
# =========================

def generate_house_of_cards_structure(n_particles=N_PARTICLES):
    """
    Build a structural "house-of-cards-like" network of N platelets.

    - With probability PROB_NEW_SEED: new free seed (random position + random orientation),
      placed so that it is at least MIN_SEED_DISTANCE away from existing platelets.
    - With probability 1 - PROB_NEW_SEED: attach to an existing platelet
      with T/parallel/random orientation rules.
    """
    platelets = []

    # 1. first seed at box center
    center = np.array([BOX_SIZE/2, BOX_SIZE/2, BOX_SIZE/2])
    n0 = random_unit_vector()
    platelets.append(ClayPlatelet(center, n0))

    # 2. grow rest
    for k in range(1, n_particles):
        if np.random.rand() < PROB_NEW_SEED:
            # new free seed, separated from existing ones
            existing_positions = np.array([p.pos for p in platelets], dtype=float)
            pos_new = place_new_seed_separated(existing_positions)
            n_new = random_unit_vector()
            platelets.append(ClayPlatelet(pos_new, n_new))
            continue

        # attached mode
        parent_idx = np.random.randint(len(platelets))
        parent = platelets[parent_idx]

        r = np.random.rand()
        if r < PROB_T:
            contact_type = "T"
        elif r < PROB_T + PROB_PARALLEL:
            contact_type = "parallel"
        else:
            contact_type = "random"

        if contact_type == "T":
            n_new = make_T_orientation(parent.n)
        elif contact_type == "parallel":
            n_new = make_parallel_orientation(parent.n)
        else:
            n_new = random_unit_vector()

        d_hat = random_unit_vector()
        dist = random_distance()
        pos_new = parent.pos + dist * d_hat
        pos_new = np.clip(pos_new, 0.0, BOX_SIZE)

        platelets.append(ClayPlatelet(pos_new, n_new))

    return platelets


# =========================
#  Fast contact search with numba + cell list
# =========================

@njit
def _build_cell_list(positions, box_size, cell_size):
    """
    Build a simple 3D cell list for neighbour search (numba JIT).
    """
    N = positions.shape[0]
    ncell = int(box_size // cell_size) + 1
    counts = np.zeros((ncell, ncell, ncell), dtype=np.int64)
    cell_index = np.empty((N, 3), dtype=np.int64)

    # count per cell
    for i in range(N):
        x = positions[i,0]
        y = positions[i,1]
        z = positions[i,2]
        cx = int(x // cell_size)
        cy = int(y // cell_size)
        cz = int(z // cell_size)
        if cx < 0: cx = 0
        if cy < 0: cy = 0
        if cz < 0: cz = 0
        if cx >= ncell: cx = ncell - 1
        if cy >= ncell: cy = ncell - 1
        if cz >= ncell: cz = ncell - 1
        cell_index[i,0] = cx
        cell_index[i,1] = cy
        cell_index[i,2] = cz
        counts[cx,cy,cz] += 1

    # prefix sum offsets
    offsets = np.zeros_like(counts)
    acc = 0
    for ix in range(ncell):
        for iy in range(ncell):
            for iz in range(ncell):
                offsets[ix,iy,iz] = acc
                acc += counts[ix,iy,iz]

    # fill cell_particles
    cell_particles = np.empty(N, dtype=np.int64)
    temp_counts = np.zeros_like(counts)

    for i in range(N):
        cx = cell_index[i,0]
        cy = cell_index[i,1]
        cz = cell_index[i,2]
        idx = offsets[cx,cy,cz] + temp_counts[cx,cy,cz]
        cell_particles[idx] = i
        temp_counts[cx,cy,cz] += 1

    return cell_index, offsets, counts, cell_particles, ncell


@njit(parallel=True)
def _find_contacts_with_cells(positions, normals,
                              box_size, cutoff,
                              cell_size):
    """
    Use cell list and numba-parallel loops to find contacting pairs.
    Returns arrays (pair_i, pair_j, angles_deg).
    """
    N = positions.shape[0]
    cutoff2 = cutoff * cutoff

    cell_index, offsets, counts, cell_particles, ncell = \
        _build_cell_list(positions, box_size, cell_size)

    # heuristic: allow up to 40 neighbours per particle
    max_pairs = 40 * N
    pair_i = np.empty(max_pairs, dtype=np.int64)
    pair_j = np.empty(max_pairs, dtype=np.int64)
    pair_ang = np.empty(max_pairs, dtype=np.float64)

    pair_count_arr = np.zeros(1, dtype=np.int64)  # mutable scalar

    for i in prange(N):
        cx = cell_index[i,0]
        cy = cell_index[i,1]
        cz = cell_index[i,2]

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

                    start = offsets[nx,ny,nz]
                    cnt = counts[nx,ny,nz]
                    for k in range(cnt):
                        j = cell_particles[start + k]
                        if j <= i:
                            continue

                        dx_ = positions[j,0] - positions[i,0]
                        dy_ = positions[j,1] - positions[i,1]
                        dz_ = positions[j,2] - positions[i,2]
                        r2 = dx_*dx_ + dy_*dy_ + dz_*dz_
                        if r2 > cutoff2:
                            continue

                        # angle between normals i and j
                        n1x = normals[i,0]
                        n1y = normals[i,1]
                        n1z = normals[i,2]
                        n2x = normals[j,0]
                        n2y = normals[j,1]
                        n2z = normals[j,2]
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


def classify_contact(angle_deg):
    """
    Classify contact type based on angle between normals (in degrees).

    Returns one of: "T", "parallel", "other"
    """
    if angle_deg > 90.0:
        angle_eff = 180.0 - angle_deg
    else:
        angle_eff = angle_deg

    if abs(angle_deg - T_ANGLE_CENTER_DEG) < T_ANGLE_WIDTH_DEG:
        return "T"

    if angle_eff < PAR_ANGLE_WIDTH_DEG:
        return "parallel"

    return "other"


def build_contact_network_fast(platelets):
    """
    Wrapper that:
      - extracts positions/normals into NumPy arrays
      - calls numba-accelerated neighbour search
      - builds adjacency list & contact type dict in Python
    """
    N = len(platelets)
    positions = np.empty((N, 3), dtype=np.float64)
    normals = np.empty((N, 3), dtype=np.float64)
    for i, p in enumerate(platelets):
        positions[i, :] = p.pos
        normals[i, :] = p.n

    cell_size = CONTACT_DISTANCE_THRESHOLD  # reasonable choice
    i_idx, j_idx, angles_deg = _find_contacts_with_cells(
        positions, normals,
        BOX_SIZE, CONTACT_DISTANCE_THRESHOLD,
        cell_size
    )

    adjacency = [set() for _ in range(N)]
    contact_types = {}
    angles_list = []

    for i, j, ang in zip(i_idx, j_idx, angles_deg):
        ctype = classify_contact(float(ang))
        adjacency[i].add(j)
        adjacency[j].add(i)
        contact_types[(int(i), int(j))] = ctype
        angles_list.append(float(ang))

    return adjacency, contact_types, np.array(angles_list, dtype=float)


# =========================
#  Network connectivity analysis
# =========================

def connected_components(adjacency):
    """
    Find connected components in an undirected graph.
    """
    N = len(adjacency)
    visited = [False] * N
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


def degree_distribution(adjacency):
    """
    Return array of degrees for each node.
    """
    return np.array([len(neigh) for neigh in adjacency], dtype=int)


# =========================
#  Visualization
# =========================

def disc_polygon_global(platelet, radius=RADIUS, n_segments=32):
    """
    Construct a polygon approximating the disc face in 3D for plotting.
    """
    u, v, n = platelet.basis_vectors()
    angles = np.linspace(0, 2*np.pi, n_segments, endpoint=True)
    verts = []
    for ang in angles:
        x = radius * math.cos(ang)
        y = radius * math.sin(ang)
        local = x * u + y * v
        g = platelet.pos + local
        verts.append(g)
    return verts


def plot_structure(platelets, degrees, phi, filename):
    """
    Plot the structure in 3D as discs colored by degree.
    For large N, draw only a subset of discs to keep plotting manageable.
    """
    N = len(platelets)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    deg_min = degrees.min()
    deg_max = degrees.max()
    if deg_min == deg_max:
        deg_max += 1
    norm = Normalize(vmin=deg_min, vmax=deg_max)
    cmap = plt.get_cmap("viridis")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # subset for drawing discs
    if N > MAX_DISCS_TO_DRAW:
        draw_idx = np.random.choice(N, size=MAX_DISCS_TO_DRAW, replace=False)
    else:
        draw_idx = np.arange(N, dtype=int)

    for idx in draw_idx:
        p = platelets[idx]
        deg = degrees[idx]
        color = cmap(norm(deg))
        verts = disc_polygon_global(p, radius=RADIUS, n_segments=32)
        poly = Poly3DCollection([verts], alpha=0.8)
        poly.set_edgecolor("k")
        poly.set_facecolor(color)
        ax.add_collection3d(poly)

    centers = np.array([p.pos for p in platelets])
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
               s=3, c="black", alpha=0.4)

    ax.set_xlim(0, BOX_SIZE)
    ax.set_ylim(0, BOX_SIZE)
    ax.set_zlim(0, BOX_SIZE)
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"House-of-cards structural model\n(color = degree, φ = {phi:.4e})")

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Degree (number of neighbours)")

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"[INFO] Structure figure saved to {filename}")


# =========================
#  Saving results
# =========================

def save_coordinates(platelets, filename):
    """
    Save x,y,z,nx,ny,nz for all platelets.
    """
    arr = []
    for p in platelets:
        arr.append([p.pos[0], p.pos[1], p.pos[2],
                    p.n[0], p.n[1], p.n[2]])
    arr = np.array(arr, dtype=float)
    header = "x,y,z,nx,ny,nz"
    np.savetxt(filename, arr, delimiter=",", header=header, comments="")
    print(f"[INFO] Coordinates saved to {filename}")


def save_stats(stats, filename):
    """
    Save structural summary to a text file.
    """
    lines = []
    lines.append("# House-of-cards structural model statistics\n\n")
    for key, val in stats.items():
        lines.append(f"{key}: {val}\n")

    with open(filename, "w") as f:
        f.writelines(lines)
    print(f"[INFO] Stats saved to {filename}")


# =========================
#  Volume fraction
# =========================

def compute_volume_fraction(n_particles):
    """
    Compute disc volume fraction in the box.
    """
    disc_volume = math.pi * RADIUS**2 * THICKNESS    # nm^3 per disc
    box_volume = BOX_SIZE**3                         # nm^3
    phi = n_particles * disc_volume / box_volume
    return phi, disc_volume, box_volume


# =========================
#  Main
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # progress bar wrapper
    if ENABLE_PROGRESS and HAVE_TQDM:
        cfg_iter = trange(N_CONFIGS, desc="Configs")
    else:
        cfg_iter = range(N_CONFIGS)

    for cfg_idx in cfg_iter:
        seed = BASE_SEED + cfg_idx
        np.random.seed(seed)

        print(f"\n[INFO] Configuration {cfg_idx+1}/{N_CONFIGS}, seed={seed}")
        print("[INFO] Generating house-of-cards-like structure (with seeds)...")
        platelets = generate_house_of_cards_structure(N_PARTICLES)
        print(f"[INFO] Generated {len(platelets)} platelets.")

        phi, disc_volume, box_volume = compute_volume_fraction(len(platelets))
        print(f"[INFO] Disc volume (per platelet): {disc_volume:.3f} nm^3")
        print(f"[INFO] Box volume               : {box_volume:.3f} nm^3")
        print(f"[INFO] Volume fraction φ        : {phi:.6e}")

        print("[INFO] Building contact network (fast numba + cell list)...")
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
            angles_eff = np.where(angles_deg > 90.0,
                                  180.0 - angles_deg,
                                  angles_deg)
            mean_angle = angles_deg.mean()
            std_angle = angles_deg.std()
            mean_angle_eff = angles_eff.mean()
            std_angle_eff = angles_eff.std()
        else:
            mean_angle = std_angle = mean_angle_eff = std_angle_eff = float("nan")

        mean_degree = degrees.mean()
        max_degree = degrees.max()
        largest_cluster = max(comp_sizes) if comp_sizes else 0
        num_clusters = len(comp_sizes)

        stats = {
            "config_index": cfg_idx,
            "seed": seed,
            "N_particles": len(platelets),
            "disc_volume_nm3": disc_volume,
            "box_volume_nm3": box_volume,
            "volume_fraction_phi": phi,
            "PROB_NEW_SEED": PROB_NEW_SEED,
            "MIN_SEED_DISTANCE": MIN_SEED_DISTANCE,
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

        for k, v in stats.items():
            print(f"{k}: {v}")

        postfix = f"_{cfg_idx:03d}"
        coords_path = os.path.join(OUTPUT_DIR, f"hoc_structure_coords{postfix}.csv")
        stats_path = os.path.join(OUTPUT_DIR, f"hoc_stats{postfix}.txt")
        fig_path = os.path.join(OUTPUT_DIR, f"hoc_structure{postfix}.png")

        save_coordinates(platelets, coords_path)
        save_stats(stats, stats_path)
        plot_structure(platelets, degrees, phi, fig_path)

    print("\n[INFO] All configurations finished.")


if __name__ == "__main__":
    main()
