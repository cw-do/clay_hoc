"""
simulation.py

Coarse-grained Monte Carlo simulation of charged clay nanoplatelets
(disc-like platelets) with orientation-dependent Yukawa interactions.

Features:
- Numba-accelerated site-site Yukawa interactions
- Multiple independent configurations in one run:
    N_CONFIGS configurations, each with its own random seed
    → outputs: output/final_configuration_000.csv, ...
- Initialization modes:
    "stack"  : staggered + jittered face-parallel stacks on 2D grids in multiple layers (default)
    "grow"   : grow cluster via face–edge–like attachments
    "random" : random placement with distance rejection
- Snapshots show actual disc shape with correct radius and orientation.
  Disc color encodes the number of touching neighbours, with a colorbar.
- Snapshot figure title shows:
    - number of particles N
    - volume fraction φ
    - number density ρ (nm^-3) for that configuration

Outputs per configuration k (under OUTPUT_DIR):
  - final_configuration_kkk.csv : x,y,z,nx,ny,nz for final configuration k
  - snapshots_kkk.png           : 2x2 3D plots for 0%, 30%, 60%, 100% MC steps
"""

import os
import numpy as np
import math
from math import sin, cos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
from numba import njit

# =========================
#  Output folder
# =========================

OUTPUT_DIR = "output"   # 원하는 이름으로 바꿔도 됨
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
#  Global parameters
# =========================

# Number of independent configurations to generate
N_CONFIGS = 20
BASE_SEED = 0  # base random seed; each config uses BASE_SEED + config_index

# System size
N_PARTICLES = 100           # number of clay platelets
BOX_SIZE = 200.0            # nm, cubic box length

# Particle geometry
RADIUS = 25.0               # nm, disc radius (≈ 50 nm diameter)
THICKNESS = 1.0             # nm, disc thickness

# Electrostatic parameters (reduced units)
KAPPA = 1.0 / 30.0          # nm^-1, inverse Debye length
FACE_CHARGE = -1.0          # charge on each face site
EDGE_CHARGE_BASE = +0.25    # base edge charge per rim site (before mode adjustment)

# Monte Carlo parameters
MC_STEPS = 5000             # total attempted moves per configuration
TEMPERATURE = 1.0           # k_B T in reduced units
MAX_TRANSLATE = 3.0         # nm, maximum translation per move
MAX_ROTATE = 0.15           # radians, maximum rotation angle

# Initialization mode: "stack", "grow", or "random"
INIT_MODE = "random"

# Snapshots to save (fractions of total steps)
SNAPSHOT_FRACTIONS = [0.0, 0.3, 0.6, 1.0]

# Stack initialization jitter (for x,y) as fraction of RADIUS
STACK_JITTER_FRAC = 0.3  # 0이면 완전 lattice, 0.2~0.3 정도면 살짝 흐트러짐


# =========================
#  Particle representation
# =========================

class ClayParticle:
    """
    Rigid disc-like platelet.

    pos : np.ndarray, shape (3,)  - center position
    n   : np.ndarray, shape (3,)  - unit normal vector (disc normal)
    u   : np.ndarray, shape (3,)  - unit vector in disc plane, ⟂ n
    """

    __slots__ = ("pos", "n", "u")

    def __init__(self, pos, normal, in_plane):
        self.pos = np.array(pos, dtype=float)

        n = np.array(normal, dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            raise ValueError("Normal vector length too small.")
        self.n = n / n_norm

        u = np.array(in_plane, dtype=float)
        # make u perpendicular to n
        u = u - np.dot(u, self.n) * self.n
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-8:
            # choose arbitrary perpendicular
            base = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(base, self.n)) > 0.9:
                base = np.array([0.0, 1.0, 0.0])
            u = np.cross(self.n, base)
            u_norm = np.linalg.norm(u)
        self.u = u / u_norm

    def orientation_matrix(self):
        """
        Return 3x3 rotation matrix with columns (u, v, n).
        v is constructed as n × u.
        """
        v = np.cross(self.n, self.u)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-8:
            base = np.array([0.0, 1.0, 0.0])
            v = np.cross(self.n, base)
            v_norm = np.linalg.norm(v)
        v = v / v_norm
        # re-orthogonalize u
        u = np.cross(v, self.n)
        return np.vstack((u, v, self.n)).T


# =========================
#  Local site geometry (face + rim)
# =========================

# Face sites (top/bottom center)
FACE_SITES_LOCAL = [
    np.array([0.0, 0.0, +0.5 * THICKNESS]),
    np.array([0.0, 0.0, -0.5 * THICKNESS]),
]

# Rim site configuration
N_RIM_SITES = 12   # number of rim charge sites around the disc

# Charge distribution mode for rim:
# "preserve_total"  : total edge charge same as 4 * EDGE_CHARGE_BASE
# "preserve_density": per-site EDGE_CHARGE_BASE kept → total edge charge increases
EDGE_MODE = "preserve_total"  # or "preserve_density"

if EDGE_MODE == "preserve_total":
    ORIGINAL_TOTAL_EDGE = 4 * EDGE_CHARGE_BASE
    EDGE_CHARGE = ORIGINAL_TOTAL_EDGE / N_RIM_SITES
elif EDGE_MODE == "preserve_density":
    EDGE_CHARGE = EDGE_CHARGE_BASE
else:
    raise ValueError("EDGE_MODE must be 'preserve_total' or 'preserve_density'")

print(f"[INFO] Using {N_RIM_SITES} rim sites with EDGE_CHARGE={EDGE_CHARGE:.4f} ({EDGE_MODE})")

# Generate rim local coordinates
RIM_SITES_LOCAL = []
for k in range(N_RIM_SITES):
    ang = 2.0 * math.pi * k / N_RIM_SITES
    x = RADIUS * math.cos(ang)
    y = RADIUS * math.sin(ang)
    RIM_SITES_LOCAL.append(np.array([x, y, 0.0]))

# Combine: 2 face sites + rim
LOCAL_SITES = FACE_SITES_LOCAL + RIM_SITES_LOCAL

# Charges per site
SITE_CHARGES_LOCAL = np.array(
    [FACE_CHARGE, FACE_CHARGE] + [EDGE_CHARGE] * len(RIM_SITES_LOCAL),
    dtype=np.float64,
)
N_SITES = SITE_CHARGES_LOCAL.shape[0]
print(f"[INFO] Total interaction sites per particle: {N_SITES}")


# =========================
#  Utilities
# =========================

def random_unit_vector():
    """
    Uniform random unit vector on the sphere.
    """
    z = 2.0 * np.random.rand() - 1.0
    phi = 2.0 * math.pi * np.random.rand()
    r_xy = math.sqrt(max(0.0, 1.0 - z * z))
    x = r_xy * math.cos(phi)
    y = r_xy * math.sin(phi)
    return np.array([x, y, z])


def minimum_image(vec, box_size=BOX_SIZE):
    """
    Apply minimum image convention with periodic boundary conditions. (Python version)
    """
    return vec - np.rint(vec / box_size) * box_size


def random_rotation_matrix(max_angle):
    """
    Generate a small random rotation matrix with rotation angle ∈ [-max_angle, max_angle].
    """
    axis = random_unit_vector()
    angle = (2.0 * np.random.rand() - 1.0) * max_angle
    # Rodrigues' formula
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    I = np.eye(3)
    R = I + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    return R


# =========================
#  Site position helpers
# =========================

def update_particle_sites(index, particle, site_pos):
    """
    Update site_pos[index, :, :] from a single ClayParticle.
    """
    Rmat = particle.orientation_matrix()
    for s, rel in enumerate(LOCAL_SITES):
        site_pos[index, s, :] = particle.pos + Rmat.dot(rel)


def update_all_sites(particles, site_pos):
    """
    Initialize / refresh all site positions from particles.
    """
    for i, p in enumerate(particles):
        update_particle_sites(i, p, site_pos)


# =========================
#  Numba-accelerated energy kernels
# =========================

@njit
def minimum_image_numba(dx, box_size):
    """
    Minimum image for a single component (Numba-safe).
    """
    return dx - box_size * np.round(dx / box_size)


@njit
def pair_energy_numba(i, j, site_pos, site_charges, kappa, box_size):
    """
    Yukawa site-site pair energy between particle i and j.
    site_pos: (N_particles, N_sites, 3)
    site_charges: (N_sites,)
    """
    E = 0.0
    Ns = site_charges.shape[0]

    for a in range(Ns):
        xia = site_pos[i, a, 0]
        yia = site_pos[i, a, 1]
        zia = site_pos[i, a, 2]
        qa = site_charges[a]

        for b in range(Ns):
            xjb = site_pos[j, b, 0]
            yjb = site_pos[j, b, 1]
            zjb = site_pos[j, b, 2]
            qb = site_charges[b]

            dx = xjb - xia
            dy = yjb - yia
            dz = zjb - zia

            dx = minimum_image_numba(dx, box_size)
            dy = minimum_image_numba(dy, box_size)
            dz = minimum_image_numba(dz, box_size)

            r2 = dx*dx + dy*dy + dz*dz
            if r2 < 1e-12:
                continue

            r = math.sqrt(r2)
            E += qa * qb * math.exp(-kappa * r) / r

    return E


@njit
def particle_energy_numba(idx, site_pos, site_charges, kappa, box_size):
    """
    Interaction energy of particle idx with all others.
    """
    E = 0.0
    Np = site_pos.shape[0]
    for j in range(Np):
        if j == idx:
            continue
        E += pair_energy_numba(idx, j, site_pos, site_charges, kappa, box_size)
    return E


@njit
def total_energy_numba(site_pos, site_charges, kappa, box_size):
    """
    Total pair energy of the system (sum over i<j).
    """
    E = 0.0
    Np = site_pos.shape[0]
    for i in range(Np):
        for j in range(i + 1, Np):
            E += pair_energy_numba(i, j, site_pos, site_charges, kappa, box_size)
    return E


# =========================
#  Initialization: random
# =========================

def initialize_random(N, box_size=BOX_SIZE, min_dist_factor=0.6):
    """
    Random placement of N particles in the box with random orientations.
    Rejects placements that are too close (rough center distance cutoff).
    """
    particles = []
    attempts = 0
    max_attempts = 20000
    min_center_dist = min_dist_factor * RADIUS

    while len(particles) < N and attempts < max_attempts:
        attempts += 1

        pos = np.random.rand(3) * box_size
        n = random_unit_vector()

        base = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(base, n)) > 0.9:
            base = np.array([0.0, 1.0, 0.0])
        u0 = np.cross(n, base)
        u0 /= (np.linalg.norm(u0) + 1e-12)

        phi = 2.0 * math.pi * np.random.rand()
        K = np.array([
            [0.0, -n[2], n[1]],
            [n[2], 0.0, -n[0]],
            [-n[1], n[0], 0.0],
        ])
        I = np.eye(3)
        Rn = I + math.sin(phi) * K + (1.0 - math.cos(phi)) * (K @ K)
        u = Rn.dot(u0)

        new_particle = ClayParticle(pos, n, u)

        too_close = False
        for p in particles:
            d = minimum_image(new_particle.pos - p.pos, box_size)
            if np.linalg.norm(d) < min_center_dist:
                too_close = True
                break

        if too_close:
            continue

        particles.append(new_particle)
        attempts = 0

    if len(particles) < N:
        print(f"[WARN] Random init placed only {len(particles)} / {N}. "
              f"Remaining will be placed without distance check.")
        while len(particles) < N:
            pos = np.random.rand(3) * box_size
            n = random_unit_vector()
            base = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(base, n)) > 0.9:
                base = np.array([0.0, 1.0, 0.0])
            u0 = np.cross(n, base)
            u0 /= (np.linalg.norm(u0) + 1e-12)
            phi = 2.0 * math.pi * np.random.rand()
            K = np.array([
                [0.0, -n[2], n[1]],
                [n[2], 0.0, -n[0]],
                [-n[1], n[0], 0.0],
            ])
            I = np.eye(3)
            Rn = I + math.sin(phi) * K + (1.0 - math.cos(phi)) * (K @ K)
            u = Rn.dot(u0)
            particles.append(ClayParticle(pos, n, u))

    return particles


# =========================
#  Initialization: grow
# =========================

def initialize_grow(N, box_size=BOX_SIZE):
    """
    Grow-based initialization: start with one particle,
    then attach new particles near edges of existing ones in a face–edge fashion.
    """
    particles = []

    pos0 = np.random.rand(3) * box_size
    n0 = random_unit_vector()
    base = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(base, n0)) > 0.9:
        base = np.array([0.0, 1.0, 0.0])
    u0 = np.cross(n0, base)
    u0 /= (np.linalg.norm(u0) + 1e-12)
    phi0 = 2.0 * math.pi * np.random.rand()
    K0 = np.array([
        [0.0, -n0[2], n0[1]],
        [n0[2], 0.0, -n0[0]],
        [-n0[1], n0[0], 0.0],
    ])
    I = np.eye(3)
    Rn0 = I + math.sin(phi0) * K0 + (1.0 - math.cos(phi0)) * (K0 @ K0)
    u0 = Rn0.dot(u0)
    particles.append(ClayParticle(pos0, n0, u0))

    target = N
    attempts = 0
    max_attempts = 30000
    min_center_dist = 0.6 * RADIUS

    while len(particles) < target and attempts < max_attempts:
        attempts += 1
        parent = particles[np.random.randint(len(particles))]

        u_p = parent.u
        v_p = np.cross(parent.n, u_p)
        v_p /= (np.linalg.norm(v_p) + 1e-12)

        theta = 2.0 * math.pi * np.random.rand()
        dir_plane = math.cos(theta) * u_p + math.sin(theta) * v_p
        dir_plane /= (np.linalg.norm(dir_plane) + 1e-12)

        attach_dist = 1.2 * RADIUS
        vertical_shift = 0.5 * RADIUS
        sign = 1.0 if np.random.rand() < 0.5 else -1.0

        pos_new = parent.pos + attach_dist * dir_plane + sign * vertical_shift * parent.n
        pos_new = pos_new % box_size

        n_new = np.cross(dir_plane, parent.n)
        if np.linalg.norm(n_new) < 1e-6:
            n_new = random_unit_vector()
        n_new = n_new / (np.linalg.norm(n_new) + 1e-12)

        u_new = -dir_plane
        u_new = u_new - np.dot(u_new, n_new) * n_new
        u_new = u_new / (np.linalg.norm(u_new) + 1e-12)

        new_particle = ClayParticle(pos_new, n_new, u_new)

        too_close = False
        for p in particles:
            d = minimum_image(new_particle.pos - p.pos, box_size)
            if np.linalg.norm(d) < min_center_dist:
                too_close = True
                break

        if too_close:
            continue

        particles.append(new_particle)
        attempts = 0

    if len(particles) < N:
        print(f"[WARN] Grow init placed only {len(particles)} / {N}. "
              f"Remaining will be placed randomly (overlaps possible).")
        remaining = N - len(particles)
        extra = initialize_random(remaining, box_size, min_dist_factor=0.3)
        particles.extend(extra)

    return particles


# =========================
#  Initialization: staggered stack
# =========================

def initialize_stack(N, box_size=BOX_SIZE, jitter_frac=STACK_JITTER_FRAC):
    """
    Stack-based initialization (staggered + jittered):
    - Particles are placed in multiple layers along z.
    - Each layer is a 2D grid in x-y (face-parallel discs).
    - Alternate layers are shifted by half a grid cell (AB-staggered).
    - Each particle has a small random jitter in x,y to avoid perfect lattice.

    All normals = (0,0,1) initially, u = (1,0,0).
    """
    particles = []

    if 2 * RADIUS > box_size:
        n_side = 1
    else:
        n_side = int(box_size // (2 * RADIUS))
        if n_side < 1:
            n_side = 1

    per_layer = n_side * n_side
    n_layers = int(np.ceil(N / per_layer))

    dx = box_size / n_side
    dy = box_size / n_side
    dz = box_size / (n_layers + 1)

    jitter_amp = jitter_frac * RADIUS

    count = 0
    for k in range(n_layers):
        z = (k + 1) * dz

        # Staggered offset: even/odd layers have half-cell shift
        if k % 2 == 0:
            offset_x = 0.0
            offset_y = 0.0
        else:
            offset_x = 0.5 * dx
            offset_y = 0.5 * dy

        for i in range(n_side):
            for j in range(n_side):
                if count >= N:
                    break

                x = (i + 0.5) * dx + offset_x
                y = (j + 0.5) * dy + offset_y

                # random jitter in x,y
                jitter = (np.random.rand(2) - 0.5) * 2.0 * jitter_amp
                x += jitter[0]
                y += jitter[1]

                # apply PBC
                x %= box_size
                y %= box_size

                pos = np.array([x, y, z])
                n = np.array([0.0, 0.0, 1.0])  # all faces parallel initially
                u = np.array([1.0, 0.0, 0.0])  # in-plane axis
                particles.append(ClayParticle(pos, n, u))
                count += 1

            if count >= N:
                break

        if count >= N:
            break

    print(f"[INFO] Stack init placed {len(particles)} particles in {n_layers} layers "
          f"(staggered + jittered).")
    return particles


# =========================
#  Monte Carlo
# =========================

def run_mc(particles, site_pos, steps=MC_STEPS, box_size=BOX_SIZE):
    """
    Run Metropolis Monte Carlo simulation.
    Uses Numba kernels for energy.
    Returns:
      snapshots: dict[step] -> list[ClayParticle]
      E_final : final total energy
    """
    Np = len(particles)
    site_charges = SITE_CHARGES_LOCAL

    snapshot_steps = sorted(set(int(f * steps) for f in SNAPSHOT_FRACTIONS))
    snapshots = {}

    E_current = total_energy_numba(site_pos, site_charges, KAPPA, box_size)
    print(f"Initial total energy: {E_current:.3f}")

    snapshots[0] = [ClayParticle(p.pos.copy(), p.n.copy(), p.u.copy()) for p in particles]

    for step in tqdm(range(1, steps + 1), desc="MC Simulation", ncols=80):
        i = np.random.randint(Np)
        p = particles[i]

        old_pos = p.pos.copy()
        old_n = p.n.copy()
        old_u = p.u.copy()
        old_sites_i = site_pos[i].copy()

        E_old_i = particle_energy_numba(i, site_pos, site_charges, KAPPA, box_size)

        dpos = (np.random.rand(3) - 0.5) * 2.0 * MAX_TRANSLATE
        new_pos = (p.pos + dpos) % box_size

        Rrot = random_rotation_matrix(MAX_ROTATE)
        Rmat_old = p.orientation_matrix()
        Rmat_new = Rrot @ Rmat_old
        new_n = Rmat_new[:, 2]
        new_u = Rmat_new[:, 0]

        new_n = new_n / (np.linalg.norm(new_n) + 1e-12)
        new_u = new_u - np.dot(new_u, new_n) * new_n
        new_u = new_u / (np.linalg.norm(new_u) + 1e-12)

        p.pos = new_pos
        p.n = new_n
        p.u = new_u

        update_particle_sites(i, p, site_pos)

        overlap = False
        for j in range(Np):
            if j == i:
                continue
            d = minimum_image(particles[j].pos - p.pos, box_size)
            if np.linalg.norm(d) < 0.5 * RADIUS:
                overlap = True
                break

        if overlap:
            p.pos = old_pos
            p.n = old_n
            p.u = old_u
            site_pos[i] = old_sites_i
        else:
            E_new_i = particle_energy_numba(i, site_pos, site_charges, KAPPA, box_size)
            dE = E_new_i - E_old_i

            if dE <= 0.0 or np.random.rand() < math.exp(-dE / TEMPERATURE):
                E_current += dE
            else:
                p.pos = old_pos
                p.n = old_n
                p.u = old_u
                site_pos[i] = old_sites_i

        if step in snapshot_steps:
            snapshots[step] = [ClayParticle(p_i.pos.copy(),
                                            p_i.n.copy(),
                                            p_i.u.copy())
                               for p_i in particles]

    print(f"Final total energy: {E_current:.3f}")
    return snapshots, E_current


# =========================
#  Output helpers
# =========================

def save_final_coordinates(particles, filename):
    """
    Save final configuration to CSV with columns: x,y,z,nx,ny,nz
    """
    arr = []
    for p in particles:
        arr.append([p.pos[0], p.pos[1], p.pos[2],
                    p.n[0], p.n[1], p.n[2]])
    arr = np.array(arr)
    header = "x,y,z,nx,ny,nz"
    np.savetxt(filename, arr, delimiter=",", header=header, comments="")
    print(f"Final coordinates saved to {filename}")


def _disc_polygon_global(p: ClayParticle, radius=RADIUS, n_segments=32):
    """
    Create a circle (disc) polygon in global coordinates for one particle.
    """
    Rmat = p.orientation_matrix()
    angles = np.linspace(0, 2*np.pi, n_segments, endpoint=True)
    verts = []
    for ang in angles:
        x = radius * np.cos(ang)
        y = radius * np.sin(ang)
        z = 0.0
        local = np.array([x, y, z])
        g = p.pos + Rmat.dot(local)
        verts.append(g)
    return verts


def compute_contact_numbers(particles, box_size=BOX_SIZE, cutoff=None):
    """
    Compute how many neighbours each particle is 'touching'.
    Contact if center-center distance (with PBC) < cutoff.
    """
    N = len(particles)
    contacts = np.zeros(N, dtype=int)
    if cutoff is None:
        cutoff = 2.05 * RADIUS  # slightly larger than diameter
    cutoff2 = cutoff * cutoff

    for i in range(N):
        for j in range(i + 1, N):
            d = minimum_image(particles[j].pos - particles[i].pos, box_size)
            r2 = np.dot(d, d)
            if r2 < cutoff2:
                contacts[i] += 1
                contacts[j] += 1
    return contacts


def plot_snapshots(
    snapshots,
    box_size=BOX_SIZE,
    filename="snapshots.png",
    volume_fraction=None,
    number_density=None,
):
    """
    Plot up to 4 snapshots in a 2x2 3D figure as actual discs.
    Disc color encodes the number of touching neighbours.

    If volume_fraction and number_density are given, show them in the figure title.
    """
    steps_sorted = sorted(snapshots.keys())
    if len(steps_sorted) > 4:
        steps_sorted = steps_sorted[:4]

    fig = plt.figure(figsize=(10, 10))

    all_contact_values = []
    per_step_contacts = {}
    for step in steps_sorted:
        snap = snapshots[step]
        contacts = compute_contact_numbers(snap, box_size)
        per_step_contacts[step] = contacts
        all_contact_values.extend(list(contacts))

    if len(all_contact_values) == 0:
        vmin, vmax = 0, 1
    else:
        vmin = min(all_contact_values)
        vmax = max(all_contact_values)
        if vmin == vmax:
            vmax = vmin + 1

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for idx, step in enumerate(steps_sorted):
        snap = snapshots[step]
        contacts = per_step_contacts[step]

        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")

        for p, cval in zip(snap, contacts):
            color = cmap(norm(cval))
            verts = _disc_polygon_global(p, radius=RADIUS, n_segments=32)
            poly = Poly3DCollection([verts], alpha=0.7)
            poly.set_edgecolor("k")
            poly.set_facecolor(color)
            ax.add_collection3d(poly)

        xs = [p.pos[0] for p in snap]
        ys = [p.pos[1] for p in snap]
        zs = [p.pos[2] for p in snap]
        ax.scatter(xs, ys, zs, s=5, c=cmap(norm(contacts)), alpha=0.9)

        ax.set_title(f"Step {step}")
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_zlim(0, box_size)
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        ax.set_zlabel("Z (nm)")
        ax.set_box_aspect((1, 1, 1))

        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Number of touching neighbours")

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    try:
        some_step = steps_sorted[0]
        N_snapshot = len(snapshots[some_step])
    except Exception:
        N_snapshot = 0

    if volume_fraction is not None and number_density is not None:
        title_str = (
            f"N = {N_snapshot}, "
            f"ϕ = {volume_fraction:.4f}, "
            f"ρ = {number_density:.3e} nm$^{{-3}}$"
        )
        fig.suptitle(title_str, fontsize=12)

    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Snapshot figure saved to {filename}")


# =========================
#  Single-configuration wrapper
# =========================

def run_single_configuration(config_index: int):
    """
    Run one full MC simulation for a single configuration index.
    Uses a seed BASE_SEED + config_index to generate different initial configs.
    Outputs final_configuration_kkk.csv and snapshots_kkk.png under OUTPUT_DIR.
    """
    seed = BASE_SEED + config_index
    np.random.seed(seed)
    print(f"\n==============================")
    print(f"Configuration {config_index + 1}/{N_CONFIGS} (seed={seed})")
    print(f"Initialization mode: {INIT_MODE}")

    # Initialize
    if INIT_MODE == "stack":
        particles = initialize_stack(N_PARTICLES, BOX_SIZE, jitter_frac=STACK_JITTER_FRAC)
    elif INIT_MODE == "grow":
        particles = initialize_grow(N_PARTICLES, BOX_SIZE)
    else:
        particles = initialize_random(N_PARTICLES, BOX_SIZE, min_dist_factor=0.6)

    Np = len(particles)
    print(f"Placed {Np} particles.")

    # Volume fraction and number density for this config
    disc_volume = math.pi * RADIUS**2 * THICKNESS      # nm^3
    box_volume = BOX_SIZE**3                           # nm^3
    volume_fraction = Np * disc_volume / box_volume
    number_density = Np / box_volume                   # nm^-3

    print(f"Disc volume       : {disc_volume:.3f} nm^3")
    print(f"Box volume        : {box_volume:.3f} nm^3")
    print(f"Volume fraction ϕ : {volume_fraction:.6f}")
    print(f"Number density ρ  : {number_density:.6e} nm^-3")

    # Allocate site_pos
    site_pos = np.zeros((Np, N_SITES, 3), dtype=np.float64)
    update_all_sites(particles, site_pos)

    print("Running Monte Carlo (Numba accelerated)...")
    snapshots, E_final = run_mc(particles, site_pos, MC_STEPS, BOX_SIZE)

    # Save final configuration
    coords_filename = os.path.join(OUTPUT_DIR, f"final_configuration_{config_index:03d}.csv")
    save_final_coordinates(particles, coords_filename)

    # Snapshots
    snapshot_filename = os.path.join(OUTPUT_DIR, f"snapshots_{config_index:03d}.png")
    plot_snapshots(
        snapshots,
        BOX_SIZE,
        snapshot_filename,
        volume_fraction=volume_fraction,
        number_density=number_density,
    )

    print(f"Configuration {config_index + 1} finished.\n")


# =========================
#  Main
# =========================

if __name__ == "__main__":
    print(f"[INFO] Starting simulations for {N_CONFIGS} configurations...")
    for conf in range(N_CONFIGS):
        run_single_configuration(conf)
    print(f"[INFO] All configurations finished.")
