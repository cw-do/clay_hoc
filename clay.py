#!/usr/bin/env python3
# Clay Nanodisc Monte Carlo Simulation (PBC + rim charges + cell list)
# Produces: final_configuration.csv with columns x,y,z,nx,ny,nz

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass
from math import ceil

# =========================
# Parameters (edit freely)
# =========================
np.random.seed(42)

# --- physics / system ---
N              = 4000          # number of discs
L              = 1000.0        # box size (nm), cubic
R              = 25.0          # disc radius (nm)
T              = 1.0           # disc thickness (nm)

debye_length   = 10.0          # nm
kappa          = 1.0 / debye_length
q_face_total   = -1.0          # total face charge (split between two faces)
q_edge_total   = +1.0          # total rim charge (uniform along rim)
M_edge         = 24            # number of rim charges per disc
prefactor      = 10.0          # overall interaction strength (tune to see EF bonding)
beta           = 1.0           # 1/kT

# --- MC control ---
n_steps        = 10000
max_trans      = 1.0           # nm
max_rot        = 0.2           # radians
cutoff         = 50.0          # neighbor cutoff (>= interaction range)
rebuild_every  = 200           # rebuild cell list this often
reject_on_overlap = True       # center hard-core check

# --- initialization mode ---
# choose one: 'lattice' | 'poisson' | 'grow'
INIT_MODE      = 'lattice'

# lattice+jitter params
LATTICE_MARGIN = 0.5           # nm minimum extra clearance beyond 2R
# poisson-disk params
POISSON_K      = 30            # Bridson attempts per active point
# growth params
GROW_STEPS     = 8000
GROW_DR        = 0.05          # radius increment per growth step (nm)
GROW_JITTER    = 0.3           # nm random shuffle per growth step

# --- snapshots / plotting ---
take_snapshots = True
num_show       = 1500
plot_figsize   = (12, 10)
save_csv_path  = "final_configuration.csv"
do_plot        = True

# =========================
# Math & geometry helpers
# =========================

def rand_unit_vec():
    v = np.random.normal(0, 1, 3)
    n = np.linalg.norm(v)
    return v/n if n > 0 else np.array([0.0, 0.0, 1.0])

def rot_axis_angle(axis, theta):
    ax = axis / np.linalg.norm(axis)
    x, y, z = ax
    K = np.array([[0, -z, y],
                  [z,  0,-x],
                  [-y, x, 0]])
    I = np.eye(3)
    return I + np.sin(theta)*K + (1.0 - np.cos(theta))*(K @ K)

def minimum_image_vec(dr, L):
    return dr - L * np.rint(dr / L)

def dist_pbc(a, b, L):
    return np.linalg.norm(minimum_image_vec(a - b, L))

def orthobasis_from_normal(n):
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    u = np.cross(n, a); nu = np.linalg.norm(u)
    u = u/nu if nu > 0 else np.array([1.0, 0.0, 0.0])
    v = np.cross(n, u)
    return u, v

def get_charge_sites(pos, n, R, T, M_edge, q_face_total, q_edge_total):
    # two face-centered charges + M_edge rim charges uniformly spaced
    u, v = orthobasis_from_normal(n)
    qf = q_face_total / 2.0
    sites = [(pos + 0.5*T*n, qf), (pos - 0.5*T*n, qf)]
    qe = q_edge_total / M_edge
    angles = np.linspace(0, 2*np.pi, M_edge, endpoint=False)
    for th in angles:
        rim = pos + R*(np.cos(th)*u + np.sin(th)*v)
        sites.append((rim, qe))
    return sites

def yukawa(r, q1, q2, kappa, pref):
    if r < 1e-9:
        return 0.0
    return pref * q1 * q2 * np.exp(-kappa * r) / r

# =========================
# Cell list (linked cells)
# =========================

@dataclass
class CellList:
    L: float
    cell_size: float
    ncell: int
    heads: dict  # (ix,iy,iz) -> list of indices

def build_cell_list(positions, L, cutoff):
    ncell = max(int(np.floor(L / cutoff)), 1)
    cell_size = L / ncell
    heads = {}
    for idx, r in enumerate(positions):
        ix = int(np.floor(r[0] / cell_size)) % ncell
        iy = int(np.floor(r[1] / cell_size)) % ncell
        iz = int(np.floor(r[2] / cell_size)) % ncell
        key = (ix, iy, iz)
        heads.setdefault(key, []).append(idx)
    return CellList(L=L, cell_size=cell_size, ncell=ncell, heads=heads)

def cell_neighbors(ix, iy, iz, ncell):
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            for dz in (-1,0,1):
                yield ((ix+dx) % ncell, (iy+dy) % ncell, (iz+dz) % ncell)

def nearby_indices(pos, cell_list: CellList):
    ix = int(np.floor(pos[0] / cell_list.cell_size)) % cell_list.ncell
    iy = int(np.floor(pos[1] / cell_list.cell_size)) % cell_list.ncell
    iz = int(np.floor(pos[2] / cell_list.cell_size)) % cell_list.ncell
    for key in cell_neighbors(ix, iy, iz, cell_list.ncell):
        for j in cell_list.heads.get(key, []):
            yield j

# =========================
# Energy + sterics
# =========================

def overlaps_any(idx, new_pos, positions, cell_list: CellList, L, R):
    for j in nearby_indices(new_pos, cell_list):
        if j == idx:
            continue
        if dist_pbc(new_pos, positions[j], L) < 2.0*R:
            return True
    return False

def compute_energy_idx(i, positions, normals, cell_list: CellList, L, R, T,
                       M_edge, q_face_total, q_edge_total, kappa, pref, cutoff):
    pos_i, n_i = positions[i], normals[i]
    sites_i = get_charge_sites(pos_i, n_i, R, T, M_edge, q_face_total, q_edge_total)
    E = 0.0
    for j in nearby_indices(pos_i, cell_list):
        if j == i:
            continue
        # quick sphere prefilter
        if dist_pbc(pos_i, positions[j], L) > cutoff:
            continue
        pos_j, n_j = positions[j], normals[j]
        sites_j = get_charge_sites(pos_j, n_j, R, T, M_edge, q_face_total, q_edge_total)
        for (ri, qi) in sites_i:
            for (rj, qj) in sites_j:
                dr = minimum_image_vec(ri - rj, L)
                r = np.linalg.norm(dr)
                if r <= cutoff:
                    E += yukawa(r, qi, qj, kappa, pref)
    return E

# =========================
# Initializers (3 modes)
# =========================

def init_lattice_jitter(N, L, R, margin=0.5):
    n = ceil(N ** (1/3))
    spacing = L / n
    min_spacing = 2*R + margin
    if spacing < min_spacing:
        raise ValueError(
            f"Box too tight for lattice init: spacing={spacing:.3f} < 2R+margin={min_spacing:.3f}."
        )
    coords = np.linspace(spacing/2, L - spacing/2, n)
    grid = np.array(np.meshgrid(coords, coords, coords, indexing='ij')).reshape(3, -1).T
    if grid.shape[0] < N:
        n += 1
        spacing = L / n
        coords = np.linspace(spacing/2, L - spacing/2, n)
        grid = np.array(np.meshgrid(coords, coords, coords, indexing='ij')).reshape(3, -1).T
    jitter_amp = max(0.0, (spacing - (2*R + margin)) / 2.0)
    jitter = (np.random.rand(N,3) - 0.5) * 2.0 * jitter_amp
    pos = (grid[:N] + jitter) % L
    nrm = np.random.normal(0, 1, (N,3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return pos, nrm

def bridson_poisson_disk_pbc(N_target, L, min_dist, k=30):
    # periodic Bridson Poisson-disk in 3D
    cell = min_dist / np.sqrt(3.0)
    nx = int(np.floor(L / cell)); nx = max(nx, 1)
    cell = L / nx
    grid = -np.ones((nx, nx, nx), dtype=int)
    def grid_idx(p): return tuple((np.floor(p / cell).astype(int)) % nx)
    def in_neighborhood_ok(p, pts):
        gi = grid_idx(p)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    idx = grid[(gi[0]+dx)%nx, (gi[1]+dy)%nx, (gi[2]+dz)%nx]
                    if idx >= 0:
                        q = pts[idx]
                        if np.linalg.norm(minimum_image_vec(p - q, L)) < min_dist:
                            return False
        return True

    p0 = np.random.uniform(0, L, 3)
    points = [p0]
    active = [0]
    grid[grid_idx(p0)] = 0

    while active and len(points) < N_target:
        ai = np.random.choice(active)
        base = points[ai]
        found = False
        for _ in range(k):
            r = np.random.uniform(min_dist, 2.0*min_dist)
            u = rand_unit_vec()
            cand = (base + r*u) % L
            if in_neighborhood_ok(cand, points):
                points.append(cand)
                grid[grid_idx(cand)] = len(points)-1
                active.append(len(points)-1)
                found = True
                if len(points) >= N_target:
                    break
        if not found:
            active.remove(ai)

    pts = np.array(points)[:N_target]
    nrm = np.random.normal(0,1,(pts.shape[0],3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return pts, nrm

def grow_nonoverlapping(N, L, R_target, steps=5000, dr_grow=0.02, max_trans=0.5):
    # start tiny, shuffle, and grow to R_target rejecting overlaps
    pos = np.random.uniform(0, L, (N,3))
    nrm = np.random.normal(0,1,(N,3)); nrm /= np.linalg.norm(nrm,axis=1,keepdims=True)
    R_cur = 0.1
    def build_cells(P, cutoff):
        m = max(int(np.floor(L/cutoff)), 1)
        cs = L/m
        heads = {}
        for i,p in enumerate(P):
            key = tuple((np.floor(p/cs).astype(int)) % m)
            heads.setdefault(key, []).append(i)
        return heads, m, cs
    def nearby(p, heads, m, cs):
        idx = (np.floor(p/cs).astype(int)) % m
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    yield from heads.get(((idx[0]+dx)%m,(idx[1]+dy)%m,(idx[2]+dz)%m), [])
    for s in range(steps):
        heads, m, cs = build_cells(pos, cutoff=2*R_cur + 2.0)
        pos = (pos + (np.random.rand(N,3)-0.5)*2.0*max_trans) % L
        R_try = min(R_cur + dr_grow, R_target)
        ok = True
        for i in range(N):
            p = pos[i]
            for j in nearby(p, heads, m, cs):
                if j <= i: continue
                if np.linalg.norm(minimum_image_vec(p - pos[j], L)) < 2.0*R_try:
                    ok = False; break
            if not ok: break
        if ok:
            R_cur = R_try
        if R_cur >= R_target:
            break
    if R_cur < R_target:
        raise RuntimeError("Growth stalled before reaching target radius; increase steps or reduce dr_grow.")
    return pos, nrm

# =========================
# Snapshots / plotting
# =========================

def create_disc_polygon(center, normal, radius, resolution=20):
    theta = np.linspace(0, 2*np.pi, resolution)
    u, v = orthobasis_from_normal(normal)
    circle = np.array([radius*np.cos(theta), radius*np.sin(theta)])
    pts = center + (u[:,None]*circle[0] + v[:,None]*circle[1]).T
    return pts

def contact_counts(positions, cell_list: CellList, L, R, margin=0.5):
    Nloc = len(positions)
    cnt = np.zeros(Nloc, dtype=int)
    rcut = 2.0*R + margin
    for i in range(Nloc):
        p = positions[i]
        for j in nearby_indices(p, cell_list):
            if j <= i: continue
            if dist_pbc(p, positions[j], L) <= rcut:
                cnt[i] += 1; cnt[j] += 1
    return cnt

# =========================
# Main MC
# =========================

def initialize_system():
    if INIT_MODE == 'lattice':
        return init_lattice_jitter(N, L, R, margin=LATTICE_MARGIN)
    elif INIT_MODE == 'poisson':
        return bridson_poisson_disk_pbc(N, L, min_dist=2.0*R, k=POISSON_K)
    elif INIT_MODE == 'grow':
        return grow_nonoverlapping(N, L, R_target=R, steps=GROW_STEPS, dr_grow=GROW_DR, max_trans=GROW_JITTER)
    else:
        raise ValueError("INIT_MODE must be 'lattice', 'poisson', or 'grow'.")

def run_mc():
    print(f"Initializing with mode: {INIT_MODE}")
    positions, normals = initialize_system()
    cell_list = build_cell_list(positions, L, cutoff)

    snapshots = []

    def take_snapshot(label):
        if not take_snapshots: 
            return
        clc = build_cell_list(positions, L, max(cutoff, 2.1*R))
        counts = contact_counts(positions, clc, L, R)
        snapshots.append((positions.copy(), normals.copy(), counts, label))

    take_snapshot("Initial")

    accepted = 0
    for step in range(n_steps):
        i = np.random.randint(N)
        pos_old, n_old = positions[i].copy(), normals[i].copy()

        dr = np.random.uniform(-max_trans, max_trans, 3)
        axis = rand_unit_vec()
        dtheta = max_rot * (2.0*np.random.rand() - 1.0)
        Rmat = rot_axis_angle(axis, dtheta)

        pos_new = np.mod(pos_old + dr, L)
        n_new  = Rmat @ n_old
        n_new /= np.linalg.norm(n_new)

        if reject_on_overlap and overlaps_any(i, pos_new, positions, cell_list, L, R):
            accept = False
        else:
            E_old = compute_energy_idx(i, positions, normals, cell_list, L, R, T,
                                       M_edge, q_face_total, q_edge_total, kappa, prefactor, cutoff)
            positions[i] = pos_new; normals[i] = n_new
            E_new = compute_energy_idx(i, positions, normals, cell_list, L, R, T,
                                       M_edge, q_face_total, q_edge_total, kappa, prefactor, cutoff)
            dE = E_new - E_old
            accept = (dE <= 0.0) or (np.random.rand() < np.exp(-beta * dE))
            if not accept:
                positions[i] = pos_old; normals[i] = n_old

        if accept:
            accepted += 1

        if (step + 1) % rebuild_every == 0:
            cell_list = build_cell_list(positions, L, cutoff)

        if (step + 1) in (0, n_steps//3, 2*n_steps//3, n_steps-1):
            take_snapshot(f"Step {step+1}/{n_steps} (acc={accepted})")

        if (step + 1) % 1000 == 0:
            print(f"Step {step+1}/{n_steps} | acc={accepted/(step+1):.3f}")

    take_snapshot(f"Final (acc={accepted}/{n_steps})")
    return positions, normals, snapshots

# =========================
# Run + output
# =========================

if __name__ == "__main__":
    pos, nrm, snaps = run_mc()

    # Save final configuration for scattering
    with open(save_csv_path, "w") as f:
        for p, n in zip(pos, nrm):
            f.write(f"{p[0]},{p[1]},{p[2]},{n[0]},{n[1]},{n[2]}\n")
    print(f"Saved final configuration to {save_csv_path}")

    if do_plot and len(snaps) > 0:
        nplots = min(4, len(snaps))
        nrows = 2; ncols = 2
        fig = plt.figure(figsize=plot_figsize)
        for k in range(nplots):
            ax = fig.add_subplot(nrows, ncols, k+1, projection='3d')
            P, Nrm, Counts, label = snaps[k]
            # subsample for speed
            idx = np.random.choice(P.shape[0], size=min(num_show, P.shape[0]), replace=False)
            P = P[idx]; Nrm = Nrm[idx]; Counts = Counts[idx]
            cmax = max(Counts.max(), 1)
            for p, n, c in zip(P, Nrm, Counts):
                poly = create_disc_polygon(p, n, R, resolution=20)
                col = plt.cm.viridis(c / cmax)
                poly3d = Poly3DCollection([poly], facecolors=col, alpha=0.25, linewidths=0.2, edgecolors='k')
                ax.add_collection3d(poly3d)
            ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
            ax.set_title(label)
        plt.tight_layout()
        plt.savefig('clay_simulation_snapshots.png', dpi=150, bbox_inches='tight')
        plt.show()
