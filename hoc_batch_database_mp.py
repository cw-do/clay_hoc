"""
hoc_batch_database_mp.py

Multiprocess batch generator for HoC clay disc structures.

- Uses hoc_structural_model.py for structure generation + contact analysis.
- Scans over:
    PHI_VALUES            = [0.005, 0.01, 0.02, 0.04, 0.06]
    PROB_NEW_SEED_VALUES  = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
- For each (phi, prob_new_seed) pair, generates N_CONFIGS_PER_COMBO configurations.

- Each configuration (job) does:
    * Build structure
    * Build contact network
    * Compute descriptors
    * Save:
        - coords_XXX.csv
        - stats_XXX.txt
        - degree_hist_XXX.csv
        - cluster_sizes_XXX.csv
        - structure_XXX.png
        - rows/row_XXXXXX.csv   <-- 1-row CSV for this config only

- Master CSV:
    hoc_database_v1/hoc_database_stats.csv

  is built at the END by merging all rows/row_*.csv.
  If the run stops midway, you still keep all finished row_*.csv files,
  and rerunning will resume remaining jobs and then re-merge.
"""

import os
import math
import csv
import glob
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
import multiprocessing as mp

import hoc_structural_model as hoc  # make sure this is in the same directory


# ==============================
# User-defined grids & settings
# ==============================

# Target volume fractions
PHI_VALUES = [0.005, 0.01, 0.02, 0.04, 0.06]

# Seed probabilities
PROB_NEW_SEED_VALUES = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# Number of configs per (phi, prob_new_seed)
N_CONFIGS_PER_COMBO = 200  # <-- 큰 시뮬레이션 할 거라면 여기 조절

# Box size (nm)
BOX_SIZE = 2000.0

# Base RNG seed
BASE_SEED = 0

# Output root directory
OUTPUT_ROOT = "hoc_database_v1"

# Directory for per-job row CSVs
ROWS_DIR = os.path.join(OUTPUT_ROOT, "rows")

# Number of worker processes (None -> use mp.cpu_count())
N_WORKERS = None


# ==============================
# Helper functions / job struct
# ==============================

@dataclass
class Job:
    global_config_id: int
    phi_target: float
    prob_new_seed: float
    N_particles: int
    local_index: int


def compute_N_for_phi(phi: float, box_size: float) -> int:
    """
    Compute number of particles N to achieve target volume fraction phi
    in a cubic box of size box_size (nm).

        phi = N * V_disc / V_box  =>  N = phi * V_box / V_disc
    """
    v_disc = math.pi * hoc.RADIUS**2 * hoc.THICKNESS
    v_box = box_size**3
    N_float = phi * v_box / v_disc
    return int(round(N_float))


def build_jobs():
    """
    Build list of Job objects covering all (phi, prob_new_seed) pairs and configs.

    If rows/row_XXXXXX.csv already exists for a global_config_id, that job is skipped.
    This allows resuming a partially completed batch.
    """
    os.makedirs(ROWS_DIR, exist_ok=True)

    jobs = []
    global_id = 0

    for phi in PHI_VALUES:
        Np = compute_N_for_phi(phi, BOX_SIZE)
        for prob in PROB_NEW_SEED_VALUES:
            for local_idx in range(N_CONFIGS_PER_COMBO):
                row_path = os.path.join(ROWS_DIR, f"row_{global_id:06d}.csv")
                if os.path.exists(row_path):
                    # this config already done in a previous run → skip
                    global_id += 1
                    continue

                jobs.append(Job(
                    global_config_id=global_id,
                    phi_target=phi,
                    prob_new_seed=prob,
                    N_particles=Np,
                    local_index=local_idx,
                ))
                global_id += 1

    return jobs


class DummyPbar:
    """Simple object with .update() doing nothing, for use in worker."""
    def update(self, n):
        pass


def ensure_combo_dir(phi_target: float, prob_new_seed: float) -> str:
    """
    Ensure directory for given (phi, pseed) exists, and return its path.
    """
    combo_dir = os.path.join(
        OUTPUT_ROOT,
        f"phi_{phi_target:.4f}",
        f"pseed_{prob_new_seed:.2f}",
    )
    os.makedirs(combo_dir, exist_ok=True)
    return combo_dir


# ==============================
# Worker function (runs in child)
# ==============================

def run_job(job: Job):
    """
    Run one configuration in a separate process.

    Returns
    -------
    row_path : str
        Path to the per-job row CSV that was written.
    """
    # --- Set global parameters in hoc module (per-process) ---
    hoc.BOX_SIZE = BOX_SIZE
    hoc.PROB_T = 0.0
    hoc.PROB_PARALLEL = 0.0
    hoc.PROB_RANDOM = 1.0
    hoc.PROB_NEW_SEED = job.prob_new_seed

    # RNG seed
    cfg_seed = BASE_SEED + job.global_config_id
    np.random.seed(cfg_seed)

    # Output directory for this (phi, prob_new_seed)
    combo_dir = ensure_combo_dir(job.phi_target, job.prob_new_seed)

    # Progress bar dummy (to satisfy build_structure_with_progress interface)
    pbar_dummy = DummyPbar()

    # --- 1) Build structure ---
    platelets = hoc.build_structure_with_progress(job.N_particles, pbar_dummy)
    Np = len(platelets)
    phi_actual = hoc.compute_volume_fraction(Np)

    # Positions array
    positions = np.array([p.pos for p in platelets], dtype=float)

    # --- 2) Build contact network ---
    adjacency, contact_types, angles_deg = hoc.build_contact_network_fast(platelets)
    degrees = hoc.degree_distribution(adjacency)
    comps = hoc.connected_components(adjacency)
    comp_sizes = [len(c) for c in comps]

    # --- 3) Compute descriptors ---
    v_disc = math.pi * hoc.RADIUS**2 * hoc.THICKNESS
    v_box = hoc.BOX_SIZE**3

    n_contacts = len(contact_types)
    n_T = sum(1 for c in contact_types.values() if c == "T")
    n_parallel = sum(1 for c in contact_types.values() if c == "parallel")
    n_other = n_contacts - n_T - n_parallel

    T_fraction = n_T / n_contacts if n_contacts > 0 else 0.0
    parallel_fraction = n_parallel / n_contacts if n_contacts > 0 else 0.0

    # angle statistics
    if angles_deg.size > 0:
        angles_eff = np.where(
            angles_deg > 90.0,
            180.0 - angles_deg,
            angles_deg
        )
        mean_angle = float(angles_deg.mean())
        std_angle = float(angles_deg.std())
        mean_angle_eff = float(angles_eff.mean())
        std_angle_eff = float(angles_eff.std())
    else:
        mean_angle = float("nan")
        std_angle = float("nan")
        mean_angle_eff = float("nan")
        std_angle_eff = float("nan")

    # degree statistics
    if degrees.size > 0:
        mean_degree = float(degrees.mean())
        max_degree = int(degrees.max())
        P_z_eq_0 = float(np.mean(degrees == 0))
        P_z_ge_3 = float(np.mean(degrees >= 3))
    else:
        mean_degree = float("nan")
        max_degree = 0
        P_z_eq_0 = float("nan")
        P_z_ge_3 = float("nan")

    # cluster statistics
    largest_cluster_size = max(comp_sizes) if comp_sizes else 0
    num_clusters = len(comp_sizes)
    largest_cluster_fraction = (
        largest_cluster_size / Np if (Np > 0 and largest_cluster_size > 0) else 0.0
    )

    # radius of gyration of largest cluster + span + percolation flag
    if comp_sizes:
        idx_max = int(np.argmax(comp_sizes))
        largest_cluster_indices = list(comps[idx_max])
        cluster_pos = positions[largest_cluster_indices]

        # center of mass and Rg
        r_cm = cluster_pos.mean(axis=0)
        diffs = cluster_pos - r_cm
        Rg2 = float(np.mean(np.sum(diffs**2, axis=1)))
        Rg = math.sqrt(Rg2) if Rg2 >= 0.0 else float("nan")

        # spans
        span_x = float(cluster_pos[:, 0].max() - cluster_pos[:, 0].min())
        span_y = float(cluster_pos[:, 1].max() - cluster_pos[:, 1].min())
        span_z = float(cluster_pos[:, 2].max() - cluster_pos[:, 2].min())

        # percolation: spanning the box along any axis
        # allow for disc radius at both ends: L - 2*R
        span_threshold = BOX_SIZE - 2.0 * hoc.RADIUS
        percolation_flag = int(
            (span_x >= span_threshold) or
            (span_y >= span_threshold) or
            (span_z >= span_threshold)
        )
    else:
        Rg = float("nan")
        Rg2 = float("nan")
        span_x = float("nan")
        span_y = float("nan")
        span_z = float("nan")
        percolation_flag = 0

    # contact distance statistics
    if n_contacts > 0:
        contact_dists = []
        for i, neigh in enumerate(adjacency):
            for j in neigh:
                if j > i:
                    d = np.linalg.norm(positions[i] - positions[j])
                    contact_dists.append(d)
        if contact_dists:
            contact_dists = np.asarray(contact_dists, dtype=float)
            mean_contact_distance = float(contact_dists.mean())
            std_contact_distance = float(contact_dists.std())
        else:
            mean_contact_distance = float("nan")
            std_contact_distance = float("nan")
    else:
        mean_contact_distance = float("nan")
        std_contact_distance = float("nan")

    # --- 4) Save per-config outputs ---
    postfix = f"{job.local_index:03d}"

    coords_file = f"coords_{postfix}.csv"
    stats_file = f"stats_{postfix}.txt"
    degree_hist_file = f"degree_hist_{postfix}.csv"
    cluster_sizes_file = f"cluster_sizes_{postfix}.csv"
    structure_png_file = f"structure_{postfix}.png"

    coords_path = os.path.join(combo_dir, coords_file)
    stats_path = os.path.join(combo_dir, stats_file)
    degree_hist_path = os.path.join(combo_dir, degree_hist_file)
    cluster_sizes_path = os.path.join(combo_dir, cluster_sizes_file)
    structure_png_path = os.path.join(combo_dir, structure_png_file)

    # coordinates
    hoc.save_coordinates(platelets, coords_path)

    # detailed stats text (include spans & percolation)
    stats_dict = {
        "global_config_id": job.global_config_id,
        "local_config_index": job.local_index,
        "phi_target": job.phi_target,
        "phi_actual": phi_actual,
        "seed": cfg_seed,
        "N_particles": Np,
        "disc_volume_nm3": v_disc,
        "box_volume_nm3": v_box,
        "PROB_NEW_SEED": job.prob_new_seed,
        "CONTACT_DISTANCE_THRESHOLD_nm": hoc.CONTACT_DISTANCE_THRESHOLD,
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
        "P_z_eq_0": P_z_eq_0,
        "P_z_ge_3": P_z_ge_3,
        "num_clusters": num_clusters,
        "largest_cluster_size": largest_cluster_size,
        "largest_cluster_fraction": largest_cluster_fraction,
        "percolation_flag": percolation_flag,
        "largest_cluster_Rg": Rg,
        "largest_cluster_Rg2": Rg2,
        "largest_cluster_span_x": span_x,
        "largest_cluster_span_y": span_y,
        "largest_cluster_span_z": span_z,
        "mean_contact_distance": mean_contact_distance,
        "std_contact_distance": std_contact_distance,
        "coords_file": coords_file,
        "degree_hist_file": degree_hist_file,
        "cluster_sizes_file": cluster_sizes_file,
        "structure_png_file": structure_png_file,
    }
    hoc.save_stats(stats_dict, stats_path)

    # degree histogram
    if degrees.size > 0:
        unique_deg, counts_deg = np.unique(degrees, return_counts=True)
        deg_arr = np.column_stack([unique_deg, counts_deg])
        np.savetxt(
            degree_hist_path,
            deg_arr,
            delimiter=",",
            header="degree,count",
            comments=""
        )
    else:
        np.savetxt(
            degree_hist_path,
            np.zeros((0, 2)),
            delimiter=",",
            header="degree,count",
            comments=""
        )

    # cluster size list
    if comp_sizes:
        np.savetxt(
            cluster_sizes_path,
            np.array(comp_sizes, dtype=int),
            delimiter=",",
            header="cluster_size",
            comments=""
        )
    else:
        np.savetxt(
            cluster_sizes_path,
            np.zeros((0, 1), dtype=int),
            delimiter=",",
            header="cluster_size",
            comments=""
        )

    # structure snapshot (degree-colored)
    hoc.plot_structure_colored_by_degree(
        platelets, degrees, phi_actual, structure_png_path
    )

    # --- 5) Build row for this config (same order as master_headers in main) ---
    row = [
        job.global_config_id,
        job.phi_target,
        phi_actual,
        job.prob_new_seed,
        cfg_seed,
        Np,
        v_disc,
        v_box,
        n_contacts,
        n_T,
        n_parallel,
        n_other,
        T_fraction,
        parallel_fraction,
        mean_angle,
        std_angle,
        mean_angle_eff,
        std_angle_eff,
        mean_degree,
        max_degree,
        P_z_eq_0,
        P_z_ge_3,
        num_clusters,
        largest_cluster_size,
        largest_cluster_fraction,
        percolation_flag,
        Rg,
        Rg2,
        span_x,
        span_y,
        span_z,
        mean_contact_distance,
        std_contact_distance,
        os.path.relpath(coords_path, OUTPUT_ROOT),
        os.path.relpath(stats_path, OUTPUT_ROOT),
        os.path.relpath(degree_hist_path, OUTPUT_ROOT),
        os.path.relpath(cluster_sizes_path, OUTPUT_ROOT),
        os.path.relpath(structure_png_path, OUTPUT_ROOT),
    ]

    # --- 6) Save per-job row CSV (atomic output for this config) ---
    os.makedirs(ROWS_DIR, exist_ok=True)
    row_path = os.path.join(ROWS_DIR, f"row_{job.global_config_id:06d}.csv")
    with open(row_path, "w", newline="") as f_row:
        writer = csv.writer(f_row)
        writer.writerow(row)

    return row_path


# ==============================
# Master CSV merge
# ==============================

def merge_rows_to_master(master_csv_path: str, master_headers: list):
    """
    Merge all rows/row_*.csv files into a single master CSV.

    Safe even if only 일부 row 파일만 존재.
    """
    row_files = sorted(glob.glob(os.path.join(ROWS_DIR, "row_*.csv")))
    if not row_files:
        print("[WARN] No row_*.csv files found in", ROWS_DIR)
        return

    with open(master_csv_path, "w", newline="") as f_master:
        writer = csv.writer(f_master)
        writer.writerow(master_headers)

        for path in row_files:
            with open(path, "r", newline="") as f_row:
                reader = csv.reader(f_row)
                for row in reader:
                    if not row:
                        continue
                    writer.writerow(row)

    print(f"[INFO] Master CSV written with {len(row_files)} rows.")
    print(f"[INFO]   -> {master_csv_path}")


# ==============================
# Main (parent process)
# ==============================

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(ROWS_DIR, exist_ok=True)

    # Prepare job list (skips already-completed configs)
    jobs = build_jobs()

    # Master CSV header (must match row order in run_job)
    master_csv_path = os.path.join(OUTPUT_ROOT, "hoc_database_stats.csv")
    master_headers = [
        "global_config_id",
        "phi_target",
        "phi_actual",
        "prob_new_seed",
        "seed",
        "N_particles",
        "disc_volume_nm3",
        "box_volume_nm3",
        "N_contacts",
        "N_T_contacts",
        "N_parallel_contacts",
        "N_other_contacts",
        "T_contact_fraction",
        "parallel_contact_fraction",
        "mean_contact_angle_deg",
        "std_contact_angle_deg",
        "mean_contact_angle_folded_deg",
        "std_contact_angle_folded_deg",
        "mean_degree",
        "max_degree",
        "P_z_eq_0",
        "P_z_ge_3",
        "num_clusters",
        "largest_cluster_size",
        "largest_cluster_fraction",
        "percolation_flag",
        "largest_cluster_Rg",
        "largest_cluster_Rg2",
        "largest_cluster_span_x",
        "largest_cluster_span_y",
        "largest_cluster_span_z",
        "mean_contact_distance",
        "std_contact_distance",
        "coords_file",
        "stats_file",
        "degree_hist_file",
        "cluster_sizes_file",
        "structure_png_file",
    ]

    if jobs:
        print(f"[INFO] Pending jobs: {len(jobs)}")
        print(f"[INFO] Running with up to {N_WORKERS or mp.cpu_count()} workers...")

        with mp.Pool(processes=N_WORKERS) as pool:
            for _ in tqdm(
                pool.imap_unordered(run_job, jobs),
                total=len(jobs),
                desc="Configs",
                unit="cfg"
            ):
                pass
    else:
        print("[INFO] No pending jobs; all configs appear to be done already.")

    # Merge all per-job rows into master CSV
    merge_rows_to_master(master_csv_path, master_headers)

    print("[INFO] Done.")


if __name__ == "__main__":
    mp.freeze_support()  # for Windows safety
    main()
