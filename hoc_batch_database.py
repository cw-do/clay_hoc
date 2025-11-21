"""
hoc_batch_database.py

Batch generator for HoC clay disc structures to build a structure–scattering database.

- Uses hoc_structural_model.py for structure generation + contact analysis.
- Scans over:
    PHI_VALUES            = [0.005, 0.01, 0.02, 0.04, 0.06]
    PROB_NEW_SEED_VALUES  = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
- For each (phi, prob_new_seed) pair, generates N_CONFIGS_PER_COMBO configurations.
- For each configuration, saves:
    * coords_XXX.csv          (x,y,z,nx,ny,nz)
    * stats_XXX.txt           (detailed structural stats)
    * structure_XXX.png       (degree-colored snapshot)
    * degree_hist_XXX.csv     (degree histogram)
    * cluster_sizes_XXX.csv   (cluster size list)
- Also builds a master CSV:
    hoc_database_stats.csv
  with one row per configuration (compact descriptors for ML / analysis).
"""

import os
import math
import csv

import numpy as np
from tqdm import tqdm

import hoc_structural_model as hoc  # make sure this is in the same directory


# ==============================
# User-defined grids & settings
# ==============================

# Target volume fractions
PHI_VALUES = [0.005, 0.01, 0.02, 0.04, 0.06]

# Seed probabilities
PROB_NEW_SEED_VALUES = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# Number of configs per (phi, prob_new_seed)
N_CONFIGS_PER_COMBO = 20

# Box size (nm)
BOX_SIZE = 2000.0

# Base RNG seed
BASE_SEED = 0

# Output root directory
OUTPUT_ROOT = "hoc_database_v1"

# Percolation threshold for largest cluster fraction
PERCOLATION_THRESHOLD = 0.30


# ==============================
# Helper functions
# ==============================

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


def compute_total_steps() -> int:
    """
    Total number of particle placements for the global tqdm progress bar.
    """
    total = 0
    for phi in PHI_VALUES:
        N = compute_N_for_phi(phi, BOX_SIZE)
        total += N * N_CONFIGS_PER_COMBO * len(PROB_NEW_SEED_VALUES)
    return total


# ==============================
# Main batch driver
# ==============================

def main():
    # Fix global parameters in hoc_structural_model
    hoc.BOX_SIZE = BOX_SIZE
    hoc.PROB_T = 0.0
    hoc.PROB_PARALLEL = 0.0
    hoc.PROB_RANDOM = 1.0

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Master CSV for compact descriptors
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
        "mean_contact_distance",
        "std_contact_distance",
        "coords_file",
        "stats_file",
        "degree_hist_file",
        "cluster_sizes_file",
        "structure_png_file",
    ]

    total_steps = compute_total_steps()
    pbar = tqdm(total=total_steps, desc="Total particles placed", unit="particle", leave=True)

    global_cfg_id = 0  # running index over all configs

    with open(master_csv_path, "w", newline="") as f_master:
        writer = csv.writer(f_master)
        writer.writerow(master_headers)

        for phi_target in PHI_VALUES:
            # compute particle number for this phi
            N_particles = compute_N_for_phi(phi_target, BOX_SIZE)

            for prob_seed in PROB_NEW_SEED_VALUES:
                # set PROB_NEW_SEED in hoc module
                hoc.PROB_NEW_SEED = prob_seed

                # directory for this parameter combo
                combo_dir = os.path.join(
                    OUTPUT_ROOT,
                    f"phi_{phi_target:.4f}",
                    f"pseed_{prob_seed:.2f}",
                )
                os.makedirs(combo_dir, exist_ok=True)

                hoc.info(
                    f"\n[INFO] Starting batch for "
                    f"phi_target={phi_target:.44f}, PROB_NEW_SEED={prob_seed:.2f}, "
                    f"N_particles≈{N_particles}"
                )

                for local_cfg_idx in range(N_CONFIGS_PER_COMBO):
                    cfg_seed = BASE_SEED + global_cfg_id
                    np.random.seed(cfg_seed)

                    hoc.info(
                        f"[INFO]   Config {local_cfg_idx+1}/{N_CONFIGS_PER_COMBO} "
                        f"(global_id={global_cfg_id}, seed={cfg_seed})"
                    )

                    # 1) Build structure
                    platelets = hoc.build_structure_with_progress(N_particles, pbar)
                    Np = len(platelets)
                    phi_actual = hoc.compute_volume_fraction(Np)
                    hoc.info(
                        f"[INFO]       N_particles = {Np}, "
                        f"phi_actual ≈ {phi_actual:.4e}"
                    )

                    # positions array for later descriptors
                    positions = np.array([p.pos for p in platelets], dtype=float)

                    # 2) Build contact network
                    hoc.info("[INFO]       Building contact network...")
                    adjacency, contact_types, angles_deg = hoc.build_contact_network_fast(platelets)

                    degrees = hoc.degree_distribution(adjacency)
                    comps = hoc.connected_components(adjacency)
                    comp_sizes = [len(c) for c in comps]

                    # 3) Compute statistics / descriptors
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

                    # percolation flag (heuristic, by fraction)
                    percolation_flag = int(largest_cluster_fraction >= PERCOLATION_THRESHOLD)

                    # radius of gyration of largest cluster
                    if comp_sizes:
                        idx_max = int(np.argmax(comp_sizes))
                        largest_cluster_indices = list(comps[idx_max])
                        cluster_pos = positions[largest_cluster_indices]
                        r_cm = cluster_pos.mean(axis=0)
                        diffs = cluster_pos - r_cm
                        Rg2 = float(np.mean(np.sum(diffs**2, axis=1)))
                        Rg = math.sqrt(Rg2) if Rg2 >= 0.0 else float("nan")
                    else:
                        Rg = float("nan")
                        Rg2 = float("nan")

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

                    # 4) Save per-config outputs
                    postfix = f"{local_cfg_idx:03d}"

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

                    # detailed stats text
                    stats_dict = {
                        "global_config_id": global_cfg_id,
                        "local_config_index": local_cfg_idx,
                        "phi_target": phi_target,
                        "phi_actual": phi_actual,
                        "seed": cfg_seed,
                        "N_particles": Np,
                        "disc_volume_nm3": v_disc,
                        "box_volume_nm3": v_box,
                        "PROB_NEW_SEED": prob_seed,
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
                        hoc.info(f"[INFO]       Saved degree histogram: {degree_hist_path}")
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
                    hoc.info(f"[INFO]       Saved cluster sizes: {cluster_sizes_path}")

                    # structure snapshot (degree-colored)
                    hoc.plot_structure_colored_by_degree(
                        platelets, degrees, phi_actual, structure_png_path
                    )

                    # 5) Append to master CSV
                    writer.writerow([
                        global_cfg_id,
                        phi_target,
                        phi_actual,
                        prob_seed,
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
                        mean_contact_distance,
                        std_contact_distance,
                        os.path.relpath(coords_path, OUTPUT_ROOT),
                        os.path.relpath(stats_path, OUTPUT_ROOT),
                        os.path.relpath(degree_hist_path, OUTPUT_ROOT),
                        os.path.relpath(cluster_sizes_path, OUTPUT_ROOT),
                        os.path.relpath(structure_png_path, OUTPUT_ROOT),
                    ])

                    global_cfg_id += 1

    pbar.close()
    hoc.info("\n[INFO] All batch simulations complete.")
    hoc.info(f"[INFO] Master CSV saved to: {master_csv_path}")


if __name__ == "__main__":
    main()
