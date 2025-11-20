#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scattering from finite-thickness discs with overall isotropic orientation (powder average).
Reads a CSV with N rows of: x,y,z,nx,ny,nz (comma-separated, no header required by default).
Computes:
  - g(r) -> H0(r)=g(r)-1
  - C2(r)=<P2(n_i·n_j)>_r -> H2(r)=g(r) C2(r)
  - Fits Ornstein-Zernike tails for H0 and H2 beyond r_fit up to r_c=L/2
  - Hybrid integrals with j0 and j2 kernels
  - Disc shape prefactors from exact finite-thickness amplitude F(q,theta)
  - I(q) = N <|F|^2> + 4pi rho [ A0(q) I0(q) + A2(q) I2(q) ]
Outputs CSV files for: g_r.csv, H_r.csv, I_q.csv

Usage example:
  python compute_Iq_discs.py data.csv --L 200.0 --radius 12.5 --thickness 1.0 --qmin 0.002 --qmax 0.5 --nq 200 --dr 0.5 --rfit 30.0 --outprefix run1

Notes:
  - Distances/lengths in nm, q in nm^-1.
  - Requires: numpy, scipy.
"""

import argparse
import numpy as np
from numpy import pi
from scipy import optimize, integrate, special
import matplotlib.pyplot as plt

def load_csv(path, has_header=False):
    arr = np.loadtxt(path, delimiter=",", skiprows=1 if has_header else 0)
    if arr.shape[1] < 6:
        raise ValueError("Input must have at least 6 columns: x,y,z,nx,ny,nz")
    pos = arr[:, :3].astype(float)
    nrm = arr[:, 3:6].astype(float)
    # normalize normals
    nrm_norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm_norm[nrm_norm == 0] = 1.0
    nrm = nrm / nrm_norm
    return pos, nrm

def minimum_image(d, L):
    # shift into [-L/2, L/2)
    return d - L * np.rint(d / L)

def p2(x):
    return 0.5*(3.0*x*x - 1.0)

def compute_rdfs(positions, normals, L, dr):
    """
    Compute g(r) and H2(r)=g(r) C2(r) up to r_c = L/2 - dr.
    Returns r_centers, g(r), H0(r), H2(r), counts per bin, and number density rho.
    """
    N = positions.shape[0]
    vol = L**3
    rho = N / vol

    r_max = L/2.0 - dr
    if r_max <= 0:
        raise ValueError("Bin width dr is too large relative to box size L.")
    nbins = int(np.floor(r_max / dr))
    if nbins < 10:
        raise ValueError("Too few bins. Decrease dr or increase L.")
    r_edges = np.linspace(0.0, r_max, nbins+1)
    r_centers = 0.5*(r_edges[:-1] + r_edges[1:])
    shell_vol = 4.0*pi*r_centers**2 * dr

    counts = np.zeros(nbins, dtype=np.float64)
    p2_sums = np.zeros(nbins, dtype=np.float64)

    # Pair loop, O(N^2) but memory-light
    for i in range(N-1):
        rij = positions[i+1:] - positions[i]
        rij = minimum_image(rij, L)
        d = np.linalg.norm(rij, axis=1)
        # filter within r_max and >0
        m = (d > 0.0) & (d < r_max)
        if not np.any(m):
            continue
        d_sel = d[m]
        # histogram distances
        hist, _ = np.histogram(d_sel, bins=r_edges)
        counts += hist

        # orientation part
        ni = normals[i]
        nj = normals[i+1:][m]
        cosang = np.einsum('ij,j->i', nj, ni)  # nj dot ni
        p2_vals = p2(cosang)

        # bin p2_vals with same indices
        bin_idx = np.searchsorted(r_edges, d_sel, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, nbins-1)
        np.add.at(p2_sums, bin_idx, p2_vals)

    # g(r)
    with np.errstate(divide='ignore', invalid='ignore'):
        g = (2.0 * counts) / (shell_vol * rho * N)  # factor 2 because pairs counted once (i<j)
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    # C2(r) and H's
    C2 = np.zeros_like(g)
    mpos = counts > 0
    C2[mpos] = p2_sums[mpos] / counts[mpos]

    H0 = g - 1.0
    H2 = g * C2

    return r_centers, g, H0, H2, counts, rho, r_max

def oz_tail(r, A, xi):
    # Ornstein-Zernike tail: A * exp(-r/xi) / r
    return A * np.exp(-r/xi) / np.maximum(r, 1e-12)

def fit_tail_OZ(r, H, counts, r_fit, r_c):
    """
    Weighted least squares fit of H(r) to A*exp(-r/xi)/r on [r_fit, r_c].
    Weights: w ~ counts (more pairs => smaller variance).
    Returns (A, xi).
    """
    m = (r >= r_fit) & (r <= r_c) & (counts > 0)
    if np.count_nonzero(m) < 5:
        # fallback: weak regularization
        rsel = r[(r > 0) & (r <= r_c)]
        A0 = H[(r > 0) & (r <= r_c)].mean() if np.any((r > 0) & (r <= r_c)) else 0.0
        xi0 = max((rsel.max() - rsel.min())/3.0, 1.0) if rsel.size else 1.0
        return A0, xi0

    x = r[m]
    y = H[m]
    w = counts[m].astype(float)
    w[w <= 0] = 1.0

    # Initial guesses
    A0 = y[np.argmax(w)] if np.any(w) else y.mean()
    xi0 = max((r_c - r_fit) / 3.0, 1.0)

    def residual(p):
        A, xi = p
        xi = abs(xi)
        yi = oz_tail(x, A, xi)
        return (np.sqrt(w) * (yi - y))

    res = optimize.least_squares(residual, x0=np.array([A0, xi0]),
                                 bounds=([-np.inf, 1e-6],[np.inf, np.inf]), max_nfev=20000)
    A, xi = res.x
    xi = abs(xi)
    return A, xi

def spherical_bessel_j0(x):
    return special.spherical_jn(0, x)

def spherical_bessel_j2(x):
    return special.spherical_jn(2, x)

def hybrid_integral(H_num, r, A_tail, xi_tail, q, ell):
    """
    Compute integral_0^inf H(r) j_ell(q r) r^2 dr via split at r_c=r[-1].
    H_num is on r[0..], tail model is A*exp(-r/xi)/r.
    I_total = A * I_full(ell) - integral_0^{r_c} H_tail * kernel + integral_0^{r_c} H_num * kernel
    where I_full has closed form for OZ tail.
    """
    if r.size < 2:
        return 0.0
    if ell == 0:
        jfun = spherical_bessel_j0
        I_full = (xi_tail**2) / (1.0 + (q*xi_tail)**2)
    elif ell == 2:
        jfun = spherical_bessel_j2
        I_full = (xi_tail**2) / ((1.0 + (q*xi_tail)**2)**2)
    else:
        raise ValueError("ell must be 0 or 2")

    integrand_tail = oz_tail(r, A_tail, xi_tail) * (r**2) * jfun(q*r)
    I_tail_0_rc = integrate.simps(integrand_tail, r)

    integrand_num = H_num * (r**2) * jfun(q*r)
    I_num_0_rc = integrate.simps(integrand_num, r)

    I_total = A_tail * I_full - I_tail_0_rc + I_num_0_rc
    return I_total

def disc_amplitude_F(q, theta, R, t, delta_rho=1.0):
    # Finite-thickness homogeneous disc amplitude
    qRsin = q * R * np.sin(theta)
    qtcos = 0.5 * q * t * np.cos(theta)
    # j1(x)/x and sinc
    with np.errstate(divide='ignore', invalid='ignore'):
        j1x_over_x = np.ones_like(qRsin)
        mask = (qRsin != 0)
        j1x_over_x[mask] = special.j1(qRsin[mask]) / qRsin[mask]
        sinc = np.ones_like(qtcos)
        m2 = (qtcos != 0)
        sinc[m2] = np.sin(qtcos[m2]) / qtcos[m2]
    V = pi * R**2 * t
    F = delta_rho * V * 2.0 * j1x_over_x * sinc
    return F

def shape_prefactors(q_array, R, t, n_theta=64):
    """
    Compute <|F|^2>, f0, f2, A0(q), A2(q) via Gauss-Legendre in mu=cos(theta).
    """
    mu, w = np.polynomial.legendre.leggauss(n_theta)  # mu in [-1,1]
    theta = np.arccos(np.clip(mu, -1.0, 1.0))

    Fqt = np.array([disc_amplitude_F(q, theta, R, t) for q in q_array])  # (nq, n_theta)

    F2_avg = 0.5 * np.sum(w * (np.abs(Fqt)**2), axis=1)
    F_avg  = 0.5 * np.sum(w * Fqt, axis=1)

    P0 = np.ones_like(mu)
    P2 = 0.5 * (3.0*mu**2 - 1.0)
    f0 = 0.5 * np.sum(w * Fqt * P0, axis=1) * 1.0     # (2*0+1)=1
    f2 = 0.5 * np.sum(w * Fqt * P2, axis=1) * 5.0     # (2*2+1)=5

    with np.errstate(divide='ignore', invalid='ignore'):
        A0 = (F_avg**2) / F2_avg
        A0 = np.nan_to_num(A0, nan=0.0, posinf=0.0, neginf=0.0)
        A2 = (2.0 * f0 * f2) / F2_avg
        A2 = np.nan_to_num(A2, nan=0.0, posinf=0.0, neginf=0.0)

    return F2_avg, A0, A2

def compute_intensity(csv_path, L, R, t, qmin, qmax, nq, dr, rfit,
                      has_header=False, outprefix="out"):
    # Load data
    pos, nrm = load_csv(csv_path, has_header=has_header)
    N = pos.shape[0]

    # RDFs
    r, g, H0, H2, counts, rho, r_max = compute_rdfs(pos, nrm, L, dr)
    r_c = r[-1]

    # Tail fits (OZ tails)
    r_fit_eff = rfit if rfit is not None else max(0.35*L, 2.0*dr)
    r_fit_eff = min(r_fit_eff, r_c*0.9)
    A0h, xi0 = fit_tail_OZ(r, H0, counts, r_fit_eff, r_c)
    A2h, xi2 = fit_tail_OZ(r, H2, counts, r_fit_eff, r_c)

    # q grid
    q = np.logspace(np.log10(qmin), np.log10(qmax), num=nq)

    # Shape prefactors
    F2_avg, A0q, A2q = shape_prefactors(q, R, t, n_theta=64)

    # Hybrid integrals for ell=0,2

    # Hybrid integrals for ell=0,2 and 'bare' structure-factor pieces
    I0 = np.zeros_like(q)
    I2 = np.zeros_like(q)
    Snn = np.zeros_like(q)  # 1 + 4πρ ∫ H0 j0 r^2 dr
    O2  = np.zeros_like(q)  #      4πρ ∫ H2 j2 r^2 dr

    for i, qv in enumerate(q):
        # Hybrid integrals including tail stitching
        I0[i] = hybrid_integral(H0, r, A0h, xi0, qv, ell=0)
        I2[i] = hybrid_integral(H2, r, A2h, xi2, qv, ell=2)
        # 'Bare' pieces (these are just 4πρ times the same integrals; Snn includes the +1)
        Snn[i] = 1.0 + 4.0*pi*rho * I0[i]
        O2[i]  = 4.0*pi*rho * I2[i]

    # Assemble I(q)
    # Assemble I(q)
    I_self = rho * F2_avg
    I_pair0 = 4.0*pi * (rho**2) * (A0q * I0)
    I_pair2 = 4.0*pi * (rho**2) * (A2q * I2)
    I_pair = I_pair0 + I_pair2
    Iq = I_self + I_pair
    Seff = Iq / np.maximum(I_self, 1e-30)

    # Save outputs
    np.savetxt(f"{outprefix}_g_r.csv",
               np.c_[r, g, H0, H2, counts],
               delimiter=",", header="r_nm,g(r),H0(r)=g-1,H2(r)=g*C2,count_pairs", comments="")
    np.savetxt(f"{outprefix}_I_q.csv",
               np.c_[q, Iq, I_self, I_pair, I_pair0, I_pair2, Seff, Snn, O2, F2_avg, A0q, A2q, I0, I2],
               delimiter=",", header="q_nm^-1,I(q)_perVol,I_self_perVol,I_pair_perVol,I_pair0_perVol,I_pair2_perVol,Seff,Snn(q),O2(q),<|F|^2>,A0(q),A2(q),I0(q),I2(q)", comments="")
    with open(f"{outprefix}_tail_params.txt","w") as f:
        f.write(f"A0_h={A0h:.6g}, xi0={xi0:.6g} (nm)\n")
        f.write(f"A2_h={A2h:.6g}, xi2={xi2:.6g} (nm)\n")
        f.write(f"r_c={r_c:.6g} nm, r_fit={r_fit_eff:.6g} nm, dr={dr:.6g} nm\n")
        f.write(f"N={N}, rho={rho:.6g} nm^-3, L={L} nm\n")

    return {
        "r": r, "g": g, "H0": H0, "H2": H2, "counts": counts,
        "q": q, "Iq": Iq, "I_self": I_self, "I_pair": I_pair,
        "I_pair0": I_pair0, "I_pair2": I_pair2,
        "Seff": Seff, "Snn": Snn, "O2": O2,
        "F2": F2_avg, "A0q": A0q, "A2q": A2q,
        "I0": I0, "I2": I2,
        "A0h": A0h, "xi0": xi0, "A2h": A2h, "xi2": xi2, "r_c": r_c, "rho": rho, "N": N
    }

def main():
    ap = argparse.ArgumentParser(description="Compute I(q) for finite-thickness discs from COM+normal CSV using hybrid RDF with OZ tails.")
    ap.add_argument("csv", help="Input CSV with columns: x,y,z,nx,ny,nz")
    ap.add_argument("--L", type=float, required=True, help="Box length (nm), cubic box assumed")
    ap.add_argument("--radius", type=float, required=True, help="Disc radius R (nm)")
    ap.add_argument("--thickness", type=float, required=True, help="Disc thickness t (nm)")
    ap.add_argument("--qmin", type=float, default=0.002, help="Min q (nm^-1)")
    ap.add_argument("--qmax", type=float, default=0.5, help="Max q (nm^-1)")
    ap.add_argument("--nq", type=int, default=200, help="Number of q points (log-spaced)")
    ap.add_argument("--dr", type=float, default=0.5, help="Bin width for r (nm)")
    ap.add_argument("--rfit", type=float, default=None, help="Fit window start for OZ tail (nm); default 0.35*L")
    ap.add_argument("--header", action="store_true", help="Set if CSV has a header row to skip")
    ap.add_argument("--outprefix", type=str, default="out", help="Prefix for output files")
    ap.add_argument("--plot", action="store_true", help="If set, generate quick diagnostic plots and save PNGs.")
    args = ap.parse_args()

    rfit = args.rfit if args.rfit is not None else 0.35*args.L
    res = compute_intensity(args.csv, args.L, args.radius, args.thickness,
                      args.qmin, args.qmax, args.nq, args.dr, rfit,
                      has_header=args.header, outprefix=args.outprefix)
    print(f"Done. Wrote files: {args.outprefix}_g_r.csv, {args.outprefix}_I_q.csv, {args.outprefix}_tail_params.txt")
    if args.plot:
        q = res['q']; Iq = res['Iq']; I_self = res['I_self']; I_pair = res['I_pair']
        I_pair0 = res['I_pair0']; I_pair2 = res['I_pair2']
        Seff = res.get('Seff', None); Snn = res.get('Snn', None); O2 = res.get('O2', None)
        # Plot 1: I(q), I_self, I_pair
        plt.figure()
        plt.loglog(q, Iq, label='I(q) total')
        plt.loglog(q, I_self, label='I_self')
        plt.loglog(q, I_pair, label='I_pair total')
        plt.loglog(q, I_pair0, label='I_pair0 (density)')
        plt.loglog(q, I_pair2, label='I_pair2 (quadrupole)')
        plt.xlabel('q (nm$^{-1}$)'); plt.ylabel('Intensity per volume')
        plt.legend(); plt.tight_layout(); plt.savefig(f"{args.outprefix}_Iq.png", dpi=200)
        # Plot 2: Seff and Snn
        if Seff is not None:
            plt.figure()
            plt.semilogx(q, Seff, label='Seff = I/I_self')
            if Snn is not None:
                plt.semilogx(q, Snn, label='Snn(q)')
            if O2 is not None:
                plt.semilogx(q, O2, label='O2(q)')
            plt.xlabel('q (nm$^{-1}$)'); plt.ylabel('dimensionless')
            plt.legend(); plt.tight_layout(); plt.savefig(f"{args.outprefix}_Seff_Snn.png", dpi=200)
        # Plot 3: g(r) and H2(r)
        r = res['r']; g = res['g']; H2 = res['H2']
        plt.figure()
        plt.plot(r, g, label='g(r)')
        plt.plot(r, H2, label='H2(r)')
        plt.xlabel('r (nm)'); plt.ylabel('g(r), H2(r)')
        plt.legend(); plt.tight_layout(); plt.savefig(f"{args.outprefix}_gr.png", dpi=200)
        print(f"Saved plots: {args.outprefix}_Iq.png, {args.outprefix}_Seff_Snn.png, {args.outprefix}_gr.png")

if __name__ == "__main__":
    main()
