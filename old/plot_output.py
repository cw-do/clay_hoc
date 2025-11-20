"""
This code plots out_g_r.csv and out_I_q.csv files.
g_r will be plotted in linear scale
I_q will be plotted in log-log scale

The file <prefix>_I_q.csv now includes:

| Column name       | Meaning                                                        |     |                                           |
| ----------------- | -------------------------------------------------------------- | --- | ----------------------------------------- |
| `q_nm^-1`         | magnitude of scattering vector (nm⁻¹)                          |     |                                           |
| `I(q)_perVol`     | total intensity per volume                                     |     |                                           |
| `I_self_perVol`   | single-particle term ρ⟨                                        | F   | ²⟩                                        |
| `I_pair_perVol`   | total pair term (structure-factor contribution)                |     |                                           |
| `I_pair0_perVol`  | density channel (4πρ²A₀(q)I₀(q))                               |     |                                           |
| `I_pair2_perVol`  | quadrupolar channel (4πρ²A₂(q)I₂(q))                           |     |                                           |
| `Seff`            | effective structure factor (I/I_\text{self})                   |     |                                           |
| `Snn(q)`          | number–number structure factor (1 + 4πρ∫h(r)j₀(qr)r²dr)        |     |                                           |
| `O2(q)`           | quadrupolar orientational structure term (4πρ∫H₂(r)j₂(qr)r²dr) |     |                                           |
| `<                | F                                                              | ²>` | orientationally averaged self form factor |
| `A0(q)` / `A2(q)` | disc shape prefactors                                          |     |                                           |
| `I0(q)` / `I2(q)` | integrated (j₀) and (j₂) kernels                               |     |                                           |

"""

import numpy as np
import matplotlib.pyplot as plt

# Load g(r) data
g_data = np.loadtxt('out_g_r.csv', delimiter=',', skiprows=1)
r = g_data[:, 0]
g_r = g_data[:, 1]
H0 = g_data[:, 2]
H2 = g_data[:, 3]

# Plot g(r) in linear scale
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.plot(r, g_r, label='g(r)')
ax1.plot(r, H0, label='H0(r)')
ax1.plot(r, H2, label='H2(r)')
ax1.set_xlabel('r')
ax1.set_ylabel('Value')
ax1.set_title('Radial Distribution Functions')
ax1.legend()
ax1.grid(True)

# Load I(q) data
iq_data = np.loadtxt('out_I_q.csv', delimiter=',', skiprows=1)
q = iq_data[:, 0]
I_q_perVol = iq_data[:, 1]
I_self_perVol = iq_data[:, 2]
Snn = iq_data[:, 7]  # Snn(q) - number-number structure factor

# Plot g(r) in linear scale
fig, axes = plt.subplots(2, 2, figsize=(8, 5))

axes[0,0].plot(r, g_r, label='g(r)')
axes[0,0].plot(r, H0, label='H0(r)')
axes[0,0].plot(r, H2, label='H2(r)')
axes[0,0].set_xlabel('r (nm)')
axes[0,0].set_ylabel('Value')
axes[0,0].set_title('Radial Distribution Functions')
axes[0,0].legend()
axes[0,0].grid(True)

# Plot I(q) components separately in log-log scale
axes[0,1].loglog(q, I_q_perVol)
axes[0,1].set_xlabel('q (nm$^{-1}$)')
axes[0,1].set_ylabel('I(q)_perVol')
axes[0,1].set_title('Total Scattering Intensity')
axes[0,1].grid(True)

axes[1,0].loglog(q, I_self_perVol)
axes[1,0].set_xlabel('q (nm$^{-1}$)')
axes[1,0].set_ylabel('I_self_perVol')
axes[1,0].set_title('Self-Scattering Intensity')
axes[1,0].grid(True)

axes[1,1].plot(q, Snn)  # Plot Snn in linear scale
axes[1,1].set_xlabel('q (nm$^{-1}$)')
axes[1,1].set_ylabel('Snn(q)')
axes[1,1].set_title('Number-Number Structure Factor')
axes[1,1].grid(True)

plt.tight_layout()
plt.show()
