"""
read iq_sq_gr_average.csv file which has [q, iq, sq] data where 
q is scattering vector in nm-1
and first row is labels

Then, make a plot of iq in log-log scale.

read iq_sq_gr_average.csv file from multiple folders given by patterns 
and make a plot comparing iq from different folders. 
use folder name as legend label.

"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Plot iq in log-log scale
plt.figure(figsize=(8, 6))

# Read from multiple folders
folders = glob.glob('hoc_output_prob0.8_random_*/')
print(f"Found {len(folders)} folders: {folders}")
for folder in sorted(folders):
    csv_file = os.path.join(folder, 'iq_sq_gr_average.csv')
    print(f"Processing {folder}, file exists: {os.path.exists(csv_file)}")
    if os.path.exists(csv_file):
        data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        print(f"Data shape: {data.shape}")
        q_f = data[:, 0]
        iq_f = data[:, 1]
        label = os.path.basename(folder.rstrip(os.sep)).replace('hoc_output_prob', 'prob ')
        print(f"Label: '{label}'")
        plt.loglog(q_f, iq_f, label=label)

plt.xlabel('q (nm$^{-1}$)')
plt.ylabel('I(q)')
plt.title('Scattering Intensity I(q) Comparison')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig('iq_comparison_random.png', dpi=150, bbox_inches='tight')
plt.show()