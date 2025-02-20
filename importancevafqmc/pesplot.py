import pandas as pd
import glob
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
from afqmc.utils import reblock

ipieenergy = []
ipieerror = []
vafqmcenergy = []
vafqmcerror = []
distances = []

nan_distances = []
for csv_path in sorted(glob.glob("csv/h2-*.csv")):  # Now looking directly in 'csv'
    match = re.search(r"h2-(\d+\.?\d*).csv", csv_path)
    if match:
            dist = float(match.group(1))
            df = pd.read_csv(csv_path)
            print(df)
            # Check for NaNs
   #         if df['ipie-ETotal'].isna().any() or df['vafqmc-ETotal'].isna().any():
    #            nan_distances.append(dist)
     #           continue  # Skip this distance but keep going

            try:
                reblocked_ipie = reblock(df['ipie-ETotal'].dropna())
                reblocked_vafqmc = reblock(df['vafqmc-ETotal'])

                distances.append(dist)
                ipieenergy.append(reblocked_ipie['Etotal_ac'])
                vafqmcenergy.append(reblocked_vafqmc['Etotal_ac'])
                ipieerror.append(reblocked_ipie['Etotal_ac_error'])
                vafqmcerror.append(reblocked_vafqmc['Etotal_ac_error'])
            
            except Exception as e:
                print(f"Error processing distance {dist}: {e}")
                nan_distances.append(dist)
                continue  # Skip this iteration safely

# Print distances that were skipped due to NaNs
if nan_distances:
    print(f"Skipped distances due to NaNs: {sorted(nan_distances)}")
sorted_indices = np.argsort(distances)
sorted_distances = np.array(distances)[sorted_indices]
sorted_ipieenergy = np.array(ipieenergy)[sorted_indices]
sorted_vafqmcenergy = np.array(vafqmcenergy)[sorted_indices]
sorted_ipieerror = np.array(ipieerror)[sorted_indices]
sorted_vafqmcerror = np.array(vafqmcerror)[sorted_indices]

# Create the plot
plt.figure(figsize=(8, 6))

# Plot IPIE data with error bars
plt.errorbar(sorted_distances, sorted_ipieenergy, yerr=sorted_ipieerror, fmt='o-',
             label="IPIE", capsize=3, markersize=5, color='blue')

# Plot VAFQMC data with error bars
plt.errorbar(sorted_distances, sorted_vafqmcenergy, yerr=sorted_vafqmcerror, fmt='s-',
             label="VAFQMC", capsize=3, markersize=5, color='red')

# Labels and legend
plt.xlabel("H2 Bond Distance (Å)", fontsize=12)
plt.ylabel("Total Energy (Ha)", fontsize=12)
plt.title("Energy Comparison: IPIE vs. VAFQMC", fontsize=14)
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig("energy_comparison.png", dpi=300, bbox_inches="tight")
