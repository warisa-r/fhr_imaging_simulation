import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_dolfin = pd.read_csv("forward_sim_data_bottom.csv")
df_matlab = pd.read_csv("matlab_measurements.csv")

# Check data alignment. Round x values to avoid floating point precision issues
df_dolfin['x_rounded'] = df_dolfin['x'].round(10)
df_dolfin = df_dolfin.sort_values(by='x_rounded')
df_matlab['x_rounded'] = df_matlab['x'].round(10)

# Merge on rounded x values
df_merged = pd.merge(df_dolfin, df_matlab, on="x_rounded", suffixes=('_dolfin', '_matlab'))

print(f"Dolfin data points: {len(df_dolfin)}")
print(f"Matlab data points: {len(df_matlab)}")
print(f"Merged data points: {len(df_merged)}")

# Calculate statistics about the differences
if len(df_merged) > 0:
    differences = df_merged["u_dolfin"] - df_merged["u_matlab"]
    print(f"Mean difference: {differences.mean():.6f}")
    print(f"Max difference: {differences.max():.6f}")
    print(f"Min difference: {differences.min():.6f}")
    print(f"Std difference: {differences.std():.6f}")

# Plot u vs x for both datasets
plt.figure(figsize=(10, 8))

plt.plot(df_dolfin["x"].to_numpy(), df_dolfin["u"].to_numpy(), label="Dolfin", marker='o', markersize=2, linestyle='-', alpha=0.7)
plt.plot(df_matlab["x"].to_numpy(), df_matlab["u"].to_numpy(), label="Matlab", marker='x', markersize=2, linestyle='--', alpha=0.7)
plt.xlabel("x")
plt.ylabel("u")
plt.title("Comparison of Dolfin and Matlab results")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("outputs/difference_dolfin_matlab_sin_1.0.png", dpi=300, bbox_inches='tight')
plt.show()