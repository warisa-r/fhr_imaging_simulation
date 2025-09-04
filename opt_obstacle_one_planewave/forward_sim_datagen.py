from dolfin import *
import numpy as np
from scipy.special import hankel1
import subprocess
import os
import sys
import json
import gmsh
import matplotlib.pyplot as plt

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HH_shape_opt.helmholtz_solve import HelmholtzSetup, plane_wave, helmholtz_solve
from HH_shape_opt.initialize_opt import initialize_opt_xml, initialize_opt_xdmf

import pandas as pd

msh_file_path = "square_with_kite_obstacle.msh"

all_results = []

frequencies = np.arange(2.5e9, 5.0e9 + 1, 0.5e9)  # [2.5e9, 3.0e9, ..., 5.0e9]

for frequency in frequencies:
    incident_field_func = plane_wave
    hh_setup = HelmholtzSetup(frequency, incident_field_func)
    h, mesh, markers = initialize_opt_xdmf(msh_file_path)
    
    # Get the solution from helmholtz_solve
    u_tot_mag, ds_bottom, V_CG5 = helmholtz_solve(
        mesh, markers, h, hh_setup,
        obstacle_marker, side_wall_marker, bottom_wall_marker, True
    )
    
    # Get DOF coordinates and values
    dof_coordinates = V_CG5.tabulate_dof_coordinates()
    u_values = u_tot_mag.vector().get_local()
    
    # Simply record all DOF positions and values on bottom boundary (y ≈ 0)
    tolerance = 1e-10
    for dof_idx, coord in enumerate(dof_coordinates):
        if abs(coord[1]) < tolerance:  # Bottom boundary
            all_results.append({
                "x": coord[0],
                "y": coord[1],
                "u": u_values[dof_idx],
                "frequency": frequency
            })

# Save all results to CSV
df = pd.DataFrame(all_results)
df.to_csv("forward_sim_data_bottom_sweep_sin.csv", index=False)

print(f"Generated data for {len(frequencies)} frequencies")
print(f"Total data points: {len(all_results)}")

# Plot for the last frequency
plt.figure(figsize=(10, 8))
p = plot(u_tot_mag, title=f"Magnitude of total field at {frequency/1e9:.1f} GHz", cmap="hot")
plt.colorbar(p)
plt.xlabel("x")
plt.ylabel("y")

# Mark the boundary DOF points
boundary_points_last = df[df['frequency'] == frequency]
plt.scatter(boundary_points_last['x'], boundary_points_last['y'], 
           c='blue', s=20, marker='o', alpha=0.7, label='Boundary DOFs')
plt.legend()
plt.axis("equal")
plt.savefig("forward_sim_with_boundary_dofs.png", dpi=300, bbox_inches='tight')
plt.show()