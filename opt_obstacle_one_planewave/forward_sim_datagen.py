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

msh_file_path = "meshes/square_with_halfsin_perturbed_rect_obstacle.msh"

all_results = []

frequencies = np.arange(2.5e9, 5.0e9 + 1, 0.5e9)  # [2.5e9, 3.0e9, ..., 5.0e9]

for frequency in frequencies:
    incident_field_func = plane_wave
    hh_setup = HelmholtzSetup(frequency, incident_field_func)
    h, mesh, markers = initialize_opt_xdmf(msh_file_path)
    u_tot_mag_dg0, _, V_DG0 = helmholtz_solve(
        mesh, markers, h, hh_setup,
        obstacle_marker, side_wall_marker, bottom_wall_marker
    )

    for facet in SubsetIterator(markers, bottom_wall_marker):
        cell = Cell(mesh, facet.entities(2)[0])  # cell adjacent to facet
        dof_idx = V_DG0.dofmap().cell_dofs(cell.index())[0]
        u_val = u_tot_mag_dg0.vector()[dof_idx]

        midpoint = facet.midpoint()
        all_results.append({
            "x": midpoint.x(),
            "y": midpoint.y(),
            "u": u_val,
            "frequency": frequency
        })

# Save all results to a single CSV
df = pd.DataFrame(all_results)
df.to_csv("forward_sim_data_bottom_sweep_halfsin.csv", index=False)

# Optionally, plot for the last frequency
plt.figure()
p = plot(u_tot_mag_dg0, title=f"Magnitude of total field (u_inc + u_sol) at {frequency/1e9} GHz", cmap="hot")
plt.colorbar(p)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.savefig("picwave_last.png")