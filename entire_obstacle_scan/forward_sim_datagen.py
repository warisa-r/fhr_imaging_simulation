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

want_plot = True
AMP = 1

def plane_wave_angle(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    direction_x = np.cos(angle_rad)
    direction_y = np.sin(angle_rad)

    def wave_func(x, k_background):
        return AMP * np.exp(1j * k_background * (x[0] * direction_x + x[1] * direction_y))
    
    return wave_func

msh_file_path = "meshes/square_with_sin_perturbed_rect_obstacle.msh"

all_results = []

frequency = 5e9
angles = [0, 90, 180, 270]

for angle in angles:
    incident_field_func = plane_wave_angle(angle)
    hh_setup = HelmholtzSetup(frequency, incident_field_func)
    h, mesh, markers = initialize_opt_xdmf(msh_file_path)
    u_tot_mag_dg0, _, V_DG0 = helmholtz_solve(
        mesh, markers, h, hh_setup,
        obstacle_marker, side_wall_marker, bottom_wall_marker
    )

    # Save data for both bottom wall and side wall
    for wall_marker, wall_name in [(bottom_wall_marker, "bottom"), (side_wall_marker, "side")]:
        for facet in SubsetIterator(markers, wall_marker):
            cell = Cell(mesh, facet.entities(2)[0])  # cell adjacent to facet
            dof_idx = V_DG0.dofmap().cell_dofs(cell.index())[0]
            u_val = u_tot_mag_dg0.vector()[dof_idx]

            midpoint = facet.midpoint()
            all_results.append({
                "x": midpoint.x(),
                "y": midpoint.y(),
                "u": u_val,
                "angle": angle
            })
    # Plot for this angle
    if want_plot == True:
        plt.figure()
        p = plot(u_tot_mag_dg0, title=f"Magnitude of total field at {frequency/1e9} GHz, angle={angle}°", cmap="hot")
        plt.colorbar(p)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.savefig(f"picwave_angle_{angle}.png")
        plt.close()

# Save all results to a single CSV
df = pd.DataFrame(all_results)
df.to_csv("forward_sim_data.csv", index=False)