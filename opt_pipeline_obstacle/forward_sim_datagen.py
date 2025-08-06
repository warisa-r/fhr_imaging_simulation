from dolfin import *
import numpy as np
from scipy.special import hankel1
import subprocess
import os
import sys
import json
import gmsh
import matplotlib.pyplot as plt
import pandas as pd

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HH_shape_opt.helmholtz_solve import HelmholtzSetup, plane_wave, helmholtz_solve
from HH_shape_opt.initialize_opt import initialize_opt_xml

# Simulation setup
msh_file_path = "meshes/square_with_gaussian_perturbed_rect.msh"

frequency = 5e9

angles = [-45, 0, 45]

h, mesh, markers = initialize_opt_xml(msh_file_path)
for angle in angles:
    # Define the plane wave from different direction
    def plane_wave(x, k_background):
        # Note: angle is given as a degree
        direction_x = np.sin(np.deg2rad(angle))  # x-component of the direction
        direction_y = np.cos(np.deg2rad(angle))  # y-component of the direction

        # Dot product of the direction vector with the spatial coordinates
        dot_product = direction_x * x[0] + direction_y * x[1]

        return np.exp(1j * k_background * dot_product)
    
    incident_field_func = plane_wave
    hh_setup = HelmholtzSetup(frequency, incident_field_func)

    # Run forward simulation
    u_tot_mag_dg0, _, V_DG0 = helmholtz_solve(mesh, markers, h, hh_setup,
                                                                obstacle_marker, side_wall_marker, bottom_wall_marker)

    ### Save the data ###
    u_vals_bottom = []
    x_vals = []
    y_vals = []

    for facet in SubsetIterator(markers, bottom_wall_marker):
        cell = Cell(mesh, facet.entities(2)[0])  # cell adjacent to facet
        dof_idx = V_DG0.dofmap().cell_dofs(cell.index())[0]
        u_val = u_tot_mag_dg0.vector()[dof_idx]

        midpoint = facet.midpoint()
        x_vals.append(midpoint.x())
        y_vals.append(midpoint.y())
        u_vals_bottom.append(u_val)

    df = pd.DataFrame({
        "angle": angle,
        "x": x_vals,
        "y": y_vals,
        "u": u_vals_bottom,
    })

    # Append data to the CSV file
    output_file = "forward_sim_data_bottom.csv"
    if not os.path.exists(output_file):
        # Write header if file does not exist
        df.to_csv(output_file, index=False)
    else:
        # Append data without writing the header
        df.to_csv(output_file, mode='a', header=False, index=False)

    # Plot magnitude of total field
    plt.figure()
    p = plot(u_tot_mag_dg0, title="Magnitude of total field (u_inc + u_sol)", cmap="hot")
    plt.colorbar(p)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()