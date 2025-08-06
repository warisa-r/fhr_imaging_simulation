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
from HH_shape_opt.initialize_opt import initialize_opt

msh_file_path = "meshes/square_with_gaussian_perturbed_rect.msh"

frequency = 5e9
incident_field_func = plane_wave
hh_setup = HelmholtzSetup(frequency, incident_field_func)
# Run forward simulation
h, mesh, markers = initialize_opt(msh_file_path)
u_tot_mag_dg0, _, V_DG0 = helmholtz_solve(mesh, markers, h, hh_setup,
                                                             obstacle_marker, side_wall_marker, bottom_wall_marker)

### Save the data ###

# We have to project the data as a constant on every grid point in order to make it a
# correct approximation of the value of u
import pandas as pd

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
    "x": x_vals,
    "y": y_vals,
    "u": u_vals_bottom
})
df.to_csv("forward_sim_data_bottom.csv", index=False)

# Plot magnitude of total field
plt.figure()
p = plot(u_tot_mag_dg0, title="Magnitude of total field (u_inc + u_sol)", cmap="hot")
plt.colorbar(p)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()