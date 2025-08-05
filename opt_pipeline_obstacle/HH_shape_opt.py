import h5py
import json
from dolfin import *
from dolfin_adjoint import * 
import numpy as np
import pandas as pd

from scipy.special import hankel1
import subprocess
import os
import sys
import gmsh
import matplotlib.pyplot as plt

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HH_shape_opt.helmholtz_solve import helmholtz_solve
from HH_shape_opt.initialize_opt import initialize_opt
from HH_shape_opt.save_result import save_optimization_result

k_background = 2* np.pi * 5e9 / 299792458 # 2pi f / c
incident_wave_amp = 1

def load_forward_simulation_data_bottomwall(V_DG0):
    df = pd.read_csv("forward_sim_data_bottom.csv")
    points = df[["x", "y"]].values
    values = df["u"].values

    # Set up the assignment
    u_ref_dg0 = Function(V_DG0)
    tree = mesh.bounding_box_tree()
    dofmap = V_DG0.dofmap()
    u_vec = u_ref_dg0.vector().get_local()

    # For tracking which cells we've already assigned (to avoid duplicates)
    assigned = np.zeros(mesh.num_cells(), dtype=bool)

    for (x, y), val in zip(points, values):
        point = Point(x, y)
        cell_id = tree.compute_first_entity_collision(point)
        if cell_id < mesh.num_cells() and not assigned[cell_id]:
            dof_idx = dofmap.cell_dofs(cell_id)[0]
            u_vec[dof_idx] = val
            assigned[cell_id] = True
        elif cell_id < mesh.num_cells() and assigned[cell_id]:
            print(f"Warning: cell {cell_id} already assigned, skipping duplicate point.")
        else:
            print(f"Warning: No cell found containing point ({x}, {y})")

    # Push the updated values into the Function
    u_ref_dg0.vector().set_local(u_vec)
    u_ref_dg0.vector().apply("insert")

    return u_ref_dg0

######################################

iteration = 0
num_iterations = 500

msh_file_path = "meshes/square_with_gaussian_perturbed_rect.msh"

# Initialization by copying the mesh we want to perform the forward sim on and
# get the first initial guesses of h (all zero by default)
h, mesh, markers = initialize_opt(msh_file_path)

# Solve the forward problem
u_tot_mag_dg0, ds_bottom, V_DG0= helmholtz_solve(mesh, markers, h, k_background, incident_wave_amp,
                                                             obstacle_marker, side_wall_marker, bottom_wall_marker)
# Load reference data
u_ref_dg0 = load_forward_simulation_data_bottomwall(V_DG0)

J = assemble((inner(u_tot_mag_dg0 - u_ref_dg0, u_tot_mag_dg0 - u_ref_dg0)* ds_bottom))
Jhat = ReducedFunctional(J, Control(h))

## Start optimizing ##
h_opt, opt_result  = minimize(
    Jhat,
    tol=1e-6,
    method="L-BFGS-B",
    options={"gtol": 1e-7, "maxiter": num_iterations, "disp": True}
)

save_optimization_result(
    h_opt,
    opt_result,
    msh_file_path,
    checkpoint_file = "result.h5",
)
