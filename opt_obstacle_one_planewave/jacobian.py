import h5py
import json
from dolfin import *
from dolfin_adjoint import * 
import scipy
import numpy as np
from matplotlib.pyplot import show, savefig

import moola
import subprocess
import os
import sys
import gmsh
import matplotlib.pyplot as plt
import warnings

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker, obstacle_opt_marker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HH_shape_opt.helmholtz_solve import HelmholtzSetup, plane_wave, helmholtz_solve, preprocess_reference_data, assign_reference_data
from HH_shape_opt.initialize_opt import initialize_opt_xdmf
from HH_shape_opt.process_result import save_optimization_result, plot_mesh_deformation_from_result

set_log_level(LogLevel.ERROR)

######################################

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

goal_geometry_msh_path = "meshes/square_with_sin_perturbed_rect_obstacle.msh"
#msh_file_path = "meshes/square_with_halfsin_perturbed_rect_obstacle.msh"
msh_file_path = "meshes/square_with_rect_obstacle_all.msh"
forward_sim_result_file_path = "forward_sim_data_bottom_sweep_halfsin.csv"
result_path = "outputs/result_sin_hybrid_75.h5"

frequencies = np.arange(2.5e9, 5.0e9 + 1, 0.5e9)

h, mesh, markers = initialize_opt_xdmf(msh_file_path)
V_initial = FunctionSpace(mesh, "CG", 5)
reference_data_maps = []

iteration_counter = [0]
frequencies = [frequencies[-1]]

for frequency in frequencies:
    reference_data_map = preprocess_reference_data(V_initial, forward_sim_result_file_path, frequency)
    reference_data_maps.append(reference_data_map)

for i, frequency in enumerate(frequencies):
    incident_field_func = plane_wave
    hh_setup = HelmholtzSetup(frequency, incident_field_func, 50)

    # Initialization by copying the mesh we want to perform the forward sim on and
    # get the first initial guesses of h (all zero by default)

    # Solve the forward problem
    u_tot_mag, ds_bottom, V_CG5 = helmholtz_solve(mesh, markers, h, hh_setup,
                                                    obstacle_marker, side_wall_marker, bottom_wall_marker)
    # Load reference data
    u_ref = assign_reference_data(V_CG5, reference_data_maps[i])

    if i ==0:
        J = assemble((inner(u_tot_mag - u_ref, u_tot_mag - u_ref)* ds_bottom))
    else:
        J += assemble((inner(u_tot_mag - u_ref, u_tot_mag - u_ref)* ds_bottom))

Jhat = ReducedFunctional(
    J,
    Control(h)
)

############# Compute the jacobian #################
dJ = Jhat.derivative()
V_h = h.function_space()

# dJ is a dolfin-adjoint UFL form, so project to the same space as h
dJ_projected = project(dJ, V_h)

# Extract coefficient vector
dJ_values = dJ_projected.vector().get_local()

import numpy as np

b_mesh = BoundaryMesh(mesh, "exterior")
S_b = VectorFunctionSpace(b_mesh, "CG", 1)

hx, hy = dJ_projected.split()
hx_vertex = hx.compute_vertex_values(b_mesh)
hy_vertex = hy.compute_vertex_values(b_mesh)

dJ_vertices = np.vstack((hx_vertex, hy_vertex))  # shape (2, n_vertices)
#print(dJ_vertices)

_, s, _ = np.linalg.svd(dJ_vertices)

print("s: ", s)
print("s1/s2", s[0]/s[1])