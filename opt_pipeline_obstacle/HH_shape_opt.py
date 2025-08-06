import h5py
import json
from dolfin import *
from dolfin_adjoint import * 
import numpy as np


import moola
import subprocess
import os
import sys
import gmsh
import matplotlib.pyplot as plt

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HH_shape_opt.helmholtz_solve import helmholtz_solve, load_forward_simulation_data_bottomwall
from HH_shape_opt.initialize_opt import initialize_opt
from HH_shape_opt.process_result import save_optimization_result, plot_mesh_deformation_from_result

k_background = 2* np.pi * 5e9 / 299792458 # 2pi f / c
incident_wave_amp = 1

######################################

iteration = 0
num_iterations = 500

msh_file_path = "meshes/square_with_gaussian_perturbed_rect.msh"
forward_sim_result_file_path = "forward_sim_data_bottom.csv"
result_path = "result.h5"

# Initialization by copying the mesh we want to perform the forward sim on and
# get the first initial guesses of h (all zero by default)
h, mesh, markers = initialize_opt(msh_file_path)

# Solve the forward problem
u_tot_mag_dg0, ds_bottom, V_DG0 = helmholtz_solve(mesh, markers, h, k_background, incident_wave_amp,
                                                             obstacle_marker, side_wall_marker, bottom_wall_marker)
# Load reference data
u_ref_dg0 = load_forward_simulation_data_bottomwall(V_DG0, forward_sim_result_file_path)

J = assemble((inner(u_tot_mag_dg0 - u_ref_dg0, u_tot_mag_dg0 - u_ref_dg0)* ds_bottom))

Jhat = ReducedFunctional(J, Control(h))

problem = MoolaOptimizationProblem(Jhat)
h_moola = moola.DolfinPrimalVector(h)

solver = moola.BFGS(problem, h_moola, options={'jtol': 1e-8,
                                               'gtol': 1e-7,
                                               'Hinit': "default",
                                               'maxiter': 1,
                                               'mem_lim': 10})

# Solve
sol = solver.solve()

save_optimization_result(sol, msh_file_path, result_path)

plot_mesh_deformation_from_result(
    result_path,
    msh_file_path,
    obstacle_marker,
    side_wall_marker,
    bottom_wall_marker,
)