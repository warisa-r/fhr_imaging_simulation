import h5py
import json
from dolfin import *
from dolfin_adjoint import * 
import numpy as np
from matplotlib.pyplot import show, savefig

import moola
import subprocess
import os
import sys
import gmsh
import matplotlib.pyplot as plt

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HH_shape_opt.helmholtz_solve import HelmholtzSetup, plane_wave, helmholtz_solve, preprocess_reference_data, assign_reference_data
from HH_shape_opt.initialize_opt import initialize_opt_xdmf
from HH_shape_opt.process_result import save_optimization_result, plot_mesh_deformation_from_result

set_log_level(LogLevel.ERROR)

######################################

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

#msh_file_path = "meshes/square_with_sym_exp_perturbed_rect.msh"
msh_file_path = "meshes/square_with_rect_obstacle.msh"
#goal_geometry_msh_path = "meshes/square_with_sym_exp_perturbed_rect.msh"
forward_sim_result_file_path = "matlab_measurement.csv"
result_path = "outputs/result_sin_freq3_matlab.h5"

frequency = 5e9
incident_field_func = plane_wave
hh_setup = HelmholtzSetup(frequency, incident_field_func)

# Initialization by copying the mesh we want to perform the forward sim on and
# get the first initial guesses of h (all zero by default)
h, mesh, markers = initialize_opt_xdmf(msh_file_path)
V_DG0_initial = FunctionSpace(mesh, "DG", 0)
reference_data_map = preprocess_reference_data(V_DG0_initial, forward_sim_result_file_path, None)

# Solve the forward problem
u_tot_mag_dg0, ds_bottom, V_DG0 = helmholtz_solve(mesh, markers, h, hh_setup,
                                                             obstacle_marker, side_wall_marker, bottom_wall_marker)
# Load reference data
u_ref_dg0 = assign_reference_data(V_DG0, reference_data_map)

J = assemble((inner(u_tot_mag_dg0 - u_ref_dg0, u_tot_mag_dg0 - u_ref_dg0)* ds_bottom))

Jhat = ReducedFunctional(J, Control(h))
#dJdh = Jhat.derivative()
#plot(dJdh, title=f"Gradient of J with respect to h for symmetric exponential perturbation")
#savefig("outputs/gradient_sy,.png")

problem = MoolaOptimizationProblem(Jhat)
h_moola = moola.DolfinPrimalVector(h)

solver = moola.BFGS(problem, h_moola, options={'jtol': 1e-8,
                                               'gtol': 1e-7,
                                               'Hinit': "default",
                                               'maxiter': 500,
                                               'mem_lim': 10})

# Solve
sol = solver.solve()


save_optimization_result(sol, msh_file_path, result_path)

#plot_mesh_deformation_from_result(
#    result_path,
#    msh_file_path,
#    goal_geometry_msh_path,
#    obstacle_marker,
#    side_wall_marker,
#    bottom_wall_marker,
#    "outputs/mesh_deformation_sym_exp_100.png"
#)