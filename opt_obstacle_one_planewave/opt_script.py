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

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker, obstacle_opt_marker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HH_shape_opt.helmholtz_solve import HelmholtzSetup, plane_wave, helmholtz_solve, preprocess_reference_data, assign_reference_data
from HH_shape_opt.initialize_opt import initialize_opt_xdmf
from HH_shape_opt.process_result import save_optimization_result, plot_mesh_deformation_from_result

set_log_level(LogLevel.ERROR)

######################################

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

#msh_file_path = "meshes/square_with_halfsin_perturbed_rect_obstacle.msh"
msh_file_path = "meshes/square_with_rect_obstacle_all.msh"
#goal_geometry_msh_path = "meshes/square_with_sym_exp_perturbed_rect.msh"
forward_sim_result_file_path = "forward_sim_data_bottom_sweep_halfsin.csv"
result_path = "outputs/result_halfsin_15.h5"

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

#dJ = Jhat.derivative()
#plot(dJ, title = "derivative of J with respect to h")
#savefig("djdh.png")


problem = MoolaOptimizationProblem(Jhat)
h_moola = moola.DolfinPrimalVector(h)

alpha = 1.0
solver = moola.BFGS(problem, h_moola, 
    options={
    "maxiter": 15,
    "mem_lim": 1
    })

#"line_search_options": {"ftol": 1e-4, "start_stp": 10.0, "stpmin" : 1e-10, "stpmax":10000}
#})
sol = solver.solve()

save_optimization_result(sol, msh_file_path, hh_setup.obstacle_stiffness, result_path, False)

"""

problem = MinimizationProblem(Jhat)

parameters = {
    "acceptable_tol": 1e-3,
    "max_iter": 100,
    "linear_solver": "ma97",
    "hsllib": 'libcoinhsl.so',
    "print_level" : 5
}
solver = IPOPTSolver(problem, parameters=parameters)
sol = solver.solve()
#save_optimization_result(sol, msh_file_path, hh_setup.obstacle_stiffness, result_path, True)

problem = MoolaOptimizationProblem(Jhat)
h_moola = moola.DolfinPrimalVector(h)

solver = moola.SteepestDescent(problem, h_moola, 
options={
"maxiter": 10,
'line_search': "fixed", 
"line_search_options": {"start_stp": 2.5}})
#"line_search_options": {"ftol": 1e-4, "start_stp": 10.0, "stpmin" : 1e-10, "stpmax":10000}
#})
sol = solver.solve()
save_optimization_result(sol, msh_file_path, hh_setup.obstacle_stiffness, result_path, False)

#plot_mesh_deformation_from_result(
#    result_path,
#    msh_file_path,
#    goal_geometry_msh_path,
#    obstacle_marker,
#    side_wall_marker,
#    bottom_wall_marker,
#    "outputs/mesh_deformation_sym_exp_100.png"
#)
"""