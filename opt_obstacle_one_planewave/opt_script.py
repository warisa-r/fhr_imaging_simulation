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


#msh_file_path = "meshes/square_with_sin_perturbed_rect_obstacle.msh"
msh_file_path = "meshes/square_with_rect_obstacle_all.msh"
#goal_geometry_msh_path = "meshes/square_with_sym_exp_perturbed_rect.msh"
forward_sim_result_file_path = "forward_sim_data_bottom_sweep2.csv"
result_path = "outputs_sweep_scipy/result_cosbump_200_1freq.h5"

frequencies = np.arange(2.5e9, 5.0e9 + 1, 0.5e9)

h, mesh, markers = initialize_opt_xdmf(msh_file_path)
V_DG0_initial = FunctionSpace(mesh, "DG", 0)
reference_data_maps = []

frequencies = [frequencies[-1]]

for frequency in frequencies:
    reference_data_map = preprocess_reference_data(V_DG0_initial, forward_sim_result_file_path, frequency)
    reference_data_maps.append(reference_data_map)

for i, frequency in enumerate(frequencies):
    incident_field_func = plane_wave
    hh_setup = HelmholtzSetup(frequency, incident_field_func, 50)

    # Initialization by copying the mesh we want to perform the forward sim on and
    # get the first initial guesses of h (all zero by default)

    # Solve the forward problem
    u_tot_mag_dg0, ds_bottom, V_DG0 = helmholtz_solve(mesh, markers, h, hh_setup,
                                                    obstacle_marker, side_wall_marker, bottom_wall_marker)
    # Load reference data
    u_ref_dg0 = assign_reference_data(V_DG0, reference_data_maps[i])

    if i ==0:
        J = assemble((inner(u_tot_mag_dg0 - u_ref_dg0, u_tot_mag_dg0 - u_ref_dg0)* ds_bottom))
    else:
        J += assemble((inner(u_tot_mag_dg0 - u_ref_dg0, u_tot_mag_dg0 - u_ref_dg0)* ds_bottom))
comm = MPI.comm_world
def eval_cb(j, h):
    if comm.rank == 0:
        print("objective = %f, norm of h = %f." % (j, float(norm(h))))
#dJdh = Jhat.derivative()
#plot(dJdh, title=f"Gradient of J with respect to h")
#savefig("outputs_sweep/gradient_sin2.png")
Jhat = ReducedFunctional(
    J,
    Control(h), eval_cb_post = eval_cb
)
sol = minimize(Jhat, tol=1e-6, options={"gtol": 1e-6, "maxiter": 200, "disp": True})
sys.stdout.flush()

save_optimization_result(sol, msh_file_path, hh_setup.obstacle_stiffness, result_path, True)

#plot_mesh_deformation_from_result(
#    result_path,
#    msh_file_path,
#    goal_geometry_msh_path,
#    obstacle_marker,
#    side_wall_marker,
#    bottom_wall_marker,
#    "outputs/mesh_deformation_sym_exp_100.png"
#)