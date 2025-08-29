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

def main(alpha_value=1.0):
    set_log_level(LogLevel.ERROR)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    #msh_file_path = "meshes/square_with_sin_perturbed_rect_obstacle.msh"
    msh_file_path = "meshes/square_with_rect_obstacle_all.msh"
    #goal_geometry_msh_path = "meshes/square_with_sym_exp_perturbed_rect.msh"
    forward_sim_result_file_path = "forward_sim_data_bottom_sweep_halfsin.csv"

    frequencies = np.arange(2.5e9, 5.0e9 + 1, 0.5e9)

    h, mesh, markers = initialize_opt_xdmf(msh_file_path)
    V_DG0_initial = FunctionSpace(mesh, "DG", 0)
    reference_data_maps = []

    iteration_counter = [0]
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

    Jhat = ReducedFunctional(
        J,
        Control(h)
    )

    # Use the provided alpha value
    problem = MoolaOptimizationProblem(Jhat)
    h_moola = moola.DolfinPrimalVector(h)

    solver = moola.SteepestDescent(problem, h_moola, 
        options={
            "maxiter": 1,
            "line_search": "fixed",
            "line_search_options": {"start_stp": alpha_value}
        })

    sol = solver.solve()
    
    # Print objective value in a parseable format
    print(f"FINAL_OBJECTIVE: {sol['objective']}")
    return sol['objective']

if __name__ == "__main__":
    # Get alpha from command line arguments
    alpha_value = 1.0  # default
    if len(sys.argv) > 1:
        try:
            alpha_value = float(sys.argv[1])
        except ValueError:
            print(f"Warning: Invalid alpha value '{sys.argv[1]}', using default 1.0")
    
    main(alpha_value)