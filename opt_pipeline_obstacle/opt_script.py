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

# MPI setup
comm = MPI.comm_world
rank = MPI.rank(comm)
size = MPI.size(comm)

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HH_shape_opt.helmholtz_solve import HelmholtzSetup, plane_wave, helmholtz_solve, preprocess_reference_data, assign_reference_data
from HH_shape_opt.initialize_opt import initialize_opt_xdmf
from HH_shape_opt.process_result import save_optimization_result, plot_mesh_deformation_from_result


######################################
#msh_file_path = "meshes/square_with_rect_obstacle.msh"
msh_file_path = "meshes/square_with_gaussian_perturbed_rect.msh" # To check and see the degree of 
forward_sim_result_file_path = "forward_sim_data_bottom.csv"
result_path = "result.h5"

frequency = 5e9
incident_field_func = plane_wave

angles = [-45, 0, 45]

h, initial_mesh, _ = initialize_opt_xdmf(msh_file_path)

# Pre-process reference data on the initial, undeformed mesh
V_DG0_initial = FunctionSpace(initial_mesh, "DG", 0)
reference_data_maps = {
    angle: preprocess_reference_data(V_DG0_initial, forward_sim_result_file_path, angle)
    for angle in angles
}

# Initialize list to store individual cost function contributions
J_contributions = []

# A factory function to correctly capture the angle in the closure
def make_plane_wave(angle):
    def plane_wave(x, k_background):
        # Note: angle is given as a degree
        direction_x = np.sin(np.deg2rad(angle))  # x-component of the direction
        direction_y = np.cos(np.deg2rad(angle))  # y-component of the direction

        # Dot product of the direction vector with the spatial coordinates
        dot_product = direction_x * x[0] + direction_y * x[1]

        return np.exp(1j * k_background * dot_product)
    return plane_wave

# Each process calculates its part of the functional
for angle in angles:
    # Create the incident field function for the current angle
    incident_field_func = make_plane_wave(angle)
    hh_setup = HelmholtzSetup(frequency, incident_field_func)

    # CREATE FRESH MESH COPIES FOR EACH SOLVE
    _, mesh_copy, markers_copy = initialize_opt_xdmf(msh_file_path)

    # Solve the forward problem
    u_tot_mag_dg0, ds_bottom, V_DG0 = helmholtz_solve(mesh_copy, markers_copy, h, hh_setup,
                                                                obstacle_marker, side_wall_marker, bottom_wall_marker)

    # Load reference data from pre-processed map
    u_ref_dg0 = assign_reference_data(V_DG0, reference_data_maps[angle])

    # Assemble and store the contribution to the functional
    J_i = assemble(inner(u_tot_mag_dg0 - u_ref_dg0, u_tot_mag_dg0 - u_ref_dg0)* ds_bottom)
    J_contributions.append(J_i)

Jhat = ReducedFunctional(sum(J_contributions), Control(h))

problem = MoolaOptimizationProblem(Jhat)

problem = MoolaOptimizationProblem(Jhat)
h_moola = moola.DolfinPrimalVector(h)

solver = moola.BFGS(problem, h_moola, options={'jtol': 1e-8,
                                            'gtol': 1e-7,
                                            'Hinit': "default",
                                            'maxiter': 1,
                                            'mem_lim': 10})

# Solve
sol = solver.solve()

comm.Barrier()

if rank == 0:
    save_optimization_result(sol, msh_file_path, result_path)

    plot_mesh_deformation_from_result(
        result_path,
        msh_file_path,
        obstacle_marker,
        side_wall_marker,
        bottom_wall_marker
    )