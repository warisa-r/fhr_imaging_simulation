import h5py
from dolfin import *
from dolfin_adjoint import * 
import numpy as np
import moola
import pandas as pd

import subprocess
import os
import gmsh
import matplotlib.pyplot as plt

from HH_shape_opt.initialize_opt import MeshUtil
from HH_shape_opt.process_result import save_optimization_result, plot_mesh_deformation_from_result
from HH_shape_opt.helmholtz_solve import forward_solve, load_forward_simulation_data_bottomwall, IncidentWaveSetup, plane_wave
from HH_shape_opt.mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker, obstacle_opt_marker

# Ensure this can be run from root dir
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

set_log_level(LogLevel.ERROR)

frequency = 5e9
inc_wave_setup = IncidentWaveSetup(frequency, plane_wave)

measurement_data_file_path = "matlab_measurements.csv"
msh_file_path = "meshes/square_with_rect_obstacle_opt.msh"
markers_dict = {
    "obstacle": obstacle_marker,
    "side_wall": side_wall_marker,
    "bottom_wall": bottom_wall_marker,
    "obstacle_opt": obstacle_opt_marker
}
obstacle_stiffness = 25

initial_guess_mesh_util = MeshUtil(msh_file_path, markers_dict, obstacle_stiffness)
mesh, _ = initial_guess_mesh_util.get_mesh_and_markers()

##### Initialization #####
# Create boundary mesh and design variables
b_mesh = BoundaryMesh(mesh, "exterior")
S_b = VectorFunctionSpace(b_mesh, "CG", 1)
h = Function(S_b, name="Design")
h.vector()[:] = 0.0
h.vector().apply("insert")

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="Mesh perturbation field")
h_V = transfer_from_boundary(h, mesh)
h_V.rename("Volume extension of h", "")
##########################

# Solve the forward problem
u_tot_mag_dg0, ds_bottom, V_DG0 = forward_solve(h, inc_wave_setup, initial_guess_mesh_util)

# Load the reference data in the same function space as the projected result of the forward solve
u_ref_dg0 = load_forward_simulation_data_bottomwall(measurement_data_file_path, V_DG0)

J = assemble((inner(u_tot_mag_dg0 - u_ref_dg0, u_tot_mag_dg0 - u_ref_dg0)* ds_bottom))
Jhat = ReducedFunctional(J, Control(h))

## Start optimizing ##
problem = MoolaOptimizationProblem(Jhat)
h_moola = moola.DolfinPrimalVector(h)

solver = moola.BFGS(problem, h_moola,
    options={
        "maxiter": 10,
        "gtol": 1e-7,
    })

sol = solver.solve()
h_opt = sol['control'].data

result_path = "outputs/result_sin_1.0_DG0_restricted_matlab.h5"
goal_geometry_msh_path = "meshes/square_with_sin_perturbed_rect_obstacle.msh"

save_optimization_result(
    sol,
    msh_file_path,
    obstacle_stiffness,
    result_file = result_path,
    use_scipy = False
)

plot_mesh_deformation_from_result(
    result_path,
    msh_file_path,
    goal_geometry_msh_path,
    obstacle_marker,
    side_wall_marker,
    bottom_wall_marker,
    obstacle_opt_marker,
    plot_file_name="outputs/mesh_deformation_sin_1.0_DG0_restricted_matlab.png",
    obstacle_stiffness = obstacle_stiffness,
)

# Print optimization summary
print("\n=== Optimization Summary ===")
print(f"Initial design: all zeros")
print(f"Optimal design range: [{np.min(h_opt.vector().get_local()):.6e}, {np.max(h_opt.vector().get_local()):.6e}]")
print(f"Max displacement: {np.max(np.abs(h_opt.vector().get_local())):.6e}")
