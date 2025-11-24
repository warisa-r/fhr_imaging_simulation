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

from HH_shape_opt import *

# Ensure this can be run from root dir
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

set_log_level(LogLevel.ERROR)

frequency = 5e9
wave_source = (0.5, -2.0)
amp = 50
inc_wave_setup = IncidentWaveSetup(frequency, cylindrical_wave(amp, wave_source))

measurement_data_file_path = "measurements/matlab_fullfield_sin0.5_scatter_noisy.csv"
msh_file_path = "meshes/square_with_rect_obstacle.msh"
markers_dict = {
    "obstacle": OBSTACLE_MARKER, # Markers importee from our mesh generation module
    "side_wall": SIDE_WALL_MARKER,
    "bottom_wall": RECEIVER_EDGE_MARKER,
    "obstacle_opt": None
}
obstacle_stiffness = 25

initial_guess_mesh_util = MeshUtil(
    msh_file_path, markers_dict, obstacle_stiffness)
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
u_scat_re, u_scat_im, ds_receiver, V_DG0 = forward_solve(
    h, inc_wave_setup, initial_guess_mesh_util, True)

# Load the reference data in the same function space as the projected result of the forward solve
u_ref_re, u_ref_im, _ = load_forward_simulation_data_bottomwall(
    measurement_data_file_path, V_DG0)

J = assemble(
    (((u_scat_re - u_ref_re)**2 + (u_scat_im - u_ref_im)**2) * ds_receiver))
Jhat = ReducedFunctional(J, Control(h))

## Start optimizing ##
problem = MoolaOptimizationProblem(Jhat)
h_moola = moola.DolfinPrimalVector(h)

solver = moola.BFGS(problem, h_moola,
                    options={
                        "maxiter": 30
                    })

sol = solver.solve()
h_opt = sol['control'].data

result_path = "outputs/result_sin0.5_scat_noisy.h5"
goal_geometry_msh_path = "meshes/square_with_halfsin_perturbed_rect_obstacle.msh"

save_optimization_result(
    sol,
    result_file=result_path,
    use_scipy=False
)

plot_mesh_deformation_from_result(
    result_path,
    goal_geometry_msh_path,
    initial_guess_mesh_util,
    plot_file_name="outputs/mesh_deformation_sin0.5_scat_noisy.png",
    mesh_overlay_plot_file_name = "outputs/mesh_overlay_sin0.5_scat_noisy.png"
)

matlab_fullfield_csv_path = "measurements/matlab_fullfield_sin0.5_scatter.csv"
results = calculate_magnitude_and_phase_error(matlab_fullfield_csv_path, result_path,
                                        initial_guess_mesh_util, inc_wave_setup, True)

plot_projected_errors(results, "outputs/error_sin0.5_scat_noisy.png", True)

# Print optimization summary
print("\n=== Optimization Summary ===")
print(f"Initial design: all zeros")
print(
    f"Optimal design range: [{np.min(h_opt.vector().get_local()):.6e}, {np.max(h_opt.vector().get_local()):.6e}]")
print(f"Max displacement: {np.max(np.abs(h_opt.vector().get_local())):.6e}")
