import moola
import pandas as pd
import numpy as np

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
inc_wave_setup = IncidentWaveSetup(frequency, plane_wave)

measurement_data_file_path = "measurements/matlab_measurements_sin0.5_refraction_3.csv"
msh_file_path = "meshes/square_with_meshed_rect_obstacle.msh"
markers_dict = {
    "obstacle": OBSTACLE_MARKER,  # Markers imported from our mesh generation module
    "side_wall": SIDE_WALL_MARKER,
    "bottom_wall": RECEIVER_EDGE_MARKER,
    "obstacle_opt": None,
    "domain_marker": DOMAIN_MARKER,
    "obstacle_domain_marker": OBSTACLE_DOMAIN_MARKER
}
obstacle_stiffness = 25

initial_guess_mesh_util = MeshUtil(
    msh_file_path, markers_dict, obstacle_stiffness)
mesh, markers, domain_markers = initial_guess_mesh_util.get_mesh_and_markers()

##### Initialization #####
# Initialize the control variable of this problem
sub_mesh = SubMesh(mesh, domain_markers, OBSTACLE_DOMAIN_MARKER)
b_mesh = BoundaryMesh(sub_mesh, "exterior")

plot(b_mesh)
#plt.show()

S = VectorFunctionSpace(mesh, "CG", 1)
h = Function(S, name="Design")
h.vector()[:] = 0.0
h.vector().apply("insert")
##########################

# Solve the forward problem
u_scat_mag_dg0, u_scat_re_projected, u_scat_im_projected, ds_receiver, V_DG0 = forward_solve_refraction(
    h, inc_wave_setup, initial_guess_mesh_util, True)

# Load the reference data in the same function space as the projected
# result of the forward solve
u_ref_dg0, _ = load_forward_simulation_data_bottomwall(
    measurement_data_file_path, V_DG0)

J = assemble(
    (inner(u_scat_mag_dg0 - u_ref_dg0, u_scat_mag_dg0 - u_ref_dg0) * ds_receiver)
)
Jhat = ReducedFunctional(J, Control(h))

## Start optimizing ##
problem = MoolaOptimizationProblem(Jhat)
h_moola = moola.DolfinPrimalVector(h)

solver = moola.BFGS(problem, h_moola,
                    options={
                        "maxiter": 20
                    })

sol = solver.solve()
#s_opt = sol['control'].data

"""
result_path = "outputs/result_sin0.5_refraction_3_DG0_matlab.h5"
goal_geometry_msh_path = "meshes/square_with_sin_perturbed_rect_obstacle.msh"

save_optimization_result(
    sol,
    result_file=result_path,
    use_scipy=False
)

plot_mesh_deformation_from_result(
    result_path,
    goal_geometry_msh_path,
    initial_guess_mesh_util,
    plot_file_name="outputs/mesh_deformation_sin0.5_refraction_3_DG0_matlab.png",
    mesh_overlay_plot_file_name="outputs/mesh_overlay_sin0.5_refraction_3_DG0_matlab.png"
)

matlab_fullfield_csv_path = "measurements/matlab_fullfield_sin0.5_refraction_3.csv"
results = calculate_magnitude_and_phase_error(matlab_fullfield_csv_path, result_path,
                                              initial_guess_mesh_util, inc_wave_setup, True)

plot_projected_errors(
    results,
    "outputs/error_sin0.5_refraction_3_DG0_matlab.png")
"""