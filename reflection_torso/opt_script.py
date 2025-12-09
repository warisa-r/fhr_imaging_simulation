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

frequency = 5.5e9
offset = 1e-8
wave_sources = [(-0.1, -offset), (0.1, -offset)]
amp = 1

measurement_data_file_path = "measurements/scattering_results_5per_noise.csv"
df_measurement = pd.read_csv(measurement_data_file_path)

msh_file_path = "meshes/torso_sim_mesh_initial.msh"
markers_dict = {
    "obstacle": OBSTACLE_MARKER, # Markers imported from our mesh generation module
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

scale_fac = 500

for tx_i, tx in enumerate(wave_sources):
    inc_wave_setup = IncidentWaveSetup(frequency, cylindrical_wave(amp, tx))
    # Solve the forward problem
    u_scat_re, u_scat_im, ds_receiver, V_CG1 = forward_solve(
                                    h, inc_wave_setup, initial_guess_mesh_util, True, 1)
    
    tol = 1e-7
    mask = (
        np.isclose(df_measurement["tx_x"].to_numpy(), tx[0], atol=tol, rtol=0.0) &
        np.isclose(df_measurement["tx_y"].to_numpy(), tx[1], atol=tol, rtol=0.0)
    )
    df_tx = df_measurement.loc[mask]

    u_ref_re, u_ref_im, _ = assign_ref_value(df_tx, V_CG1, 1)
    
    if tx_i == 0:
        J = assemble(
            (scale_fac * ((u_scat_re - u_ref_re)**2 + (u_scat_im - u_ref_im)**2) * ds_receiver))
    else:
        J += assemble(
            (scale_fac * ((u_scat_re - u_ref_re)**2 + (u_scat_im - u_ref_im)**2) * ds_receiver))

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

result_path = "outputs/result_5per_noise.h5"
goal_geometry_msh_path = "meshes/torso_sim_mesh_solution.msh"

save_optimization_result(
    sol,
    result_file=result_path,
    use_scipy=False
)

plot_mesh_deformation_from_result(
    result_path,
    goal_geometry_msh_path,
    initial_guess_mesh_util,
    plot_file_name="outputs/mesh_deformation_5per_noise.png",
    mesh_overlay_plot_file_name = "outputs/mesh_overlay_5per_noise.png"
)