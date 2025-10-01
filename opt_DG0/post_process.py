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

frequency = 5e9
inc_wave_setup = IncidentWaveSetup(frequency, plane_wave)

measurement_data_file_path = "measurements/matlab_measurements_sin0.5_amp2.csv"
msh_file_path = "meshes/square_with_rect_obstacle.msh"
markers_dict = {
    "obstacle": OBSTACLE_MARKER,
    "side_wall": SIDE_WALL_MARKER,
    "bottom_wall": RECEIVER_EDGE_MARKER,
    "obstacle_opt": None
}
obstacle_stiffness = 25

initial_guess_mesh_util = MeshUtil(
    msh_file_path, markers_dict, obstacle_stiffness)

result_path = "outputs/result_sin0.5_DG0_matlab.h5"
goal_geometry_msh_path = "meshes/square_with_halfsin_perturbed_rect_obstacle.msh"
matlab_fullfield_csv_path = "measurements/matlab_fullfield_sin0.5.csv"
results = calculate_magnitude_and_phase_error(matlab_fullfield_csv_path, result_path,
                                        initial_guess_mesh_util, inc_wave_setup)

plot_projected_errors(results, "outputs/error_sin0.5_amp2_DG0_matlab.png")
plot_mesh_deformation_from_result(
    result_path,
    goal_geometry_msh_path,
    initial_guess_mesh_util,
    plot_file_name="outputs/mesh_deformation_sin0.5_amp2_DG0_matlab.png",
    mesh_overlay_plot_file_name = "outputs/mesh_overlay_sin0.5_DG0_matlab.png"
)