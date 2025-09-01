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
import warnings

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker, obstacle_opt_marker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HH_shape_opt.helmholtz_solve import HelmholtzSetup, plane_wave, helmholtz_solve, preprocess_reference_data, assign_reference_data
from HH_shape_opt.initialize_opt import initialize_opt_xdmf
from HH_shape_opt.process_result import save_optimization_result, plot_mesh_deformation_from_result

set_log_level(LogLevel.ERROR)

######################################

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

goal_geometry_msh_path = "meshes/square_with_halfsin_perturbed_rect_obstacle.msh"
#msh_file_path = "meshes/square_with_halfsin_perturbed_rect_obstacle.msh"
msh_file_path = "meshes/square_with_rect_obstacle_all.msh"
forward_sim_result_file_path = "forward_sim_data_bottom_sweep_halfsin.csv"
result_path = "outputs/result_halfsin_sq_200.h5"

plot_mesh_deformation_from_result(
    result_path,
    msh_file_path,
    goal_geometry_msh_path,
    obstacle_marker,
    side_wall_marker,
    bottom_wall_marker,
    None,
    "outputs/mesh_deformation_halfsin_sq_200.png",
    50
)