import sys
import os

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HH_shape_opt.helmholtz_solve import HelmholtzSetup, plane_wave, helmholtz_solve, preprocess_reference_data, assign_reference_data
from HH_shape_opt.initialize_opt import initialize_opt_xdmf
from HH_shape_opt.process_result import save_optimization_result, plot_mesh_deformation_from_result


######################################
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

msh_file_path = "meshes/square_with_rect_obstacle.msh"
#msh_file_path = "meshes/square_with_gaussian_perturbed_rect.msh" # To check and see the degree of 
#msh_file_path = "meshes/square_with_perturbed_rect_obstacle.msh"
goal_geometry_msh_path = "meshes/square_with_perturbed_rect_obstacle.msh"
result_path = "result_sym_exp.h5"

plot_mesh_deformation_from_result(
    result_path,
    msh_file_path,
    goal_geometry_msh_path,
    obstacle_marker,
    side_wall_marker,
    bottom_wall_marker,
    "mesh_deformation_test.png"
)