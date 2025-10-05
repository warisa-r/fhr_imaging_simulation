from HH_shape_opt import *

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

goal_geometry_msh_path = "meshes/square_with_meshed_halfsin_perturbed_rect_obstacle.msh"

result_path = "outputs/result_sin0.5_refraction_3_DG0_matlab.h5"

plot_mesh_deformation_from_result(
    result_path,
    goal_geometry_msh_path,
    initial_guess_mesh_util,
    plot_file_name="outputs/mesh_deformation_sin0.5_refraction_3_DG0_matlab.png",
    mesh_overlay_plot_file_name="outputs/mesh_overlay_sin0.5_refraction_3_DG0_matlab.png",
    refraction = True
)