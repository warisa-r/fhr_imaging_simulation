from HH_shape_opt.mesh_generation import obstacle_marker, side_wall_marker, receiver_edge_marker
from HH_shape_opt.mesh_generation import generate_square_with_sin_perturbed_rect_obstacle_mesh, convert_msh_to_xdmf

if __name__ == "__main__":
    print("Generating square with hole mesh...")

    c = 299792458
    freq_max = 5e9  # 5GHz

    # Parameters
    wavelength = c / freq_max  # Physical wavelength
    mesh_size = wavelength / 5

    mesh_file = generate_square_with_sin_perturbed_rect_obstacle_mesh(
        width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=mesh_size,
        output_name="meshes/square_with_sin_perturbed_top_bottom_rect_obstacle",
        n_points_bottom=100, n_points_rect_bottom=40,
        perturb_amplitude=0.01, perturb_frequency=0.5,
        perturb_top=True, top_perturb_amplitude=0.01, top_perturb_frequency=2
    )

    """
    mesh_file =  generate_square_with_rect_obstacle_mesh(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=mesh_size,
    output_name="meshes/square_with_rect_obstacle_opt",
    n_points_bottom=100, n_points_rect_bottom=100,
    use_opt_marker = True
    )
    """

    convert_msh_to_xdmf(mesh_file)
