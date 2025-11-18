from HH_shape_opt import *

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
        n_points_bottom=100, n_points_rect_bottom=100,
        perturb_amplitude=0.0, perturb_frequency=0.5,
        perturb_top=False
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_mesh(mesh_file, ax)
    ax.get_legend().remove()
    plt.savefig("simple_obstacle_initial_guess.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()

    """
    mesh_file =  generate_square_with_rect_obstacle_mesh(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=mesh_size,
    output_name="meshes/square_with_rect_obstacle_opt",
    n_points_bottom=100, n_points_rect_bottom=100,
    use_opt_marker = True
    )
    """
    #convert_msh_to_xdmf(mesh_file)
