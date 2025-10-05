from HH_shape_opt import *

if __name__ == "__main__":
    print("Generating square with hole mesh...")

    c = 299792458
    freq_max = 5e9  # 5GHz

    # Parameters
    wavelength = c / freq_max  # Physical wavelength
    mesh_size = wavelength / 5

    mesh_file = generate_square_with_meshed_rect_obstacle(
        width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=mesh_size,
        output_name="meshes/square_with_meshed_rect_obstacle",
        n_points_bottom=100, n_points_rect_bottom=40,
        use_opt_marker=False
    )

    # Create a figure containing a single Axes.
    fig, ax = plt.subplots()
    plot_mesh(mesh_file, ax)
    plt.show()

    convert_msh_to_xdmf(mesh_file)
