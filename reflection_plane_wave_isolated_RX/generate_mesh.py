from HH_shape_opt import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Generating square with hole mesh...")

    c = 299792458
    freq_max = 5e9  # 5GHz

    # Parameters
    wavelength = c / freq_max  # Physical wavelength
    mesh_size = wavelength / 5

    
    mesh_file = generate_square_with_rect_obstacle_and_receiver_segments(
        width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=0.05,
        output_name="meshes/square_with_rect_obstacle_receivers",
        n_points_bottom=100, n_points_rect_bottom=40,
        receiver_segments=[(0.2, 0.35),(0.35, 0.5), (0.7, 0.9)],
        use_opt_marker=False
    )

    # Plot the generated mesh
    fig, ax = plt.subplots(figsize=(10, 8))
    # Hide receiver patches (show only predefined markers)
    plot_mesh(mesh_file, ax, title="Mesh without Receiver Patches", show_receiver_patches=True)
    plt.show()
    plt.close()

    # Convert to XDMF format for FEniCS
    convert_msh_to_xdmf(mesh_file)