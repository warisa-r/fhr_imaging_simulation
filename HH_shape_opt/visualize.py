import os

from dolfin import *
from dolfin_adjoint import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from .initialize_opt import msh2xml_path, initialize_opt_xdmf
from .helmholtz_solve import mesh_deformation
from .process_result import calculate_magnitude_and_phase_error


def gather_and_plot_mesh(mesh, ax, color="k", linewidth=0.3, title=None):
    comm = MPI.comm_world

    coords = mesh.coordinates()
    cells = mesh.cells()

    # Gather coordinates and cells
    all_coords = comm.gather(coords, root=0)
    all_cells = comm.gather(cells, root=0)

    if comm.rank == 0:
        # Offset each partition's cell indices so they refer to the global coords array
        global_coords = []
        global_cells = []
        offset = 0
        for coords_part, cells_part in zip(all_coords, all_cells):
            global_coords.append(coords_part)
            global_cells.append(cells_part + offset)
            offset += coords_part.shape[0]

        global_coords = np.vstack(global_coords)
        global_cells = np.vstack(global_cells)

        # Build triangulation
        triang = mtri.Triangulation(
            global_coords[:, 0], global_coords[:, 1], global_cells)
        ax.triplot(triang, color=color, linewidth=linewidth)
        if title:
            ax.set_title(title)
        ax.set_aspect("equal")


def extract_and_overlay_mesh_outlines(original_mesh, goal_mesh, optimized_mesh, plot_file_name="mesh_outlines.png"):
#TODO: Make this compatible with parallel run
    # Extract boundary meshes
    boundary_original = BoundaryMesh(original_mesh, "exterior")
    boundary_goal = BoundaryMesh(goal_mesh, "exterior")
    boundary_optimized = BoundaryMesh(optimized_mesh, "exterior")

    # Helper function to plot a boundary mesh
    def plot_boundary(ax, boundary_mesh, color, label):
        coords = boundary_mesh.coordinates()
        cells = boundary_mesh.cells()

        for cell in cells:
            pts = coords[cell]
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.0, label=label)
            label = None

    # Create the figure
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Plot outlines
    plot_boundary(ax, boundary_original, "blue", "Original")
    plot_boundary(ax, boundary_goal, "red", "Goal")
    plot_boundary(ax, boundary_optimized, "green", "Optimized")

    # Add legend and styling
    ax.set_aspect('equal', 'box')
    ax.set_title("Overlay of Mesh Outlines")
    ax.legend()

    # Save or show the figure
    if MPI.comm_world.rank == 0:
        plt.savefig(plot_file_name, dpi=300)
        plt.close()
        print(f"Overlay mesh outline saved to {plot_file_name}")


def plot_mesh_deformation_from_result(
    h5_file_path,
    goal_geometry_msh_path,
    initial_guess_mesh_util,
    plot_file_name="mesh_deformation.png",
    mesh_overlay_plot_file_name = "outlines.png",
    subplot_titles=None,
):

    if subplot_titles is None:
        subplot_titles = [
            "Original mesh",
            "Reference/perturbed mesh of amplitude 2cm",
            ""
        ]

    msh_file_path = initial_guess_mesh_util.msh_file_path

    # Create fresh new mesh out of msh_file_path instead of the already modified mesh saved in initial_guess_mesh_util
    mesh, markers = initial_guess_mesh_util.get_mesh_and_markers(True)

    # Extract the number of the marker of each object in the simulation
    obstacle_marker = initial_guess_mesh_util.markers_dict["obstacle"]
    side_wall_marker = initial_guess_mesh_util.markers_dict["side_wall"]
    bottom_wall_marker = initial_guess_mesh_util.markers_dict["bottom_wall"]
    obstacle_opt_marker = initial_guess_mesh_util.markers_dict["obstacle_opt"]

    obstacle_stiffness = initial_guess_mesh_util.obstacle_stiffness

    # Load h and optimization info from checkpoint
    b_mesh = BoundaryMesh(mesh, "exterior")
    S_b = VectorFunctionSpace(b_mesh, "CG", 1)
    h = Function(S_b, name="Design")
    final_residual = None
    num_iterations = None
    with HDF5File(MPI.comm_world, h5_file_path, "r") as h5f:
        h = Function(S_b, name="Design")
        h5f.read(h, "/h_opt")
        # h_opt_vec = h.vector()
        # h_mean_abs = np.mean(np.abs(h_opt_vec.get_local()))
        # print("h_mean_abs:", h_mean_abs)
        try:
            final_residual = h5f.attributes("/h_opt")["objective"]
        except Exception:
            final_residual = None
        try:
            num_iterations = h5f.attributes("/h_opt")["nit"]
        except Exception:
            num_iterations = None

    # Create fresh new mesh out of msh_file_path instead of the already modified mesh saved in initial_guess_mesh_util
    mesh_copy, markers_copy = initial_guess_mesh_util.get_mesh_and_markers(
        True)

    h_vol = transfer_from_boundary(h, mesh_copy)

    # Deform the mesh using the imported mesh_deformation
    s_final = mesh_deformation(
        h_vol, mesh_copy, markers_copy,
        obstacle_marker, side_wall_marker, bottom_wall_marker, obstacle_opt_marker, obstacle_stiffness
    )
    ALE.move(mesh_copy, s_final)

    # Load goal geometry mesh
    _, mesh_goal, markers_goal = initialize_opt_xdmf(goal_geometry_msh_path)

    plt.figure(figsize=(18, 6))

    ax1 = plt.subplot(1, 3, 1)
    gather_and_plot_mesh(mesh, ax1, color="b",
                         linewidth=0.5, title=subplot_titles[0])

    ax2 = plt.subplot(1, 3, 2)
    gather_and_plot_mesh(mesh_goal, ax2, color="r",
                         linewidth=0.5, title=subplot_titles[1])

    ax3 = plt.subplot(1, 3, 3)
    title = subplot_titles[2]
    if num_iterations is not None or final_residual is not None:
        title += f"\n(iters={num_iterations}, residual={final_residual:.2e})"
    gather_and_plot_mesh(mesh_copy, ax3, color="r", linewidth=0.5, title=title)

    plt.tight_layout()

    if MPI.comm_world.rank == 0:
        plt.savefig(plot_file_name)
        plt.close()
        print(f"Mesh deformation plot saved to {plot_file_name}")

    extract_and_overlay_mesh_outlines(mesh, mesh_goal, mesh_copy, mesh_overlay_plot_file_name)


def plot_projected_errors(results, error_plot_file, show=False, projection_degree=0):

    # Plot in rank 0 only
    if MPI.comm_world.rank == 0:
        points = np.asarray(results["points"])
        x = points[:, 0]
        proj_mag = np.asarray(results["projected_mag"])
        matlab_mag = np.asarray(results["matlab_mag"])
        mag_err = np.asarray(results["mag_error"])
        phase_err_rad = np.asarray(results["phase_error"])
        phase_err_deg = np.degrees(phase_err_rad)

        # Sort by x for a clean plot
        order = np.argsort(x)
        x_s = x[order]
        proj_mag_s = proj_mag[order]
        matlab_mag_s = matlab_mag[order]
        mag_err_s = mag_err[order]
        phase_err_deg_s = phase_err_deg[order]

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        ax0, ax1, ax2 = axes

        ax0.plot(x_s, proj_mag_s, marker="o", markersize=3,
                linestyle="-", color="tab:blue", label="Optimized")
        ax0.plot(x_s, matlab_mag_s, marker="x", markersize=3,
                linestyle="-", color="tab:red", label="Matlab ref")
        ax0.set_ylabel("|u|")
        ax0.set_title("Magnitude of u_total")
        ax0.legend()

        ax1.plot(x_s, mag_err_s, marker="o", markersize=3,
                 linestyle="-", color="tab:orange")
        ax1.axhline(0.0, color="k", linewidth=0.6, linestyle="--")
        ax1.set_ylabel("Magnitude error (optimized - matlab ref)")

        ax2.plot(x_s, phase_err_deg_s, marker="o",
                 markersize=3, linestyle="-", color="tab:green")
        ax2.axhline(0.0, color="k", linewidth=0.6, linestyle="--")
        ax2.set_ylabel("Phase error in degree")
        ax2.set_xlabel("x")

        for ax in axes:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plt.savefig(error_plot_file, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        # TODO: Make this a test
        print(f"Sum of magnitude error square: {np.sum(mag_err ** 2)}") # If divided by 1/num points-1 i think u can recover the residual
        print(f"Projected error plots saved to {error_plot_file}")


def plot_comparison(dolfin_csv_path, matlab_csv_path, output_image_path):
    # Load data
    df_dolfin = pd.read_csv(dolfin_csv_path)
    df_matlab = pd.read_csv(matlab_csv_path)

    # Check data alignment. Round x values to avoid floating point precision issues
    df_dolfin['x_rounded'] = df_dolfin['x'].round(10)
    df_dolfin = df_dolfin.sort_values(by='x_rounded')
    df_matlab['x_rounded'] = df_matlab['x'].round(10)

    # Merge on rounded x values
    df_merged = pd.merge(df_dolfin, df_matlab, on="x_rounded",
                         suffixes=('_dolfin', '_matlab'))

    print(f"Dolfin data points: {len(df_dolfin)}")
    print(f"Matlab data points: {len(df_matlab)}")
    print(f"Merged data points: {len(df_merged)}")

    # Calculate statistics about the differences

    if len(df_merged) > 0:
        differences = df_merged["u_dolfin"] - df_merged["u_matlab"]
        print(f"\n--- Difference Statistics ---")
        print(f"Mean difference: {differences.mean():.6f}")
        print(f"Max difference: {differences.max():.6f}")
        print(f"Min difference: {differences.min():.6f}")
        print(f"Std difference: {differences.std():.6f}")
        print(f"---------------------------\n")

    # Plot u vs x for both datasets
    plt.figure(figsize=(10, 8))

    plt.plot(df_dolfin["x"].to_numpy(), df_dolfin["u"].to_numpy(
    ), label="Dolfin", marker='o', markersize=2, linestyle='-', alpha=0.7)
    plt.plot(df_matlab["x"].to_numpy(), df_matlab["u"].to_numpy(
    ), label="Matlab", marker='x', markersize=2, linestyle='--', alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Comparison of Dolfin and Matlab results")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_image_path}")
    plt.show()
