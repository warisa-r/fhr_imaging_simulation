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


def extract_and_overlay_mesh_outlines(original_mesh, goal_mesh, optimized_mesh, plot_file_name="mesh_outlines.png", print_at_x = None):
    # Parallel-safe version: gather boundary coordinates and cells on rank 0,
    # then perform comparisons and plotting there.
    comm = MPI.comm_world

    def gather_boundary_arrays(boundary_mesh):
        coords = np.asarray(boundary_mesh.coordinates())
        cells = np.asarray(boundary_mesh.cells(), dtype=np.int64)
        packed = comm.gather((coords, cells), root=0)
        if comm.rank != 0:
            return None, None
        # On root: concatenate with index offsets
        global_coords_list = []
        global_cells_list = []
        offset = 0
        for coords_part, cells_part in packed:
            if coords_part.size == 0:
                continue
            global_coords_list.append(coords_part)
            global_cells_list.append(cells_part + offset)
            offset += coords_part.shape[0]
        if len(global_coords_list) == 0:
            return np.empty((0, 2)), np.empty((0, 2), dtype=int)
        global_coords = np.vstack(global_coords_list)
        global_cells = np.vstack(global_cells_list)
        return global_coords, global_cells

    # Build boundary meshes locally (works in parallel)
    b_orig = BoundaryMesh(original_mesh, "exterior")
    b_goal = BoundaryMesh(goal_mesh, "exterior")
    b_opt = BoundaryMesh(optimized_mesh, "exterior")

    # Gather arrays to rank 0
    coords_orig, cells_orig = gather_boundary_arrays(b_orig)
    coords_goal, cells_goal = gather_boundary_arrays(b_goal)
    coords_opt, cells_opt = gather_boundary_arrays(b_opt)

    if comm.rank == 0:
        import csv
        
        def save_boundary_to_csv(coords, filename):
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['x', 'y'])
                for coord in coords:
                    writer.writerow([coord[0], coord[1]])
            print(f"Saved boundary points to {filename}")
        
        save_boundary_to_csv(coords_orig, "boundary_original.csv")
        save_boundary_to_csv(coords_goal, "boundary_goal.csv")
        save_boundary_to_csv(coords_opt, "boundary_optimized.csv")

    # Helper: calculate boundary difference using assembled coordinate arrays
    def calculate_boundary_difference_arrays(coords1, coords2, print_at_x=None, tolerance=0.01):
        if coords1.shape[0] == 0 or coords2.shape[0] == 0:
            print("Warning: one of the boundaries is empty.")
            return None

        if coords1.shape[0] == coords2.shape[0]:
            # Sort by x-coordinate for consistent comparison
            sorted_idx1 = np.argsort(coords1[:, 0])
            sorted_idx2 = np.argsort(coords2[:, 0])

            coords1_sorted = coords1[sorted_idx1]
            coords2_sorted = coords2[sorted_idx2]

            if print_at_x is not None:
                mask1 = np.abs(coords1_sorted[:, 0] - print_at_x) < tolerance
                mask2 = np.abs(coords2_sorted[:, 0] - print_at_x) < tolerance

                points1_near = coords1_sorted[mask1]
                points2_near = coords2_sorted[mask2]

                print(f"\n{'='*60}")
                print(f"Points near x = {print_at_x} (tolerance = {tolerance})")
                print(f"{'='*60}")

                print(f"\nGoal boundary points ({len(points1_near)} found):")
                print(f"{'Index':<8} {'x':<12} {'y':<12}")
                print(f"{'-'*32}")
                for i, pt in enumerate(points1_near):
                    print(f"{i:<8} {pt[0]:<12.6f} {pt[1]:<12.6f}")

                print(f"\nOptimized boundary points ({len(points2_near)} found):")
                print(f"{'Index':<8} {'x':<12} {'y':<12}")
                print(f"{'-'*32}")
                for i, pt in enumerate(points2_near):
                    print(f"{i:<8} {pt[0]:<12.6f} {pt[1]:<12.6f}")

                if len(points1_near) > 0 and len(points2_near) > 0:
                    print(f"\nPairwise differences (closest matches):")
                    print(f"{'Goal (x,y)':<28} {'Opt (x,y)':<28} {'Δx':<12} {'Δy':<12} {'|Δ|²':<12}")
                    print(f"{'-'*92}")

                    for pt1 in points1_near:
                        distances = np.linalg.norm(points2_near - pt1, axis=1)
                        closest_idx = np.argmin(distances)
                        pt2 = points2_near[closest_idx]
                        diff = pt1 - pt2

                        print(f"({pt1[0]:.6f}, {pt1[1]:.6f})  ({pt2[0]:.6f}, {pt2[1]:.6f})  "
                              f"{diff[0]:+.6e}  {diff[1]:+.6e}  {np.sum(diff**2):.6e}")

                print(f"{'='*60}\n")

            diff = coords1_sorted - coords2_sorted
            squared_diff = np.sum(diff**2)
            return squared_diff / coords1.shape[0]
        else:
            print(f"Warning: Different number of boundary points ({coords1.shape[0]} vs {coords2.shape[0]})")
            return None

    boundary_diff = calculate_boundary_difference_arrays(coords_goal, coords_opt, print_at_x, tolerance=0.01)
    if boundary_diff is not None:
        print(f"Normalized sum of squared boundary differences (goal vs optimized): {boundary_diff:.6e}")

    # Helper to plot using assembled arrays
    def plot_boundary_from_arrays(ax, coords, cells, color, label):
        if coords.shape[0] == 0 or cells.shape[0] == 0:
            return
        for i, cell in enumerate(cells):
            pts = coords[cell]
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.0, label=label if i == 0 else None)

    # Create figure and plot outlines
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    plot_boundary_from_arrays(ax, coords_orig, cells_orig, "blue", "Original")
    plot_boundary_from_arrays(ax, coords_goal, cells_goal, "red", "Goal")
    plot_boundary_from_arrays(ax, coords_opt, cells_opt, "green", "Optimized")

    ax.set_aspect('equal', 'box')
    title = "Overlay of Mesh Outlines"
    ax.set_title(title)
    ax.legend()

    plt.savefig(plot_file_name, dpi=300)
    plt.close()
    print(f"Overlay mesh outline saved to {plot_file_name}")

def plot_mesh_deformation_from_result(
    h5_file_path,
    goal_geometry_msh_path,
    initial_guess_mesh_util,
    plot_file_name="mesh_deformation.png",
    mesh_overlay_plot_file_name = "outlines.png",
    print_at_x = None,
    subplot_titles=None,
):

    if subplot_titles is None:
        subplot_titles = [
            "Original mesh",
            "Reference/perturbed mesh",
            ""
        ]

    msh_file_path = initial_guess_mesh_util.msh_file_path

    # Create fresh new mesh out of msh_file_path instead of the already modified mesh saved in initial_guess_mesh_util
    mesh, markers = initial_guess_mesh_util.get_mesh_and_markers(True)

    # Extract the number of the marker of each object in the simulation
    obstacle_marker = initial_guess_mesh_util.markers_dict["obstacle"]
    side_wall_marker = initial_guess_mesh_util.markers_dict["side_wall"]
    receiver_edge_marker = initial_guess_mesh_util.markers_dict["bottom_wall"]
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
        obstacle_marker, side_wall_marker, receiver_edge_marker, obstacle_opt_marker, obstacle_stiffness
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

    extract_and_overlay_mesh_outlines(mesh, mesh_goal, mesh_copy, mesh_overlay_plot_file_name, print_at_x)


def plot_projected_errors(results, error_plot_file, 
    use_u_scat = False, show=False, projection_degree=0):

    if use_u_scat:
        u_to_plot = "u_scatter"
    else:
        u_to_plot = "u_total"

    # Plot in rank 0 only
    if MPI.comm_world.rank == 0:
        points = np.asarray(results["points"])
        x = points[:, 0]
        proj_mag = np.asarray(results["projected_mag"])
        matlab_mag = np.asarray(results["matlab_mag"])
        matlab_phase = np.degrees(np.asarray(results["matlab_phase"]))
        proj_phase = np.degrees(np.asarray(results["projected_phase"]))
        mag_err = np.asarray(results["mag_error"])
        phase_err_rad = np.asarray(results["phase_error"])

        phase_err_deg = np.degrees(phase_err_rad)
        phase_err_deg = np.where(phase_err_deg < -90, phase_err_deg + 180,
                        np.where(phase_err_deg > 90, phase_err_deg - 180, phase_err_deg))

        # Sort by x for a clean plot
        order = np.argsort(x)
        x_s = x[order]
        proj_mag_s = proj_mag[order]
        matlab_mag_s = matlab_mag[order]
        proj_phase_s = proj_phase[order]
        matlab_phase_s = matlab_phase[order]
        mag_err_s = mag_err[order]
        phase_err_deg_s = phase_err_deg[order]

        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        ax0, ax1, ax2, ax3 = axes

        ax0.plot(x_s, proj_mag_s, marker="o", markersize=3,
                linestyle="-", color="tab:blue", label="Optimized")
        ax0.plot(x_s, matlab_mag_s, marker="x", markersize=3,
                linestyle="-", color="tab:red", label="Matlab ref")
        ax0.set_ylabel("|u|")
        ax0.set_title(f"Magnitude of " + u_to_plot)
        ax0.legend()

        ax1.plot(x_s, proj_phase_s, marker="o", markersize=3,
                linestyle="-", color="tab:blue", label="Optimized")
        ax1.plot(x_s, matlab_phase_s, marker="x", markersize=3,
                linestyle="-", color="tab:red", label="Matlab ref")
        ax1.set_ylabel("Phase of u")
        ax1.set_title("Phase of " + u_to_plot)
        ax1.legend()

        ax2.plot(x_s, mag_err_s, marker="o", markersize=3,
                 linestyle="-", color="tab:orange")
        ax2.axhline(0.0, color="k", linewidth=0.6, linestyle="--")
        ax2.set_ylabel("Magnitude error (optimized - matlab ref)")

        ax3.plot(x_s, phase_err_deg_s, marker="o",
                 markersize=3, linestyle="-", color="tab:green")
        ax3.axhline(0.0, color="k", linewidth=0.6, linestyle="--")
        ax3.set_ylabel("Phase error in degree")
        ax3.set_xlabel("x")

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
