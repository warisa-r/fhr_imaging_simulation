from dolfin import *
from dolfin_adjoint import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from .initialize_opt import msh2xml_path, initialize_opt_xdmf
from .helmholtz_solve import mesh_deformation

def save_optimization_result(
    sol,
    msh_file_path,
    obstacle_stiffness,
    result_file = "result.h5",
    use_scipy = True
):
    if use_scipy == False:
        with HDF5File(MPI.comm_world, result_file, "w") as h5f:
            h5f.write(sol['control'].data, "/h_opt")
            h5f.attributes("/h_opt")["nit"] = sol['iteration']
            h5f.attributes("/h_opt")["objective"] = sol['objective']
            h5f.attributes("/h_opt")["grad_norm"] = sol['grad_norm']
            h5f.attributes("/h_opt")["msh_file_path"] = msh_file_path
            h5f.attributes("/h_opt")["obstacle_stiffness"] = obstacle_stiffness
            h_opt_vec = sol['control'].data.vector()
    else:
        with HDF5File(MPI.comm_world, result_file, "w") as h5f:
            h5f.write(sol, "/h_opt")
            h5f.attributes("/h_opt")["msh_file_path"] = msh_file_path
            h5f.attributes("/h_opt")["obstacle_stiffness"] = obstacle_stiffness
            h_opt_vec = sol.vector()
    
    h_min = h_opt_vec.min()
    h_max = h_opt_vec.max()
    h_mean_abs = np.mean(np.abs(h_opt_vec.get_local()))
    print(f"h_opt min: {h_min}, h_opt max: {h_max}, mean|h_opt|: {h_mean_abs}")
    print(f"Optimization result saved to {result_file}")


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
        triang = mtri.Triangulation(global_coords[:, 0], global_coords[:, 1], global_cells)
        ax.triplot(triang, color=color, linewidth=linewidth)
        if title:
            ax.set_title(title)
        ax.set_aspect("equal")

def plot_mesh_deformation_from_result(
    h5_file_path,
    msh_file_path,
    goal_geometry_msh_path,
    obstacle_marker,
    side_wall_marker,
    bottom_wall_marker,
    obstacle_opt_marker = None,
    plot_file_name="mesh_deformation.png",
    obstacle_stiffness = 50,
    subplot_titles=None,
):

    if subplot_titles is None:
        subplot_titles = [
            "Original mesh",
            "Reference/perturbed mesh",
            "Mesh resulted from the optimization"
        ]

    _, mesh, markers = initialize_opt_xdmf(msh_file_path)

    # Load h and optimization info from checkpoint
    b_mesh = BoundaryMesh(mesh, "exterior")
    S_b = VectorFunctionSpace(b_mesh, "CG", 1)
    h = Function(S_b, name="Design")
    final_residual = None
    num_iterations = None
    with HDF5File(MPI.comm_world, h5_file_path, "r") as h5f:
        h = Function(S_b, name="Design")
        h5f.read(h, "/h_opt")
        #h_opt_vec = h.vector()
        #h_mean_abs = np.mean(np.abs(h_opt_vec.get_local()))
        #print("h_mean_abs:", h_mean_abs)
        try:
            final_residual = h5f.attributes("/h_opt")["objective"]
        except Exception:
            final_residual = None
        try:
            num_iterations = h5f.attributes("/h_opt")["nit"]
        except Exception:
            num_iterations = None

    # Make a copy of the mesh for deformation to get the optimized mesh
    # Load mesh and markers from XDMF files
    _, mesh_copy, markers_copy = initialize_opt_xdmf(msh_file_path)

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
    gather_and_plot_mesh(mesh, ax1, color="b", linewidth=0.5, title=subplot_titles[0])

    ax2 = plt.subplot(1, 3, 2)
    gather_and_plot_mesh(mesh_goal, ax2, color="r", linewidth=0.5, title=subplot_titles[1])

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

def plot_comparison(dolfin_csv_path, matlab_csv_path, output_image_path):
    # Load data
    df_dolfin = pd.read_csv(dolfin_csv_path)
    df_matlab = pd.read_csv(matlab_csv_path)

    # Check data alignment. Round x values to avoid floating point precision issues
    df_dolfin['x_rounded'] = df_dolfin['x'].round(10)
    df_dolfin = df_dolfin.sort_values(by='x_rounded')
    df_matlab['x_rounded'] = df_matlab['x'].round(10)

    # Merge on rounded x values
    df_merged = pd.merge(df_dolfin, df_matlab, on="x_rounded", suffixes=('_dolfin', '_matlab'))

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

    plt.plot(df_dolfin["x"].to_numpy(), df_dolfin["u"].to_numpy(), label="Dolfin", marker='o', markersize=2, linestyle='-', alpha=0.7)
    plt.plot(df_matlab["x"].to_numpy(), df_matlab["u"].to_numpy(), label="Matlab", marker='x', markersize=2, linestyle='--', alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Comparison of Dolfin and Matlab results")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_image_path}")
    plt.show()