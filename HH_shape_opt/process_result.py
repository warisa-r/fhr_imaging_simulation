from dolfin import *
from dolfin_adjoint import *
import numpy as np
import matplotlib.pyplot as plt

from .initialize_opt import msh2xml_path, initialize_opt_xdmf
from .helmholtz_solve import mesh_deformation

def save_optimization_result(
    sol,
    msh_file_path,
    result_file = "result.h5"
):
    #TODO: Save obstacle stiffness
    with HDF5File(MPI.comm_world, result_file, "w") as h5f:
        h5f.write(sol['control'].data, "/h_opt")
        h5f.attributes("/h_opt")["nit"] = sol['iteration']
        h5f.attributes("/h_opt")["objective"] = sol['objective']
        h5f.attributes("/h_opt")["grad_norm"] = sol['grad_norm']
        h5f.attributes("/h_opt")["msh_file_path"] = msh_file_path
    print(f"Optimization result saved to {result_file}")

def plot_mesh_deformation_from_result(
    h5_file_path,
    msh_file_path,
    goal_geometry_msh_path,
    obstacle_marker,
    side_wall_marker,
    bottom_wall_marker,
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
        h_temp = Function(S_b, name="Design")
        h5f.read(h_temp, "/h_opt")
        h.vector()[:] = h_temp.vector().get_local()
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
        obstacle_marker, side_wall_marker, bottom_wall_marker, obstacle_stiffness
    )
    ALE.move(mesh_copy, s_final)

    # Load goal geometry mesh
    _, mesh_goal, _ = initialize_opt_xdmf(goal_geometry_msh_path)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plot(mesh, color="b", linewidth=0.5)
    plt.title(subplot_titles[0])
    plt.axis("equal")

    plt.subplot(1, 3, 2)
    plot(mesh_goal, color="r", linewidth=0.5)
    plt.title(subplot_titles[1])
    plt.axis("equal")

    plt.subplot(1, 3, 3)
    plot(mesh_copy, color="r", linewidth=0.5)
    title = subplot_titles[2]
    if num_iterations is not None or final_residual is not None:
        title += f"\n(iters={num_iterations}, residual={final_residual:.2e})"
    plt.title(title)
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig(plot_file_name)
    plt.close()
    print(f"Mesh deformation plot saved to {plot_file_name}")