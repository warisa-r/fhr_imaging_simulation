from dolfin import *
from dolfin_adjoint import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from .initialize_opt import msh2xml_path, initialize_opt_xdmf
from .helmholtz_solve import mesh_deformation, forward_solve

# TODO: Some redundancy here with msh_file_path and obstacle_stiffness
# since it's already saved in MeshUtil


def save_optimization_result(
    sol,
    result_file="result.h5",
    use_scipy=False
):
    if use_scipy == False:
        with HDF5File(MPI.comm_world, result_file, "w") as h5f:
            h5f.write(sol['control'].data, "/h_opt")
            h5f.attributes("/h_opt")["nit"] = sol['iteration']
            h5f.attributes("/h_opt")["objective"] = sol['objective']
            h5f.attributes("/h_opt")["grad_norm"] = sol['grad_norm']
            h_opt_vec = sol['control'].data.vector()
    else:
        with HDF5File(MPI.comm_world, result_file, "w") as h5f:
            h5f.write(sol, "/h_opt")
            h_opt_vec = sol.vector()

    h_min = h_opt_vec.min()
    h_max = h_opt_vec.max()
    h_mean_abs = np.mean(np.abs(h_opt_vec.get_local()))
    print(f"h_opt min: {h_min}, h_opt max: {h_max}, mean|h_opt|: {h_mean_abs}")
    print(f"Optimization result saved to {result_file}")


def calculate_magnitude_and_phase_error(matlab_fullfield_csv_path, h5_file_path,
                                        initial_guess_mesh_util, inc_wave_setup,
                                        use_u_scat=False):

    mesh, markers = initial_guess_mesh_util.get_mesh_and_markers()
    b_mesh = BoundaryMesh(mesh, "exterior")
    S_b = VectorFunctionSpace(b_mesh, "CG", 1)
    h = Function(S_b, name="Design")
    with HDF5File(MPI.comm_world, h5_file_path, "r") as h5f:
        h5f.read(h, "/h_opt")

    u_tot_mag_projected, u_tot_re_projected, u_tot_im_projected, _, V_projection = forward_solve(h,
                                                                                                 inc_wave_setup, initial_guess_mesh_util, use_u_scat)

    mag_vec = u_tot_mag_projected.vector().get_local()
    re_vec = u_tot_re_projected.vector().get_local()
    im_vec = u_tot_im_projected.vector().get_local()

    # Read the data from matlab
    df = pd.read_csv(matlab_fullfield_csv_path)
    points = df[["x", "y"]].values
    mag_values_matlab = df["mag_u"].values
    real_values_matlab = df["real_u"].values
    imag_values_matlab = df["imag_u"].values

    mesh = V_projection.mesh()
    tree = mesh.bounding_box_tree()
    dofmap = V_projection.dofmap()

    # For CG spaces we need coordinates of dofs
    dof_coords = None
    if not V_projection.ufl_element().degree() == 0:
        dof_coords = V_projection.tabulate_dof_coordinates().reshape(
            (-1, mesh.geometry().dim()))

    projected_mag = np.empty(len(points))
    projected_re = np.empty(len(points))
    projected_im = np.empty(len(points))

    assigned_cells = np.zeros(mesh.num_cells(
    ), dtype=bool) if V_projection.ufl_element().degree() == 0 else None
    assigned_dofs = set() if V_projection.ufl_element().degree() != 0 else None

    for i, (x, y) in enumerate(points):
        pt = Point(x, y)
        cell_id = tree.compute_first_entity_collision(pt)

        if cell_id >= mesh.num_cells():
            # point outside mesh: set NaN
            projected_mag[i] = np.nan
            projected_re[i] = np.nan
            projected_im[i] = np.nan
            continue

        if V_projection.ufl_element().degree() == 0:
            # Logic for DG0: one DOF per cell.
            dof_idx = dofmap.cell_dofs(cell_id)[0]
            projected_mag[i] = mag_vec[dof_idx]
            projected_re[i] = re_vec[dof_idx]
            projected_im[i] = im_vec[dof_idx]
        else:
            # Logic for CG > 0 or DG > 0: find the closest DOF within the cell.
            cell_dofs = dofmap.cell_dofs(cell_id)
            cell_dof_coords = dof_coords[cell_dofs]

            # Find the closest DOF in this cell to the point
            distances = np.linalg.norm(
                cell_dof_coords - np.array([x, y]), axis=1)
            closest_local_dof_idx = np.argmin(distances)
            closest_global_dof = cell_dofs[closest_local_dof_idx]

            projected_mag[i] = mag_vec[closest_global_dof]
            projected_re[i] = re_vec[closest_global_dof]
            projected_im[i] = im_vec[closest_global_dof]

    # Now projected_* arrays are in the same order as csv points

    # Compute magnitude error
    mag_error = projected_mag - mag_values_matlab

    # Calculate phase error
    projected_u = projected_re + 1j * projected_im
    matlab_u = real_values_matlab + 1j * imag_values_matlab
    matlab_u_phase = np.angle(matlab_u)
    projected_u_phase = np.angle(projected_u)
    phase_error = np.angle(projected_u * np.conjugate(matlab_u))  # in rad

    results = {
        "points": points,
        "matlab_mag": mag_values_matlab,
        "projected_mag": projected_mag,
        "matlab_phase": matlab_u_phase,
        "projected_phase": projected_u_phase,
        "mag_error": mag_error,
        "phase_error": phase_error,
    }

    return results
