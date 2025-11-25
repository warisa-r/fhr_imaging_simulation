from dolfin import *
from dolfin_adjoint import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from .initialize_opt import msh2xml_path, initialize_opt_xdmf
from .helmholtz_solve import mesh_deformation, forward_solve

# TODO: Some redundancy here with msh_file_path and obstacle_stiffness since it's already saved in MeshUtil


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

def calculate_magnitude_and_phase_error(matlab_fullfield_csv_path, h5_file_path, initial_guess_mesh_util, inc_wave_setup, 
                                        use_u_scat = True, projection_degree = 0):
    df = pd.read_csv(matlab_fullfield_csv_path)

    return calculate_magnitude_and_phase_error_from_dataframe(df, h5_file_path,
                                        initial_guess_mesh_util, inc_wave_setup, 
                                        use_u_scat, projection_degree)


def calculate_magnitude_and_phase_error_from_dataframe(df, h5_file_path,
                                        initial_guess_mesh_util, inc_wave_setup, 
                                        use_u_scat = True, projection_degree = 0):

    mesh, markers = initial_guess_mesh_util.get_mesh_and_markers()
    b_mesh = BoundaryMesh(mesh, "exterior")
    S_b = VectorFunctionSpace(b_mesh, "CG", 1)
    h = Function(S_b, name="Design")
    with HDF5File(MPI.comm_world, h5_file_path, "r") as h5f:
        h5f.read(h, "/h_opt")
    
    u_re_projected, u_im_projected, _, V_projection = forward_solve(
        h, inc_wave_setup, initial_guess_mesh_util, use_u_scat, projection_degree
    )
    u_mag = sqrt(u_re_projected**2 + u_im_projected**2)
    u_mag_projected = project(u_mag, V_projection)

    # Read the data from matlab
    points = df[["x", "y"]].values
    
    real_values_matlab = df["real_u"].values
    imag_values_matlab = df["imag_u"].values

    # Calculate magnitude if not present in dataframe
    if "mag_u" in df.columns:
        mag_values_matlab = df["mag_u"].values
    else:
        mag_values_matlab = np.sqrt(real_values_matlab**2 + imag_values_matlab**2)

    mesh = V_projection.mesh()
    tree = mesh.bounding_box_tree()

    comm = MPI.comm_world
    rank = comm.rank

    # Each rank evaluates only points that fall inside its local partition
    local_idx = []
    local_mag = []
    local_re = []
    local_im = []

    for i, (x, y) in enumerate(points):
        pt = Point(x, y)
        cell_id = tree.compute_first_entity_collision(pt)
        if cell_id >= mesh.num_cells():
            continue  # not on this rank
        try:
            # Evaluate functions at the point (safe because cell is local)
            m = float(u_mag_projected(pt))
            r = float(u_re_projected(pt))
            im = float(u_im_projected(pt))
        except RuntimeError:
            # Fallback: leave for other ranks or mark later as NaN
            continue
        local_idx.append(i)
        local_mag.append(m)
        local_re.append(r)
        local_im.append(im)

    # Gather (index, values) from all ranks
    gathered = comm.gather(
        (np.asarray(local_idx, dtype=np.int64),
         np.asarray(local_mag, dtype=float),
         np.asarray(local_re, dtype=float),
         np.asarray(local_im, dtype=float)),
        root=0
    )

    # Root merges into full arrays
    if rank == 0:
        N = len(points)
        projected_mag = np.full(N, np.nan, dtype=float)
        projected_re  = np.full(N, np.nan, dtype=float)
        projected_im  = np.full(N, np.nan, dtype=float)
        for idxs, mags, res, ims in gathered:
            if idxs.size == 0:
                continue
            projected_mag[idxs] = mags
            projected_re[idxs]  = res
            projected_im[idxs]  = ims
    else:
        projected_mag = projected_re = projected_im = None

    # Broadcast merged arrays so all ranks have consistent results
    projected_mag = comm.bcast(projected_mag, root=0)
    projected_re  = comm.bcast(projected_re,  root=0)
    projected_im  = comm.bcast(projected_im,  root=0)

    # Compute errors (will be NaN where points lie outside the mesh)
    mag_error = projected_mag - mag_values_matlab
    projected_u = projected_re + 1j * projected_im
    matlab_u = real_values_matlab + 1j * imag_values_matlab
    matlab_u_phase = np.angle(matlab_u)
    projected_u_phase = np.angle(projected_u)
    phase_error = np.angle(projected_u * np.conjugate(matlab_u))  # rad

    results = {
        "points": points,
        "matlab_mag": mag_values_matlab,
        "projected_mag": projected_mag,
        "matlab_phase":  matlab_u_phase,
        "projected_phase": projected_u_phase,
        "mag_error": mag_error,
        "phase_error": phase_error,
    }
    return results
