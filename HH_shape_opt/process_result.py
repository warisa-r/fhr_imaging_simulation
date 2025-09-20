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
    result_file="result.h5",
    use_scipy=True
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
