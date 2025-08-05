from dolfin import HDF5File, MPI
import numpy as np

def save_optimization_result(
    h_opt,
    opt_result,
    msh_file_path,
    checkpoint_file = "result.h5",
):
    with HDF5File(MPI.comm_world, checkpoint_file, "w") as h5f:
        h5f.write(h_opt, "/h_opt") # The final design variable
        h5f.attributes("/h_opt")["num_iterations"] = opt_result.get("nit", None)
        h5f.attributes("/h_opt")["final_residual"] = opt_result.get("fun", None)
        h5f.attributes("/h_opt")["final_gradient"] = np.linalg.norm(opt_result.get("jac", np.zeros_like(h_opt.vector().get_local())))
        h5f.attributes("/h_opt")["termination_message"] = opt_result.get("message", "")
        h5f.attributes("/h_opt")["msh_file_path"] = msh_file_path
    print(f"Optimization result saved to {checkpoint_file}")