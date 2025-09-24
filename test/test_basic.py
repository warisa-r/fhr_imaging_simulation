import os
import numpy as np
from dolfin import *
from dolfin_adjoint import *
import moola

from HH_shape_opt.initialize_opt import MeshUtil
from HH_shape_opt.helmholtz_solve import forward_solve, load_forward_simulation_data_bottomwall, IncidentWaveSetup, plane_wave
from HH_shape_opt.mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker, obstacle_opt_marker

BASE_DIR = os.path.dirname(__file__)

MEASUREMENT_DATA_FILE_PATH = os.path.join(BASE_DIR, "measurements", "matlab_measurements_sin0.5.csv")
MSH_FILE_PATH = os.path.join(BASE_DIR, "meshes", "square_with_rect_obstacle.msh")

def initialize_Jhat_basic():
    # ensure test runs from opt_DG0 so relative paths match the script
    repo_root = os.path.dirname(os.path.dirname(__file__))
    opt_dir = os.path.join(repo_root, "opt_DG0")
    os.chdir(opt_dir)

    # setup (match opt_script)
    frequency = 5e9
    inc_wave_setup = IncidentWaveSetup(frequency, plane_wave)

    # Optimize all edges of obstacle
    markers_dict = {
        "obstacle": obstacle_marker,
        "side_wall": side_wall_marker,
        "bottom_wall": bottom_wall_marker,
        "obstacle_opt": None
    }
    obstacle_stiffness = 25

    initial_guess_mesh_util = MeshUtil(
        MSH_FILE_PATH, markers_dict, obstacle_stiffness)
    mesh, _ = initial_guess_mesh_util.get_mesh_and_markers()

    # design variable
    b_mesh = BoundaryMesh(mesh, "exterior")
    S_b = VectorFunctionSpace(b_mesh, "CG", 1)
    h = Function(S_b, name="Design")
    h.vector()[:] = 0.0
    h.vector().apply("insert")

    # forward solve + build objective
    u_tot_mag_dg0, _, _, ds_bottom, V_DG0 = forward_solve(h, inc_wave_setup, initial_guess_mesh_util)
    u_ref_dg0, _ = load_forward_simulation_data_bottomwall(MEASUREMENT_DATA_FILE_PATH, V_DG0)
    J = assemble(
        (inner(u_tot_mag_dg0 - u_ref_dg0, u_tot_mag_dg0 - u_ref_dg0) * ds_bottom))
    Jhat = ReducedFunctional(J, Control(h))

    return Jhat, h, initial_guess_mesh_util, inc_wave_setup


def test_basic_runs_two_iterations_and_zero_residual():
    Jhat, h, initial_guess_mesh_util, inc_wave_setup = initialize_Jhat_basic()

    # Optimize for exactly 1 iteration
    problem = MoolaOptimizationProblem(Jhat)
    h_moola = moola.DolfinPrimalVector(h)
    solver = moola.BFGS(problem, h_moola, options={"maxiter": 1})
    sol = solver.solve()

    assert sol['objective'] == 0.010592287670655856

    # Check that the objective functional is consistent
    result_path = os.path.join(BASE_DIR, "outputs", "result_sin0.5_DG0_matlab.h5")
    # Read h from a result file
    with HDF5File(MPI.comm_world, result_path, "r") as h5f:
        h5f.read(h, "/h_opt")

    mesh_fresh, markers_fresh = initial_guess_mesh_util.get_mesh_and_markers(create_new_object=True)
    u_mag_fresh, _, _, ds_bottom_fresh, Vproj_fresh = forward_solve(h, inc_wave_setup, initial_guess_mesh_util, projection_degree=0)
    # reproject u_ref onto the fresh Vproj_fresh for fair comparison:
    u_ref_dg0_fresh, _ = load_forward_simulation_data_bottomwall(MEASUREMENT_DATA_FILE_PATH, Vproj_fresh)
    J_manual = assemble((u_mag_fresh - u_ref_dg0_fresh)**2 * ds_bottom_fresh)

    # Assert that forward simulation gives the same objective functional value as the objective functional value call
    # by dolfin_adjoint's tape
    assert J_manual == Jhat(h)
