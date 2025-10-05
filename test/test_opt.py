import os
import numpy as np
from dolfin import *
from dolfin_adjoint import *
import moola

from HH_shape_opt import *

BASE_DIR = os.path.dirname(__file__)


def test_opt_runs_two_iterations_and_zero_residual():
    """
    Test that optimization with only one movable bottom edge of the obstacle works
    """

    # ensure test runs from opt_DG0 so relative paths match the script
    repo_root = os.path.dirname(os.path.dirname(__file__))
    opt_dir = os.path.join(repo_root, "reflection_plane_wave")
    os.chdir(opt_dir)

    # setup (match opt_script)
    frequency = 5e9
    inc_wave_setup = IncidentWaveSetup(frequency, plane_wave)

    measurement_data_file_path = os.path.join(
        BASE_DIR, "measurements", "matlab_measurements_sin1.csv")
    msh_file_path = os.path.join(
        BASE_DIR, "meshes", "square_with_rect_obstacle_opt.msh")
    markers_dict = {
        "obstacle": OBSTACLE_MARKER,
        "side_wall": SIDE_WALL_MARKER,
        "bottom_wall": RECEIVER_EDGE_MARKER,
        "obstacle_opt": OBSTACLE_OPT_MARKER
    }
    obstacle_stiffness = 25

    initial_guess_mesh_util = MeshUtil(
        msh_file_path, markers_dict, obstacle_stiffness)
    mesh, _ = initial_guess_mesh_util.get_mesh_and_markers()

    # design variable
    b_mesh = BoundaryMesh(mesh, "exterior")
    S_b = VectorFunctionSpace(b_mesh, "CG", 1)
    h = Function(S_b, name="Design")
    h.vector()[:] = 0.0
    h.vector().apply("insert")

    # forward solve + build objective
    u_tot_mag_dg0, _, _, ds_receiver, V_DG0 = forward_solve(
        h, inc_wave_setup, initial_guess_mesh_util)
    u_ref_dg0, _ = load_forward_simulation_data_bottomwall(
        measurement_data_file_path, V_DG0)
    J = assemble(
        (inner(u_tot_mag_dg0 - u_ref_dg0, u_tot_mag_dg0 - u_ref_dg0) * ds_receiver))
    Jhat = ReducedFunctional(J, Control(h))

    # optimize for exactly 2 iterations
    problem = MoolaOptimizationProblem(Jhat)
    h_moola = moola.DolfinPrimalVector(h)
    solver = moola.BFGS(problem, h_moola, options={"maxiter": 1})
    sol = solver.solve()

    # For some reasons, running in Github pipeline with multiple processes seem to yield a level of randomness to the
    # objective functional value 
    assert abs(sol['objective'] - 0.1225758306856595) < 1e-9
