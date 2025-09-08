import h5py
from dolfin import *
from dolfin_adjoint import * 
import numpy as np
import moola
import pandas as pd

import subprocess
import os
import gmsh
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HH_shape_opt.initialize_opt import MeshUtil
from HH_shape_opt.process_result import save_optimization_result, plot_mesh_deformation_from_result
from HH_shape_opt.helmholtz_solve import mesh_deformation, load_forward_simulation_data_bottomwall, HelmholtzSetup, plane_wave

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker, obstacle_opt_marker

set_log_level(LogLevel.ERROR)

frequency = 5e9
sim_setup = HelmholtzSetup(frequency, plane_wave)

measurement_data_file_path = "matlab_measurements.csv"
msh_file_path = "meshes/square_with_rect_obstacle_opt.msh"
initial_guess_mesh_util = MeshUtil(msh_file_path)
mesh, _ = initial_guess_mesh_util.get_mesh_and_markers()

# Create boundary mesh and design variables
b_mesh = BoundaryMesh(mesh, "exterior")
S_b = VectorFunctionSpace(b_mesh, "CG", 1)
h = Function(S_b, name="Design")
h.vector()[:] = 0.0
h.vector().apply("insert")

zero = Constant([0] * mesh.geometric_dimension())

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="Mesh perturbation field")
h_V = transfer_from_boundary(h, mesh)
h_V.rename("Volume extension of h", "")

V_DG0 = FunctionSpace(mesh, "DG", 0)
u_ref_dg0 = load_forward_simulation_data_bottomwall(measurement_data_file_path, V_DG0)

#TODO: Try to use HHSetup with this forward solve
#TODO: Move forward solve to modularize
def forward_solve(h_control, obstacle_opt_marker = None):
    # Copy the “master” mesh and its facet markers
    mesh_copy, markers_copy = initial_guess_mesh_util.get_mesh_and_markers()

    # Transfer h → volume and deform the copy since we want to preserve always the original
    h_vol = transfer_from_boundary(h_control, mesh_copy)
    s = mesh_deformation(h_vol, mesh_copy, markers_copy, obstacle_marker, side_wall_marker, bottom_wall_marker, obstacle_opt_marker, sim_setup.obstacle_stiffness)
    ALE.move(mesh_copy, s)

    V = FunctionSpace(mesh_copy, "CG", 5)
    u_inc_re = project(sim_setup.u_inc_re, V)
    u_inc_im = project(sim_setup.u_inc_im, V)

    ds_bottom = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=bottom_wall_marker)
    ds_sides = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=side_wall_marker)
    ds_obstacle = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=obstacle_marker)

    if obstacle_opt_marker != None:
        # Since obstacle_marker excludes the to-be-optimized outline of the obstacle
        # we need to add the to-be-optimized outline to ds_obstacle
        ds_obstacle = ds_obstacle + Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=obstacle_opt_marker)

    ds_outer = ds_bottom + ds_sides

    W = FunctionSpace(mesh_copy, MixedElement([V.ufl_element(),
                                               V.ufl_element()]))
    (u_re, u_im), (v_re, v_im) = TrialFunctions(W), TestFunctions(W)

    k_background = sim_setup.k_background

    a = (inner(grad(u_re), grad(v_re)) - k_background**2*u_re*v_re)*dx \
        + k_background*u_im*v_re*ds_outer \
        + (inner(grad(u_im), grad(v_im)) - k_background**2*u_im*v_im)*dx \
        - k_background*u_re*v_im*ds_outer

    L = Constant(0.0)*(v_re + v_im)*dx

    # Dirichlet BCs on the obstacle u_s = - u_in on the reflective surface
    uinc_re_neg = Function(V); uinc_re_neg.vector()[:] = -u_inc_re.vector()[:]
    uinc_im_neg = Function(V); uinc_im_neg.vector()[:] = -u_inc_im.vector()[:]

    bcs = [
      DirichletBC(W.sub(0), uinc_re_neg, markers_copy, obstacle_marker),
      DirichletBC(W.sub(1), uinc_im_neg, markers_copy, obstacle_marker),
      
    ]

    if obstacle_opt_marker != None:
        # Since obstacle_marker excludes the to-be-optimized outline of the obstacle
        # we need to add the to-be-optimized outline to ds_obstacle
        bcs.append(DirichletBC(W.sub(0), uinc_re_neg, markers_copy, obstacle_opt_marker))
        bcs.append(DirichletBC(W.sub(1), uinc_im_neg, markers_copy, obstacle_opt_marker))


    w = Function(W)
    solve(a == L, w, bcs)
    
    # Extract solutions
    u_sol_re, u_sol_im = w.split()

    # Total field expressions
    u_tot_re = u_inc_re + u_sol_re
    u_tot_im = u_inc_im + u_sol_im

    u_tot_mag = sqrt(u_tot_re**2 + u_tot_im**2)

    V_DG0 = FunctionSpace(mesh_copy, "DG", 0)
    u_tot_mag_dg0 = project(u_tot_mag, V_DG0)
    
    ds_bottom = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=bottom_wall_marker, 
                metadata={"quadrature_degree": 0})
    return u_tot_mag_dg0, ds_bottom, V_DG0

# Solve the forward problem
u_tot_mag_dg0, ds_bottom, V_DG0 = forward_solve(h, obstacle_opt_marker)

J = assemble((inner(u_tot_mag_dg0 - u_ref_dg0, u_tot_mag_dg0 - u_ref_dg0)* ds_bottom))
Jhat = ReducedFunctional(J, Control(h))

## Start optimizing ##
problem = MoolaOptimizationProblem(Jhat)
h_moola = moola.DolfinPrimalVector(h)

solver = moola.BFGS(problem, h_moola,
    options={
        "maxiter": 50,
        "gtol": 1e-7,
    })

sol = solver.solve()
h_opt = sol['control'].data

result_path = "outputs/result_sin_1.0_DG0_restricted_matlab.h5"
goal_geometry_msh_path = "meshes/square_with_sin_perturbed_rect_obstacle.msh"

save_optimization_result(
    sol,
    msh_file_path,
    sim_setup.obstacle_stiffness,
    result_file = result_path,
    use_scipy = False
)

plot_mesh_deformation_from_result(
    result_path,
    msh_file_path,
    goal_geometry_msh_path,
    obstacle_marker,
    side_wall_marker,
    bottom_wall_marker,
    obstacle_opt_marker,
    plot_file_name="outputs/mesh_deformation_sin_1.0_DG0_restricted_matlab.png",
    obstacle_stiffness = sim_setup.obstacle_stiffness,
)

# Print optimization summary
print("\n=== Optimization Summary ===")
print(f"Initial design: all zeros")
print(f"Optimal design range: [{np.min(h_opt.vector().get_local()):.6e}, {np.max(h_opt.vector().get_local()):.6e}]")
print(f"Max displacement: {np.max(np.abs(h_opt.vector().get_local())):.6e}")
