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
from HH_shape_opt.process_result import save_optimization_result, plot_mesh_deformation_from_result
from HH_shape_opt.helmholtz_solve import mesh_deformation

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker, obstacle_opt_marker

set_log_level(LogLevel.ERROR)

k_background = 2* np.pi * 5e9 / 299792458 # 2pi f / c
incident_wave_amp = 1

# Define Incident-based incident field (real part)
class IncidentReal(UserExpression):
    def eval(self, values, x):
        values[0] = np.real(incident_wave_amp * np.exp(1j * k_background * x[1]))
    def value_shape(self):
        return ()

# Define Incident-based incident field (imaginary part)
class IncidentImag(UserExpression):
    def eval(self, values, x):
        values[0] = np.imag(incident_wave_amp * np.exp(1j * k_background * x[1]))
    def value_shape(self):
        return ()

def load_forward_simulation_data_bottomwall(measurement_data_file_path, V_ref, projection_degree=0):
    df = pd.read_csv(measurement_data_file_path)
    points = df[["x", "y"]].values
    values = df["u"].values

    # Set up the assignment
    u_ref = Function(V_ref)
    mesh = V_ref.mesh()
    tree = mesh.bounding_box_tree()
    dofmap = V_ref.dofmap()
    u_vec = u_ref.vector().get_local()

    if projection_degree == 0:
        # Logic for DG0: one DOF per cell.
        assigned = np.zeros(mesh.num_cells(), dtype=bool)
        for (x, y), val in zip(points, values):
            point = Point(x, y)
            cell_id = tree.compute_first_entity_collision(point)
            if cell_id < mesh.num_cells() and not assigned[cell_id]:
                dof_idx = dofmap.cell_dofs(cell_id)[0]
                u_vec[dof_idx] = val
                assigned[cell_id] = True
            elif cell_id < mesh.num_cells() and assigned[cell_id]:
                print(f"Warning: cell {cell_id} already assigned, skipping duplicate point.")
    else:
        # Logic for CG > 0 or DG > 0: find the closest DOF within the cell.
        dof_coords = V_ref.tabulate_dof_coordinates()
        assigned_dofs = set()
        for (x, y), val in zip(points, values):
            point = Point(x, y)
            cell_id = tree.compute_first_entity_collision(point)
            if cell_id < mesh.num_cells():
                cell_dofs = dofmap.cell_dofs(cell_id)
                cell_dof_coords = dof_coords[cell_dofs]
                
                # Find the closest DOF in this cell to the point
                distances = np.linalg.norm(cell_dof_coords - np.array([x, y]), axis=1)
                closest_local_dof_idx = np.argmin(distances)
                closest_global_dof = cell_dofs[closest_local_dof_idx]

                if closest_global_dof not in assigned_dofs:
                    u_vec[closest_global_dof] = val
                    assigned_dofs.add(closest_global_dof)
                else:
                    # This can happen if a DOF is shared by multiple cells that contain points
                    pass

    # Push the updated values into the Function
    u_ref.vector().set_local(u_vec)
    u_ref.vector().apply("insert")

    return u_ref

measurement_data_file_path = "forward_sim_data_bottom.csv"
mesh = Mesh()
# meshes/square_with_sin_perturbed_rect_obstacle.xdmf
with XDMFFile("meshes/square_with_rect_obstacle.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("meshes/square_with_rect_obstacle_facets.xdmf") as infile:
    infile.read(mvc, "name_to_read")
    boundary_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)

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

def forward_solve(h_control, obstacle_opt_marker = None):
    # Copy the “master” mesh and its facet markers
    mesh_copy = Mesh(mesh)
    mvc_copy = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile("meshes/square_with_rect_obstacle_facets.xdmf") as infile:
        infile.read(mvc_copy, "name_to_read")
        markers_copy = cpp.mesh.MeshFunctionSizet(mesh_copy, mvc_copy)

    # Transfer h → volume and deform the copy since we want to preserve always the original
    h_vol = transfer_from_boundary(h_control, mesh_copy)
    s = mesh_deformation(h_vol, mesh_copy, markers_copy, obstacle_marker, side_wall_marker, bottom_wall_marker, obstacle_opt_marker, 25)
    ALE.move(mesh_copy, s)

    V = FunctionSpace(mesh_copy, "CG", 5)
    u_inc_re = project(IncidentReal(degree=2), V)
    u_inc_im = project(IncidentImag(degree=2), V)

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

    # Magnitude as UFL expression -> autodiffbar hopefully 
    u_tot_mag = sqrt(u_tot_re**2 + u_tot_im**2)

    V_DG0 = FunctionSpace(mesh_copy, "DG", 0)
    u_tot_mag_dg0 = project(u_tot_mag, V_DG0)
    
    ds_bottom = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=bottom_wall_marker, 
                metadata={"quadrature_degree": 0})
    return u_tot_mag_dg0, ds_bottom, V_DG0

######################################

# Initial guess
import os
# Solve the forward problem
u_tot_mag_dg0, ds_bottom, V_DG0 = forward_solve(h)

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

msh_file_path = "meshes/square_with_rect_obstacle.msh"
result_path = "outputs/result_sin_1.0_DG0_restricted_matlab.h5"
goal_geometry_msh_path = "meshes/square_with_sin_perturbed_rect_obstacle.msh"

save_optimization_result(
    sol,
    msh_file_path,
    25,
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
    obstacle_stiffness = 25,
)

# Print optimization summary
print("\n=== Optimization Summary ===")
print(f"Initial design: all zeros")
print(f"Optimal design range: [{np.min(h_opt.vector().get_local()):.6e}, {np.max(h_opt.vector().get_local()):.6e}]")
print(f"Max displacement: {np.max(np.abs(h_opt.vector().get_local())):.6e}")
