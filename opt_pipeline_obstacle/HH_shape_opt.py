import h5py
import json
from dolfin import *
from dolfin_adjoint import * 
import numpy as np
import pandas as pd

from scipy.special import hankel1
import subprocess
import os
import gmsh
import matplotlib.pyplot as plt

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker

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

def load_forward_simulation_data_bottomwall(V_DG0):
    df = pd.read_csv("forward_sim_data_bottom.csv")
    points = df[["x", "y"]].values
    values = df["u"].values

    # Set up the assignment
    u_ref_dg0 = Function(V_DG0)
    tree = mesh.bounding_box_tree()
    dofmap = V_DG0.dofmap()
    u_vec = u_ref_dg0.vector().get_local()

    # For tracking which cells we've already assigned (to avoid duplicates)
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
            print(f"Warning: No cell found containing point ({x}, {y})")

    # Push the updated values into the Function
    u_ref_dg0.vector().set_local(u_vec)
    u_ref_dg0.vector().apply("insert")

    return u_ref_dg0

# Try to convert the mesh 
print(f"Converting square_with_rect_obstacle to XML format...")
result = subprocess.run([
    "dolfin-convert", 
    f"meshes/square_with_rect_obstacle.msh", 
    f"meshes/square_with_rect_obstacle.xml"
], capture_output=True, text=True)

mesh = Mesh(f"meshes/square_with_rect_obstacle.xml")
boundary_markers = MeshFunction("size_t", mesh, f"meshes/square_with_rect_obstacle_facet_region.xml")

# Create boundary mesh and design variables
b_mesh = BoundaryMesh(mesh, "exterior")
S_b = VectorFunctionSpace(b_mesh, "CG", 1)
h = Function(S_b, name="Design")

zero = Constant([0] * mesh.geometric_dimension())

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="Mesh perturbation field")
h_V = transfer_from_boundary(h, mesh)
h_V.rename("Volume extension of h", "")

def mesh_deformation(h, mesh_local, markers_local):

    V = FunctionSpace(mesh_local, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a  = inner(grad(u), grad(v)) * dx
    L0 = Constant(0.0) * v * dx
    bcs0 = [
        DirichletBC(V, Constant(1.0), markers_local, side_wall_marker),
        DirichletBC(V, Constant(1.0), markers_local, bottom_wall_marker),
        DirichletBC(V, Constant(25), markers_local, obstacle_marker),
    ]
    mu = Function(V, name="mu")
    LinearVariationalSolver(LinearVariationalProblem(a, L0, mu, bcs0)).solve()

    S = VectorFunctionSpace(mesh_local, "CG", 1)
    u_vec, v_vec = TrialFunction(S), TestFunction(S)
    dObs = Measure("ds",
        domain=mesh_local,
        subdomain_data=markers_local,
        subdomain_id=obstacle_marker
    )

    def ε(w):    return sym(grad(w))
    def σ(w):    return 2 * mu * ε(w)

    a_el = inner(σ(u_vec), grad(v_vec)) * dx
    L_el = inner(h, v_vec) * dObs

    bc_el = [ DirichletBC(S, Constant((0.0, 0.0)), markers_local, bottom_wall_marker),
              DirichletBC(S, Constant((0.0, 0.0)), markers_local, side_wall_marker)
     ]
    s = Function(S, name="deformation")
    LinearVariationalSolver(LinearVariationalProblem(a_el, L_el, s, bc_el)).solve()

    return s

def forward_solve(h_control):
    # Copy the “master” mesh and its facet markers
    mesh_copy = Mesh(mesh)
    markers_copy = MeshFunction("size_t", mesh_copy, f"meshes/square_with_rect_obstacle_facet_region.xml")

    # Transfer h → volume and deform the copy since we want to preserve always the original
    h_vol = transfer_from_boundary(h_control, mesh_copy)
    s    = mesh_deformation(h_vol, mesh_copy, markers_copy)
    ALE.move(mesh_copy, s)

    V = FunctionSpace(mesh_copy, "CG", 5)
    u_inc_re = project(IncidentReal(degree=2), V)
    u_inc_im = project(IncidentImag(degree=2), V)

    ds_bottom = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=bottom_wall_marker)
    ds_sides = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=side_wall_marker)
    ds_obstacle = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=obstacle_marker)

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
from dolfin import HDF5File, MPI
checkpoint_file = "h_checkpoint.h5"
iteration = 0

h_vec = h.vector().get_local()
h_vec[:] = 0.0
h.vector()[:] = h_vec
print("No checkpoint found, starting from zero initial guess")

num_iterations = 500
# Solve the forward problem
u_tot_mag_dg0, ds_bottom, V_DG0 = forward_solve(h)

# Load reference data
u_ref_dg0 = load_forward_simulation_data_bottomwall(V_DG0)

J = assemble((inner(u_tot_mag_dg0 - u_ref_dg0, u_tot_mag_dg0 - u_ref_dg0)* ds_bottom))
Jhat = ReducedFunctional(J, Control(h))


## Start optimizing ##
h_opt = minimize(
    Jhat,
    tol=1e-6,
    method="L-BFGS-B",
    options={"gtol": 1e-7, "maxiter": num_iterations, "disp": True}
)

# Save the current checkpoint
iteration += num_iterations

with HDF5File(MPI.comm_world, checkpoint_file, "w") as h5f:
    h5f.write(h_opt, "/h_opt")
    h5f.attributes("/h_opt")["iteration"] = iteration
print(f"Checkpoint saved to h_checkpoint.h5 (iteration {iteration})")

# Print optimization summary
print("\n=== Optimization Summary ===")
print(f"Initial design: all zeros")
print(f"Optimal design range: [{np.min(h_opt.vector().get_local()):.6e}, {np.max(h_opt.vector().get_local()):.6e}]")
print(f"Max displacement: {np.max(np.abs(h_opt.vector().get_local())):.6e}")
