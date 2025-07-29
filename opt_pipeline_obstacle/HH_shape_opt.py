import h5py
import json
from dolfin import *
from dolfin_adjoint import * 
import numpy as np
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

def load_forward_simulation_data_bottomwall(func_space):
    import pandas as pd
    try:
        # Load the CSV data with only 'u' values along the bottom wall
        df = pd.read_csv("forward_sim_data_bottom.csv")
        print(f"Loaded bottom wall reference data from CSV:")
        print(f"  Data points: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  u range: [{df.u.min():.6e}, {df.u.max():.6e}]")
        # Return as a NumPy array
        u_ref_bottom =  df['u'].values

        u_ref_func = Function(func_space)
        u_ref_func.vector()[:] = 0.0  # Initialize to zero
        
        bottom_vertex_indices = set()
        for facet in SubsetIterator(boundary_markers, bottom_wall_marker):
            for v in vertices(facet):
                bottom_vertex_indices.add(v.index())

        dof_coords = u_ref_func.function_space().tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
        vertex_coords = mesh.coordinates()
        tol = 1e-10
        bottom_dof_indices = []
        for vi in bottom_vertex_indices:
            v_coord = vertex_coords[vi]
            matches = np.where(np.linalg.norm(dof_coords - v_coord, axis=1) < tol)[0]
            bottom_dof_indices.extend(matches)
        bottom_dof_indices = np.unique(bottom_dof_indices)

        # Assign loaded values to the correct DOFs
        u_ref_func.vector()[bottom_dof_indices] = u_ref_bottom
        return u_ref_func
        
    except FileNotFoundError:
        print("Error: forward_sim_data_bottom.csv not found")
        print("Please run forward_sim_datagen.py first")
        return None
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return None    

# Try to convert the mesh 
print(f"Converting square_with_hole to XML format...")
result = subprocess.run([
    "dolfin-convert", 
    f"square_with_hole.msh", 
    f"square_with_hole.xml"
], capture_output=True, text=True)

mesh = Mesh(f"square_with_hole.xml")
boundary_markers = MeshFunction("size_t", mesh, f"square_with_hole_facet_region.xml")

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
        DirichletBC(V, Constant(50.0), markers_local, obstacle_marker),
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
    markers_copy = MeshFunction("size_t", mesh_copy, f"square_with_hole_facet_region.xml")

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
    u_sol_re_proj = project(u_sol_re, V)
    u_sol_im_proj = project(u_sol_im, V)

    u_sol_mag = Function(V)
    u_sol_mag.vector()[:] = np.sqrt(u_sol_re_proj.vector().get_local()**2 + u_sol_im_proj.vector().get_local()**2)
    
    # Create total field
    u_tot_re = Function(V)
    u_tot_im = Function(V)
    u_tot_re.vector()[:] = u_inc_re.vector()[:] + u_sol_re_proj.vector()[:]
    u_tot_im.vector()[:] = u_inc_im.vector()[:] + u_sol_im_proj.vector()[:]
    
    # Calculate magnitude
    u_tot_mag = Function(V)
    u_tot_mag.vector()[:] = np.sqrt(u_tot_re.vector().get_local()**2 + u_tot_im.vector().get_local()**2)
    
    return u_tot_mag, ds_bottom

######### Mesh deformation test #######
"""
np.random.seed(42)  # For reproducibility
h_random = 2 * (2 * np.random.random(len(h_V.vector()[:])) - 1)  # Random values between -0.02 and 0.02
print(h_random)

h_V.vector()[:] = h_random
mesh_copy = Mesh(mesh)
boundary_markers_copy = MeshFunction("size_t", mesh_copy, f"square_with_hole_facet_region.xml")
s = mesh_deformation(h_V, mesh_copy, boundary_markers_copy)
ALE.move(mesh_copy, s)
plot(mesh_copy, color="green", linewidth=1.0)
plt.title("Random deformation")
plt.axis("equal")
#plt.show()
"""
######################################

# Initial guess
import os
from dolfin import HDF5File, MPI
checkpoint_file = "h_checkpoint.h5"
iteration = 0

h_vec = h.vector().get_local()
h_vec[:] = 1.0
h.vector()[:] = h_vec
print("No checkpoint found, starting from zero initial guess")

num_iterations = 20
u_tot_mag_initial, ds_bottom = forward_solve(h)

u_ref_func = load_forward_simulation_data_bottomwall(u_tot_mag_initial.function_space())

J_data = assemble((u_tot_mag_initial - u_ref_func)**2 * ds_bottom)

# Combine
J = J_data

Jhat = ReducedFunctional(J, Control(h))

## Start optimizing ##
h_opt = minimize(Jhat,
#                bounds=[-7.0, 7.0],
                tol=1e-6, 
                options={"gtol": 1e-7, "maxiter": num_iterations, "disp": True})

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
