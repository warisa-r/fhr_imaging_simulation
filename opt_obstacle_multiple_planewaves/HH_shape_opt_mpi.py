import h5py
import json
from dolfin import *
from dolfin_adjoint import * 
import numpy as np
import pandas as pd
import os

from scipy.special import hankel1
import subprocess
import os
import gmsh
import matplotlib.pyplot as plt

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker


# Suppress output on non-root ranks
#import sys
#if rank != 0:
#    sys.stdout = open(os.devnull, 'w')

# Use ONLY this communicator throughout
comm = MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

# DEBUG: Print from ALL ranks to see what's happening
print(f"DEBUG: Process {rank} of {size} is alive")

# Force output to flush
import sys
sys.stdout.flush()

# Small delay to see output from all ranks
import time
time.sleep(0.1 * rank)

if rank == 0:
    print("DEBUG: Rank 0 is working")
    print(f"DEBUG: Total processes: {size}")

comm.Barrier()

if rank == 0:
    print("DEBUG: After first barrier")

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
    print(f"DEBUG: Process {rank} entering load_forward_simulation_data_bottomwall")
    sys.stdout.flush()

    # COLLECTIVE OPERATION: All processes must create the Function.
    u_ref_dg0 = Function(V_DG0)
    
    # On rank 0, load the raw data from CSV
    if rank == 0:
        print("DEBUG: Rank 0 loading CSV data")
        sys.stdout.flush()
        try:
            df = pd.read_csv("forward_sim_data_bottom.csv")
            values = df["u"].values
            print(f"DEBUG: Rank 0 loaded {len(values)} data points")
            sys.stdout.flush()
        except Exception as e:
            print(f"DEBUG: Rank 0 failed to load CSV: {e}")
            sys.stdout.flush()
            values = np.array([]) # Send an empty array on failure
    else:
        # Placeholder for other ranks
        values = None
    
    # Broadcast the RAW data array (the `values` from the CSV) from rank 0 to all other ranks.
    # This is small and safe to broadcast.
    values = comm.bcast(values, root=0)
    
    print(f"DEBUG: Process {rank} received broadcast of {len(values)} data points")
    sys.stdout.flush()
    
    # Now that all processes have the raw data, each process populates its OWN local vector.
    u_vec = u_ref_dg0.vector().get_local() # Gets a zero-filled array of the correct local size for THIS rank
    n_dofs_local = len(u_vec)
    n_points = len(values)
    
    if n_points > 0 and n_dofs_local > 0:
        # Each rank applies the same distribution logic to its local part of the vector.
        for i in range(min(n_dofs_local, n_points)):
            dof_idx = int((i * n_dofs_local) / n_points)
            if dof_idx < n_dofs_local:
                u_vec[dof_idx] = values[i]
        print(f"DEBUG: Process {rank} prepared values in its local u_vec of size {n_dofs_local}")
        sys.stdout.flush()

    # Each process sets its own local vector with its own correctly-sized array.
    u_ref_dg0.vector().set_local(u_vec)
    u_ref_dg0.vector().apply("insert") # Synchronize after setting local values

    print(f"DEBUG: Process {rank} exiting load_forward_simulation_data_bottomwall")
    sys.stdout.flush()
    
    return u_ref_dg0

mesh_xdmf_file = "meshes/square_with_perturbed_rect_obstacle.xdmf"
facet_xdmf_file = "meshes/square_with_perturbed_rect_obstacle_f.xdmf"

# All processes load the mesh and markers in parallel from the XDMF files.
# This is a collective operation.
print(f"Process {rank}: Loading mesh from {mesh_xdmf_file}...")
sys.stdout.flush()

mesh = Mesh()
with XDMFFile(comm, mesh_xdmf_file) as infile:
    infile.read(mesh)

mvc = MeshValueCollection("size_t", mesh, 1)  # '1' for 1D facets in 2D mesh
with XDMFFile(comm, facet_xdmf_file) as infile:
    infile.read(mvc, "name_to_read")

boundary_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# Add a barrier to ensure the mesh is loaded everywhere before proceeding.
print(f"Process {rank}: Finished loading mesh and markers.")
sys.stdout.flush()
comm.Barrier()

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
    
    problem = LinearVariationalProblem(a_el, L_el, s, bc_el)
    solver = LinearVariationalSolver(problem)
    # 'mumps' is a good direct solver. If not available, 'lu' is an alternative.
    solver.parameters["linear_solver"] = "mumps"
    solver.solve()

    return s

def forward_solve(h_control):
    # Copy the "master" mesh and its facet markers
    mesh_copy = Mesh(mesh)
    markers_copy = MeshFunction("size_t", mesh_copy, mesh_copy.topology().dim() - 1)
    markers_copy.set_values(boundary_markers.array())

    # Transfer h → volume and deform the copy
    h_vol = transfer_from_boundary(h_control, mesh_copy)
    s = mesh_deformation(h_vol, mesh_copy, markers_copy)
    ALE.move(mesh_copy, s)

    # Add barrier after mesh deformation
    comm.Barrier()
    
    V = FunctionSpace(mesh_copy, "CG", 5)
    u_inc_re = project(IncidentReal(degree=2), V)
    u_inc_im = project(IncidentImag(degree=2), V)

    ds_bottom = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=bottom_wall_marker)
    ds_sides = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=side_wall_marker)
    ds_obstacle = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=obstacle_marker)

    ds_outer = ds_bottom + ds_sides

    W = FunctionSpace(mesh_copy, MixedElement([V.ufl_element(), V.ufl_element()]))
    (u_re, u_im), (v_re, v_im) = TrialFunctions(W), TestFunctions(W)

    a = (inner(grad(u_re), grad(v_re)) - k_background**2*u_re*v_re)*dx \
        + k_background*u_im*v_re*ds_outer \
        + (inner(grad(u_im), grad(v_im)) - k_background**2*u_im*v_im)*dx \
        - k_background*u_re*v_im*ds_outer

    L = Constant(0.0)*(v_re + v_im)*dx

    uinc_re_neg = Function(V); uinc_re_neg.vector()[:] = -u_inc_re.vector()[:]
    uinc_im_neg = Function(V); uinc_im_neg.vector()[:] = -u_inc_im.vector()[:]

    bcs = [
        DirichletBC(W.sub(0), uinc_re_neg, markers_copy, obstacle_marker),
        DirichletBC(W.sub(1), uinc_im_neg, markers_copy, obstacle_marker),
    ]

    w = Function(W)
    solve(a == L, w, bcs)
    
    # Add barrier after solve
    comm.Barrier()
    
    u_sol_re, u_sol_im = w.split()
    u_tot_re = u_inc_re + u_sol_re
    u_tot_im = u_inc_im + u_sol_im
    u_tot_mag = sqrt(u_tot_re**2 + u_tot_im**2)

    V_DG0 = FunctionSpace(mesh_copy, "DG", 0)
    u_tot_mag_dg0 = project(u_tot_mag, V_DG0)
    
    ds_bottom = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=bottom_wall_marker, 
                metadata={"quadrature_degree": 0})
    return u_tot_mag_dg0, ds_bottom, V_DG0, mesh_copy

######################################

# Initial guess
# Add after the mesh loading debug prints:

print(f"DEBUG: Process {rank} about to start optimization section")
sys.stdout.flush()

# Use consistent communicator
comm = MPI.comm_world
rank = comm.Get_rank()

######################################

# Initial guess
checkpoint_file = "h_checkpoint.h5"
iteration = 0

h_vec = h.vector().get_local()
h_vec[:] = 0.0
h.vector()[:] = h_vec

if rank == 0:
    print("No checkpoint found, starting from zero initial guess")
    sys.stdout.flush()

print(f"DEBUG: Process {rank} about to start forward solve")
sys.stdout.flush()

num_iterations = 1
# Solve the forward problem
u_tot_mag_dg0, ds_bottom, V_DG0, mesh_copy = forward_solve(h)

print(f"DEBUG: Process {rank} finished forward solve")
sys.stdout.flush()

# Load reference data
u_ref_dg0 = load_forward_simulation_data_bottomwall(V_DG0)

print(f"DEBUG: Process {rank} loaded reference data")
sys.stdout.flush()

J = assemble((inner(u_tot_mag_dg0 - u_ref_dg0, u_tot_mag_dg0 - u_ref_dg0)* ds_bottom))
Jhat = ReducedFunctional(J, Control(h))

print(f"DEBUG: Process {rank} about to start optimization")
sys.stdout.flush()

## Start optimizing ##
h_opt = minimize(
    Jhat,
    tol=1e-6,
    method="L-BFGS-B",
    options={"gtol": 1e-7, "maxiter": num_iterations, "disp": True}
)

print(f"DEBUG: Process {rank} finished optimization")
sys.stdout.flush()

# Save the current checkpoint - ONLY RANK 0
iteration += num_iterations

if rank == 0:
    with HDF5File(comm, checkpoint_file, "w") as h5f:
        h5f.write(h_opt, "/h_opt")
        h5f.attributes("/h_opt")["iteration"] = iteration
    print(f"Checkpoint saved to h_checkpoint.h5 (iteration {iteration})")
    sys.stdout.flush()

# Synchronize all ranks after file writing
comm.Barrier()

# Print optimization summary - ONLY RANK 0
if rank == 0:
    print("\n=== Optimization Summary ===")
    print(f"Initial design: all zeros")
    print(f"Optimal design range: [{np.min(h_opt.vector().get_local()):.6e}, {np.max(h_opt.vector().get_local()):.6e}]")
    print(f"Max displacement: {np.max(np.abs(h_opt.vector().get_local())):.6e}")
    sys.stdout.flush()

print(f"DEBUG: Process {rank} finished everything")
sys.stdout.flush()