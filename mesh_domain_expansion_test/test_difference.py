from dolfin import *
import numpy as np
from scipy.special import hankel1
import subprocess
import os
import gmsh
import matplotlib.pyplot as plt

### In this file, we solve instead the scatter field ###

# Set PETSc options for solver
parameters["linear_algebra_backend"] = "PETSc"
PETScOptions.set("ksp_type", "gmres")
PETScOptions.set("pc_type", "lu")
PETScOptions.set("pc_factor_mat_solver_type", "mumps")

# Parameters
k_background = 2* np.pi * 5e9 / 299792458 # 2pi f / c
x0 = np.array([0.0, -1.0])  # source location

# Define Incident-based incident field (real part)
class IncidentReal(UserExpression):
    def eval(self, values, x):
        r = np.linalg.norm(x - x0)
        if r < 1e-12:
            values[0] = 0.0
        else:
            values[0] = np.real(-0.25 * 1j *hankel1(0, k_background * r))
    def value_shape(self):
        return ()

# Define Incident-based incident field (imaginary part)
class IncidentImag(UserExpression):
    def eval(self, values, x):
        r = np.linalg.norm(x - x0)
        if r < 1e-12:
            values[0] = 0.0
        else:
            values[0] = np.imag(-0.25 * 1j *hankel1(0, k_background * r))
    def value_shape(self):
        return ()

def solve_for_mesh(mesh_radius):
    
    # Load mesh
    print(f"Converting mesh_{mesh_radius} to XML format...")
    result = subprocess.run([
        "dolfin-convert", 
        f"mesh_test_{mesh_radius}.msh", 
        f"mesh_test_{mesh_radius}.xml"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error converting mesh_{mesh_radius}: {result.stderr}")
        return None, None, None
    
    mesh = Mesh(f"mesh_test_{mesh_radius}.xml")
    boundary_markers = MeshFunction("size_t", mesh, f"mesh_test_{mesh_radius}_facet_region.xml")

    # Define function space
    V_element = FiniteElement("Lagrange", mesh.ufl_cell(), 5)
    V = FunctionSpace(mesh, V_element)

    # Instantiate expressions
    u_inc_re = project(IncidentReal(degree=2), V)
    u_inc_im = project(IncidentImag(degree=2), V)

    # Define the outward unit normal vector
    n = FacetNormal(mesh)

    # Define boundary measures
    ds_outer = Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=1)
    ds_circle = Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=2)

    # Define mixed function space
    W = FunctionSpace(mesh, V_element * V_element)
    (u_re, u_im) = TrialFunctions(W)
    (v_re, v_im) = TestFunctions(W)

    # Coupled bilinear form
    a = (inner(grad(u_re), grad(v_re)) - k_background**2 * u_re * v_re) * dx + k_background * u_im * v_re * ds_outer + \
        (inner(grad(u_im), grad(v_im)) - k_background**2 * u_im * v_im) * dx - k_background * u_re * v_im * ds_outer

    # Homogeneous RHS
    L = Constant(0.0) * (v_re + v_im) * dx

    # Dirichlet boundary conditions on circle boundary (u = -u_inc)
    u_inc_re_neg = Function(V)
    u_inc_im_neg = Function(V)
    u_inc_re_neg.vector()[:] = -u_inc_re.vector()[:]
    u_inc_im_neg.vector()[:] = -u_inc_im.vector()[:]

    bc_circle_re = DirichletBC(W.sub(0), u_inc_re_neg, boundary_markers, 2)
    bc_circle_im = DirichletBC(W.sub(1), u_inc_im_neg, boundary_markers, 2)
    bcs = [bc_circle_re, bc_circle_im]

    # Solve the coupled system
    w = Function(W)
    solve(a == L, w, bcs)

    # Extract real and imaginary parts
    u_sol_re, u_sol_im = w.split()
    u_sol_re_proj = project(u_sol_re, V)
    u_sol_im_proj = project(u_sol_im, V)

    # Create total field: u_tot = u_inc + u_sol
    u_tot_re = Function(V)
    u_tot_im = Function(V)
    u_tot_re.vector()[:] = u_inc_re.vector()[:] + u_sol_re_proj.vector()[:]
    u_tot_im.vector()[:] = u_inc_im.vector()[:] + u_sol_im_proj.vector()[:]

    # Calculate magnitude of total field
    u_tot_mag = Function(V)
    u_tot_mag.vector()[:] = np.sqrt(u_tot_re.vector().get_local()**2 + u_tot_im.vector().get_local()**2)

    return mesh, u_tot_mag, V

def extract_inner_region_values(mesh, u_tot_mag, V, radius=0.4):
    # Extract u_mag of only within given radius

    # Get coordinates of all DOFs
    dof_coordinates = V.tabulate_dof_coordinates()
    
    # Find DOFs within the specified radius from origin
    inner_indices = []
    inner_values = []
    
    for i, coord in enumerate(dof_coordinates):
        r = np.linalg.norm(coord)
        if r <= radius:
            inner_indices.append(i)
            inner_values.append(u_tot_mag.vector()[i])
    
    return np.array(inner_indices), np.array(inner_values)

# Main loop
mesh_radiuss = [0.4, 0.5, 0.6]
results = {}

print("Solving for different mesh sizes...")
for mesh_radius in mesh_radiuss:
    print(f"\nProcessing mesh size: {mesh_radius}")
    mesh, u_tot_mag, V = solve_for_mesh(mesh_radius)
    
    if mesh is not None:
        # Extract values in inner region (r <= 0.4)
        inner_indices, inner_values = extract_inner_region_values(mesh, u_tot_mag, V, radius=0.4)
        
        results[mesh_radius] = {
            'mesh': mesh,
            'u_tot_mag': u_tot_mag,
            'V': V,
            'inner_indices': inner_indices,
            'inner_values': inner_values
        }
        
        print(f"Mesh {mesh_radius}: {len(inner_indices)} DOFs in inner region (r <= 0.4)")
        print(f"Mean magnitude in inner region: {np.mean(inner_values):.6f}")
        print(f"Max magnitude in inner region: {np.max(inner_values):.6f}")

if len(results) >= 2:

    reference_mesh_radius = 0.6
    if reference_mesh_radius not in results:
        reference_mesh_radius = min(results.keys())
    
    reference_values = results[reference_mesh_radius]['inner_values']
    
    print(f"Using mesh size {reference_mesh_radius} as reference")
    
    for mesh_radius in sorted(results.keys()):
        if mesh_radius != reference_mesh_radius:
            current_values = results[mesh_radius]['inner_values']
            
            # For comparison, we need to interpolate values at same points
            # This is a simplified comparison - for more accurate results,
            # you'd want to interpolate to the same set of points
            
            # Calculate basic statistics for comparison
            ref_mean = np.mean(reference_values)
            curr_mean = np.mean(current_values)
            
            ref_max = np.max(reference_values)
            curr_max = np.max(current_values)
            
            mean_error = abs(curr_mean - ref_mean)
            max_error = abs(curr_max - ref_max)
            
            relative_mean_error = mean_error / ref_mean * 100 if ref_mean != 0 else 0
            relative_max_error = max_error / ref_max * 100 if ref_max != 0 else 0
            
            print(f"\nMesh {mesh_radius} vs Reference {reference_mesh_radius}:")
            print(f"  Mean value error: {mean_error:.6f} ({relative_mean_error:.2f}%)")
            print(f"  Max value error: {max_error:.6f} ({relative_max_error:.2f}%)")

# Plot comparison
plt.figure(figsize=(15, 10))

plot_idx = 1
for mesh_radius in sorted(results.keys()):
    plt.subplot(2, 3, plot_idx)
    p = plot(results[mesh_radius]['u_tot_mag'], 
             title=f"Total field magnitude (mesh {mesh_radius})", 
             cmap="hot")
    plt.colorbar(p)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plot_idx += 1

# Plot mesh comparison
for i, mesh_radius in enumerate(sorted(results.keys())):
    plt.subplot(2, 3, plot_idx)
    plot(results[mesh_radius]['mesh'])
    plt.title(f"Mesh {mesh_radius}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plot_idx += 1

plt.tight_layout()
plt.show()