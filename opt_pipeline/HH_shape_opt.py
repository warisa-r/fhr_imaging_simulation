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

from mesh_generation import side_wall_marker, bottom_wall_marker, obstacle_marker

k_background = 2* np.pi * 5e9 / 299792458 # 2pi f / c
x0 = np.array([0.0, 0.0])  # source location
incident_wave_amp = 100

# Define Hankel-based incident field (real part)
class HankelReal(UserExpression):
    def eval(self, values, x):
        r = np.linalg.norm(x - x0)
        if r < 1e-12:
            values[0] = 0.0
        else:
            values[0] = np.real(incident_wave_amp * hankel1(0, k_background * r))
    def value_shape(self):
        return ()

# Define Hankel-based incident field (imaginary part)
class HankelImag(UserExpression):
    def eval(self, values, x):
        r = np.linalg.norm(x - x0)
        if r < 1e-12:
            values[0] = 0.0
        else:
            values[0] = np.imag(incident_wave_amp * hankel1(0, k_background * r))
    def value_shape(self):
        return ()

def load_forward_simulation_data():
    
    # Load NumPy data
    data = np.load(f"forward_sim_data.npz")
    
    # Load metadata
    with open(f"forward_sim_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return {
        'coordinates': data['coordinates'],
        'cells': data['cells'],
        'u_total_real': data['u_total_real'],
        'u_total_imag': data['u_total_imag'],
        'u_total_magnitude': data['u_total_magnitude'],
        'u_scattered_real': data['u_scattered_real'],
        'u_scattered_imag': data['u_scattered_imag'],
        'u_incident_real': data['u_incident_real'],
        'u_incident_imag': data['u_incident_imag'],
        'metadata': metadata
    }

# Try to convert the mesh 
print(f"Converting rectangle_mesh to XML format...")
result = subprocess.run([
    "dolfin-convert", 
    f"rectangle_mesh.msh", 
    f"rectangle_mesh.xml"
], capture_output=True, text=True)

mesh = Mesh(f"rectangle_mesh.xml")
boundary_markers = MeshFunction("size_t", mesh, f"rectangle_mesh_facet_region.xml")

# Load reference data BEFORE any mesh deformation
reference_data = load_forward_simulation_data()
reference_u_mag = reference_data['u_total_magnitude']
reference_coords = reference_data['coordinates']

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
    #TODO: If we want it to work with this shape we need to modify the BC -> echo data inverse problem paper as
    # example of modifying the formulation to fit the shape u want to optimize.
    # The drag minimization restrict all the BC very strictly

    V = FunctionSpace(mesh_local, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a  = inner(grad(u), grad(v)) * dx
    L0 = Constant(0.0) * v * dx
    bcs0 = [
        DirichletBC(V, Constant(10.0), markers_local, side_wall_marker),
        DirichletBC(V, Constant(1.0), markers_local, bottom_wall_marker),
        DirichletBC(V, Constant(20.0), markers_local, obstacle_marker),
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

    #TODO: Only set bottom wall to be (0.0, 0.0), side walls should be only s_x = 0
    bc_el = [ DirichletBC(S, Constant((0.0, 0.0)), markers_local, bottom_wall_marker),
              DirichletBC(S.sub(0), Constant(0.0), markers_local, side_wall_marker)
     ]
    s = Function(S, name="deformation")
    LinearVariationalSolver(LinearVariationalProblem(a_el, L_el, s, bc_el)).solve()

    return s

def forward_solve(h_control):
    # 1) Copy the “master” mesh and its facet markers
    mesh_copy = Mesh(mesh)
    markers_copy = MeshFunction("size_t", mesh_copy, f"rectangle_mesh_facet_region.xml")

    # 2) Transfer h → volume and deform the copy since we want to preserve always the original
    h_vol = transfer_from_boundary(h_control, mesh_copy)
    s    = mesh_deformation(h_vol, mesh_copy, markers_copy)
    ALE.move(mesh_copy, s)

    # 3) Now build your forward Helmholtz solve entirely on mesh_copy
    V = FunctionSpace(mesh_copy, "CG", 5)
    u_inc_re = project(HankelReal(degree=2), V)
    u_inc_im = project(HankelImag(degree=2), V)

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
    
    # Create total field
    u_tot_re = Function(V)
    u_tot_im = Function(V)
    u_tot_re.vector()[:] = u_inc_re.vector()[:] + u_sol_re_proj.vector()[:]
    u_tot_im.vector()[:] = u_inc_im.vector()[:] + u_sol_im_proj.vector()[:]
    
    # Calculate magnitude
    u_tot_mag = Function(V)
    u_tot_mag.vector()[:] = np.sqrt(u_tot_re.vector().get_local()**2 + u_tot_im.vector().get_local()**2)
    
    return u_tot_mag, V # return V as target_function_space to use in the interpolating function

def interpolate_reference_data_to_mesh(reference_data, target_function_space):
    from scipy.spatial import cKDTree
    
    # Get reference coordinates and field values
    ref_coords = reference_data['coordinates']
    ref_field = reference_data['u_total_magnitude']
    
    # Get current mesh coordinates
    current_coords = target_function_space.tabulate_dof_coordinates()
    
    # Build KDTree for fast nearest neighbor search
    tree = cKDTree(ref_coords)
    
    # Find nearest neighbors for each current coordinate
    distances, indices = tree.query(current_coords, k=1)
    
    # Create interpolated values by taking nearest neighbor values
    interpolated_values = ref_field[indices]
    
    # Create function and assign interpolated values
    reference_func = Function(target_function_space)
    reference_func.vector()[:] = interpolated_values
    
    return reference_func


######### Mesh deformation test #######

np.random.seed(42)  # For reproducibility
h_random = 2 * (2 * np.random.random(len(h_V.vector()[:])) - 1)  # Random values between -0.02 and 0.02
print(h_random)

h_V.vector()[:] = h_random
mesh_copy = Mesh(mesh)
boundary_markers_copy = MeshFunction("size_t", mesh_copy, f"rectangle_mesh_facet_region.xml")
s = mesh_deformation(h_V, mesh_copy, boundary_markers_copy)
ALE.move(mesh_copy, s)
plot(mesh_copy, color="green", linewidth=1.0)
plt.title("Random deformation")
plt.axis("equal")
#plt.show()

######################################

u_tot_mag_initial, V = forward_solve(h)
reference_u_mag_func = interpolate_reference_data_to_mesh(reference_data, V)

# Create the functional directly
J = assemble((u_tot_mag_initial - reference_u_mag_func)**2 * dx)
print(f"Initial cost: {float(J):.6e}")

# Create ReducedFunctional
Jhat = ReducedFunctional(J, Control(h))

## Start optimizing ##

bound_value = 4.0

# Create bounds as arrays, not Functions
n_dofs = h.vector().size()
lb_array = np.full(n_dofs, -bound_value)
ub_array = np.full(n_dofs, bound_value)

print(f"Setting design variable bounds: [{-bound_value}, {bound_value}]")
print(f"Number of design variables: {n_dofs}")

# Run optimization with bounds as tuples of arrays
h_opt = minimize(Jhat,
                tol=1e-6, 
                options={"gtol": 1e-6, "maxiter": 2, "disp": True})

plt.figure(figsize=(12, 5))

# Plot initial mesh
plt.subplot(1, 2, 1)
Jhat(h)  # Reset to initial design - this evaluates the functional with h=0
plot(mesh, color="b", linewidth=0.5)  # Use original mesh directly
plt.title("Initial Mesh")
plt.axis("equal")

# Apply optimal design and plot
plt.subplot(1, 2, 2)
print(f"Applying optimal design with max displacement: {np.max(np.abs(h_opt.vector().get_local())):.6e}")

# Apply optimal design to the mesh (you must do this yourself!)
mesh_copy = Mesh(mesh)  # Copy again to preserve original
boundary_markers_copy = MeshFunction("size_t", mesh_copy, f"rectangle_mesh_facet_region.xml")

# Transfer optimal boundary control to volume
h_opt_volume = transfer_from_boundary(h_opt, mesh_copy)

# Recompute the mesh deformation
s_final = mesh_deformation(h_opt_volume, mesh_copy, boundary_markers_copy)
ALE.move(mesh_copy, s_final)  # Move the mesh using the final deformation

# Now plot it
plot(mesh_copy, color="r", linewidth=0.5)

# Check if optimization actually changed anything
if np.max(np.abs(h_opt.vector().get_local())) < 1e-10:
    print("Warning: Optimal design shows no significant change!")
    print("This might indicate:")
    print("  - Optimization converged to initial guess")
    print("  - Cost function gradient is zero")
    print("  - Bounds are too restrictive")
else:
    print(f"✓ Optimization found non-trivial design with max displacement: {np.max(np.abs(h_opt.vector().get_local())):.6e}")

plt.tight_layout()
plt.savefig("meshes_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# Print optimization summary
print("\n=== Optimization Summary ===")
print(f"Initial design: all zeros")
print(f"Optimal design range: [{np.min(h_opt.vector().get_local()):.6e}, {np.max(h_opt.vector().get_local()):.6e}]")
print(f"Max displacement: {np.max(np.abs(h_opt.vector().get_local())):.6e}")

# Evaluate costs
initial_cost = float(Jhat(h))
optimal_cost = float(Jhat(h_opt))
print(f"Initial cost: {initial_cost:.6e}")
print(f"Optimal cost: {optimal_cost:.6e}")
print(f"Cost reduction: {initial_cost - optimal_cost:.6e}")
if initial_cost > 0:
    print(f"Relative improvement: {(initial_cost - optimal_cost)/initial_cost*100:.2f}%")