import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import moola
from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker
from dolfin import *
from dolfin_adjoint import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HH_shape_opt.process_result import save_optimization_result, plot_mesh_deformation_from_result
set_log_level(LogLevel.ERROR)

# Next, we load the facet marker values used in the mesh, as well as some
# geometrical quantities mesh-generator file.

def mesh_deformation(h):
    # Compute variable μ
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)

    a = inner(grad(u), grad(v)) * dx
    L = Constant(0) * v * dx

    mu_min = Constant(1, name="mu_min")
    mu_max = Constant(400, name="mu_max")
    bcs = []
    for marker in [side_wall_marker, bottom_wall_marker]:
        bcs.append(DirichletBC(V, mu_min, mf, marker))
    bcs.append(DirichletBC(V, mu_max, mf, obstacle_marker))

    mu = Function(V, name="mesh deformation mu")
    
    # Use LinearVariationalProblem instead of solve(a == L, ...)
    problem = LinearVariationalProblem(a, L, mu, bcs)
    solver = LinearVariationalSolver(problem)
    solver.solve()

    # Compute the mesh deformation
    S = VectorFunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(S), TestFunction(S)
    dObstacle = Measure("ds", subdomain_data=mf, subdomain_id=obstacle_marker)

    def epsilon(u):
        return sym(grad(u))

    def sigma(u, mu=400, lmb=0):
        return 2 * mu * epsilon(u) + lmb * tr(epsilon(u)) * Identity(2)

    a = inner(sigma(u, mu=mu), grad(v)) * dx
    L = inner(h, v) * dObstacle

    bcs = []
    for marker in [side_wall_marker, bottom_wall_marker]:
        bcs.append(DirichletBC(S, zero, mf, marker))

    s = Function(S, name="mesh deformation")
    
    # Use LinearVariationalProblem instead of solve(a == L, ...)
    problem = LinearVariationalProblem(a, L, s, bcs)
    solver = LinearVariationalSolver(problem)
    solver.solve()
    
    return s


mesh = Mesh()
# meshes/square_with_sin_perturbed_rect_obstacle.xdmf
with XDMFFile("meshes/square_with_kite_obstacle.xdmf") as infile:
#with XDMFFile("meshes/square_with_rect_obstacle_all.xdmf") as infile:
    infile.read(mesh)
# meshes/square_with_sin_perturbed_rect_obstacle_facets.xdmf
mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("meshes/square_with_kite_obstacle_facets.xdmf") as infile:
#with XDMFFile("meshes/square_with_rect_obstacle_all_facets.xdmf") as infile:
    infile.read(mvc, "name_to_read")
    mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

ds_bottom = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=bottom_wall_marker)
ds_sides = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=side_wall_marker)
ds_outer = ds_bottom +ds_sides

b_mesh = BoundaryMesh(mesh, "exterior")
S_b = VectorFunctionSpace(b_mesh, "CG", 1)
h = Function(S_b, name="Design")

zero = Constant([0] * mesh.geometric_dimension())


S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="Mesh perturbation field")
h_V = transfer_from_boundary(h, mesh)
h_V.rename("Volume extension of h", "")

s = mesh_deformation(h_V)
ALE.move(mesh, s)


V1 = FiniteElement("CG", mesh.ufl_cell(), 5)
V2 = FiniteElement("CG", mesh.ufl_cell(), 5)
W = FunctionSpace(mesh, V1 * V2)

# Then, we define the test and trial functions, as well as the variational form

(u_re, u_im) = TrialFunctions(W)
(v_re, v_im) = TestFunctions(W)

AMP = 1.0
frequency = 5e9
LIGHT_SPEED = 299792458
PI = 3.14159265359
k_background = 2* PI * frequency / LIGHT_SPEED

a = (inner(grad(u_re), grad(v_re)) - k_background**2*u_re*v_re)*dx \
    + k_background*u_im*v_re*ds_outer \
    + (inner(grad(u_im), grad(v_im)) - k_background**2*u_im*v_im)*dx \
    - k_background*u_re*v_im*ds_outer

l = Constant(0.0)*(v_re + v_im)*dx

# The Dirichlet boundary conditions on :math:`\Gamma` is defined as follows

uinc_re_neg = Expression("-AMP * cos(k_background * x[1])",
                         AMP=AMP, k_background=k_background,
                         degree=3)

uinc_im_neg = Expression("-AMP * sin(k_background * x[1])",
                         AMP=AMP, k_background=k_background,
                         degree=3)

uinc_re = Expression("AMP * cos(k_background * x[1])",
                         AMP=AMP, k_background=k_background,
                         degree=3)

uinc_im = Expression("AMP * sin(k_background * x[1])",
                         AMP=AMP, k_background=k_background,
                         degree=3)

bcs = [
    DirichletBC(W.sub(0), uinc_re_neg, mf, obstacle_marker),
    DirichletBC(W.sub(1), uinc_im_neg, mf, obstacle_marker),
]

w = Function(W, name="Mixed State Solution")
problem = LinearVariationalProblem(a, l, w, bcs)
solver = LinearVariationalSolver(problem)
solver.solve()

u_re, u_im = w.split()

# Plotting the initial velocity and pressure

u_tot_re = u_re + uinc_re
u_tot_im = u_im + uinc_im
u_tot_mag = sqrt(u_tot_re * u_tot_re + u_tot_im * u_tot_im)

plt.figure()
plt.subplot(1, 2, 1)
plot(mesh, color="k", linewidth=0.2, zorder=0)
plot(u_tot_mag, zorder=1)
plt.axis("off")
plt.subplot(1, 2, 2)
plot(u_tot_im, zorder=1)
plt.axis("off")
plt.savefig("intial.png", dpi=800, bbox_inches="tight", pad_inches=0)

# Define the reference function
V_sub0 = W.sub(0).collapse()
u_ref = Function(V_sub0, name="u")
u_ref.vector()[:] = 0.0


reference_data_path = "forward_sim_data_bottom_sweep_kite.csv"
# Only rank 0 reads the CSV file
if MPI.comm_world.rank == 0:
    df = pd.read_csv(reference_data_path)
    if frequency is not None:
        df = df.loc[df['frequency'] == frequency]
    points = df[["x", "y"]].values
    values = df["u"].values
else:
    points = None
    values = None

# Broadcast data to all processes
points = MPI.comm_world.bcast(points, root=0)
values = MPI.comm_world.bcast(values, root=0)

# This is the correct, MPI-safe way to assign point data to a distributed vector.
if points is not None:
    # Get the mapping from local DOF index on the process to the global DOF index
    dofmap = V_sub0.dofmap()
    my_first_dof, my_last_dof = dofmap.ownership_range()
    
    # Create a vector of the correct local size, initialized to the current values
    u_ref_local_vec = u_ref.vector().get_local()

    # Use the bounding box tree to find which points belong to the local process
    tree = mesh.bounding_box_tree()
    
    # Get coordinates of all DOFs (owned and ghosted) on this process
    dof_coords = V_sub0.tabulate_dof_coordinates()
    
    tolerance = 1e-8

    for (px, py), val in zip(points, values):
        cell_id = tree.compute_first_entity_collision(Point(px, py))
        
        if cell_id < mesh.num_cells():
            # Find the DOFs associated with that cell
            cell_dofs = dofmap.cell_dofs(cell_id)
            
            # Find the specific DOF in that cell closest to the point
            min_dist_sq = float('inf')
            target_dof_local = -1
            
            for dof_local in cell_dofs:
                # Check array bounds before accessing dof_coords
                if dof_local < len(dof_coords):
                    dist_sq = (dof_coords[dof_local][0] - px)**2 + (dof_coords[dof_local][1] - py)**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        target_dof_local = dof_local
            
            # If the closest DOF is within tolerance and is owned by this process
            if target_dof_local != -1 and min_dist_sq < tolerance**2:
                # Check if the global index of the dof is owned by the current process
                global_dof = dofmap.local_to_global_index(target_dof_local)
                if my_first_dof <= global_dof < my_last_dof:
                    u_ref_local_vec[target_dof_local] = val

    # Set the local values and then apply changes across all processes
    u_ref.vector().set_local(u_ref_local_vec)
    u_ref.vector().apply("insert")


# Define objective function
J = assemble(inner((u_tot_mag - u_ref), (u_tot_mag - u_ref)) * ds_bottom)

result_file = "result.h5"
Jhat = ReducedFunctional(J, Control(h))


# Set up and solve the optimization problem with moola
problem = MoolaOptimizationProblem(Jhat)
h_moola = moola.DolfinPrimalVector(h)

solver = moola.HybridCG(problem, h_moola,
    options={
        "maxiter": 2,
        "gtol": 1e-6,
    })

sol = solver.solve()

msh_file_path = "meshes/square_with_sin_perturbed_rect_obstacle.msh"
goal_geometry_msh_path = "meshes/square_with_halfsin_perturbed_rect_obstacle.msh"

save_optimization_result(sol, msh_file_path,
                         400,
                         result_file, False)


plot_mesh_deformation_from_result(
    result_file,
    msh_file_path,
    goal_geometry_msh_path,
    obstacle_marker,
    side_wall_marker,
    bottom_wall_marker,
    None,
    "pic.png",
    400
)
"""
s_opt = sol['control'].data



with HDF5File(MPI.comm_world, result_file, "w") as h5f:
    # store under a well-named path
    h5f.write(s_opt, "/h_opt")


# === 4) Read it back (simulate later run / verification) ===
h_opt_loaded = Function(S_b)
with HDF5File(MPI.comm_world, result_file, "r") as h5f:
    h5f.read(h_opt_loaded, "/h_opt")

# === 5) Evaluate the ReducedFunctional Jhat at loaded optimum and at the initial h0 ===
# Jhat is the ReducedFunctional you created earlier: Jhat = ReducedFunctional(J, Control(h))
J_at_hopt_loaded = Jhat(h_opt_loaded)
print(J_at_hopt_loaded)

plt.figure()
Jhat(h)
initial, _ = plot(mesh, color="b", linewidth=0.25)
Jhat(s_opt)
optimal, _ = plot(mesh, color="r", linewidth=0.25)
#plt.legend(handles=[initial, optimal])
plt.axis("off")
plt.savefig("meshes.png", dpi=800, bbox_inches="tight", pad_inches=0)
"""