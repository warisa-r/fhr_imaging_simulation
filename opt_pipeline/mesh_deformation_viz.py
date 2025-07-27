import os
import matplotlib.pyplot as plt
from dolfin import *
from dolfin_adjoint import *
import numpy as np

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker

# Load mesh and boundary markers
mesh = Mesh("rectangle_mesh.xml")
boundary_markers = MeshFunction("size_t", mesh, "rectangle_mesh_facet_region.xml")

# Create boundary mesh and function space
b_mesh = BoundaryMesh(mesh, "exterior")
S_b = VectorFunctionSpace(b_mesh, "CG", 1)
h = Function(S_b, name="Design")

# Load h from checkpoint
checkpoint_file = "h_checkpoint.h5"
iteration = 0
if os.path.exists(checkpoint_file):
    with HDF5File(MPI.comm_world, checkpoint_file, "r") as h5f:
        h_temp = Function(S_b, name="Design")
        h5f.read(h_temp, "/h_opt")
        h.vector()[:] = h_temp.vector().get_local()
        try:
            iteration = h5f.attributes("/h_opt")["iteration"]
        except Exception:
            iteration = 0
    print(f"Loaded checkpoint from h_checkpoint.h5 (iteration {iteration})")
else:
    print("Checkpoint file not found.")
    exit(1)

# Transfer h to volume mesh
h_V = transfer_from_boundary(h, mesh)

# Mesh deformation function (copied from your main script)
def mesh_deformation(h, mesh_local, markers_local):
    V = FunctionSpace(mesh_local, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a  = inner(grad(u), grad(v)) * dx
    L0 = Constant(0.0) * v * dx
    bcs0 = [
        DirichletBC(V, Constant(2.0), markers_local, side_wall_marker),
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

# Make a copy of the mesh for deformation
mesh_copy = Mesh(mesh)
boundary_markers_copy = MeshFunction("size_t", mesh_copy, "rectangle_mesh_facet_region.xml")

# Deform the mesh
s_final = mesh_deformation(h_V, mesh_copy, boundary_markers_copy)
ALE.move(mesh_copy, s_final)

# Plot the deformed mesh
plt.figure(figsize=(8, 6))
plot(mesh_copy, color="r", linewidth=0.5)
plt.title(f"Deformed mesh (iteration {iteration})")
plt.axis("equal")
plt.tight_layout()
plt.show()