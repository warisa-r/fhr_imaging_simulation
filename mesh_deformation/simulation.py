import numpy as np
import pyvista
import gmsh
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from mpi4py import MPI
import ufl

import dolfinx

# === Problem parameters ===
L, H = 2.0, 1.0
r = 0.2
#mu = 0.1
beta = 1.25
traction_vector = np.array([0.0, -0.6], dtype=np.float64)  # Increased force for visibility

# === Geometry with GMSH ===
gmsh.initialize()
gmsh.model.add("circle_force")

# Create rectangle
rect = gmsh.model.occ.addRectangle(0, 0, 0, L, H)

# Create circle
center_x, center_y = L/2, H/2
circle = gmsh.model.occ.addCircle(center_x, center_y, 0, r)
circle_loop = gmsh.model.occ.addCurveLoop([circle])

# First synchronize to ensure entities exist
gmsh.model.occ.synchronize()

# Add the rectangle and circle curve to the model
gmsh.model.addPhysicalGroup(2, [rect], 2)  # Tag rectangle as domain
gmsh.model.setPhysicalName(2, 2, "domain")

gmsh.model.addPhysicalGroup(1, [circle], 1)  # Tag circle as boundary 1
gmsh.model.setPhysicalName(1, 1, "circle")

# Embed the circle curve in the rectangle surface
gmsh.model.mesh.embed(1, [circle], 2, rect)
gmsh.model.occ.synchronize()

# Set mesh size
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.02)

# Generate mesh
gmsh.model.mesh.generate(2)

# For debugging - optionally save the mesh to view in GMSH GUI
# gmsh.write("debug_mesh.msh")

# Convert to DOLFINx mesh
domain, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()

# Print the unique tags to verify
print("Available facet tags:", np.unique(facet_tags.values))

# === Assigning material parameter ===
W = dolfinx.fem.functionspace(domain, ("DG", 0))
# Function: A finite element function that is represented by a function space (domain, element and dofmap) and a vector holding the degrees-of-freedom.
mu = dolfinx.fem.Function(W) # k is a function whose values vary across the mesh (input: cell. output: value of k at that cell)
mu.x.array[:] = 0.2 # x here just represent the variable of a degree of freedom

mu.x.array[cell_tags.find(2)] = 0.3 # Refraction geometry


# === Function space ===
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

# === Boundary conditions ===
def clamped_boundary(x):
    return (
        np.isclose(x[0], 0) | np.isclose(x[0], L) |
        np.isclose(x[1], 0) | np.isclose(x[1], H)
    ) # The box is clamped

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
u_D = np.array([0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# === Stress-strain definitions ===
def epsilon(u): return ufl.sym(ufl.grad(u))
def sigma(u): return 2 * mu * epsilon(u) # Term with lambda is left out

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

traction = fem.Constant(domain, traction_vector)
def circle_force_region(x):
    center_x, center_y = L/2, H/2
    distance = np.sqrt((x[0] - center_x)**2 + (x[1] - center_y)**2)
    return distance <= r

# Create a function to mark the circle region
circle_cells = mesh.locate_entities(domain, domain.topology.dim, circle_force_region)
circle_marker = mesh.meshtags(domain, domain.topology.dim, circle_cells, np.full(len(circle_cells), 1))

# Define body force in the circle region
f_body = fem.Function(V)
f_body.x.array[:] = 0.0

# Apply distributed force in circle region
dV_circle = ufl.Measure("dx", domain=domain, subdomain_data=circle_marker)
body_force = fem.Constant(domain, traction_vector)

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(body_force, v) * dV_circle(1)  # Apply body force in circle region

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Visualization
topo, cell_types, geom = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topo, cell_types, geom)

u_2d = uh.x.array.reshape((geom.shape[0], 2))
u_3d = np.zeros((geom.shape[0], 3))
u_3d[:, :2] = u_2d

grid = pyvista.UnstructuredGrid(topo, cell_types, geom)
grid["u"] = u_3d
warped = grid.warp_by_vector("u", factor=5.0)  # Exaggerate for visibility

# Plot only the deformed mesh
plotter = pyvista.Plotter()
plotter.add_mesh(warped, show_edges=True, scalars="u", cmap="viridis")
plotter.show_axes()
if not pyvista.OFF_SCREEN:
    plotter.show()