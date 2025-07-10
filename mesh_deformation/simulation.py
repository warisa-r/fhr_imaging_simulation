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
L, H = 2.0, 2.0
r = 0.2
beta = 1.25
traction_vector = np.array([0.0, -0.25], dtype=np.float64)  # This is what we are aiming to optimize

# === Geometry with GMSH ===
gmsh.initialize()
gmsh.model.add("circle_in_box")

# Add circle
c1 = gmsh.model.occ.addCircle(1.0, 1.0, 0.0, 0.25)
gmsh.model.occ.addCurveLoop([c1], tag=c1)
gmsh.model.occ.addPlaneSurface([c1], tag=c1)

# Add domain
r0 = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, H)
inclusive_rectangle, _ = gmsh.model.occ.fragment([(2, r0)], [(2, c1)])

gmsh.model.occ.synchronize()

# === Tag both surfaces (cell tags) AND curves (facet tags) ===

# Add physical groups for surfaces (cell tags)
gmsh.model.addPhysicalGroup(2, [c1], tag=1)  # Circle surface
gmsh.model.addPhysicalGroup(2, [r0], tag=2)  # Rectangle surface

# Add physical groups for curves (facet tags)
# Tag the circle curve (1D boundary)
gmsh.model.addPhysicalGroup(1, [c1], tag=1)  # Circle curve with tag 1
gmsh.model.setPhysicalName(1, 1, "circle_boundary")

# Get all boundaries of the fragmented geometry
all_boundaries = gmsh.model.getBoundary(inclusive_rectangle, oriented=False)

# Filter out the circle boundary to get only the outer rectangle boundaries
outer_boundaries = []
for boundary in all_boundaries:
    if boundary[1] != c1:  # Not the circle curve
        outer_boundaries.append(boundary[1])

# Tag the outer rectangle boundaries
if outer_boundaries:
    gmsh.model.addPhysicalGroup(1, outer_boundaries, tag=2)  # Outer boundaries with tag 2
    gmsh.model.setPhysicalName(1, 2, "outer_boundary")

# Set mesh size
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.02)

# Generate mesh
gmsh.model.mesh.generate(2)

# Convert to DOLFINx mesh
domain, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

gmsh.finalize()

# Print the unique tags to verify
print("Available cell tags:", np.unique(cell_tags.values))
print("Available facet tags:", np.unique(facet_tags.values))
print("Number of facets with tag 1 (circle):", np.sum(facet_tags.values == 1))
print("Number of facets with tag 2 (outer):", np.sum(facet_tags.values == 2))

# === Assigning material parameter ===

# THIS IS THE STIFFNESS OF THE MATERIAL PART
W = dolfinx.fem.functionspace(domain, ("DG", 0))
# Function: A finite element function that is represented by a function space (domain, element and dofmap) and a vector holding the degrees-of-freedom.
mu = dolfinx.fem.Function(W) # k is a function whose values vary across the mesh (input: cell. output: value of k at that cell)
mu.x.array[:] = 0.2 # x here just represent the variable of a degree of freedom

mu.x.array[cell_tags.find(2)] = 0.15 # The outer material

# === Function space ===
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

# === Boundary conditions ===
#TODO: Think if we want to define the mesh like the paper
# Or what shall we do in order to make the domain such that we can change
def clamped_boundary(x):
    return (
        np.isclose(x[0], 0) | np.isclose(x[0], L)
    ) # The box is clamped.only in certain directions

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

# Use the boundary integral (ds) with facet tags, not volume integral (dx)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(traction, v) * ds(1)  # Apply traction on facet tag 1 (circle boundary)

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
warped = grid.warp_by_vector("u")

# Plot only the deformed mesh
plotter = pyvista.Plotter()
plotter.add_mesh(warped, show_edges=True, scalars="u", cmap="viridis")
plotter.show_axes()
if not pyvista.OFF_SCREEN:
    plotter.show()

# == Save the deformed mesh... to run another simulation again? ==#
# Create deformed coordinates (original + displacement)

from dolfinx.io import XDMFFile

# Create new mesh coordinates (original + displacement)
deformed_coords = domain.geometry.x + u_3d

# Overwrite the domain mesh coordinates
domain.geometry.x[:, :] = deformed_coords

# Save mesh with deformed geometry and properly named tags
with XDMFFile(MPI.COMM_WORLD, "deformed_mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    
    # Write cell tags with a name
    cell_tags.name = "cell_tags"
    xdmf.write_meshtags(cell_tags, domain.geometry)
