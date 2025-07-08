from mpi4py import MPI
import sys

import numpy as np
import scipy.io

import dolfinx.fem.petsc
import ufl
import basix.ufl
from petsc4py import PETSc
from dolfinx.io import gmshio
from dolfinx.plot import vtk_mesh
import matplotlib.pyplot as plt
import pyvista

from mesh_generation import generate_mesh

pathname_calib = r'./2d_RT_helmholtz/'

# Load the frequency vector
freq_vec_init = scipy.io.loadmat(pathname_calib + 'freq_vec.mat')['freq_vec'].flatten()
freq_vec = np.flip(freq_vec_init) # Flip to get up-chirp

# Load the antenna position
ant_pos = np.loadtxt(pathname_calib + 'ant_pos.txt')
ant_pos_2D = ant_pos[:, [0, 2]] # Take receiver position in x and z dims only

# Define some constants
c = 299792458
interested_f = freq_vec.min()
lam0 = c / interested_f

# wavenumber in free space (air)
k0 = 2* np.pi * interested_f / c

# Polynomial degree
degree = 6

# Mesh order
mesh_order = 2

# MPI communicator
comm = MPI.COMM_WORLD

# Generate the mesh
file_name = "domain.msh"
generate_mesh(file_name, lam0, order=mesh_order)
mesh, cell_tags, _ = gmshio.read_from_msh(file_name, comm, rank=0, gdim=2)

# Assigning material parameter
W = dolfinx.fem.functionspace(mesh, ("DG", 0))
# Function: A finite element function that is represented by a function space (domain, element and dofmap) and a vector holding the degrees-of-freedom.
k = dolfinx.fem.Function(W) # k is a function whose values vary across the mesh (input: cell. output: value of k at that cell)
k.x.array[:] = k0 # x here just represent the variable of a degree of freedom

# Set refractive index for differrent geometry withing the computational domain
n_refractive = 1.5 # Refractive index for the refraction geometry
n_refractive_metal = 500 # Refractive index for the metal inlets
k.x.array[cell_tags.find(1)] = n_refractive_metal * k0 # Refraction geometry
k.x.array[cell_tags.find(2)] = n_refractive_metal * k0

# Visualize the mesh
#pyvista.start_xvfb()
pyvista.set_plot_theme("paraview")
sargs = dict(
    title_font_size=25,
    label_font_size=20,
    fmt="%.2e",
    color="black",
    position_x=0.1,
    position_y=0.8,
    width=0.8,
    height=0.1,
)


def export_function(grid, name, show_mesh=False, tessellate=False):
    grid.set_active_scalars(name)
    plotter = pyvista.Plotter(window_size=(700, 700))
    t_grid = grid.tessellate() if tessellate else grid
    plotter.add_mesh(t_grid, show_edges=False, scalar_bar_args=sargs)
    if show_mesh:
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        grid_mesh = pyvista.UnstructuredGrid(*vtk_mesh(V))
        plotter.add_mesh(grid_mesh, style="wireframe", line_width=0.1, color="k")
        plotter.view_xy()
    plotter.view_xy()
    plotter.camera.zoom(1.3)
    plotter.export_html(f"./{name}.html")

grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh))
grid.cell_data["wavenumber"] = k.x.array.real
export_function(grid, "wavenumber", show_mesh=True, tessellate=True)

# Define source term at boundary
n = ufl.FacetNormal(mesh)
x = ufl.SpatialCoordinate(mesh)

# Spherical wave as superpositions from all the antenna
#source_pos = ant_pos_2D[1]
#r = ufl.sqrt((x[0] - source_pos[0])**2 + (x[1] - source_pos[1])**2)
#uinc = ufl.exp(1j * k * r)/ ( 4 * np.pi * r)
uinc = 0
source_pos = ant_pos_2D[0, :]
r = ufl.sqrt((x[0] - source_pos[0])**2 + (x[1] - source_pos[1])**2)
uinc += ufl.exp(1j * k * r)/ ( 4 * np.pi * r)
g = ufl.dot(ufl.grad(uinc), n) - 1j * k * uinc

# Define the weak form of the problem
# https://fenics.readthedocs.io/projects/ufl/en/latest/manual/form_language.html
element = basix.ufl.element("Lagrange", mesh.topology.cell_name(), degree)
V = dolfinx.fem.functionspace(mesh, element)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
# LHS
a = (
    -ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    + k**2 * ufl.inner(u, v) * ufl.dx
    - 1j * k * ufl.inner(u, v) * ufl.ds # ds represents the computational domain boundary
)
L = ufl.inner(g, v) * ufl.ds # RHS

opt = {"ksp_type": "preonly", "pc_type": "lu"}
problem = dolfinx.fem.petsc.LinearProblem(a, L, petsc_options=opt)
uh = problem.solve()
uh.name = "u"

topology, cells, geometry = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cells, geometry)
grid.point_data["Abs(u)"] = np.abs(uh.x.array)
export_function(grid, "Abs(u)", show_mesh=False, tessellate=True)

grid.point_data["Re(u)"] = np.real(uh.x.array)
export_function(grid, "Re(u)", show_mesh=False, tessellate=True)