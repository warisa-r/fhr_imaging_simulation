from mpi4py import MPI

import numpy as np

import dolfinx.fem.petsc
import ufl

import sys

from petsc4py import PETSc

if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("This tutorial requires complex number support")
    sys.exit(0)
else:
    print(f"Using {PETSc.ScalarType}.")

## Define constants in the equations and important FEM setup information
# wavenumber in free space (air)
k0 = 10 * np.pi

# Corresponding wavelength
lmbda = 2 * np.pi / k0

# Polynomial degree
degree = 6

# Mesh order
mesh_order = 2

# Generate mesh using 'mesh_generation'
from dolfinx.io import gmshio
from mesh_generation import generate_mesh

# MPI communicator
comm = MPI.COMM_WORLD

file_name = "domain.msh"
generate_mesh(file_name, lmbda, order=mesh_order)
mesh, cell_tags, _ = gmshio.read_from_msh(file_name, comm, rank=0, gdim=2) # MUST BE A .msh file for dolfinx

W = dolfinx.fem.functionspace(mesh, ("DG", 0))
k = dolfinx.fem.Function(W)
k.x.array[:] = k0
k.x.array[cell_tags.find(1)] = 3 * k0 # All cells that has been tagged with physical group ID 1 has 3 * k0

import matplotlib.pyplot as plt
import pyvista

from dolfinx.plot import vtk_mesh
pyvista.OFF_SCREEN = True # So that we don't need extra library

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
    
    # Create plotter with off_screen=True for HTML export
    plotter = pyvista.Plotter(window_size=(700, 700), off_screen=True)
    
    t_grid = grid.tessellate() if tessellate else grid
    plotter.add_mesh(t_grid, show_edges=False, scalar_bar_args=sargs)
    
    if show_mesh:
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        grid_mesh = pyvista.UnstructuredGrid(*vtk_mesh(V))
        plotter.add_mesh(grid_mesh, style="wireframe", line_width=0.1, color="k")
        
    plotter.view_xy()
    plotter.camera.zoom(1.3)
    plotter.export_html(f"./{name}.html")
    plotter.close()
    print(f"Interactive HTML saved as {name}.html")

grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh))
grid.cell_data["wavenumber"] = k.x.array.real

export_function(grid, "wavenumber", show_mesh=True, tessellate=False)