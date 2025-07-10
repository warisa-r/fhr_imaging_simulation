import numpy as np
import pyvista
from dolfinx import mesh, fem, plot, io
from dolfinx.io import XDMFFile
from mpi4py import MPI
import matplotlib.pyplot as plt

# Read the deformed mesh
with XDMFFile(MPI.COMM_WORLD, "deformed_mesh.xdmf", "r") as xdmf:
    deformed_domain = xdmf.read_mesh()
    
    # Try to read cell tags if they exist
    try:
        deformed_cell_tags = xdmf.read_meshtags(deformed_domain, "cell_tags")
        print("Cell tags loaded successfully")
        print("Available cell tag values:", np.unique(deformed_cell_tags.values))
    except:
        print("No cell tags found in file")
        deformed_cell_tags = None

# Create function space for visualization
V = fem.functionspace(deformed_domain, ("Lagrange", 1))

# Get mesh data for PyVista
topo, cell_types, geom = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topo, cell_types, geom)

# If cell tags exist, add them to the grid
if deformed_cell_tags is not None:
    # Create a function to hold cell tag values
    cell_tag_function = fem.Function(fem.functionspace(deformed_domain, ("DG", 0)))
    
    # Map cell tags to function values
    for i, tag_value in enumerate(deformed_cell_tags.values):
        cell_index = deformed_cell_tags.indices[i]
        cell_tag_function.x.array[cell_index] = tag_value
    
    # Add to grid
    cell_tag_values = cell_tag_function.x.array
    grid["cell_tags"] = cell_tag_values
    
    print(f"Cell tag range: {cell_tag_values.min()} to {cell_tag_values.max()}")

# Visualization
plotter = pyvista.Plotter()

if deformed_cell_tags is not None:
    # Color by cell tags
    plotter.add_mesh(grid, show_edges=True, scalars="cell_tags", 
                    cmap="Set1", show_scalar_bar=True, 
                    scalar_bar_args={"title": "Cell Tags"})
else:
    # Just show the mesh
    plotter.add_mesh(grid, show_edges=True, color="lightblue")

plotter.add_title("Deformed Mesh with Cell Tags")
plotter.show_axes()
plotter.show_grid()

if not pyvista.OFF_SCREEN:
    plotter.show()

# Print mesh info
print(f"Deformed mesh has {deformed_domain.topology.index_map(2).size_local} cells")
print(f"Deformed mesh has {deformed_domain.topology.index_map(0).size_local} vertices")

# Optional: Save screenshot
# plotter.screenshot("deformed_mesh_with_tags.png")