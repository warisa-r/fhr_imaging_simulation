import pyvista
import dolfinx.fem.petsc
from dolfinx.plot import vtk_mesh

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

def export_function(mesh, grid, name, show_mesh=False, tessellate=False):
    # visualizes and exports DOLFINx simulation results as interactive HTML plots
    grid.set_active_scalars(name)
    plotter = pyvista.Plotter(window_size=(700, 700))
    t_grid = grid.tessellate() if tessellate else grid
    plotter.add_mesh(t_grid, show_edges=False, scalar_bar_args=sargs)
    if show_mesh:
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1)) #TODO: This should be changable
        grid_mesh = pyvista.UnstructuredGrid(*vtk_mesh(V))
        plotter.add_mesh(grid_mesh, style="wireframe", line_width=0.1, color="k")
        plotter.view_xy()
    plotter.view_xy()
    plotter.camera.zoom(1.3)
    plotter.export_html(f"./{name}.html")