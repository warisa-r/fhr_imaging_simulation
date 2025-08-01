import meshio

# Load the .msh file
msh = meshio.read("meshes/square_with_perturbed_rect_obstacle.msh")

# Drop Z-dimension from points (keep only x, y)
points_2d = msh.points[:, :2]

# Get cell blocks
cells = msh.cells_dict
cell_data = msh.cell_data_dict.get("gmsh:physical", {})

# Export triangle mesh (domain)
if "triangle" in cells:
    mesh = meshio.Mesh(
        points=points_2d,
        cells=[("triangle", cells["triangle"])],
        cell_data={"name_to_read": [cell_data["triangle"]]} if "triangle" in cell_data else None
    )
    mesh.write("meshes/square_with_perturbed_rect_obstacle.xdmf")

# Export boundary markers (line facets)
if "line" in cells:
    boundary = meshio.Mesh(
        points=points_2d,
        cells=[("line", cells["line"])],
        cell_data={"name_to_read": [cell_data["line"]]} if "line" in cell_data else None
    )
    boundary.write("meshes/square_with_perturbed_rect_obstacle_f.xdmf")