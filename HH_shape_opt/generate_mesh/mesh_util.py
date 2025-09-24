# Global mesh markers

SIDE_WALL_MARKER = 1
RECEIVER_EDGE_MARKER = 2
OBSTACLE_MARKER = 3
OBSTACLE_OPT_MARKER = 4
DOMAIN_MARKER = 5

def calculate_mesh_size(freq_max = 5e9, num_mesh_points_per_wavelength = 5):
    c = 299792458

    # Parameters
    wavelength = c / freq_max  # Physical wavelength
    mesh_size = wavelength / num_mesh_points_per_wavelength

    return mesh_size

def plot_mesh(filename, ax, title=""):
    mesh = meshio.read(filename)
    points = mesh.points[:, :2]
    # Find triangle cells
    cells = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            cells = cell_block.data
            break
    if cells is None:
        raise RuntimeError("No triangle cells found in mesh.")
    ax.triplot(points[:, 0], points[:, 1], cells, color="gray", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def convert_msh_to_xdmf(msh_file_path):

    # Define output paths based on the input .msh file
    base_path, _ = os.path.splitext(msh_file_path)
    xdmf_path = f"{base_path}.xdmf"
    facet_xdmf_path = f"{base_path}_facets.xdmf"

    # --- Convert .msh to .xdmf using meshio (only on rank 0) ---
    print(f"[INFO] Converting {msh_file_path} to XDMF format...")
    msh = meshio.read(msh_file_path)

    # Extract 2D points from the 3D points read by meshio
    points_2d = msh.points[:, :2]

    # Create and write the domain mesh (triangles) using 2D points
    triangle_cells = msh.get_cells_type("triangle")
    domain_mesh = meshio.Mesh(points=points_2d, cells=[
                              ("triangle", triangle_cells)])
    domain_mesh.write(xdmf_path)
    print(f"[INFO] Wrote domain mesh to {xdmf_path}")

    # Create and write the facet mesh (lines) using 2D points
    line_cells = msh.get_cells_type("line")
    line_data = msh.get_cell_data("gmsh:physical", "line")
    facet_mesh = meshio.Mesh(
        points=points_2d,
        cells=[("line", line_cells)],
        cell_data={"name_to_read": [line_data]}
    )
    facet_mesh.write(facet_xdmf_path)
    print(f"[INFO] Wrote facet markers to {facet_xdmf_path}")

    return xdmf_path, facet_xdmf_path