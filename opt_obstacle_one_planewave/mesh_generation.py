import meshio
import gmsh
import matplotlib.pyplot as plt
import numpy as np
import os

side_wall_marker = 1
bottom_wall_marker = 2
obstacle_marker = 3
domain_marker = 4

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
    domain_mesh = meshio.Mesh(points=points_2d, cells=[("triangle", triangle_cells)])
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

def generate_square_with_circle_hole_mesh(
    width=1.0, height=1.0, circle_radius=0.2, mesh_size=0.1, output_name="square_with_circle_hole", n_points_bottom = None
):
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_circle_obstacle")

    # --- 1. Outer square points ---
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(width, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(width, height, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, height, 0, mesh_size)

    # --- 2. Outer square lines ---
    l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # Right
    l3 = gmsh.model.geo.addLine(p3, p4)  # Top
    l4 = gmsh.model.geo.addLine(p4, p1)  # Left

    if n_points_bottom != None:
        gmsh.model.geo.mesh.setTransfiniteCurve(l1, n_points_bottom)

    # --- 3. Circle center and perimeter points ---
    cx, cy = width / 2, height / 2
    center = gmsh.model.geo.addPoint(cx, cy, 0, mesh_size)

    # Four points on the circle (N, E, S, W)
    pN = gmsh.model.geo.addPoint(cx, cy + circle_radius, 0, mesh_size)
    pE = gmsh.model.geo.addPoint(cx + circle_radius, cy, 0, mesh_size)
    pS = gmsh.model.geo.addPoint(cx, cy - circle_radius, 0, mesh_size)
    pW = gmsh.model.geo.addPoint(cx - circle_radius, cy, 0, mesh_size)

    # --- 4. Circle arcs ---
    arc1 = gmsh.model.geo.addCircleArc(pN, center, pE)
    arc2 = gmsh.model.geo.addCircleArc(pE, center, pS)
    arc3 = gmsh.model.geo.addCircleArc(pS, center, pW)
    arc4 = gmsh.model.geo.addCircleArc(pW, center, pN)

    circle_loop = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])

    # --- 5. Outer boundary loop ---
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

    # --- 6. Create surface with circular hole ---
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, circle_loop])

    gmsh.model.geo.synchronize()

    # --- 7. Physical groups for FEniCS ---
    gmsh.model.addPhysicalGroup(1, [l1], bottom_wall_marker, "bottom_wall")
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4], side_wall_marker, "outer_walls")
    gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4], obstacle_marker, "circle_obstacle_boundary")
    gmsh.model.addPhysicalGroup(2, [surface], domain_marker, "domain")

    # --- 8. Generate mesh ---
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # FEniCS compatibility
    mesh_path = f"{output_name}.msh"
    gmsh.write(mesh_path)
    gmsh.finalize()

    print(f"[INFO] Generated mesh file: {mesh_path}")
    return mesh_path

def generate_square_with_kite_obstacle_mesh(
    width=2.0,
    height=2.0,
    mesh_size=0.05,
    output_name="square_with_kite_obstacle",
    n_points_bottom=50,
    n_kite_points=200,
    scale_factor=1.0  # <--- NEW
):
    """
    Generate a square domain with a kite-shaped obstacle inside.
    The kite can be scaled using `scale_factor`.
    """
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_kite_obstacle")

    # Gmsh tolerances to avoid intersection issues
    gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
    gmsh.option.setNumber("Mesh.ToleranceEdgeLength", 1e-8)
    gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-8)

    # --- 1. Outer square points ---
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(width, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(width, height, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, height, 0, mesh_size)

    # Outer square lines
    l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # Right
    l3 = gmsh.model.geo.addLine(p3, p4)  # Top
    l4 = gmsh.model.geo.addLine(p4, p1)  # Left

    # Control mesh refinement along the bottom boundary
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, n_points_bottom)

    # --- 2. Generate kite shape points with scaling ---
    t_vals = np.linspace(0, 2 * np.pi, n_kite_points, endpoint=False)

    # Parametric equations scaled
    x_vals = scale_factor * (np.cos(t_vals) + 0.65 * np.cos(2 * t_vals) - 1)
    y_vals = scale_factor * (1.5 * np.sin(t_vals))

    # Center the kite in the square domain
    cx, cy = width / 2, height / 2
    kite_points = []
    for x, y in zip(x_vals, y_vals):
        px = cx + x
        py = cy + y
        kite_points.append(gmsh.model.geo.addPoint(px, py, 0, mesh_size))

    # Create a closed spline curve for the kite boundary
    kite_curve = gmsh.model.geo.addSpline(kite_points + [kite_points[0]])

    # --- 3. Define curve loops ---
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    kite_loop = gmsh.model.geo.addCurveLoop([kite_curve])

    # Create surface with kite as a hole
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, kite_loop])

    gmsh.model.geo.synchronize()

    # --- 4. Physical groups for FEniCS ---
    gmsh.model.addPhysicalGroup(1, [l1], bottom_wall_marker, "bottom_wall")
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4], side_wall_marker, "outer_walls")
    gmsh.model.addPhysicalGroup(1, [kite_curve], obstacle_marker, "kite_boundary")
    gmsh.model.addPhysicalGroup(2, [surface], domain_marker, "domain")

    # --- 5. Mesh generation ---
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # FEniCS compatibility
    msh_file = f"{output_name}.msh"
    gmsh.write(msh_file)
    gmsh.finalize()

    print(f"[INFO] Generated mesh saved to {msh_file}")
    return msh_file

if __name__ == "__main__":
    print("Generating square with hole mesh...")

    c = 299792458
    freq_max = 1e9 # 5GHz
    
    # Parameters
    wavelength = c / freq_max  # Physical wavelength
    mesh_size = wavelength / 4
    
    
    mesh_file = generate_square_with_circle_hole_mesh(
        width=9, height=9, circle_radius=1.2, mesh_size=mesh_size,
        output_name="meshes/square_with_circle_obstacle", n_points_bottom = 100
    )

    convert_msh_to_xdmf(mesh_file)
    

    """    
    mesh_file = generate_square_with_kite_obstacle_mesh(
        width=9,
        height=9,
        mesh_size=mesh_size,
        output_name="meshes/square_with_kite_obstacle",
        n_points_bottom=100,
        n_kite_points=150,
        scale_factor = 1
    )

    convert_msh_to_xdmf(mesh_file)
    """
    
    
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_mesh(mesh_file, ax, title="Mesh")
    plt.savefig("meshpic.png")
    plt.show()
