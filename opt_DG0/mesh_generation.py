import meshio
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

def generate_square_with_hole_mesh(
    width=1.0, height=1.0, hole_radius=0.2, mesh_size=0.05, output_name="square_with_hole",
    n_circle=40, n_points_bottom=100
):
    import gmsh

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_hole")

    # Outer square points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)         # Bottom-left
    p2 = gmsh.model.geo.addPoint(width, 0, 0, mesh_size)     # Bottom-right
    p3 = gmsh.model.geo.addPoint(width, height, 0, mesh_size)# Top-right
    p4 = gmsh.model.geo.addPoint(0, height, 0, mesh_size)    # Top-left

    # Outer square lines
    l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # Right
    l3 = gmsh.model.geo.addLine(p3, p4)  # Top
    l4 = gmsh.model.geo.addLine(p4, p1)  # Left

    # Use TransfiniteCurve for bottom wall discretization
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, n_points_bottom)

    # Circle (hole) center
    cx, cy = width/2, height/2

    # Circle boundary (hole)
    hole_points = []
    for i in range(n_circle):
        angle = 2 * np.pi * i / n_circle
        x = cx + hole_radius * np.cos(angle)
        y = cy + hole_radius * np.sin(angle)
        hole_points.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size))
    hole_lines = []
    for i in range(n_circle):
        p_start = hole_points[i]
        p_end = hole_points[(i + 1) % n_circle]
        line = gmsh.model.geo.addLine(p_start, p_end)
        gmsh.model.geo.mesh.setTransfiniteCurve(line, 2)  # 2 = 1 segment
        hole_lines.append(line)

    # Curve loops
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    hole_loop = gmsh.model.geo.addCurveLoop(hole_lines)

    # Plane surface with hole
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [l1], bottom_wall_marker, "bottom_wall")
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4], side_wall_marker, "outer_walls")
    gmsh.model.addPhysicalGroup(1, hole_lines, obstacle_marker, "hole_boundary")
    gmsh.model.addPhysicalGroup(2, [surface], domain_marker, "domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(f"{output_name}.msh")
    gmsh.finalize()
    return f"{output_name}.msh"

def generate_square_with_eccentric_hole_mesh(
    width=1.0, height=1.0, hole_radius=0.2, mesh_size=0.05,
    output_name="square_with_eccentric_hole",
    n_circle=40, n_points_bottom=100,
    eccentricity_x=1.2, eccentricity_y=0.8
):
    import gmsh

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_eccentric_hole")

    # Outer square points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)         # Bottom-left
    p2 = gmsh.model.geo.addPoint(width, 0, 0, mesh_size)     # Bottom-right
    p3 = gmsh.model.geo.addPoint(width, height, 0, mesh_size)# Top-right
    p4 = gmsh.model.geo.addPoint(0, height, 0, mesh_size)    # Top-left

    # Outer square lines
    l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # Right
    l3 = gmsh.model.geo.addLine(p3, p4)  # Top
    l4 = gmsh.model.geo.addLine(p4, p1)  # Left

    # Use TransfiniteCurve for bottom wall discretization
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, n_points_bottom)

    # Eccentric (elliptical) hole
    cx, cy = width/2, height/2
    hole_points = []
    for i in range(n_circle):
        angle = 2 * np.pi * i / n_circle
        x = cx + hole_radius * eccentricity_x * np.cos(angle)
        y = cy + hole_radius * eccentricity_y * np.sin(angle)
        hole_points.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size))
    hole_lines = []
    for i in range(n_circle):
        hole_lines.append(gmsh.model.geo.addLine(hole_points[i], hole_points[(i+1)%n_circle]))

    # Curve loops
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    hole_loop = gmsh.model.geo.addCurveLoop(hole_lines)

    # Plane surface with eccentric hole
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])

    gmsh.model.geo.synchronize()

    # Physical groups (same as original)
    gmsh.model.addPhysicalGroup(1, [l1], bottom_wall_marker, "bottom_wall")
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4], side_wall_marker, "outer_walls")
    gmsh.model.addPhysicalGroup(1, hole_lines, obstacle_marker, "hole_boundary")
    gmsh.model.addPhysicalGroup(2, [surface], domain_marker, "domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(f"{output_name}.msh")
    gmsh.finalize()
    return f"{output_name}.msh"

def generate_square_with_rect_obstacle_mesh(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=0.05,
    output_name="square_with_rect_obstacle",
    n_points_bottom=100, n_points_rect_bottom=40
):
    import gmsh

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_rect_obstacle")

    # Outer square points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)         # Bottom-left
    p2 = gmsh.model.geo.addPoint(width, 0, 0, mesh_size)     # Bottom-right
    p3 = gmsh.model.geo.addPoint(width, height, 0, mesh_size)# Top-right
    p4 = gmsh.model.geo.addPoint(0, height, 0, mesh_size)    # Top-left

    # Outer square lines
    l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # Right
    l3 = gmsh.model.geo.addLine(p3, p4)  # Top
    l4 = gmsh.model.geo.addLine(p4, p1)  # Left

    # Use TransfiniteCurve for bottom wall discretization
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, n_points_bottom)

    # Rectangle obstacle center
    cx, cy = width/2, height/2
    rx1 = cx - rect_w/2
    rx2 = cx + rect_w/2
    ry1 = cy - rect_h/2
    ry2 = cy + rect_h/2

    # Rectangle obstacle points (counterclockwise)
    rp1 = gmsh.model.geo.addPoint(rx1, ry1, 0, mesh_size)
    rp2 = gmsh.model.geo.addPoint(rx2, ry1, 0, mesh_size)
    rp3 = gmsh.model.geo.addPoint(rx2, ry2, 0, mesh_size)
    rp4 = gmsh.model.geo.addPoint(rx1, ry2, 0, mesh_size)

    # Rectangle obstacle lines
    rl1 = gmsh.model.geo.addLine(rp1, rp2)
    rl2 = gmsh.model.geo.addLine(rp2, rp3)
    rl3 = gmsh.model.geo.addLine(rp3, rp4)
    rl4 = gmsh.model.geo.addLine(rp4, rp1)
    rect_lines = [rl1, rl2, rl3, rl4]

    gmsh.model.geo.mesh.setTransfiniteCurve(rl1, n_points_rect_bottom)

    # Curve loops
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    rect_loop = gmsh.model.geo.addCurveLoop(rect_lines)

    # Plane surface with rectangle obstacle
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, rect_loop])

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [l1], bottom_wall_marker, "bottom_wall")
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4], side_wall_marker, "outer_walls")
    gmsh.model.addPhysicalGroup(1, rect_lines, obstacle_marker, "rect_obstacle_boundary")
    gmsh.model.addPhysicalGroup(2, [surface], domain_marker, "domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(f"{output_name}.msh")
    gmsh.finalize()
    return f"{output_name}.msh"

def generate_square_with_perturbed_rect_obstacle_mesh(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=0.05,
    output_name="square_with_perturbed_rect_obstacle",
    n_points_bottom=100, n_points_rect_bottom=40,
    perturb_amplitude=0.03, perturb_frequency=3
):
    import gmsh

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_perturbed_rect_obstacle")

    # Outer square points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(width, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(width, height, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, height, 0, mesh_size)

    # Outer square lines
    l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # Right
    l3 = gmsh.model.geo.addLine(p3, p4)  # Top
    l4 = gmsh.model.geo.addLine(p4, p1)  # Left

    # Use TransfiniteCurve for bottom wall discretization
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, n_points_bottom)

    # Rectangle obstacle center
    cx, cy = width/2, height/2
    rx1 = cx - rect_w/2
    rx2 = cx + rect_w/2
    ry1 = cy - rect_h/2
    ry2 = cy + rect_h/2

    # Perturbed bottom edge of rectangle
    rect_bottom_points = []
    for i in range(n_points_rect_bottom):
        t = i / (n_points_rect_bottom - 1)
        x = rx1 + t * (rx2 - rx1)
        y = ry1 + perturb_amplitude * np.cos(perturb_frequency * np.pi * t)
        rect_bottom_points.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size))

    # Other rectangle points (no perturbation)
    rp2 = gmsh.model.geo.addPoint(rx2, ry1, 0, mesh_size)
    rp3 = gmsh.model.geo.addPoint(rx2, ry2, 0, mesh_size)
    rp4 = gmsh.model.geo.addPoint(rx1, ry2, 0, mesh_size)

    # Rectangle lines
    rect_lines = []
    # Bottom (perturbed)
    for i in range(n_points_rect_bottom - 1):
        rect_lines.append(gmsh.model.geo.addLine(rect_bottom_points[i], rect_bottom_points[i+1]))
    # Right
    rect_lines.append(gmsh.model.geo.addLine(rect_bottom_points[-1], rp3))
    rect_lines.append(gmsh.model.geo.addLine(rp3, rp4))
    # Left
    rect_lines.append(gmsh.model.geo.addLine(rp4, rect_bottom_points[0]))

    # Curve loops
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    rect_loop = gmsh.model.geo.addCurveLoop(rect_lines)

    # Plane surface with perturbed rectangle obstacle
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, rect_loop])

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [l1], bottom_wall_marker, "bottom_wall")
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4], side_wall_marker, "outer_walls")
    gmsh.model.addPhysicalGroup(1, rect_lines, obstacle_marker, "perturbed_rect_obstacle_boundary")
    gmsh.model.addPhysicalGroup(2, [surface], domain_marker, "domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(f"{output_name}.msh")
    gmsh.finalize()
    return f"{output_name}.msh"

def generate_square_with_flattened_circle_mesh(
    width=1.0, height=1.0, hole_radius=0.2, mesh_size=0.05,
    output_name="square_with_flattened_circle",
    n_circle=100, n_points_bottom=100,
    flatten_y=None  # If None, flatten at the lowest point of the circle
):
    import gmsh

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_flattened_circle")

    # Outer square points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(width, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(width, height, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, height, 0, mesh_size)

    # Outer square lines
    l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # Right
    l3 = gmsh.model.geo.addLine(p3, p4)  # Top
    l4 = gmsh.model.geo.addLine(p4, p1)  # Left

    # Use TransfiniteCurve for bottom wall discretization
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, n_points_bottom)

    # Circle (hole) center
    cx, cy = width/2, height/2

    # Determine flatten_y if not provided (default: lowest point of circle)
    if flatten_y is None:
        flatten_y = cy - hole_radius

    # Circle boundary (flattened bottom)
    hole_points = []
    for i in range(n_circle):
        angle = 2 * np.pi * i / n_circle
        x = cx + hole_radius * np.cos(angle)
        y = cy + hole_radius * np.sin(angle)
        # Flatten if below flatten_y
        if y < flatten_y:
            y = flatten_y
        hole_points.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size))
    hole_lines = []
    for i in range(n_circle):
        hole_lines.append(gmsh.model.geo.addLine(hole_points[i], hole_points[(i+1)%n_circle]))

    # Curve loops
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    hole_loop = gmsh.model.geo.addCurveLoop(hole_lines)

    # Plane surface with flattened circle
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [l1], bottom_wall_marker, "bottom_wall")
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4], side_wall_marker, "outer_walls")
    gmsh.model.addPhysicalGroup(1, hole_lines, obstacle_marker, "flattened_circle_boundary")
    gmsh.model.addPhysicalGroup(2, [surface], domain_marker, "domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(f"{output_name}.msh")
    gmsh.finalize()
    return f"{output_name}.msh"

def generate_square_with_sin_perturbed_circle_mesh(
    width=1.0, height=1.0, hole_radius=0.2, mesh_size=0.05,
    output_name="square_with_sin_perturbed_circle",
    n_circle=100, n_points_bottom=100,
    perturb_amplitude=0.02, perturb_frequency=3
):
    import gmsh

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_sin_perturbed_circle")

    # Outer square points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(width, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(width, height, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, height, 0, mesh_size)

    # Outer square lines
    l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # Right
    l3 = gmsh.model.geo.addLine(p3, p4)  # Top
    l4 = gmsh.model.geo.addLine(p4, p1)  # Left

    # Use TransfiniteCurve for bottom wall discretization
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, n_points_bottom)

    # Circle (hole) center
    cx, cy = width/2, height/2

    # Circle boundary (perturbed bottom)
    hole_points = []
    for i in range(n_circle):
        angle = 2 * np.pi * i / n_circle
        x = cx + hole_radius * np.cos(angle)
        y = cy + hole_radius * np.sin(angle)
        # Perturb only the bottom (y < cy)
        if y < cy:
            t = (x - (cx - hole_radius)) / (2 * hole_radius)  # Normalize x along bottom
            y += perturb_amplitude * np.sin(perturb_frequency * np.pi * t)
        hole_points.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size))
    hole_lines = []
    for i in range(n_circle):
        hole_lines.append(gmsh.model.geo.addLine(hole_points[i], hole_points[(i+1)%n_circle]))

    # Curve loops
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    hole_loop = gmsh.model.geo.addCurveLoop(hole_lines)

    # Plane surface with perturbed circle
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [l1], bottom_wall_marker, "bottom_wall")
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4], side_wall_marker, "outer_walls")
    gmsh.model.addPhysicalGroup(1, hole_lines, obstacle_marker, "sin_perturbed_circle_boundary")
    gmsh.model.addPhysicalGroup(2, [surface], domain_marker, "domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(f"{output_name}.msh")
    gmsh.finalize()
    return f"{output_name}.msh"

if __name__ == "__main__":
    print("Generating square with hole mesh...")

    c = 299792458
    freq_max = 5e9 # 5GHz
    
    # Parameters
    wavelength = c / freq_max  # Physical wavelength
    mesh_size = wavelength / 5
    
    
    mesh_file = "meshes/square_with_rect_obstacle.msh"

    convert_msh_to_xdmf(mesh_file)