import meshio
import matplotlib.pyplot as plt
import numpy as np

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
        hole_lines.append(gmsh.model.geo.addLine(hole_points[i], hole_points[(i+1)%n_circle]))

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

if __name__ == "__main__":
    print("Generating square with hole mesh...")

    c = 299792458
    freq_max = 5e9 # 5GHz
    
    # Parameters
    wavelength = c / freq_max  # Physical wavelength
    mesh_size = wavelength / 5

    mesh_file = generate_square_with_hole_mesh(
        width=1.0,
        height=1.0,
        hole_radius=0.2,
        mesh_size=mesh_size,
        output_name="square_with_hole",
        n_circle=40, n_points_bottom=100
    )

    mesh_file = generate_square_with_eccentric_hole_mesh(
        width=1.0,
        height=1.0,
        hole_radius=0.2,
        mesh_size=mesh_size,
        output_name="square_with_eccentric_hole",
        n_circle=40,
        n_points_bottom=100,
        eccentricity_x=1.2,
        eccentricity_y=0.8
    )

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    plot_mesh("square_with_hole.msh", ax1, "Square with Circular Hole")

    ax2 = plt.subplot(1, 2, 2)
    plot_mesh("square_with_eccentric_hole.msh", ax2, "Square with Eccentric Hole")

    plt.tight_layout()
    plt.savefig("compare_initial_mesh.png")
    plt.show()