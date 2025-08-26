import meshio
import matplotlib.pyplot as plt
import numpy as np

side_wall_marker = 1
bottom_wall_marker = 2
obstacle_marker = 3
domain_marker = 4
obstacle_opt_marker = 5

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

def generate_square_with_exp_perturbed_rect_mesh(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=0.05,
    output_name="square_with_exp_perturbed_rect",
    n_points_bottom=100, n_points_rect_bottom=100,
    amplitude=0.03, std_dev=0.1
):
    import gmsh
    import numpy as np

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_exp_perturbed_rect")

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

    # Symmetric exponential perturbation on bottom edge
    rect_bottom_points = []
    for i in range(n_points_rect_bottom):
        t = i / (n_points_rect_bottom - 1)  # Normalized [0, 1]
        x = rx1 + t * (rx2 - rx1)
        y = ry1 + amplitude * np.exp(-((t - 0.5)**2) / (2 * std_dev**2))
        rect_bottom_points.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size))

    # Other rectangle corner points (top-right and top-left)
    rp3 = gmsh.model.geo.addPoint(rx2, ry2, 0, mesh_size)
    rp4 = gmsh.model.geo.addPoint(rx1, ry2, 0, mesh_size)

    # Rectangle lines
    rect_lines = []
    # Bottom (perturbed)
    for i in range(n_points_rect_bottom - 1):
        rect_lines.append(gmsh.model.geo.addLine(rect_bottom_points[i], rect_bottom_points[i+1]))
    # Right
    rect_lines.append(gmsh.model.geo.addLine(rect_bottom_points[-1], rp3))
    # Top
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
    gmsh.model.addPhysicalGroup(1, rect_lines, obstacle_marker, "exp_perturbed_rect_boundary")
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
    n_points_bottom=100, n_points_rect_bottom=40,
    use_opt_marker=True
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
    rl1 = gmsh.model.geo.addLine(rp1, rp2)  # Bottom edge (to be optimized)
    rl2 = gmsh.model.geo.addLine(rp2, rp3)  # Right edge
    rl3 = gmsh.model.geo.addLine(rp3, rp4)  # Top edge
    rl4 = gmsh.model.geo.addLine(rp4, rp1)  # Left edge
    
    # Separate bottom edge from other edges
    rect_bottom_line = [rl1]
    rect_other_lines = [rl2, rl3, rl4]
    
    gmsh.model.geo.mesh.setTransfiniteCurve(rl1, n_points_rect_bottom)

    # Curve loops
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    rect_loop = gmsh.model.geo.addCurveLoop([rl1, rl2, rl3, rl4])

    # Plane surface with rectangle obstacle
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, rect_loop])

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [l1], bottom_wall_marker, "bottom_wall")
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4], side_wall_marker, "outer_walls")
    
    if use_opt_marker:
        # Separate marker for the bottom of the obstacle
        gmsh.model.addPhysicalGroup(1, rect_other_lines, obstacle_marker, "rect_obstacle_boundary")
        gmsh.model.addPhysicalGroup(1, rect_bottom_line, obstacle_opt_marker, "rect_obstacle_bottom")
    else:
        # Single marker for the entire obstacle boundary
        all_rect_lines = rect_bottom_line + rect_other_lines
        gmsh.model.addPhysicalGroup(1, all_rect_lines, obstacle_marker, "rect_obstacle_boundary")

    gmsh.model.addPhysicalGroup(2, [surface], domain_marker, "domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(f"{output_name}.msh")
    gmsh.finalize()
    return f"{output_name}.msh"

def generate_square_with_sin_perturbed_rect_obstacle_mesh(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=0.05,
    output_name="square_with_sin_perturbed_rect_obstacle",
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
        y = ry1 + perturb_amplitude * np.sin(2 * perturb_frequency * np.pi * t)
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

def generate_square_with_gaussian_perturbed_rect_mesh(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=0.05,
    output_name="square_with_gaussian_perturbed_rect",
    n_points_bottom=100, n_points_rect_bottom=100,
    perturbations=None  # Expects a list of dicts: [{'amplitude': A, 'center': C, 'std_dev': S}, ...]
):
    import gmsh
    import numpy as np

    # Default to a single wave packet if none are provided
    if perturbations is None:
        perturbations = [{'amplitude': 0.03, 'center': 0.5, 'std_dev': 0.1}]

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_gaussian_perturbed_rect")

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

    # Perturbed bottom edge by summing multiple Gaussian wave packets
    rect_bottom_points = []
    for i in range(n_points_rect_bottom):
        t = i / (n_points_rect_bottom - 1)  # Normalized position [0, 1]
        x = rx1 + t * (rx2 - rx1)
        
        # Sum contributions from all specified wave packets
        total_gaussian_offset = 0
        for p in perturbations:
            amp = p.get('amplitude', 0.0)
            center = p.get('center', 0.5)
            std_dev = p.get('std_dev', 0.1)
            if std_dev > 1e-9:  # Avoid division by zero for safety
                total_gaussian_offset += amp * np.exp(-((t - center)**2) / (2 * std_dev**2))
            
        y = ry1 + total_gaussian_offset
        rect_bottom_points.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size))

    # Other rectangle corner points (top-right and top-left)
    rp3 = gmsh.model.geo.addPoint(rx2, ry2, 0, mesh_size)
    rp4 = gmsh.model.geo.addPoint(rx1, ry2, 0, mesh_size)

    # Rectangle lines
    rect_lines = []
    # Bottom (perturbed) edge from the generated points
    for i in range(n_points_rect_bottom - 1):
        rect_lines.append(gmsh.model.geo.addLine(rect_bottom_points[i], rect_bottom_points[i+1]))
    
    # Right edge (from end of perturbed bottom to top-right)
    rect_lines.append(gmsh.model.geo.addLine(rect_bottom_points[-1], rp3))
    
    # Top edge
    rect_lines.append(gmsh.model.geo.addLine(rp3, rp4))
    
    # Left edge (from top-left to start of perturbed bottom)
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
    gmsh.model.addPhysicalGroup(1, rect_lines, obstacle_marker, "gaussian_perturbed_rect_boundary")
    gmsh.model.addPhysicalGroup(2, [surface], domain_marker, "domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(f"{output_name}.msh")
    gmsh.finalize()
    return f"{output_name}.msh"

def generate_square_with_cos_perturbed_rect_obstacle_mesh(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=0.05,
    output_name="square_with_sin_perturbed_rect_obstacle",
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
        y = ry1 + perturb_amplitude * (np.cos(2 * perturb_frequency * np.pi * t))
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

def generate_square_with_cos_bump_rect_obstacle_mesh(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=0.05,
    bump_w=0.08, bump_h=0.05,
    output_name="square_with_cos_bump_rect_obstacle",
    n_points_bottom=100, n_points_rect_bottom=100,
    seed=None
):
    import gmsh
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_cos_bump_rect_obstacle")

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

    gmsh.model.geo.mesh.setTransfiniteCurve(l1, n_points_bottom)

    # Rectangle obstacle center
    cx, cy = width/2, height/2
    rx1 = cx - rect_w/2
    rx2 = cx + rect_w/2
    ry1 = cy - rect_h/2
    ry2 = cy + rect_h/2

    # Random bump position (ensure bump stays within obstacle)
    bump_x_min = rx1
    bump_x_max = rx2 - bump_w
    bump_x = np.random.uniform(bump_x_min, bump_x_max)

    # Discretize the bottom edge of the rectangle
    rect_bottom_points = []
    for i in range(n_points_rect_bottom):
        t = i / (n_points_rect_bottom - 1)
        x = rx1 + t * (rx2 - rx1)
        # Cosine bump
        if bump_x <= x <= bump_x + bump_w:
            s = (x - bump_x) / bump_w  # normalized [0,1] in bump region
            y = ry1 - bump_h * (1 - np.cos(2 * np.pi * s))
        else:
            y = ry1
        rect_bottom_points.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size))

    # Other rectangle points (top-right and top-left)
    rp3 = gmsh.model.geo.addPoint(rx2, ry2, 0, mesh_size)
    rp4 = gmsh.model.geo.addPoint(rx1, ry2, 0, mesh_size)

    # Rectangle lines
    rect_lines = []
    # Bottom (with bump)
    for i in range(n_points_rect_bottom - 1):
        rect_lines.append(gmsh.model.geo.addLine(rect_bottom_points[i], rect_bottom_points[i+1]))
    # Right
    rect_lines.append(gmsh.model.geo.addLine(rect_bottom_points[-1], rp3))
    # Top
    rect_lines.append(gmsh.model.geo.addLine(rp3, rp4))
    # Left
    rect_lines.append(gmsh.model.geo.addLine(rp4, rect_bottom_points[0]))

    # Curve loops
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    rect_loop = gmsh.model.geo.addCurveLoop(rect_lines)

    # Plane surface with rectangle obstacle + smooth bump
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, rect_loop])

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [l1], bottom_wall_marker, "bottom_wall")
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4], side_wall_marker, "outer_walls")
    gmsh.model.addPhysicalGroup(1, rect_lines, obstacle_marker, "rect_obstacle_cos_bump_boundary")
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
    mesh_size = wavelength / 4

    """
    mesh_file = generate_square_with_eccentric_hole_mesh(
        width=1.0,
        height=1.0,
        hole_radius=0.2,
        mesh_size=mesh_size,
        output_name="square_with_eccentric_hole",
        n_circle=40,
        n_points_bottom=100,
        eccentricity_x=1.1,
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
    

    mesh_file = generate_square_with_rect_obstacle_mesh(
        width=1.0,
        height=1.0,
        rect_w=0.4,
        rect_h=0.2,
        mesh_size=mesh_size,
        output_name="meshes/square_with_rect_obstacle",
        n_points_bottom=100,
        n_points_rect_bottom = 100
    )


    mesh_file = generate_square_with_perturbed_rect_obstacle_mesh(
        width=1.0,
        height=1.0,
        rect_w=0.4,
        rect_h=0.2,
        mesh_size=mesh_size,
        output_name="meshes/square_with_perturbed_rect_obstacle",
        n_points_bottom=100,
        n_points_rect_bottom=100,
        perturb_amplitude=0.01,
        perturb_frequency=3
    )

    mesh_file = generate_square_with_flattened_circle_mesh(
    width=1.0,
    height=1.0,
    hole_radius=0.2,
    mesh_size=mesh_size,
    output_name="meshes/square_with_flattened_circle",
    n_circle=100,
    n_points_bottom=100,
    flatten_y=0.35
    )

    mesh_file = generate_square_with_sin_perturbed_circle_mesh(
    width=1.0,
    height=1.0,
    hole_radius=0.2,
    mesh_size=mesh_size,
    output_name="meshes/square_with_sin_perturbed_circle",
    n_circle=100,
    n_points_bottom=100,
    perturb_amplitude=0.01,
    perturb_frequency=3)

    mesh_file = generate_square_with_hole_mesh(
        width=1.0,
        height=1.0,
        hole_radius=0.2,
        mesh_size=mesh_size,
        output_name="square_with_hole",
        n_circle=200, n_points_bottom=100
    )


    """

    
    mesh_file = generate_square_with_sin_perturbed_rect_obstacle_mesh(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=mesh_size,
    output_name="meshes/square_with_sin_perturbed_rect_obstacle",
    n_points_bottom=100, n_points_rect_bottom=100,
    perturb_amplitude=0.02, perturb_frequency=1.0)

    
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_mesh(mesh_file, ax, title="Mesh")
    plt.savefig("meshpic.png")
    plt.show()
