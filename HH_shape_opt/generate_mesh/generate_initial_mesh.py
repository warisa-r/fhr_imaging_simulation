import meshio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import gmsh

from .mesh_util import SIDE_WALL_MARKER, RECEIVER_EDGE_MARKER, OBSTACLE_MARKER, OBSTACLE_OPT_MARKER, DOMAIN_MARKER
from .mesh_util import calculate_mesh_size


def generate_square_with_rect_obstacle_mesh(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=0.05,
    output_name="square_with_rect_obstacle",
    n_points_bottom=100, n_points_rect_bottom=40,
    use_opt_marker=False
):

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_rect_obstacle")

    # Outer square points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)         # Bottom-left
    p2 = gmsh.model.geo.addPoint(width, 0, 0, mesh_size)     # Bottom-right
    p3 = gmsh.model.geo.addPoint(width, height, 0, mesh_size)  # Top-right
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
    gmsh.model.addPhysicalGroup(1, [l1], RECEIVER_EDGE_MARKER, "bottom_wall")
    gmsh.model.addPhysicalGroup(
        1, [l2, l3, l4], SIDE_WALL_MARKER, "outer_walls")
    if use_opt_marker:
        # Mark the bottom of the obstacle separately for optimization
        gmsh.model.addPhysicalGroup(
            1, [rl1], OBSTACLE_OPT_MARKER, "obstacle_opt_boundary")
        # Mark the rest of the obstacle
        gmsh.model.addPhysicalGroup(
            1, [rl2, rl3, rl4], OBSTACLE_MARKER, "rect_obstacle_boundary")
    else:
        # Mark the entire obstacle with one marker
        gmsh.model.addPhysicalGroup(
            1, rect_lines, OBSTACLE_MARKER, "rect_obstacle_boundary")
    gmsh.model.addPhysicalGroup(2, [surface], DOMAIN_MARKER, "domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(f"{output_name}.msh")
    gmsh.finalize()
    return f"{output_name}.msh"

#TODO: (low priority finish this)
class MeshGenerator():
    def __init__(signal_csv_file, freq_max = 5e9, num_mesh_points_per_wavelength = 5):
        mesh_size = calculate_mesh_size(freq_max, num_mesh_points_per_wavelength)
        # The csv must have categories 'x', 'y', and 'u'
        df = pd.read_csv(measurement_data_file_path)
        points = df[["x", "y"]].values

        # Assert points are approximately equidistant and within mesh_size constraint
        self._validate_point_spacing(points, mesh_size)
        
        #TODO: Define domain and generate mesh
        return
    
    def _validate_point_spacing(self, points, mesh_size, tolerance=1e-12):
        if len(points) < 2:
            return  # Can't validate spacing with less than 2 points
        
        # Calculate distances between consecutive points
        distances = []
        for i in range(len(points) - 1):
            dist = np.sqrt((points[i+1][0] - points[i][0])**2 + 
                          (points[i+1][1] - points[i][1])**2)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Assert all distances are <= mesh_size
        assert np.all(distances <= mesh_size), \
            f"Some point distances exceed mesh_size: max distance = {np.max(distances):.2e}, mesh_size = {mesh_size:.2e}"
        
        # Assert points are approximately equidistant (within tolerance)
        if len(distances) > 1:
            mean_distance = np.mean(distances)
            max_deviation = np.max(np.abs(distances - mean_distance))
            assert max_deviation <= tolerance, \
                f"Points are not equidistant within tolerance {tolerance:.2e}: max deviation = {max_deviation:.2e}"
        
        print(f"Point spacing validation passed: {len(distances)} segments, "
              f"mean distance = {np.mean(distances):.6e}, max deviation = {np.max(np.abs(distances - np.mean(distances))):.2e}")

def generate_square_with_meshed_rect_obstacle(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=0.05,
    output_name="square_with_meshed_rect_obstacle",
    n_points_bottom=100, n_points_rect_bottom=40,
    use_opt_marker=False
):
    # Obstacle with which refraction can still happen
    
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_meshed_rect_obstacle")

    # Outer square points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)         # Bottom-left
    p2 = gmsh.model.geo.addPoint(width, 0, 0, mesh_size)     # Bottom-right
    p3 = gmsh.model.geo.addPoint(width, height, 0, mesh_size)  # Top-right
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
    rl1 = gmsh.model.geo.addLine(rp1, rp2)  # Bottom
    rl2 = gmsh.model.geo.addLine(rp2, rp3)  # Right
    rl3 = gmsh.model.geo.addLine(rp3, rp4)  # Top
    rl4 = gmsh.model.geo.addLine(rp4, rp1)  # Left
    rect_lines = [rl1, rl2, rl3, rl4]

    gmsh.model.geo.mesh.setTransfiniteCurve(rl1, n_points_rect_bottom)

    # Create curve loops
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    rect_loop = gmsh.model.geo.addCurveLoop(rect_lines)

    # Create TWO separate surfaces:
    # 1. Background domain (outer square minus rectangle)
    background_surface = gmsh.model.geo.addPlaneSurface([outer_loop, rect_loop])
    
    # 2. Obstacle domain (rectangle)
    obstacle_surface = gmsh.model.geo.addPlaneSurface([rect_loop])

    gmsh.model.geo.synchronize()

    # Physical groups for boundaries
    gmsh.model.addPhysicalGroup(1, [l1], RECEIVER_EDGE_MARKER, "bottom_wall")
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4], SIDE_WALL_MARKER, "outer_walls")
    
    if use_opt_marker:
        # Mark the bottom of the obstacle separately for optimization
        gmsh.model.addPhysicalGroup(1, [rl1], OBSTACLE_OPT_MARKER, "obstacle_opt_boundary")
        # Mark the rest of the obstacle boundary
        gmsh.model.addPhysicalGroup(1, [rl2, rl3, rl4], OBSTACLE_MARKER, "obstacle_boundary")
    else:
        # Mark the entire obstacle boundary with one marker
        gmsh.model.addPhysicalGroup(1, rect_lines, OBSTACLE_MARKER, "obstacle_boundary")

    # Physical groups for domains (2D regions)
    gmsh.model.addPhysicalGroup(2, [background_surface], DOMAIN_MARKER, "background_domain")
    gmsh.model.addPhysicalGroup(2, [obstacle_surface], OBSTACLE_MARKER, "obstacle_domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(f"{output_name}.msh")
    gmsh.finalize()
    return f"{output_name}.msh"

def generate_square_with_rect_obstacle_and_receiver_segments(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=0.05,
    output_name="square_with_rect_obstacle_receivers",
    n_points_bottom=100, n_points_rect_bottom=40,
    receiver_segments=[(0.2, 0.3), (0.7, 0.8)],  # List of (x_start, x_end) tuples
    use_opt_marker=False
):
    """
    Generate mesh with specific receiver segments on the bottom wall.
    
    receiver_segments: List of (x_start, x_end) tuples defining receiver locations
    enforce_two_points_per_receiver: If True, force exactly 2 mesh points per receiver segment
    """
    from .mesh_util import BOTTOM_WALL_MARKER, RECEIVER_SEGMENT_MARKER
    
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_rect_obstacle_receivers")

    # Outer square corner points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)      # Bottom-left
    p2 = gmsh.model.geo.addPoint(width, 0, 0, mesh_size)  # Bottom-right
    p3 = gmsh.model.geo.addPoint(width, height, 0, mesh_size)  # Top-right
    p4 = gmsh.model.geo.addPoint(0, height, 0, mesh_size)      # Top-left

    # Sort receiver segments
    sorted_segments = sorted(receiver_segments)
    
    # Create all points along the bottom wall
    bottom_points = [p1]
    all_x_coords = [0.0]
    
    for x_start, x_end in sorted_segments:
        if x_start not in all_x_coords:
            all_x_coords.append(x_start)
        if x_end not in all_x_coords:
            all_x_coords.append(x_end)
    
    if width not in all_x_coords:
        all_x_coords.append(width)
    
    all_x_coords = sorted(set(all_x_coords))
    
    # Create points for all unique x coordinates (except first which is p1)
    x_to_point = {0.0: p1, width: p2}
    for x in all_x_coords[1:-1]:  # Skip 0 and width
        pt = gmsh.model.geo.addPoint(x, 0, 0, mesh_size)
        x_to_point[x] = pt
    
    # Create lines and categorize them
    all_bottom_lines = []
    receiver_lines = []
    
    for i in range(len(all_x_coords) - 1):
        x_start = all_x_coords[i]
        x_end = all_x_coords[i + 1]
        
        p_start = x_to_point[x_start]
        p_end = x_to_point[x_end]
        
        line = gmsh.model.geo.addLine(p_start, p_end)
        all_bottom_lines.append(line)
        
        # Check if this segment is a receiver
        is_receiver = False
        for rx_start, rx_end in sorted_segments:
            if abs(x_start - rx_start) < 1e-10 and abs(x_end - rx_end) < 1e-10:
                receiver_lines.append(line)
                is_receiver = True
                # Force the line to have only 2 points so that there is 1 midpoint to integrate over
                gmsh.model.geo.mesh.setTransfiniteCurve(line, 2)
                break

    # Other walls
    l2 = gmsh.model.geo.addLine(p2, p3)  # Right
    l3 = gmsh.model.geo.addLine(p3, p4)  # Top
    l4 = gmsh.model.geo.addLine(p4, p1)  # Left

    # Rectangle obstacle
    cx, cy = width/2, height/2
    rx1 = cx - rect_w/2
    rx2 = cx + rect_w/2
    ry1 = cy - rect_h/2
    ry2 = cy + rect_h/2

    rp1 = gmsh.model.geo.addPoint(rx1, ry1, 0, mesh_size)
    rp2 = gmsh.model.geo.addPoint(rx2, ry1, 0, mesh_size)
    rp3 = gmsh.model.geo.addPoint(rx2, ry2, 0, mesh_size)
    rp4 = gmsh.model.geo.addPoint(rx1, ry2, 0, mesh_size)

    rl1 = gmsh.model.geo.addLine(rp1, rp2)
    rl2 = gmsh.model.geo.addLine(rp2, rp3)
    rl3 = gmsh.model.geo.addLine(rp3, rp4)
    rl4 = gmsh.model.geo.addLine(rp4, rp1)
    rect_lines = [rl1, rl2, rl3, rl4]

    # Curve loops
    outer_loop = gmsh.model.geo.addCurveLoop(all_bottom_lines + [l2, l3, l4])
    rect_loop = gmsh.model.geo.addCurveLoop(rect_lines)

    # Surface
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, rect_loop])

    gmsh.model.geo.synchronize()

    # Physical groups
    # Mark entire bottom wall (all segments)
    gmsh.model.addPhysicalGroup(1, all_bottom_lines, BOTTOM_WALL_MARKER, "bottom_wall")
    
    # Mark only receiver segments
    if receiver_lines:
        # Assign individual markers to each receiver segment for isolated patches
        for idx, rx_line in enumerate(receiver_lines):
            marker_id = RECEIVER_SEGMENT_MARKER + idx
            gmsh.model.addPhysicalGroup(1, [rx_line], marker_id, f"receiver_segment_{idx}")
    
    # Mark side walls
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4], SIDE_WALL_MARKER, "outer_walls")
    
    # Mark obstacle
    if use_opt_marker:
        gmsh.model.addPhysicalGroup(1, [rl1], OBSTACLE_OPT_MARKER, "obstacle_opt_boundary")
        gmsh.model.addPhysicalGroup(1, [rl2, rl3, rl4], OBSTACLE_MARKER, "rect_obstacle_boundary")
    else:
        gmsh.model.addPhysicalGroup(1, rect_lines, OBSTACLE_MARKER, "rect_obstacle_boundary")
    
    gmsh.model.addPhysicalGroup(2, [surface], DOMAIN_MARKER, "domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(f"{output_name}.msh")
    gmsh.finalize()
    return f"{output_name}.msh"

if __name__ == "__main__":
    print("Generating square with hole mesh...")

    c = 299792458
    freq_max = 5e9  # 5GHz

    # Parameters
    wavelength = c / freq_max  # Physical wavelength
    mesh_size = wavelength / 5

    """
    mesh_file = generate_square_with_cos_perturbed_rect_obstacle_mesh(
        width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=mesh_size,
        output_name="meshes/square_with_perturbed_rect_obstacle",
        n_points_bottom=100, n_points_rect_bottom=100,
        perturb_amplitude=0.01, perturb_frequency=1.5
    )
    """

    """
    mesh_file = generate_square_with_sin_perturbed_rect_obstacle_mesh(
        width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=mesh_size,
        output_name="square_with_halfsin2_perturbed_rect_obstacle",
        n_points_bottom=100, n_points_rect_bottom=100,
        perturb_amplitude=0.02, perturb_frequency=0.5
    )
    """

    """
    mesh_file =  generate_square_with_rect_obstacle_mesh(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=mesh_size,
    output_name="meshes/square_with_rect_obstacle_opt",
    n_points_bottom=100, n_points_rect_bottom=100,
    use_opt_marker = True
    )
    """

    """
    convert_msh_to_xdmf(mesh_file)
    """