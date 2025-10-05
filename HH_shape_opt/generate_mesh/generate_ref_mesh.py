from .mesh_util import SIDE_WALL_MARKER, RECEIVER_EDGE_MARKER, OBSTACLE_MARKER, OBSTACLE_DOMAIN_MARKER, DOMAIN_MARKER
import meshio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import gmsh

def generate_square_with_sin_perturbed_rect_obstacle_mesh(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=0.05,
    output_name="square_with_sin_perturbed_rect_obstacle",
    n_points_bottom=100, n_points_rect_bottom=40,
    perturb_amplitude=0.03, perturb_frequency=3,
    perturb_top=False, top_perturb_amplitude=0.03, top_perturb_frequency=3
):

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
    cx, cy = width / 2, height / 2
    rx1 = cx - rect_w / 2
    rx2 = cx + rect_w / 2
    ry1 = cy - rect_h / 2
    ry2 = cy + rect_h / 2

    # Perturbed bottom edge of rectangle
    rect_bottom_points = []
    for i in range(n_points_rect_bottom):
        t = i / (n_points_rect_bottom - 1)
        x = rx1 + t * (rx2 - rx1)
        y = ry1 + perturb_amplitude * np.sin(2 * perturb_frequency * np.pi * t)
        rect_bottom_points.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size))

    # Right edge point
    rp2 = gmsh.model.geo.addPoint(rx2, ry2, 0, mesh_size)

    # Perturbed top edge of rectangle (if enabled)
    rect_top_points = []
    if perturb_top:
        for i in range(n_points_rect_bottom):
            t = i / (n_points_rect_bottom - 1)
            x = rx2 - t * (rx2 - rx1)  # Going from right to left
            y = ry2 + top_perturb_amplitude * \
                np.sin(2 * top_perturb_frequency * np.pi * t)
            rect_top_points.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size))
    else:
        # Non-perturbed top edge
        rp3 = gmsh.model.geo.addPoint(rx1, ry2, 0, mesh_size)

    # Left edge point
    rp4 = gmsh.model.geo.addPoint(rx1, ry1, 0, mesh_size)

    # Rectangle lines
    rect_lines = []

    # Bottom (perturbed)
    for i in range(n_points_rect_bottom - 1):
        rect_lines.append(gmsh.model.geo.addLine(
            rect_bottom_points[i], rect_bottom_points[i + 1]))

    # Right edge
    if perturb_top:
        rect_lines.append(gmsh.model.geo.addLine(
            rect_bottom_points[-1], rect_top_points[0]))
    else:
        rect_lines.append(gmsh.model.geo.addLine(rect_bottom_points[-1], rp2))

    # Top edge
    if perturb_top:
        # Top (perturbed)
        for i in range(n_points_rect_bottom - 1):
            rect_lines.append(gmsh.model.geo.addLine(
                rect_top_points[i], rect_top_points[i + 1]))
        # Left edge
        rect_lines.append(gmsh.model.geo.addLine(
            rect_top_points[-1], rect_bottom_points[0]))
    else:
        # Top (non-perturbed)
        rect_lines.append(gmsh.model.geo.addLine(rp2, rp3))
        # Left edge
        rect_lines.append(gmsh.model.geo.addLine(rp3, rect_bottom_points[0]))

    # Curve loops
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    rect_loop = gmsh.model.geo.addCurveLoop(rect_lines)

    # Plane surface with perturbed rectangle obstacle
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, rect_loop])

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [l1], RECEIVER_EDGE_MARKER, "bottom_wall")
    gmsh.model.addPhysicalGroup(
        1, [l2, l3, l4], SIDE_WALL_MARKER, "outer_walls")
    gmsh.model.addPhysicalGroup(
        1, rect_lines, OBSTACLE_MARKER, "perturbed_rect_obstacle_boundary")
    gmsh.model.addPhysicalGroup(2, [surface], DOMAIN_MARKER, "domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(f"{output_name}.msh")
    gmsh.finalize()
    return f"{output_name}.msh"


def generate_square_with_cos_perturbed_rect_obstacle_mesh(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=0.05,
    output_name="square_with_cos_perturbed_rect_obstacle",
    n_points_bottom=100, n_points_rect_bottom=40,
    perturb_amplitude=0.03, perturb_frequency=3
):

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
    cx, cy = width / 2, height / 2
    rx1 = cx - rect_w / 2
    rx2 = cx + rect_w / 2
    ry1 = cy - rect_h / 2
    ry2 = cy + rect_h / 2

    # Perturbed bottom edge of rectangle
    rect_bottom_points = []
    for i in range(n_points_rect_bottom):
        t = i / (n_points_rect_bottom - 1)
        x = rx1 + t * (rx2 - rx1)
        y = ry1 + perturb_amplitude * np.cos(2 * perturb_frequency * np.pi * t)
        rect_bottom_points.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size))

    # Other rectangle points (no perturbation)
    rp2 = gmsh.model.geo.addPoint(rx2, ry1, 0, mesh_size)
    rp3 = gmsh.model.geo.addPoint(rx2, ry2, 0, mesh_size)
    rp4 = gmsh.model.geo.addPoint(rx1, ry2, 0, mesh_size)

    # Rectangle lines
    rect_lines = []
    # Bottom (perturbed)
    for i in range(n_points_rect_bottom - 1):
        rect_lines.append(gmsh.model.geo.addLine(
            rect_bottom_points[i], rect_bottom_points[i + 1]))
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
    gmsh.model.addPhysicalGroup(1, [l1], RECEIVER_EDGE_MARKER, "bottom_wall")
    gmsh.model.addPhysicalGroup(
        1, [l2, l3, l4], SIDE_WALL_MARKER, "outer_walls")
    gmsh.model.addPhysicalGroup(
        1, rect_lines, OBSTACLE_MARKER, "perturbed_rect_obstacle_boundary")
    gmsh.model.addPhysicalGroup(2, [surface], DOMAIN_MARKER, "domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(f"{output_name}.msh")
    gmsh.finalize()
    return f"{output_name}.msh"

def generate_square_with_meshed_sin_perturbed_rect_obstacle(
    width=1.0, height=1.0, rect_w=0.4, rect_h=0.2, mesh_size=0.05,
    output_name="square_with_meshed_sin_perturbed_rect_obstacle",
    n_points_bottom=100, n_points_rect_bottom=40,
    perturb_amplitude=0.03, perturb_frequency=3,
    perturb_top=False, top_perturb_amplitude=0.03, top_perturb_frequency=3,
    use_opt_marker=False
):

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("square_with_meshed_sin_perturbed_rect_obstacle")

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
    cx, cy = width / 2, height / 2
    rx1 = cx - rect_w / 2
    rx2 = cx + rect_w / 2
    ry1 = cy - rect_h / 2
    ry2 = cy + rect_h / 2

    # Perturbed bottom edge of rectangle
    rect_bottom_points = []
    for i in range(n_points_rect_bottom):
        t = i / (n_points_rect_bottom - 1)
        x = rx1 + t * (rx2 - rx1)
        y = ry1 + perturb_amplitude * np.sin(2 * perturb_frequency * np.pi * t)
        rect_bottom_points.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size))

    # Right edge point
    rp2 = gmsh.model.geo.addPoint(rx2, ry2, 0, mesh_size)

    # Perturbed top edge of rectangle (if enabled)
    rect_top_points = []
    if perturb_top:
        for i in range(n_points_rect_bottom):
            t = i / (n_points_rect_bottom - 1)
            x = rx2 - t * (rx2 - rx1)  # Going from right to left
            y = ry2 + top_perturb_amplitude * \
                np.sin(2 * top_perturb_frequency * np.pi * t)
            rect_top_points.append(gmsh.model.geo.addPoint(x, y, 0, mesh_size))
    else:
        # Non-perturbed top edge
        rp3 = gmsh.model.geo.addPoint(rx1, ry2, 0, mesh_size)

    # Left edge point
    rp4 = gmsh.model.geo.addPoint(rx1, ry1, 0, mesh_size)

    # Rectangle lines
    rect_lines = []

    # Bottom (perturbed)
    bottom_lines = []
    for i in range(n_points_rect_bottom - 1):
        line = gmsh.model.geo.addLine(
            rect_bottom_points[i], rect_bottom_points[i + 1])
        rect_lines.append(line)
        bottom_lines.append(line)

    # Right edge
    if perturb_top:
        right_line = gmsh.model.geo.addLine(
            rect_bottom_points[-1], rect_top_points[0])
    else:
        right_line = gmsh.model.geo.addLine(rect_bottom_points[-1], rp2)
    rect_lines.append(right_line)

    # Top edge
    top_lines = []
    if perturb_top:
        # Top (perturbed)
        for i in range(n_points_rect_bottom - 1):
            line = gmsh.model.geo.addLine(
                rect_top_points[i], rect_top_points[i + 1])
            rect_lines.append(line)
            top_lines.append(line)
        # Left edge
        left_line = gmsh.model.geo.addLine(
            rect_top_points[-1], rect_bottom_points[0])
    else:
        # Top (non-perturbed)
        top_line = gmsh.model.geo.addLine(rp2, rp3)
        rect_lines.append(top_line)
        top_lines.append(top_line)
        # Left edge
        left_line = gmsh.model.geo.addLine(rp3, rect_bottom_points[0])
    rect_lines.append(left_line)

    # Curve loops
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    rect_loop = gmsh.model.geo.addCurveLoop(rect_lines)

    # Create TWO separate, adjacent surfaces:
    # 1. Background domain (outer square with a hole for the obstacle)
    background_surface = gmsh.model.geo.addPlaneSurface([outer_loop, rect_loop])

    # 2. Obstacle domain (the inner perturbed rectangle)
    obstacle_surface = gmsh.model.geo.addPlaneSurface([rect_loop])

    gmsh.model.geo.synchronize()

    # Physical groups for boundaries
    gmsh.model.addPhysicalGroup(1, [l1], RECEIVER_EDGE_MARKER, "bottom_wall")
    gmsh.model.addPhysicalGroup(
        1, [l2, l3, l4], SIDE_WALL_MARKER, "outer_walls")
    
    if use_opt_marker:
        # Mark the bottom of the obstacle separately for optimization
        gmsh.model.addPhysicalGroup(
            1, bottom_lines, OBSTACLE_OPT_MARKER, "perturbed_rect_obstacle_opt_boundary")
        # Mark the rest of the obstacle
        other_lines = [right_line] + top_lines + [left_line]
        gmsh.model.addPhysicalGroup(
            1, other_lines, OBSTACLE_MARKER, "perturbed_rect_obstacle_boundary")
    else:
        # Mark the entire obstacle with one marker
        gmsh.model.addPhysicalGroup(
            1, rect_lines, OBSTACLE_MARKER, "perturbed_rect_obstacle_boundary")
    
    # Physical groups for domains (2D regions)
    gmsh.model.addPhysicalGroup(2, [background_surface], DOMAIN_MARKER, "background_domain")
    gmsh.model.addPhysicalGroup(2, [obstacle_surface], OBSTACLE_DOMAIN_MARKER, "obstacle_domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(f"{output_name}.msh")
    gmsh.finalize()
    return f"{output_name}.msh"