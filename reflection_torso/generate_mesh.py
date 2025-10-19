from HH_shape_opt import *
import meshio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import gmsh
import math

# Generate solution mesh
def generate_torso_sim_mesh(
    boundary_point_coordinate, torso_coordinate_csv, ant_pos_csv, mesh_size=0.05,
    output_name="torso_sim_mesh_solution"
):

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("torso_sim_mesh_solution")

    # Read receiver positions from csv
    df_ant_pos = pd.read_csv(ant_pos_csv)
    # Read only from the fist transmitter to reduce repetition
    df_ant_pos = df_ant_pos.loc[df_ant_pos["tx_x"] == -0.1]
    receiver_points = df_ant_pos[["rx_x", "rx_y"]].values

    # Identify sets of consecutive receiver positions by distance
    receiver_patches = []
    if len(receiver_points) > 1: # Of course
        # Calculate typical distance between consecutive receivers
        receiver_dist = np.linalg.norm(receiver_points[1] - receiver_points[0], 2)
        distance_threshold = receiver_dist
        
        current_patch = [0]  # Start first patch with first receiver
        
        for rx_i in range(1, len(receiver_points)):
            dist_to_prev = np.linalg.norm(receiver_points[rx_i] - receiver_points[rx_i-1], 2)
            
            if abs(dist_to_prev - distance_threshold) < 1e-3:
                # Still in same patch
                current_patch.append(rx_i)
            else:
                # Start new patch
                receiver_patches.append(current_patch)
                current_patch = [rx_i]
        
        # Don't forget the last patch
        receiver_patches.append(current_patch)
        print(receiver_patches)
    
    num_receiver_per_patches = len(receiver_patches[0]) # Assume equal for all patches

    # Read boundary points from CSV
    df_boundary_edge_points = pd.read_csv(boundary_point_coordinate)
    boundary_points = df_boundary_edge_points[["x", "y"]].values

    # Combine boundary points with receiver points, maintaining order
    # We need to insert receiver points into the boundary at their locations
    all_points = []
    point_to_gmsh_id = {}
    bp_receivers_indices = set()
    
    tolerance = 1e-6
    
    # Create a list to track which boundary points are receivers
    for i, bp in enumerate(boundary_points):
        # Check if this boundary point is a receiver point
        is_receiver = False
        for rp in receiver_points:
            if np.linalg.norm(bp - rp) < tolerance:
                is_receiver = True
                bp_receivers_indices.add(len(all_points))
                break
        all_points.append(bp)
    
    # Create gmsh points
    ps = []
    for point in all_points:
        p = gmsh.model.geo.addPoint(point[0], point[1], 0, mesh_size)
        ps.append(p)

    ls = []
    receiver_lines = []
    n_points = len(boundary_points)
    tolerance = 1e-6

    
    for i in range(n_points):
        next_i = (i + 1) % n_points
        
        # Special handling for the edge between i==1 and i==2
        if i == 1:
            # Create intermediate points at specific coordinates
            p_left = gmsh.model.geo.addPoint(-0.1, 0, 0, mesh_size)
            p_right = gmsh.model.geo.addPoint(0.1, 0, 0, mesh_size)
            
            # Create three line segments instead of one
            line1 = gmsh.model.geo.addLine(ps[i], p_left)
            line2 = gmsh.model.geo.addLine(p_left, p_right)
            line3 = gmsh.model.geo.addLine(p_right, ps[next_i])
            
            ls.extend([line1, line2, line3])
            
            gmsh.model.geo.mesh.setTransfiniteCurve(line2, 11)
            receiver_lines.append(line2)
        else:
            line = gmsh.model.geo.addLine(ps[i], ps[next_i])
            ls.append(line)
            if i == 0 or i == 2:
                gmsh.model.geo.mesh.setTransfiniteCurve(line, 11)
                receiver_lines.append(line)

    # Create obstacle
    df_obstacle = pd.read_csv(torso_coordinate_csv)
    obstacle_ps = []
    obstacle_points = df_obstacle[["x", "y"]].values

    for obstacle_point in obstacle_points:
        obstacle_p = gmsh.model.geo.addPoint(obstacle_point[0], obstacle_point[1], 0, mesh_size)
        obstacle_ps.append(obstacle_p)

    obstacle_ls = []
    n_obstacle_points = len(obstacle_ps)
    for i in range(n_obstacle_points):
        if i < n_obstacle_points - 1:
            obstacle_l = gmsh.model.geo.addLine(obstacle_ps[i], obstacle_ps[i+1])
        else:
            # Close the loop
            obstacle_l = gmsh.model.geo.addLine(obstacle_ps[i], obstacle_ps[0])
        obstacle_ls.append(obstacle_l)
    
    # Curve loops
    outer_loop = gmsh.model.geo.addCurveLoop(ls)
    obstacle_loop = gmsh.model.geo.addCurveLoop(obstacle_ls)

    # Plane surface with obstacle hole
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, obstacle_loop])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.removeDuplicateNodes()

    # Physical groups
    # Mark receiver edges separately
    if receiver_lines:
        gmsh.model.addPhysicalGroup(1, receiver_lines, RECEIVER_EDGE_MARKER, "receiver_edges")
    
    # Mark non-receiver outer walls
    non_receiver_lines = [l for l in ls if l not in receiver_lines]
    if non_receiver_lines:
        gmsh.model.addPhysicalGroup(1, non_receiver_lines, SIDE_WALL_MARKER, "outer_walls")
    
    gmsh.model.addPhysicalGroup(1, obstacle_ls, OBSTACLE_MARKER, "torso_boundary")
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
    # Generate and visualize mesh
    mesh_file = generate_torso_sim_mesh("meshes/boundary_points.csv",
                "meshes/torso_points.csv", "meshes/ant_pos_table.csv", mesh_size = mesh_size
                , output_name="meshes/torso_sim_mesh_solution")
    """

    # Generate an initial mesh
    mesh_file = generate_torso_sim_mesh("meshes/boundary_points.csv", 
            "meshes/torso_initial_points.csv", "meshes/ant_pos_table.csv"
            , mesh_size = mesh_size
            , output_name="meshes/torso_sim_mesh_initial")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_mesh(mesh_file, ax, title="Initial Torso Domain with Dummy Obstacle")
    plt.tight_layout()
    plt.savefig("torso_mesh_initial.png", dpi=150)
    print("Mesh visualization saved to torso_mesh_initial.png")
    plt.show()
    plt.close()

    convert_msh_to_xdmf(mesh_file)