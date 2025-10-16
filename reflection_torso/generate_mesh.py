from HH_shape_opt import *
import meshio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import gmsh

# Generate solution mesh
def generate_torso_sim_mesh(
    boundary_point_coordinate, torso_coordinate_csv, ant_pos_csv = None, mesh_size=0.05,
    output_name="torso_sim_mesh_solution"
):

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("torso_sim_mesh_solution")

    # Read boundary points from CSV
    df_boundary_edge_points = pd.read_csv(boundary_point_coordinate)
    points = df_boundary_edge_points[["x", "y"]].values

    ps = []
    # Create boundary points
    for point in points:
        p = gmsh.model.geo.addPoint(point[0], point[1], 0, mesh_size)
        ps.append(p)

    ls = []
    # Create boundary lines (connecting consecutive points)
    n_points = len(ps)
    for i in range(n_points):
        if i < n_points - 1:
            l = gmsh.model.geo.addLine(ps[i], ps[i+1])
        else:
            # Close the loop
            l = gmsh.model.geo.addLine(ps[i], ps[0])
        ls.append(l)

    # Create a simple dummy obstacle (small circle/rectangle in center)
    # Calculate center of domain
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    cx = (x_coords.min() + x_coords.max()) / 2
    cy = (y_coords.min() + y_coords.max()) / 2
    
    # Create obstacle
    df_obstacle = pd.read_csv(torso_coordinate_csv)
    obstacle_ps = []
    obstacle_points = df_obstacle[["x", "y"]].values

    for obstacle_point in obstacle_points:
        obstacle_p = gmsh.model.geo.addPoint(obstacle_point[0], obstacle_point[1], 0, mesh_size)
        obstacle_ps.append(obstacle_p)

    obstacle_ls = []
    # Create obstacle boundary lines (connecting consecutive points)
    n_points = len(obstacle_ps)
    for i in range(n_points):
        if i < n_points - 1:
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

    # Physical groups
    gmsh.model.addPhysicalGroup(1, ls, SIDE_WALL_MARKER, "outer_walls")
    gmsh.model.addPhysicalGroup(1, obstacle_ls, OBSTACLE_MARKER, "dummy_obstacle_boundary")
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
    # Generate and visualize mesh
    mesh_file = generate_torso_sim_mesh("meshes/boundary_points.csv", 
                "meshes/torso_points.csv", mesh_size = mesh_size)
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_mesh(mesh_file, ax, title="Torso Domain with Dummy Obstacle")
    plt.tight_layout()
    plt.savefig("torso_mesh.png", dpi=150)
    print("Mesh visualization saved to torso_mesh.png")
    plt.show()
    plt.close()