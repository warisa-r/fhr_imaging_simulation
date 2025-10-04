import moola
import pandas as pd
import numpy as np

import subprocess
import os
import gmsh
import matplotlib.pyplot as plt

from HH_shape_opt import *

# Ensure this can be run from root dir
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

set_log_level(LogLevel.ERROR)

frequency = 5e9
inc_wave_setup = IncidentWaveSetup(frequency, plane_wave)

#measurement_data_file_path = "measurements/matlab_measurements_sin0.5_scatter.csv"
msh_file_path = "meshes/square_with_meshed_rect_obstacle.msh"
markers_dict = {
    "obstacle": OBSTACLE_MARKER, # Markers imported from our mesh generation module
    "side_wall": SIDE_WALL_MARKER,
    "bottom_wall": RECEIVER_EDGE_MARKER,
    "obstacle_opt": None
}
obstacle_stiffness = 25

initial_guess_mesh_util = MeshUtil(
    msh_file_path, markers_dict, obstacle_stiffness)
mesh, _ = initial_guess_mesh_util.get_mesh_and_markers()

##### Initialization #####
# Create boundary mesh and design variables
b_mesh = BoundaryMesh(mesh, "exterior")
S_b = VectorFunctionSpace(b_mesh, "CG", 1)
h = Function(S_b, name="Design")
h.vector()[:] = 0.0
h.vector().apply("insert")

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="Mesh perturbation field")
h_V = transfer_from_boundary(h, mesh)
h_V.rename("Volume extension of h", "")
##########################

# Solve the forward problem
u_tot_mag_dg0, u_tot_re_projected, u_tot_im_projected, ds_receiver, V_DG0 = forward_solve_refraction(
    h, inc_wave_setup, initial_guess_mesh_util, True)


##### Plot ######
# Extract mesh coordinates and cell connectivity for plotting
coords = mesh.coordinates()
cells = mesh.cells()

def plot_function(ax, f, title):
    # Evaluate f at cell centers (DG0 works fine, CG needs projection to DG0 first)
    V = f.function_space()
    if V.ufl_element().family() != "Discontinuous Lagrange" or V.ufl_element().degree() != 0:
        f = project(f, FunctionSpace(mesh, "DG", 0))

    values = f.vector().get_local()
    tpc = ax.tripcolor(coords[:,0], coords[:,1], cells, facecolors=values, shading="flat", cmap="viridis")
    ax.set_title(title)
    ax.set_aspect("equal")
    return tpc

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot total field magnitude
c1 = plot_function(axes[0], u_tot_mag_dg0, "Total Field Magnitude |u_tot|")
fig.colorbar(c1, ax=axes[0])

# Plot real part
c2 = plot_function(axes[1], u_tot_re_projected, "Real Part of Total Field")
fig.colorbar(c2, ax=axes[1])

# Plot imaginary part
c3 = plot_function(axes[2], u_tot_im_projected, "Imaginary Part of Total Field")
fig.colorbar(c3, ax=axes[2])

plt.tight_layout()
plt.savefig("outputs/forward_simulation_fields.png", dpi=300, bbox_inches='tight')
plt.show()

##### Load MATLAB measurements and compare #####
def extract_data_along_line(mesh, function, y_value=0.0, tolerance=1e-6):
    """Extract function values at facet midpoints along a horizontal line y = y_value"""
    points = []
    values = []
    
    # Get mesh topology
    mesh.init(1, 0)  # Initialize facet-to-vertex connectivity
    
    # Iterate through all facets
    for facet in facets(mesh):
        # Get facet midpoint
        midpoint = facet.midpoint()
        
        # Check if facet midpoint is close to the target y-value
        if abs(midpoint.y() - y_value) < tolerance:
            x_coord = midpoint.x()
            
            # Evaluate function at this point
            try:
                if hasattr(function, 'function_space'):
                    # It's a FEniCS Function
                    val = function(midpoint.x(), midpoint.y())
                else:
                    # It's an expression or other callable
                    val = function(midpoint)
                
                points.append([x_coord, midpoint.y()])
                values.append(val)
            except Exception as e:
                # Skip points where evaluation fails
                continue
    
    # Sort by x-coordinate
    if points:
        sorted_indices = np.argsort([p[0] for p in points])
        points = np.array(points)[sorted_indices]
        values = np.array(values)[sorted_indices]
    
    return points, values

# Extract simulation data along y = 0.0
points_mag, values_mag = extract_data_along_line(mesh, u_tot_mag_dg0)

# Load MATLAB measurements
matlab_file = "measurements/matlab_measurements_sin0.5_scatter.csv"
if os.path.exists(matlab_file):
    matlab_df = pd.read_csv(matlab_file)
    print(f"Loaded MATLAB data with columns: {list(matlab_df.columns)}")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot simulation results
    if len(points_mag) > 0:
        plt.subplot(2, 1, 1)
        plt.plot(points_mag[:, 0], values_mag, 'b-', linewidth=2, label='FEniCS Simulation')
        plt.xlabel('x coordinate')
        plt.ylabel('Field Magnitude')
        plt.title('Simulation Results along y = 0.0')
        plt.grid(True)
        plt.legend()
        
        # Plot MATLAB measurements
        plt.subplot(2, 1, 2)
        # Assume the CSV has 'x' and 'u' columns (adjust column names as needed)
        if 'x' in matlab_df.columns and 'u' in matlab_df.columns:
            plt.plot(matlab_df['x'].values, matlab_df['u'].values, 'r-', linewidth=2, label='MATLAB Measurements')
        elif 'X' in matlab_df.columns and 'U' in matlab_df.columns:
            plt.plot(matlab_df['X'].values, matlab_df['U'].values, 'r-', linewidth=2, label='MATLAB Measurements')
        else:
            # Try to use first two columns
            col_names = list(matlab_df.columns)
            plt.plot(matlab_df.iloc[:, 0].values, matlab_df.iloc[:, 1].values, 'r-', linewidth=2, 
                    label=f'MATLAB: {col_names[1]} vs {col_names[0]}')
        
        plt.xlabel('x coordinate')
        plt.ylabel('Field Magnitude')
        plt.title('MATLAB Measurements')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("outputs/simulation_vs_matlab_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Overlay comparison
        plt.figure(figsize=(10, 6))
        plt.plot(points_mag[:, 0], values_mag, 'b-', linewidth=2, label='FEniCS Simulation')
        
        if 'x' in matlab_df.columns and 'u' in matlab_df.columns:
            plt.plot(matlab_df['x'].values, matlab_df['u'].values, 'r--', linewidth=2, label='MATLAB Measurements')
        elif 'X' in matlab_df.columns and 'U' in matlab_df.columns:
            plt.plot(matlab_df['X'].values, matlab_df['U'].values, 'r--', linewidth=2, label='MATLAB Measurements')
        else:
            col_names = list(matlab_df.columns)
            plt.plot(matlab_df.iloc[:, 0].values, matlab_df.iloc[:, 1].values, 'r--', linewidth=2, 
                    label=f'MATLAB: {col_names[1]}')
        
        plt.xlabel('x coordinate')
        plt.ylabel('Field Magnitude')
        plt.title('Simulation vs MATLAB Measurements Comparison')
        plt.grid(True)
        plt.legend()
        plt.savefig("outputs/overlay_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Simulation data points: {len(values_mag)}")
        print(f"MATLAB data points: {len(matlab_df)}")
        print(f"Simulation range: [{np.min(values_mag):.6f}, {np.max(values_mag):.6f}]")
        if 'u' in matlab_df.columns:
            print(f"MATLAB range: [{matlab_df['u'].min():.6f}, {matlab_df['u'].max():.6f}]")
        elif 'U' in matlab_df.columns:
            print(f"MATLAB range: [{matlab_df['U'].min():.6f}, {matlab_df['U'].max():.6f}]")
    else:
        print("No simulation data points found along y = 0.0")
        
else:
    print(f"MATLAB measurements file not found: {matlab_file}")
    print("Available files in measurements directory:")
    if os.path.exists("measurements"):
        for f in os.listdir("measurements"):
            print(f"  {f}")
