import numpy as np
import math
import matplotlib.pyplot as plt
import dolfinx
import dolfinx.mesh
import dolfinx.io
from dolfinx.mesh import create_mesh, meshtags_from_entities
import ufl
from mpi4py import MPI
import gmsh
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

def create_circular_mesh_with_rough_hole(outer_radius=2.0, hole_center=(0.0, 0.0), hole_radius=0.5, 
                                       num_points_outer=60, num_points_hole=40, hole_roughness=0.08, 
                                       mesh_size=0.1, smoothing_method="spline", outer_roughness=0.0):
    """
    Create a circular mesh with a rough circular hole inside
    
    Parameters:
    - outer_radius: Radius of the outer circular domain
    - hole_center: (x, y) coordinates of hole center
    - hole_radius: Base radius of the inner hole
    - num_points_outer: Number of points on outer circle
    - num_points_hole: Number of points on hole circle
    - hole_roughness: Maximum deviation from perfect circle for hole
    - mesh_size: Characteristic mesh size
    - smoothing_method: "spline", "gaussian", "fourier", or "bezier"
    - outer_roughness: Roughness for outer boundary (0 = perfect circle)
    """
    
    # Initialize GMSH
    gmsh.initialize()
    gmsh.model.add("circular_domain_with_rough_hole")
    
    # Create outer circular boundary
    if outer_roughness > 0:
        x_outer, y_outer = create_rough_circular_boundary(
            (0.0, 0.0), outer_radius, num_points_outer, outer_roughness, smoothing_method
        )
    else:
        # Perfect outer circle
        theta_outer = np.linspace(0, 2*np.pi, num_points_outer, endpoint=False)
        x_outer = outer_radius * np.cos(theta_outer)
        y_outer = outer_radius * np.sin(theta_outer)
    
    # Create rough inner hole boundary
    x_hole, y_hole = create_rough_circular_boundary(
        hole_center, hole_radius, num_points_hole, hole_roughness, smoothing_method
    )
    
    # Add outer circle points
    outer_points = []
    for i in range(len(x_outer)):
        p = gmsh.model.geo.addPoint(x_outer[i], y_outer[i], 0.0, mesh_size)
        outer_points.append(p)
    
    # Add inner hole points
    hole_points = []
    for i in range(len(x_hole)):
        p = gmsh.model.geo.addPoint(x_hole[i], y_hole[i], 0.0, mesh_size * 0.5)  # Finer mesh near hole
        hole_points.append(p)
    
    # Create outer circular boundary
    if smoothing_method == "spline" and len(outer_points) > 3:
        outer_spline = gmsh.model.geo.addSpline(outer_points + [outer_points[0]])
        outer_lines = [outer_spline]
    else:
        outer_lines = []
        for i in range(len(outer_points)):
            next_i = (i + 1) % len(outer_points)
            line = gmsh.model.geo.addLine(outer_points[i], outer_points[next_i])
            outer_lines.append(line)
    
    # Create inner hole boundary
    if smoothing_method == "spline" and len(hole_points) > 3:
        hole_spline = gmsh.model.geo.addSpline(hole_points + [hole_points[0]])
        hole_lines = [hole_spline]
    else:
        hole_lines = []
        for i in range(len(hole_points)):
            next_i = (i + 1) % len(hole_points)
            line = gmsh.model.geo.addLine(hole_points[i], hole_points[next_i])
            hole_lines.append(line)
    
    # Create curve loops
    outer_loop = gmsh.model.geo.addCurveLoop(outer_lines)
    hole_loop = gmsh.model.geo.addCurveLoop(hole_lines)
    
    # Create surface (outer circle with hole)
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])
    
    # Synchronize
    gmsh.model.geo.synchronize()
    
    # Create physical groups for boundary markers
    gmsh.model.addPhysicalGroup(1, outer_lines, tag=1, name="Outer_Circle")
    gmsh.model.addPhysicalGroup(1, hole_lines, tag=2, name="Rough_Hole")
    gmsh.model.addPhysicalGroup(2, [surface], tag=3, name="Domain")
    
    # Generate mesh
    gmsh.model.mesh.generate(2)
    
    # Convert to DOLFINx mesh
    mesh, cell_markers, facet_markers = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2
    )
    
    gmsh.finalize()
    
    return mesh, facet_markers

def create_rough_circular_boundary(center, radius, num_points, roughness, method="spline"):
    """
    Create a rough circular boundary
    
    Parameters:
    - center: (x, y) center coordinates
    - radius: Base radius
    - num_points: Number of points on circle
    - roughness: Maximum radial deviation
    - method: Smoothing method
    """
    #np.random.seed(42)  # For reproducible results
    
    # Base circle angles
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    if method == "gaussian":
        # Generate rough radial variations
        theta_coarse = np.linspace(0, 2*np.pi, num_points // 3, endpoint=False)
        r_noise_coarse = roughness * (np.random.random(len(theta_coarse)) - 0.5)
        
        # Apply Gaussian smoothing with periodic boundary conditions
        r_noise_coarse = np.concatenate([r_noise_coarse, r_noise_coarse, r_noise_coarse])
        sigma = len(theta_coarse) / 6
        r_smooth_extended = gaussian_filter1d(r_noise_coarse, sigma=sigma, mode='wrap')
        r_smooth = r_smooth_extended[len(theta_coarse):2*len(theta_coarse)]
        
        # Interpolate to final resolution
        f = interpolate.interp1d(theta_coarse, r_smooth, kind='cubic', 
                               bounds_error=False, fill_value='extrapolate')
        r_variation = f(theta)
        
    elif method == "fourier":
        # Use Fourier series for smooth periodic variations
        r_variation = np.zeros_like(theta)
        num_harmonics = 8
        
        for n in range(1, num_harmonics + 1):
            amplitude = roughness / n
            phase = np.random.random() * 2 * np.pi
            r_variation += amplitude * np.sin(n * theta + phase)
            
    elif method == "bezier":
        # Use periodic Bezier-like approach
        num_control = 8
        theta_control = np.linspace(0, 2*np.pi, num_control, endpoint=False)
        r_control = roughness * (np.random.random(num_control) - 0.5)
        
        # Make it periodic
        theta_control = np.concatenate([theta_control, [2*np.pi]])
        r_control = np.concatenate([r_control, [r_control[0]]])
        
        # Smooth the control points
        r_control = gaussian_filter1d(r_control, sigma=1.0, mode='wrap')
        
        # Interpolate
        f = interpolate.interp1d(theta_control, r_control, kind='cubic', 
                               bounds_error=False, fill_value='extrapolate')
        r_variation = f(theta)
        
    else:  # spline (default)
        # Create control points for spline
        num_control = max(8, num_points // 6)
        theta_control = np.linspace(0, 2*np.pi, num_control, endpoint=False)
        r_control = roughness * (np.random.random(num_control) - 0.5)
        
        # Apply smoothing to control points
        r_control = gaussian_filter1d(np.concatenate([r_control, r_control, r_control]), 
                                    sigma=1.0, mode='wrap')[num_control:2*num_control]
        
        # Create periodic spline
        theta_control = np.concatenate([theta_control, [2*np.pi]])
        r_control = np.concatenate([r_control, [r_control[0]]])
        
        tck = interpolate.splrep(theta_control, r_control, s=0, k=3, per=False)
        r_variation = interpolate.splev(theta, tck)
    
    # Apply variations to radius
    r_rough = radius + r_variation
    
    # Convert to Cartesian coordinates
    x_circle = center[0] + r_rough * np.cos(theta)
    y_circle = center[1] + r_rough * np.sin(theta)
    
    return x_circle, y_circle

def plot_circular_mesh(mesh, facet_markers=None):
    """Plot circular mesh with hole"""
    
    # Get mesh coordinates and connectivity
    x = mesh.geometry.x
    cells = mesh.topology.connectivity(2, 0).array.reshape(-1, 3)
    
    plt.figure(figsize=(12, 12))
    
    # Plot mesh
    for cell in cells:
        triangle = x[cell]
        triangle = np.vstack([triangle, triangle[0]])
        plt.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=0.3, alpha=0.7)
    
    # Plot boundary markers if provided
    if facet_markers is not None:
        mesh.topology.create_connectivity(1, 0)
        
        colors = {1: 'blue', 2: 'red'}
        labels = {1: 'Outer Circle', 2: 'Rough Hole'}
        
        for marker in [1, 2]:
            if marker in facet_markers.values:
                facets = facet_markers.indices[facet_markers.values == marker]
                
                for facet in facets:
                    vertices = mesh.topology.connectivity(1, 0).links(facet)
                    facet_coords = x[vertices]
                    
                    plt.plot(facet_coords[:, 0], facet_coords[:, 1], 
                            color=colors[marker], linewidth=3, label=labels[marker])
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title("Circular Domain with Rough Hole")
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_mesh_dolfinx(mesh, facet_markers, filename_base="circular_mesh_with_rough_hole"):
    """Save mesh and boundary markers"""
    
    # Save mesh and facet markers together
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{filename_base}.xdmf", "w") as file:
        file.write_mesh(mesh)
        mesh.topology.create_connectivity(1, 2)
        file.write_meshtags(facet_markers, mesh.geometry, "Grid")
    
    print(f"Mesh and boundaries saved as {filename_base}.xdmf")

# Boundary markers
OUTER_CIRCLE_MARKER = 1
ROUGH_HOLE_MARKER = 2

if __name__ == "__main__":
    
    # Generate circular mesh with rough hole
    print("Generating circular mesh with rough hole...")
    mesh, facet_markers = create_circular_mesh_with_rough_hole(
        outer_radius=2.0,         # Outer circle radius
        hole_center=(0.0, 0.0),   # Center of hole (can be eccentric)
        hole_radius=0.6,          # Base radius of hole
        num_points_outer=80,      # Points on outer circle
        num_points_hole=50,       # Points on hole
        hole_roughness=0.12,      # Roughness amplitude for hole
        mesh_size=0.06,           # Mesh resolution
        smoothing_method="fourier", # Roughness method
        outer_roughness=0.0       # 0 = perfect outer circle, >0 = rough outer
    )
    
    print(f"Generated circular mesh with rough hole")
    print(f"Mesh has {mesh.topology.index_map(0).size_local} vertices")
    print(f"Mesh has {mesh.topology.index_map(2).size_local} cells")
    
    # Plot mesh
    plot_circular_mesh(mesh, facet_markers)
    
    # Save mesh
    save_mesh_dolfinx(mesh, facet_markers, "circular_domain_rough_hole")
    
    # Print boundary marker information
    print("\nBoundary markers:")
    print(f"Outer circular boundary: {OUTER_CIRCLE_MARKER}")
    print(f"Rough hole boundary: {ROUGH_HOLE_MARKER}")