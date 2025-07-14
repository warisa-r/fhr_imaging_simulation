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

def create_smooth_rough_top_mesh_dolfinx(width=2.0, height=4.0, num_points_top=50, roughness=5, mesh_size=0.1, smoothing_method="spline"):
    # Initialize GMSH
    gmsh.initialize()
    gmsh.model.add("smooth_rough_rectangle")
    
    # Create smooth rough top edge
    x_top, y_top = create_smooth_rough_boundary(width, height, num_points_top, roughness, smoothing_method)
    
    # Add points to GMSH
    points = []
    
    # Bottom left corner
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, mesh_size)
    points.append(p1)
    
    # Bottom right corner
    p2 = gmsh.model.geo.addPoint(width, 0.0, 0.0, mesh_size)
    points.append(p2)
    
    # Top edge points (smooth rough)
    top_points = []
    for i in range(len(x_top)):
        p = gmsh.model.geo.addPoint(x_top[i], y_top[i], 0.0, mesh_size)
        top_points.append(p)
    
    # Create lines
    lines = []
    
    # Bottom edge
    bottom_line = gmsh.model.geo.addLine(p1, p2)
    lines.append(bottom_line)
    
    # Right edge
    right_line = gmsh.model.geo.addLine(p2, top_points[-1])
    lines.append(right_line)
    
    # Top edge (smooth rough) - use splines for smoother curves
    if smoothing_method == "spline" and len(top_points) > 3:
        # Create a single spline through all top points
        top_spline = gmsh.model.geo.addSpline(top_points[::-1])  # Reverse for correct orientation
        lines.append(top_spline)
        top_lines = [top_spline]
    else:
        # Fall back to line segments
        top_lines = []
        for i in range(len(top_points) - 1):
            line = gmsh.model.geo.addLine(top_points[i+1], top_points[i])
            top_lines.append(line)
            lines.append(line)
    
    # Left edge
    left_line = gmsh.model.geo.addLine(top_points[0], p1)
    lines.append(left_line)
    
    # Create curve loop and surface
    curve_loop = gmsh.model.geo.addCurveLoop([bottom_line, right_line] + top_lines + [left_line])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])
    
    # Synchronize
    gmsh.model.geo.synchronize()
    
    # Create physical groups for boundary markers
    gmsh.model.addPhysicalGroup(1, [bottom_line], tag=1, name="Bottom")
    gmsh.model.addPhysicalGroup(1, [right_line], tag=2, name="Right")
    gmsh.model.addPhysicalGroup(1, top_lines, tag=3, name="Top_Reflective")
    gmsh.model.addPhysicalGroup(1, [left_line], tag=4, name="Left")
    gmsh.model.addPhysicalGroup(2, [surface], tag=5, name="Domain")
    
    # Generate mesh
    gmsh.model.mesh.generate(2)
    
    # Convert to DOLFINx mesh
    mesh, cell_markers, facet_markers = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2
    )
    
    gmsh.finalize()
    
    return mesh, facet_markers

def create_smooth_rough_boundary(width, height, num_points, roughness, method="spline"):
    np.random.seed(42)
    
    if method == "gaussian":
        x_coarse = np.linspace(0, width, num_points // 3)
        y_noise_coarse = roughness * (np.random.random(len(x_coarse)) - 0.5)
        
        # Apply Gaussian smoothing
        sigma = len(x_coarse) / 4  # Smoothing parameter
        y_smooth = gaussian_filter1d(y_noise_coarse, sigma=sigma, mode='reflect')
        
        # Interpolate to final resolution
        f = interpolate.interp1d(x_coarse, y_smooth, kind='cubic')
        x_top = np.linspace(0, width, num_points)
        y_top = height + f(x_top)
        
    elif method == "fourier":
        x_top = np.linspace(0, width, num_points)
        y_top = np.full_like(x_top, height)
        
        # Add smooth Fourier components
        num_harmonics = 30
        for n in range(1, num_harmonics + 1):
            amplitude = roughness / n  # Decreasing amplitude
            phase = np.random.random() * 2 * np.pi
            frequency = n * np.pi / width
            y_top += amplitude * np.sin(frequency * x_top + phase)
            
    elif method == "bezier":
        num_control = 7  # Number of control points
        x_control = np.linspace(0, width, num_control)
        y_control = height + roughness * (np.random.random(num_control) - 0.5)
        
        # Ensure endpoints are at correct height
        y_control[0] = height
        y_control[-1] = height
        
        # Create Bezier curve
        t = np.linspace(0, 1, num_points)
        x_top = np.zeros(num_points)
        y_top = np.zeros(num_points)
        
        n = len(x_control) - 1
        for i in range(n + 1):
            bernstein = math.comb(n, i) * (1 - t) ** (n - i) * t ** i
            x_top += bernstein * x_control[i]
            y_top += bernstein * y_control[i]
            
    else:  # spline (default)
        # Method 4: Spline interpolation
        num_control = max(5, num_points // 8)
        x_control = np.linspace(0, width, num_control)
        y_control = height + roughness * (np.random.random(num_control) - 0.5)
        
        # Ensure endpoints are at correct height
        y_control[0] = height
        y_control[-1] = height
        
        # Apply some smoothing to control points
        if len(y_control) > 3:
            y_control = gaussian_filter1d(y_control, sigma=1.0)
            y_control[0] = height  # Restore endpoints
            y_control[-1] = height
        
        # Create spline
        tck = interpolate.splrep(x_control, y_control, s=0, k=min(3, len(x_control)-1))
        x_top = np.linspace(0, width, num_points)
        y_top = interpolate.splev(x_top, tck)
    
    # Ensure endpoints are exactly at corners
    y_top[0] = height
    y_top[-1] = height
    
    return x_top, y_top

def plot_boundary_comparison():
    width, height = 2.0, 4.0
    num_points = 50
    roughness = 0.2
    
    methods = ["spline", "gaussian", "fourier", "bezier"]
    
    plt.figure(figsize=(15, 10))
    
    for i, method in enumerate(methods):
        plt.subplot(2, 2, i+1)
        x_top, y_top = create_smooth_rough_boundary(width, height, num_points, roughness, method)
        
        plt.plot(x_top, y_top, 'r-', linewidth=2, label=f'{method.capitalize()} method')
        plt.axhline(y=height, color='k', linestyle='--', alpha=0.5, label='Original height')
        plt.fill_between([0, width], [0, 0], [height, height], alpha=0.2, color='lightblue')
        plt.fill_between(x_top, [height]*len(x_top), y_top, alpha=0.3, color='red')
        
        plt.xlim(0, width)
        plt.ylim(0, height + roughness + 0.1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{method.capitalize()} Smoothing')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_mesh_dolfinx(mesh, facet_markers=None):
    """Plot DOLFINx mesh with matplotlib"""
    
    # Get mesh coordinates and connectivity
    x = mesh.geometry.x
    cells = mesh.topology.connectivity(2, 0).array.reshape(-1, 3)
    
    plt.figure(figsize=(10, 12))
    
    # Plot mesh
    for cell in cells:
        triangle = x[cell]
        triangle = np.vstack([triangle, triangle[0]])  # Close the triangle
        plt.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=0.3, alpha=0.7)
    
    # Plot boundary markers if provided
    if facet_markers is not None:
        # Get boundary facets
        mesh.topology.create_connectivity(1, 0)  # facet to vertex connectivity
        
        colors = {1: 'blue', 2: 'green', 3: 'red', 4: 'orange'}
        labels = {1: 'Bottom', 2: 'Right', 3: 'Top (Reflective)', 4: 'Left'}
        
        for marker in [1, 2, 3, 4]:
            facets = facet_markers.indices[facet_markers.values == marker]
            
            for facet in facets:
                # Get vertices of this facet
                vertices = mesh.topology.connectivity(1, 0).links(facet)
                facet_coords = x[vertices]
                
                plt.plot(facet_coords[:, 0], facet_coords[:, 1], 
                        color=colors[marker], linewidth=2, label=labels[marker])
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title("Rough Top Edge Mesh (DOLFINx)")
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_mesh_dolfinx(mesh, facet_markers, filename_base="rough_mesh_dolfinx"):
    # Save mesh
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{filename_base}.xdmf", "w") as file:
        file.write_mesh(mesh)
    
    # Save facet markers - fix the argument order
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{filename_base}_boundaries.xdmf", "w") as file:
        file.write_meshtags(facet_markers, mesh.geometry, "Grid")
    
    print(f"Mesh saved as {filename_base}.xdmf")
    print(f"Boundaries saved as {filename_base}_boundaries.xdmf")

# Update the main execution
if __name__ == "__main__":
    print("Generating smooth rough top edge mesh with DOLFINx...")
    
    # First, show comparison of different smoothing methods
    print("Comparing smoothing methods...")
    plot_boundary_comparison()
    
    # Generate mesh with spline smoothing (recommended)
    print("Generating mesh with spline smoothing...")
    mesh, facet_markers = create_smooth_rough_top_mesh_dolfinx(
        width=2.0, 
        height=4.0, 
        num_points_top=40,
        roughness=0.15,
        mesh_size=0.05,
        smoothing_method="spline"  # Try "gaussian", "fourier", or "bezier"
    )
    
    print(f"Generated smooth rough mesh")
    print(f"Mesh has {mesh.topology.index_map(0).size_local} vertices")
    print(f"Mesh has {mesh.topology.index_map(2).size_local} cells")
    
    # Plot mesh
    plot_mesh_dolfinx(mesh, facet_markers)
    
    # Save mesh
    save_mesh_dolfinx(mesh, facet_markers, "smooth_rough_top_mesh_dolfinx")