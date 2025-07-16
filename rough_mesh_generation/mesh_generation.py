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

def create_circular_mesh_with_rough_hole(outer_radius=2.0, hole_center=(0.0, 0.0), hole_radius=0.5, 
                                       hole_roughness=0.08, lam=1.0, mesh_density=10):

    # Calculate mesh size based on wavelength
    mesh_size = lam / mesh_density
    
    # Calculate number of points based on mesh size and circumference
    outer_circumference = 2 * np.pi * outer_radius
    hole_circumference = 2 * np.pi * hole_radius
    
    num_points_outer = int(outer_circumference / mesh_size)
    #num_points_hole = int(hole_circumference / mesh_size)
    num_points_hole = num_points_outer
    
    # Initialize GMSH
    gmsh.initialize()
    gmsh.model.add("circular_domain_with_rough_hole")
    
    # Perfect outer circle
    theta_outer = np.linspace(0, 2*np.pi, num_points_outer, endpoint=False)
    x_outer = outer_radius * np.cos(theta_outer)
    y_outer = outer_radius * np.sin(theta_outer)
    
    # Create rough inner hole boundary using Fourier method
    x_hole, y_hole = create_fourier_rough_boundary(
        hole_center, hole_radius, num_points_hole, hole_roughness
    )
    
    # Add outer circle points
    outer_points = []
    for i in range(len(x_outer)):
        p = gmsh.model.geo.addPoint(x_outer[i], y_outer[i], 0.0, mesh_size)
        outer_points.append(p)
    
    # Add inner hole points with finer mesh
    hole_points = []
    fine_mesh_size = mesh_size * 0.5
    for i in range(len(x_hole)):
        p = gmsh.model.geo.addPoint(x_hole[i], y_hole[i], 0.0, fine_mesh_size)
        hole_points.append(p)
    
    # Create outer circular boundary (line segments)
    outer_lines = []
    for i in range(len(outer_points)):
        next_i = (i + 1) % len(outer_points)
        line = gmsh.model.geo.addLine(outer_points[i], outer_points[next_i])
        outer_lines.append(line)
    
    # Create inner hole boundary (line segments)
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

def create_fourier_rough_boundary(center, radius, num_points, roughness, num_harmonics=3):
    # Base circle angles
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    # Use Fourier series for smooth periodic variations
    r_variation = np.zeros_like(theta)
    
    for n in range(1, num_harmonics + 1):
        amplitude = roughness / n  # Decreasing amplitude for higher harmonics
        phase = np.random.random() * 2 * np.pi
        r_variation += amplitude * np.sin(n * theta + phase)
    
    # Apply variations to radius
    r_rough = radius + r_variation
    
    # Convert to Cartesian coordinates
    x_circle = center[0] + r_rough * np.cos(theta)
    y_circle = center[1] + r_rough * np.sin(theta)
    
    return x_circle, y_circle

def plot_circular_mesh(mesh, facet_markers=None):
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
    
    plt.title("Circular Domain with Fourier Rough Hole")
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_roughness_comparison(center=(0.0, 0.0), radius=0.5, roughness=0.08):
    harmonics_list = [3, 6, 8, 12]
    num_points = 100
    
    plt.figure(figsize=(15, 10))
    
    for i, num_harmonics in enumerate(harmonics_list):
        plt.subplot(2, 2, i+1)
        
        # Perfect circle
        theta_perfect = np.linspace(0, 2*np.pi, 200)
        x_perfect = center[0] + radius * np.cos(theta_perfect)
        y_perfect = center[1] + radius * np.sin(theta_perfect)
        
        # Rough circle
        x_rough, y_rough = create_fourier_rough_boundary(
            center, radius, num_points, roughness, num_harmonics
        )
        
        plt.plot(x_perfect, y_perfect, 'k--', linewidth=1, alpha=0.5, label='Perfect circle')
        plt.plot(x_rough, y_rough, 'r-', linewidth=2, label=f'{num_harmonics} harmonics')
        plt.fill(x_rough, y_rough, alpha=0.3, color='red')
        
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Fourier Roughness: {num_harmonics} Harmonics')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_mesh_dolfinx(mesh, facet_markers, filename_base="circular_mesh_with_rough_hole"):
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
    # Some constant necessary
    c = 299792458
    freq_max = 5e9 # 5GHz
    
    # Generate circular mesh with Fourier rough hole
    print("Generating circular mesh with Fourier rough hole...")
    
    # Parameters
    wavelength = c / freq_max  # Physical wavelength
    mesh_points_per_wavelength = 5  # Higher = finer mesh -> We can use quite high order of polynomials to prevent polution
    
    mesh, facet_markers = create_circular_mesh_with_rough_hole(
        outer_radius=0.6,         # Outer circle radius
        hole_center=(0.0, 0.0),   # Center of hole
        hole_radius=0.1,          # Base radius of hole
        hole_roughness=0.03,      # Roughness amplitude for hole
        lam=wavelength,           # Wavelength
        mesh_density=mesh_points_per_wavelength  # Points per wavelength
    )
    
    print(f"Generated circular mesh with Fourier rough hole")
    print(f"Wavelength: {wavelength}")
    print(f"Mesh size: {wavelength/mesh_points_per_wavelength:.4f}")
    print(f"Mesh has {mesh.topology.index_map(0).size_local} vertices")
    print(f"Mesh has {mesh.topology.index_map(2).size_local} cells")
    
    # Plot mesh
    plot_circular_mesh(mesh, facet_markers)
    
    # Save mesh
    save_mesh_dolfinx(mesh, facet_markers, "geometry_perturbed_mesh")
    
    # Print boundary marker information
    print("\nBoundary markers:")
    print(f"Outer circular boundary: {OUTER_CIRCLE_MARKER}")
    print(f"Rough hole boundary: {ROUGH_HOLE_MARKER}")