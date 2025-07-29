import gmsh
import numpy as np
import os

obstacle_marker = 3        # Top boundary (obstacle)
bottom_wall_marker = 1     # Bottom wall only
side_wall_marker = 2       # Left and right walls

def generate_rectangle_mesh(width=5.0, height=2.0, mesh_size=0.1, output_name="rectangle_mesh"):
    # Initialize gmsh
    gmsh.initialize()
    gmsh.clear()
    
    # Create new model
    gmsh.model.add("rectangle")
    
    # Create rectangle points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)        # Bottom-left
    p2 = gmsh.model.geo.addPoint(width, 0, 0, mesh_size)    # Bottom-right
    p3 = gmsh.model.geo.addPoint(width, height, 0, mesh_size) # Top-right
    p4 = gmsh.model.geo.addPoint(0, height, 0, mesh_size)   # Top-left
    
    # Create lines
    bottom = gmsh.model.geo.addLine(p1, p2)  # Bottom wall
    right = gmsh.model.geo.addLine(p2, p3)   # Right wall
    top = gmsh.model.geo.addLine(p3, p4)     # Top wall (obstacle)
    left = gmsh.model.geo.addLine(p4, p1)    # Left wall
    
    # Create curve loop and surface
    curve_loop = gmsh.model.geo.addCurveLoop([bottom, right, top, left])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])
    
    # Synchronize
    gmsh.model.geo.synchronize()
    
    # Add physical groups with separate markers
    gmsh.model.addPhysicalGroup(1, [top], obstacle_marker, "top_boundary")           # Top: marker 3
    gmsh.model.addPhysicalGroup(1, [bottom], bottom_wall_marker, "bottom_wall")      # Bottom: marker 1
    gmsh.model.addPhysicalGroup(1, [right, left], side_wall_marker, "side_walls")    # Sides: marker 2
    gmsh.model.addPhysicalGroup(2, [surface], 4, "domain")                          # Domain: marker 4
    
    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    
    # Write mesh files
    gmsh.write(f"{output_name}.msh")
    # Clean up
    gmsh.finalize()
    
    return f"{output_name}.msh"

def generate_rough_top_mesh(width=4.0, height=2.0, roughness_amplitude=0.02, 
                           roughness_frequency=1, mesh_size=0.1, output_name="rough_top_mesh"):
    # Initialize gmsh
    gmsh.initialize()
    gmsh.clear()
    
    # Create new model
    gmsh.model.add("rough_rectangle")
    
    # Create bottom rectangle points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)        # Bottom-left
    p2 = gmsh.model.geo.addPoint(width, 0, 0, mesh_size)    # Bottom-right
    
    # Create smooth rough top boundary
    n_points_top = max(20, int(width / mesh_size * 2))  # Ensure enough points for smooth curve
    x_coords = np.linspace(0, width, n_points_top)
    
    # Generate rough top boundary points - Gaussian-localized roughness
    top_points = []
    for x in x_coords:
        y_rough = height + roughness_amplitude * (np.sin(2 * np.pi * roughness_frequency * x / width)-1)
        p = gmsh.model.geo.addPoint(x, y_rough, 0, mesh_size)
        top_points.append(p)
    
    # Create lines for boundaries (following counterclockwise orientation)
    bottom = gmsh.model.geo.addLine(p1, p2)                    # Bottom wall
    right = gmsh.model.geo.addLine(p2, top_points[-1])         # Right wall
    top_rough = gmsh.model.geo.addSpline(top_points[::-1])     # Top: right to left (reversed)
    left = gmsh.model.geo.addLine(top_points[0], p1)           # Left wall
    
    # Create curve loop and surface (counterclockwise orientation)
    curve_loop = gmsh.model.geo.addCurveLoop([bottom, right, top_rough, left])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])
    
    # Synchronize
    gmsh.model.geo.synchronize()
    
    # Add physical groups with separate markers
    gmsh.model.addPhysicalGroup(1, [top_rough], obstacle_marker, "rough_top_boundary")  # Top: marker 3
    gmsh.model.addPhysicalGroup(1, [bottom], bottom_wall_marker, "bottom_wall")         # Bottom: marker 1
    gmsh.model.addPhysicalGroup(1, [right, left], side_wall_marker, "side_walls")       # Sides: marker 2
    gmsh.model.addPhysicalGroup(2, [surface], 4, "domain")                             # Domain: marker 4

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    
    # Write mesh files
    gmsh.write(f"{output_name}.msh")
    # Clean up
    gmsh.finalize()

    return f"{output_name}.msh"

def convert_to_xml(msh_file):
    import subprocess
    
    base_name = os.path.splitext(msh_file)[0]
    xml_file = f"{base_name}.xml"
    
    try:
        result = subprocess.run([
            "dolfin-convert", msh_file, xml_file
        ], capture_output=True, text=True, check=True)
        
        print(f"Converted to XML: {xml_file}")
        print(f"Facet regions: {base_name}_facet_region.xml")
        
        return xml_file
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting mesh: {e}")
        print(f"Error output: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: dolfin-convert not found. Make sure FEniCS is installed.")
        return None

if __name__ == "__main__":

    c = 299792458
    freq_max = 5e9 # 5GHz
    
    # Parameters
    wavelength = c / freq_max  # Physical wavelength
    mesh_size = wavelength / 5

    # Generate rectangle mesh
    print("Generating rectangle mesh...")
    rect_mesh = generate_rectangle_mesh(
        width=1.0, 
        height=1.5, 
        mesh_size=mesh_size, 
        output_name="rectangle_mesh"
    )
    
    # Generate rough top mesh
    print("Generating rough top mesh...")
    rough_mesh = generate_rough_top_mesh(
        width=1.0, 
        height=1.5, 
        roughness_amplitude=0.005, 
        roughness_frequency=5, 
        mesh_size=mesh_size, 
        output_name="rough_top_mesh"
    )
    
    # Convert both to XML
    print("Converting meshes to XML...")
    convert_to_xml(rect_mesh)
    convert_to_xml(rough_mesh)
    
    print("\nBoundary markers:")
    print(f"  Bottom wall: {bottom_wall_marker}")
    print(f"  Side walls: {side_wall_marker}")
    print(f"  Top boundary (obstacle): {obstacle_marker}")
    print(f"  Domain: 4")