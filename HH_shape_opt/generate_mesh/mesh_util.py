import meshio
import numpy as np
import os
from matplotlib.collections import PolyCollection, LineCollection

# Global mesh markers

SIDE_WALL_MARKER = 1
RECEIVER_EDGE_MARKER = 2
OBSTACLE_MARKER = 3
OBSTACLE_OPT_MARKER = 4
DOMAIN_MARKER = 5
BOTTOM_WALL_MARKER = 6
RECEIVER_SEGMENT_MARKER = 7

def calculate_mesh_size(freq_max, num_mesh_points_per_wavelength):
    c = 299792458

    # Parameters
    wavelength = c / freq_max  # Physical wavelength
    mesh_size = wavelength / num_mesh_points_per_wavelength

    return mesh_size

def plot_mesh(filename, ax, title="", show_domains=True, show_markers=True, show_receiver_patches=True):
    mesh = meshio.read(filename)
    points = mesh.points[:, :2]
    
    # Find triangle cells for domains
    triangle_cells = None
    triangle_data = None
    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type == "triangle":
            triangle_cells = cell_block.data
            if mesh.cell_data and "gmsh:physical" in mesh.cell_data:
                triangle_data = mesh.cell_data["gmsh:physical"][i]
            break
    
    if triangle_cells is None:
        raise RuntimeError("No triangle cells found in mesh.")
    
    # Define human-readable names and colors
    domain_labels = {
        DOMAIN_MARKER: "Domain",
        OBSTACLE_MARKER: "Obstacle",
        5: "Medium",
    }
    
    domain_colors = {
        DOMAIN_MARKER: "lightblue",
        OBSTACLE_MARKER: "lightcoral",
        5: "lightgreen",
    }
    
    marker_labels = {
        SIDE_WALL_MARKER: "Side Wall",
        RECEIVER_EDGE_MARKER: "Receiver Edge", 
        OBSTACLE_MARKER: "Obstacle Boundary",
        OBSTACLE_OPT_MARKER: "Optimized Obstacle",
        BOTTOM_WALL_MARKER: "Bottom Wall",
        RECEIVER_SEGMENT_MARKER: "Receiver Segments",
    }
    
    marker_colors = {
        SIDE_WALL_MARKER: "blue",
        RECEIVER_EDGE_MARKER: "red", 
        OBSTACLE_MARKER: "black",
        OBSTACLE_OPT_MARKER: "orange",
        BOTTOM_WALL_MARKER: "green",
        RECEIVER_SEGMENT_MARKER: "purple",
    }
    
    # Plot different domains
    if show_domains and triangle_data is not None:
        unique_domains = np.unique(triangle_data)
        for domain in unique_domains:
            domain_mask = triangle_data == domain
            domain_triangles = triangle_cells[domain_mask]
            
            if len(domain_triangles) == 0:
                continue
            
            color = domain_colors.get(domain, "lightgray")
            label = domain_labels.get(domain, f"Domain {domain}")
            
            # Build polygons for each triangle
            polys = [points[tri] for tri in domain_triangles]
            coll = PolyCollection(polys, facecolors=color, alpha=0.6, label=label)
            ax.add_collection(coll)
    
    # Plot mesh edges
    ax.triplot(points[:, 0], points[:, 1], triangle_cells, 
               color="gray", linewidth=0.3, alpha=0.8)
    
    # Plot boundary markers
    if show_markers:
        line_cells = None
        line_data = None
        for i, cell_block in enumerate(mesh.cells):
            if cell_block.type == "line":
                line_cells = cell_block.data
                if mesh.cell_data and "gmsh:physical" in mesh.cell_data:
                    line_data = mesh.cell_data["gmsh:physical"][i]
                break
        
        if line_cells is not None and line_data is not None:
            unique_markers = np.unique(line_data)
            
            # Track if we've already added a "Receiver Patches" label
            receiver_patches_labeled = False
            
            for marker in unique_markers:
                marker_mask = line_data == marker
                marker_lines = line_cells[marker_mask]
                
                if marker >= RECEIVER_SEGMENT_MARKER:
                    # Unknown marker > 7: treat as additional receiver patch
                    # Skip if show_receiver_patches is False
                    if not show_receiver_patches:
                        continue
                    color = "purple"  # Same color for all receiver patches
                    label = "Receiver Patches" if not receiver_patches_labeled else None
                    receiver_patches_labeled = True
                # Check if this marker is already defined
                elif marker in marker_labels:
                    # Use predefined label and color
                    color = marker_colors[marker]
                    label = marker_labels[marker]
                else:
                    # Unknown marker <= 7: use default gray
                    color = "gray"
                    label = f"Marker {marker}"
                
                # Build list of line segments (each is a 2x2 array of xy coords)
                segments = [points[line] for line in marker_lines]
                
                # Add all segments for this marker as one collection
                lc = LineCollection(segments, colors=color, linewidths=2, label=label)
                ax.add_collection(lc)
    
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def convert_msh_to_xdmf(msh_file_path):

    # Define output paths based on the input .msh file
    base_path, _ = os.path.splitext(msh_file_path)
    xdmf_path = f"{base_path}.xdmf"
    facet_xdmf_path = f"{base_path}_facets.xdmf"

    # --- Convert .msh to .xdmf using meshio (only on rank 0) ---
    print(f"[INFO] Converting {msh_file_path} to XDMF format...")
    msh = meshio.read(msh_file_path)

    # Extract 2D points from the 3D points read by meshio
    points_2d = msh.points[:, :2]

    # Create and write the domain mesh (triangles) using 2D points
    triangle_cells = msh.get_cells_type("triangle")
    domain_mesh = meshio.Mesh(points=points_2d, cells=[
                              ("triangle", triangle_cells)])
    domain_mesh.write(xdmf_path)
    print(f"[INFO] Wrote domain mesh to {xdmf_path}")

    # Create and write the facet mesh (lines) using 2D points
    line_cells = msh.get_cells_type("line")
    line_data = msh.get_cell_data("gmsh:physical", "line")
    facet_mesh = meshio.Mesh(
        points=points_2d,
        cells=[("line", line_cells)],
        cell_data={"name_to_read": [line_data]}
    )
    facet_mesh.write(facet_xdmf_path)
    print(f"[INFO] Wrote facet markers to {facet_xdmf_path}")

    return xdmf_path, facet_xdmf_path