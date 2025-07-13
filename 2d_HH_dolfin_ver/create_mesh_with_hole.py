import gmsh
import sys

def create_mesh_with_hole():
    gmsh.initialize()
    
    # Create a new model
    gmsh.model.add("domain_with_hole")
    
    # Define geometry parameters
    lc = 0.02  # characteristic length
    
    # Create outer rectangle points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(1, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(1, 1, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, 1, 0, lc)
    
    # Create circle center and points
    center = gmsh.model.geo.addPoint(0.5, 0.5, 0, lc/2)
    radius = 0.2
    
    # Create circle points
    pc1 = gmsh.model.geo.addPoint(0.5 + radius, 0.5, 0, lc/2)
    pc2 = gmsh.model.geo.addPoint(0.5, 0.5 + radius, 0, lc/2)
    pc3 = gmsh.model.geo.addPoint(0.5 - radius, 0.5, 0, lc/2)
    pc4 = gmsh.model.geo.addPoint(0.5, 0.5 - radius, 0, lc/2)
    
    # Create outer rectangle lines
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    
    # Create circle arcs
    arc1 = gmsh.model.geo.addCircleArc(pc1, center, pc2)
    arc2 = gmsh.model.geo.addCircleArc(pc2, center, pc3)
    arc3 = gmsh.model.geo.addCircleArc(pc3, center, pc4)
    arc4 = gmsh.model.geo.addCircleArc(pc4, center, pc1)
    
    # Create curve loops
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    inner_loop = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])
    
    # Create surface (domain minus hole)
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])
    
    # Synchronize before meshing
    gmsh.model.geo.synchronize()
    
    # Add physical groups for boundary conditions
    gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], 1)  # Outer boundary
    gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4], 2)  # Circle boundary -> The one we are going to set Dirichlet for
    gmsh.model.addPhysicalGroup(2, [surface], 3)  # Domain
    
    # Set names
    gmsh.model.setPhysicalName(1, 1, "outer_boundary")
    gmsh.model.setPhysicalName(1, 2, "circle_boundary") 
    gmsh.model.setPhysicalName(2, 3, "domain")
    
    # Generate mesh
    gmsh.model.mesh.generate(2)
    
    # Set mesh format to version 2.2 for compatibility
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    
    # Write mesh file
    gmsh.write("mesh_with_hole.msh")
    
    gmsh.finalize()

if __name__ == "__main__":
    create_mesh_with_hole()
    print("Mesh with hole created: mesh_with_hole.msh")