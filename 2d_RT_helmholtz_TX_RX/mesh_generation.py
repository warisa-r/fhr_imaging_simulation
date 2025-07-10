import sys

from mpi4py import MPI

import gmsh

def add_disk(x, y, r):
    circle = gmsh.model.occ.addCircle(x, y, 0.0, r)
    loop = gmsh.model.occ.addCurveLoop([circle])
    surface = gmsh.model.occ.addPlaneSurface([loop])
    return surface

def generate_mesh(filename, lmbda, order, receiver_pos, verbose = False):
    if MPI.COMM_WORLD.rank == 0:
        import gmsh
        gmsh.initialize()
        gmsh.model.add("helmholtz_domain")
        gmsh.option.setNumber("General.Terminal", verbose)
        
        # Use direct mesh size instead of CharacteristicLengthFactor
        mesh_size = lmbda / 10
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size / 4)
        
        # Turn off automatic mesh sizing to force our values
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

        delta = lmbda * 2 # A little edge so that the receiver doesn't stand exactly on BC
        comp_domain_x_low = min(-0.2, receiver_pos[0])
        comp_domain_dx = max(0.2, receiver_pos[0]) - comp_domain_x_low + delta
        comp_domain_y_low = min(-0.15, receiver_pos[1])
        comp_domain_dy = max(0.15, receiver_pos[1]) - comp_domain_y_low + delta
        rect = gmsh.model.occ.addRectangle(comp_domain_x_low, comp_domain_y_low, 0.0, comp_domain_dx, comp_domain_dy)
        
        p0 = add_disk(0.0, 0.0, 0.1)
        p1 = add_disk(-0.03, -0.015, 0.015)
        p2 = add_disk(0.07, -0.015, 0.015)
        p3 = add_disk(0.01, 0.05, 0.01)

        # Cut out metal disks from the domain (creates holes)
        metal_disks = [(2, p1), (2, p2), (2, p3)]
        domain_with_holes = gmsh.model.occ.cut([(2, rect)], metal_disks)
        
        # Fragment with P0 (which we want to mesh)
        all_surfaces = domain_with_holes[0] + [(2, p0)]
        gmsh.model.occ.fragment(all_surfaces, [])
        gmsh.model.occ.synchronize()

        # Get all surfaces and curves after fragmentation
        surfaces = gmsh.model.getEntities(2)
        curves = gmsh.model.getEntities(1)
        
        # Find surface areas
        areas = [gmsh.model.occ.getMass(dim, tag) for dim, tag in surfaces]

        # Find the largest area (background)
        max_area_idx = areas.index(max(areas))
        rect_surf = surfaces[max_area_idx][1]

        # Find P0 surface (center near origin, but not the largest area)
        p0_surf = None
        for i, (dim, tag) in enumerate(surfaces):
            if i == max_area_idx:
                continue  # Skip background
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            # P0: center near origin, radius ~0.1
            if (com[0]**2 + com[1]**2)**0.5 < 0.05:  # Increased tolerance
                p0_surf = tag
                break

        # Find boundary curves of the holes (metal surfaces)
        metal_boundary_curves = []
        for dim, tag in curves:
            # Get the curve's center of mass
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            
            # Check if this curve is on the boundary of one of the metal disks
            # P1 center: (-0.03, -0.015)
            if abs(com[0] - (-0.03)) < 0.02 and abs(com[1] - (-0.015)) < 0.02:
                metal_boundary_curves.append(tag)
            # P2 center: (0.07, -0.015)
            elif abs(com[0] - 0.07) < 0.02 and abs(com[1] - (-0.015)) < 0.02:
                metal_boundary_curves.append(tag)
            # P3 center: (0.01, 0.05)
            elif abs(com[0] - 0.01) < 0.015 and abs(com[1] - 0.05) < 0.015:
                metal_boundary_curves.append(tag)

        # Add physical groups
        gmsh.model.addPhysicalGroup(2, [rect_surf], tag=3)  # Background
        if p0_surf is not None:
            gmsh.model.addPhysicalGroup(2, [p0_surf], tag=1)    # P0
        if metal_boundary_curves:
            gmsh.model.addPhysicalGroup(1, metal_boundary_curves, tag=2)  # Metal boundaries (1D)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
        gmsh.model.mesh.optimize("HighOrder")
        gmsh.write(filename)
        gmsh.finalize()