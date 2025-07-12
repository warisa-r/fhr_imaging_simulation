import sys

from mpi4py import MPI

import gmsh

def add_disk(x, y, r):
    circle = gmsh.model.occ.addCircle(x, y, 0.0, r)
    loop = gmsh.model.occ.addCurveLoop([circle])
    surface = gmsh.model.occ.addPlaneSurface([loop])
    return surface

def generate_mesh(filename, lmbda, order, receiver_pos, verbose = False):
        import gmsh
        gmsh.initialize()
        gmsh.model.add("helmholtz_domain")
        gmsh.option.setNumber("General.Terminal", verbose)
        
        # Use direct mesh size instead of CharacteristicLengthFactor
        mesh_size = lmbda / 4
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size / 4)
        
        # Turn off automatic mesh sizing to force our values
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

        delta = lmbda * 6 # A little edge so that the receiver doesn't stand exactly on BC
        comp_domain_x_low = min(-0.2, receiver_pos[0])
        comp_domain_dx = max(0.2, receiver_pos[0]) - comp_domain_x_low + delta
        comp_domain_y_low = min(-0.15, receiver_pos[1])
        comp_domain_dy = max(0.15, receiver_pos[1]) - comp_domain_y_low + delta
        rect = gmsh.model.occ.addRectangle(comp_domain_x_low, comp_domain_y_low, 0.0, comp_domain_dx, comp_domain_dy)
        
        p0 = add_disk(0.0, 0.0, 0.1)
        p1 = add_disk(-0.03, -0.015, 0.015)
        p2 = add_disk(0.07, -0.015, 0.015)
        p3 = add_disk(0.01, 0.05, 0.01)

        all_surfaces = [(2, rect), (2, p0), (2, p1), (2, p2), (2, p3)]
        gmsh.model.occ.fragment(all_surfaces, []) # From just surfaces, we put the fragments together
        gmsh.model.occ.synchronize()

        # We now need to assign each surface to be in different physical groups
        # Because the geometry we define earlier collapse once we fragments thing together
        rect_surf = None
        p0_surf = None
        p_metal_surfs = []
        # Get all the different existing entities in the mesh
        surfaces = gmsh.model.getEntities(2)
        areas = [gmsh.model.occ.getMass(dim, tag) for dim, tag in surfaces]

        # Find the largest area (background)
        max_area_idx = areas.index(max(areas))
        rect_surf = surfaces[max_area_idx][1]

        # Find P0 (center near origin, but not the largest area)
        p0_surf = None
        p_metal_surfs = []
        for i, (dim, tag) in enumerate(surfaces):
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            if i == max_area_idx:
                continue  # NOT BACKGROUND
            # P0: center near origin, radius ~0.1
            if (com[0]**2 + com[1]**2)**0.5 < 0.01:
                p0_surf = tag
            else:
                p_metal_surfs.append(tag)

        gmsh.model.addPhysicalGroup(2, [rect_surf], tag=3)  # Background
        if p0_surf is not None:
            gmsh.model.addPhysicalGroup(2, [p0_surf], tag=1)    # P0
        if p_metal_surfs:
            gmsh.model.addPhysicalGroup(2, p_metal_surfs, tag=2)  # P1, P2, P3

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
        gmsh.model.mesh.optimize("HighOrder")
        gmsh.write(filename)
        gmsh.finalize()