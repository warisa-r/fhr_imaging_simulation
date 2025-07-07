import sys

from mpi4py import MPI

import gmsh

def add_disk(x, y, r):
    circle = gmsh.model.occ.addCircle(x, y, 0.0, r)
    loop = gmsh.model.occ.addCurveLoop([circle])
    surface = gmsh.model.occ.addPlaneSurface([loop])
    return surface

def generate_mesh(filename: str, lmbda: int, order: int, verbose: bool = False):
    if MPI.COMM_WORLD.rank == 0:
        import gmsh
        gmsh.initialize()
        gmsh.model.add("helmholtz_domain")
        gmsh.option.setNumber("General.Terminal", verbose)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 2 * lmbda)

        rect = gmsh.model.occ.addRectangle(-0.2, -0.15, 0.0, 0.4, 0.3)
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