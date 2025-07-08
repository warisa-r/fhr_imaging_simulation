import gmsh
import meshio
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI

gmsh.initialize()
gmsh.model.add("circle_in_box")

# Rectangle domain
Lx, Ly = 1.0, 1.0
gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly, tag=1)

# Internal circular object
cx, cy, r = 0.5, 0.5, 0.2
gmsh.model.occ.addDisk(cx, cy, 0, r, r, tag=2)

# Subtract circle from rectangle to define hole
gmsh.model.occ.cut([(2, 1)], [(2, 2)], tag=3)
gmsh.model.occ.synchronize()

# Add physical groups
gmsh.model.addPhysicalGroup(2, [3], tag=1)  # Main domain
gmsh.model.setPhysicalName(2, 1, "domain")
gmsh.model.addPhysicalGroup(1, [1], tag=2)  # Outer boundary
gmsh.model.setPhysicalName(1, 2, "outer")
gmsh.model.addPhysicalGroup(1, [4], tag=3)  # Circle boundary
gmsh.model.setPhysicalName(1, 3, "circle")

gmsh.model.mesh.generate(2)
mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()
