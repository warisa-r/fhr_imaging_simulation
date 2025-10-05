from dolfin import *
from dolfin_adjoint import *
import meshio
import subprocess
import numpy as np
import os

from .generate_mesh import convert_msh_to_xdmf

# To get the path of an xml file and its facet region file from msh assuming that
# dolfin-convert has been called


def msh2xml_path(msh_file_path):
    xml_path = msh_file_path.replace('.msh', '.xml')
    base, ext = os.path.splitext(xml_path)
    facet_region_xml_path = f"{base}_facet_region{ext}"
    return xml_path, facet_region_xml_path


def msh2xdmf_path(msh_file_path):
    base_path = os.path.splitext(msh_file_path)[0]
    xdmf_path = base_path + ".xdmf"
    facet_xdmf_path = base_path + "_facets.xdmf"
    domain_xdmf_path = base_path + "_domains.xdmf"
    return xdmf_path, facet_xdmf_path, domain_xdmf_path


class MeshUtil():
    def __init__(self, msh_file_path, markers_dict, obstacle_stiffness):
        self.msh_file_path = msh_file_path
        self.mesh_xdmf_file_path, self.mesh_facets_xdmf_file_path, self.mesh_domain_xdmf_file_path = msh2xdmf_path(
            msh_file_path)

        # Check if XDMF files exist, if not, generate them
        if not (os.path.exists(self.mesh_xdmf_file_path)
                and os.path.exists(self.mesh_facets_xdmf_file_path)
                and os.path.exists(self.mesh_domain_xdmf_file_path)):
            convert_msh_to_xdmf(self.msh_file_path)

        self.mesh = None
        self.boundary_markers = None
        self.domain_markers = None
        self.markers_dict = markers_dict
        self.obstacle_stiffness = obstacle_stiffness

    @classmethod
    def from_xdmf(cls, mesh_xdmf_file_path, mesh_facets_xdmf_file_path, mesh_domain_xdmf_file_path,
                  markers_dict, obstacle_stiffness):
        instance = cls.__new__(cls)  # Create an uninitialized instance
        instance.mesh_xdmf_file_path = mesh_xdmf_file_path
        instance.mesh_facets_xdmf_file_path = mesh_facets_xdmf_file_path
        instance.mesh_domain_xdmf_file_path = mesh_domain_xdmf_file_path
        instance.mesh = None
        instance.boundary_markers = None
        instance.domain_markers = None
        instance.markers_dict = markers_dict
        instance.obstacle_stiffness = obstacle_stiffness
        return instance

    def get_mesh_and_markers(self, create_new_object=False):
        # Create new object only when called. Sometimes we need fresh new mesh.
        # Not the already modified one.
        if self.mesh is None or self.boundary_markers is None or create_new_object == True:
            # Read mesh
            self.mesh = Mesh()
            with XDMFFile(self.mesh_xdmf_file_path) as infile:
                infile.read(self.mesh)
            # Read mesh boundary markers
            mf = MeshValueCollection("size_t", self.mesh, 1)
            with XDMFFile(self.mesh_facets_xdmf_file_path) as infile:
                infile.read(mf, "name_to_read")
                self.boundary_markers = cpp.mesh.MeshFunctionSizet(
                    self.mesh, mf)
            # Read mesh domain markers
            mvc = MeshValueCollection("size_t", self.mesh, self.mesh.topology().dim())
            with XDMFFile(self.mesh_domain_xdmf_file_path) as infile:
                # must match the name used in meshio
                infile.read(mvc, "name_to_read")
                self.domain_markers = cpp.mesh.MeshFunctionSizet(
                    self.mesh, mvc)

        # TODO: Also return domain_markers and change the existing code that use this version
        # TODO: Differentiate all the markers to boundary_markers not just simply markers
        return self.mesh, self.boundary_markers, self.domain_markers


def initialize_opt_xdmf(msh_file_path):
    comm = MPI.comm_world
    xdmf_path, facet_xdmf_path, domain_xdmf_path = msh2xdmf_path(msh_file_path)

    # --- Convert .msh to .xdmf using meshio (only on rank 0) ---
    if comm.rank == 0:
        msh = meshio.read(msh_file_path)

        # Extract 2D points
        points_2d = msh.points[:, :2]

        # Create and write the domain mesh (triangles)
        triangle_cells = msh.get_cells_type("triangle")
        domain_mesh = meshio.Mesh(points=points_2d, cells=[
                                  ("triangle", triangle_cells)])
        domain_mesh.write(xdmf_path)

        # Create and write the facet mesh (lines)
        line_cells = msh.get_cells_type("line")
        line_data = msh.get_cell_data("gmsh:physical", "line")
        facet_mesh = meshio.Mesh(
            points=points_2d,
            cells=[("line", line_cells)],
            cell_data={"name_to_read": [line_data]}
        )
        facet_mesh.write(facet_xdmf_path)

    # All processes wait here until the file conversion is done
    comm.barrier()

    # Load mesh and markers from .xdmf files
    mesh = Mesh()
    with XDMFFile(xdmf_path) as infile:
        infile.read(mesh)

    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
    with XDMFFile(facet_xdmf_path) as infile:
        infile.read(mvc, "name_to_read")
        markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    mvc_dom = MeshValueCollection("size_t", mesh, mesh.topology().dim())
    with XDMFFile(domain_xdmf_path) as infile:
        # must match the name used in meshio
        infile.read(mvc_dom, "name_to_read")
        domain_markers = cpp.mesh.MeshFunctionSizet(
            mesh, mvc_dom)


    # Create boundary mesh and design variable
    b_mesh = BoundaryMesh(mesh, "exterior")
    S_b = VectorFunctionSpace(b_mesh, "CG", 1)
    h = Function(S_b, name="Design")
    h.vector()[:] = 0.0  # Zero initial guess for h

    return h, mesh, markers, domain_markers
