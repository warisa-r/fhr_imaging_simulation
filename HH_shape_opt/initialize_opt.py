from dolfin import *
from dolfin_adjoint import *
import subprocess
import numpy as np
import os

from .util import msh2xml_path

def initialize_opt(msh_file_path):
    xml_path, facet_region_xml_path = msh2xml_path(msh_file_path)

    # Convert mesh if ends with .msh
    print(f"Converting {msh_file_path} to XML format...")
    subprocess.run([
            "dolfin-convert", 
            msh_file_path, 
            xml_path
    ], capture_output=True, text=True)

    # Load mesh and markers from the converted .xml files
    mesh = Mesh(xml_path)
    markers = MeshFunction("size_t", mesh, facet_region_xml_path)

    # Create boundary mesh and design variable
    b_mesh = BoundaryMesh(mesh, "exterior")
    S_b = VectorFunctionSpace(b_mesh, "CG", 1)
    h = Function(S_b, name="Design")
    h.vector()[:] = 0.0  # Zero initial guess for h

    return h, mesh, markers