from dolfin import *
from dolfin_adjoint import *
import subprocess
import numpy as np
import os

def initialize_opt(msh_file_path):
    # Convert mesh if ends with .msh
    if msh_file_path.endswith('.msh'):
        xml_path = msh_file_path.replace('.msh', '.xml')
        print(f"Converting {msh_file_path} to XML format...")
        subprocess.run([
            "dolfin-convert", 
            msh_file_path, 
            xml_path
        ], capture_output=True, text=True)
    else:
        # If fed xml path, we assume that a facet region file already exist
        xml_path = msh_file_path

    # Build facet_region_path by inserting '_facet_region' before '.xml'
    base, ext = os.path.splitext(xml_path)
    facet_region_path = f"{base}_facet_region{ext}"

    # Load mesh and markers from the converted .xml files
    mesh = Mesh(xml_path)
    markers = MeshFunction("size_t", mesh, facet_region_path)

    # Create boundary mesh and design variable
    b_mesh = BoundaryMesh(mesh, "exterior")
    S_b = VectorFunctionSpace(b_mesh, "CG", 1)
    h = Function(S_b, name="Design")
    h.vector()[:] = 0.0  # Zero initial guess for h

    return h, mesh, markers