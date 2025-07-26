from dolfin import *
import numpy as np
from scipy.special import hankel1
import subprocess
import os
import json
import gmsh
import matplotlib.pyplot as plt

from mesh_generation import domain_boundary_marker, obstacle_marker

k_background = 2* np.pi * 5e9 / 299792458 # 2pi f / c
x0 = np.array([0.5, -1.0])  # source location

# Define Hankel-based incident field (real part)
class HankelReal(UserExpression):
    def eval(self, values, x):
        r = np.linalg.norm(x - x0)
        if r < 1e-12:
            values[0] = 0.0
        else:
            values[0] = np.real(hankel1(0, k_background * r))
    def value_shape(self):
        return ()

# Define Hankel-based incident field (imaginary part)
class HankelImag(UserExpression):
    def eval(self, values, x):
        r = np.linalg.norm(x - x0)
        if r < 1e-12:
            values[0] = 0.0
        else:
            values[0] = np.imag(hankel1(0, k_background * r))
    def value_shape(self):
        return ()

print(f"Converting mesh to XML format...")
result = subprocess.run([
    "dolfin-convert", 
    f"rough_top_mesh.msh", 
    f"rough_top_mesh.xml"
], capture_output=True, text=True)

mesh = Mesh(f"rough_top_mesh.xml")
boundary_markers = MeshFunction("size_t", mesh, f"rough_top_mesh_facet_region.xml")

# Define function space
V_element = FiniteElement("Lagrange", mesh.ufl_cell(), 5)
V = FunctionSpace(mesh, V_element)

# Instantiate expressions
u_inc_re = project(HankelReal(degree=2), V)
u_inc_im = project(HankelImag(degree=2), V)

# Define the outward unit normal vector
n = FacetNormal(mesh)

# Define boundary measures
ds_outer = Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=domain_boundary_marker)
ds_obstacle = Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=obstacle_marker)

# Define mixed function space
W = FunctionSpace(mesh, V_element * V_element)
(u_re, u_im) = TrialFunctions(W)
(v_re, v_im) = TestFunctions(W)

# Coupled bilinear form
a = (inner(grad(u_re), grad(v_re)) - k_background**2 * u_re * v_re) * dx + k_background * u_im * v_re * ds_outer + \
    (inner(grad(u_im), grad(v_im)) - k_background**2 * u_im * v_im) * dx - k_background * u_re * v_im * ds_outer

# Homogeneous RHS
L = Constant(0.0) * (v_re + v_im) * dx

# Dirichlet boundary conditions on circle boundary (u = -u_inc)
u_inc_re_neg = Function(V)
u_inc_im_neg = Function(V)
u_inc_re_neg.vector()[:] = -u_inc_re.vector()[:]
u_inc_im_neg.vector()[:] = -u_inc_im.vector()[:]

bc_circle_re = DirichletBC(W.sub(0), u_inc_re_neg, boundary_markers, obstacle_marker)
bc_circle_im = DirichletBC(W.sub(1), u_inc_im_neg, boundary_markers, obstacle_marker)
bcs = [bc_circle_re, bc_circle_im]

# Solve the coupled system
w = Function(W)
solve(a == L, w, bcs)

# Extract real and imaginary parts
u_sol_re, u_sol_im = w.split()
u_sol_re_proj = project(u_sol_re, V)
u_sol_im_proj = project(u_sol_im, V)

# Create total field: u_tot = u_inc + u_sol
u_tot_re = Function(V)
u_tot_im = Function(V)
u_tot_re.vector()[:] = u_inc_re.vector()[:] + u_sol_re_proj.vector()[:]
u_tot_im.vector()[:] = u_inc_im.vector()[:] + u_sol_im_proj.vector()[:]

# Calculate magnitude of total field
u_tot_mag = Function(V)
u_tot_mag.vector()[:] = np.sqrt(u_tot_re.vector().get_local()**2 + u_tot_im.vector().get_local()**2)

simulation_data = {
    'coordinates': mesh.coordinates(),
    'cells': mesh.cells(),
    'u_total_real': u_tot_re.vector().get_local(),
    'u_total_imag': u_tot_im.vector().get_local(),
    'u_total_magnitude': u_tot_mag.vector().get_local(),
    'u_scattered_real': u_sol_re_proj.vector().get_local(),
    'u_scattered_imag': u_sol_im_proj.vector().get_local(),
    'u_incident_real': u_inc_re.vector().get_local(),
    'u_incident_imag': u_inc_im.vector().get_local(),
    'k_background': k_background,
}

np.savez_compressed(f"forward_sim_data.npz", **simulation_data)

metadata = {
    'k_background': float(k_background),
    'num_vertices': mesh.num_vertices(),
    'num_cells': mesh.num_cells(),
    'function_space_degree': 5,
    'description': 'Forward simulation results for inverse problem'
}

with open(f"forward_sim_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

# Plot magnitude of total field
plt.figure()
p = plot(u_tot_mag, title="Magnitude of total field (u_inc + u_sol)", cmap="hot")
plt.colorbar(p)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# Plot real part of total field
plt.figure()
p = plot(u_tot_re, title="Real part of total field (u_inc + u_sol)", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# Plot real part of scattered field
plt.figure()
p = plot(u_sol_re_proj, title="Real part of scattered field", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")