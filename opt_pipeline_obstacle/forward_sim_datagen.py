from dolfin import *
import numpy as np
from scipy.special import hankel1
import subprocess
import os
import json
import gmsh
import matplotlib.pyplot as plt

from mesh_generation import obstacle_marker, side_wall_marker, bottom_wall_marker

k_background = 2* np.pi * 5e9 / 299792458 # 2pi f / c

incident_wave_amp = 1
# Define Incident-based incident field (real part)
class IncidentReal(UserExpression):
    def eval(self, values, x):
        values[0] = np.real(incident_wave_amp * np.exp(1j * k_background * x[1]))
    def value_shape(self):
        return ()

# Define Incident-based incident field (imaginary part)
class IncidentImag(UserExpression):
    def eval(self, values, x):
        values[0] = np.imag(incident_wave_amp * np.exp(1j * k_background * x[1]))
    def value_shape(self):
        return ()

print(f"Converting mesh to XML format...")
result = subprocess.run([
    "dolfin-convert", 
    f"meshes/square_with_flattened_circle.msh", 
    f"meshes/square_with_flattened_circle.xml"
], capture_output=True, text=True)

mesh = Mesh(f"meshes/square_with_flattened_circle.xml")
boundary_markers = MeshFunction("size_t", mesh, f"meshes/square_with_flattened_circle_facet_region.xml")

# Define function space
V_element = FiniteElement("CG", mesh.ufl_cell(), 5)
V = FunctionSpace(mesh, V_element)

# Instantiate expressions
u_inc_re = project(IncidentReal(degree=2), V)
u_inc_im = project(IncidentImag(degree=2), V)

# Define the outward unit normal vector
n = FacetNormal(mesh)

ds_bottom = Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=bottom_wall_marker)
ds_sides = Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=side_wall_marker)
ds_obstacle = Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=obstacle_marker)

ds_outer = ds_bottom + ds_sides

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

### Save the data ###

import pandas as pd

# Extract u values (total field magnitude) along the bottom wall
bottom_vertex_indices = set()
for facet in SubsetIterator(boundary_markers, bottom_wall_marker):
    for v in vertices(facet):
        bottom_vertex_indices.add(v.index())

dof_coords = V.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
vertex_coords = mesh.coordinates()
tol = 1e-10
bottom_dof_indices = []
for vi in bottom_vertex_indices:
    v_coord = vertex_coords[vi]
    matches = np.where(np.linalg.norm(dof_coords - v_coord, axis=1) < tol)[0]
    bottom_dof_indices.extend(matches)
bottom_dof_indices = np.unique(bottom_dof_indices)

u_vals_bottom = u_tot_mag.vector().get_local()[bottom_dof_indices]
x_vals = []
y_vals = []
for idx in bottom_dof_indices:
    x_vals.append(dof_coords[idx, 0])
    y_vals.append(dof_coords[idx, 1])

df = pd.DataFrame({
    "x": x_vals,
    "y": y_vals,
    "u": u_vals_bottom
})

#df.to_csv("forward_sim_data_bottom.csv", index=False)

### Check saved data integrity

#print(simulation_df.head())
#plt.figure(figsize=(6, 5))
#plt.scatter(simulation_df['x'], simulation_df['y'], c=simulation_df['u_total_magnitude'], cmap='viridis', s=10)
#plt.colorbar(label='|u_total|')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Total Field Magnitude |u_total|')
#plt.axis('equal')
#plt.tight_layout()
#plt.show()
##############################

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