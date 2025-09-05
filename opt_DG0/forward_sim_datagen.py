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

mesh = Mesh()
# meshes/square_with_sin_perturbed_rect_obstacle.xdmf
with XDMFFile("meshes/square_with_halfsin_perturbed_rect_obstacle.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("meshes/square_with_halfsin_perturbed_rect_obstacle_facets.xdmf") as infile:
    infile.read(mvc, "name_to_read")
    boundary_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)

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

# We have to project the data as a constant on every grid point in order to make it a
# correct approximation of the value of u
import pandas as pd

V_DG0 = FunctionSpace(mesh, "DG", 0)
u_tot_mag_dg0 = project(u_tot_mag, V_DG0)

u_vals_bottom = []
x_vals = []
y_vals = []

for facet in SubsetIterator(boundary_markers, bottom_wall_marker):
    cell = Cell(mesh, facet.entities(2)[0])  # cell adjacent to facet
    dof_idx = V_DG0.dofmap().cell_dofs(cell.index())[0]
    u_val = u_tot_mag_dg0.vector()[dof_idx]

    midpoint = facet.midpoint()
    x_vals.append(midpoint.x())
    y_vals.append(midpoint.y())
    u_vals_bottom.append(u_val)

df = pd.DataFrame({
    "x": x_vals,
    "y": y_vals,
    "u": u_vals_bottom
})
df.to_csv("forward_sim_data_bottom.csv", index=False)

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