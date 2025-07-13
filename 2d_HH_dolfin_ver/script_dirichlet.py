from dolfin import *
import numpy as np
from scipy.special import hankel1
import subprocess
import os

# Install GMSH if not available
try:
    import gmsh
except ImportError:
    print("Installing GMSH...")
    subprocess.run(["apt", "update"], check=True)
    subprocess.run(["apt", "install", "-y", "gmsh", "python3-gmsh"], check=True)
    import gmsh

# Create mesh with hole using GMSH
print("Creating mesh with hole...")
subprocess.run(["python3", "create_mesh_with_hole.py"], check=True)

# Try to convert mesh to XML format
print("Converting mesh to XML format...")
result = subprocess.run(["dolfin-convert", "mesh_with_hole.msh", "mesh_with_hole.xml"], 
                        capture_output=True, text=True)
mesh = Mesh("mesh_with_hole.xml")
boundary_markers = MeshFunction("size_t", mesh, "mesh_with_hole_facet_region.xml")

# Define function space
V_element = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, V_element)

# Parameters
k_background = 20.0
x0 = np.array([0.5, -0.05])  # source location

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

# Instantiate expressions
u_inc_re = project(HankelReal(degree=2), V)
u_inc_im = project(HankelImag(degree=2), V)

# Define the outward unit normal vector
n = FacetNormal(mesh)

# Define boundary measures
ds_outer = Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=1)
ds_circle = Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=2)

# Define g for Robin BC on outer boundary
g_real = (dot(grad(u_inc_re), n) + k_background * u_inc_im)
g_imag = (dot(grad(u_inc_im), n) - k_background * u_inc_re)

# Define mixed function space
W = FunctionSpace(mesh, V_element * V_element)
(u_re, u_im) = TrialFunctions(W)
(v_re, v_im) = TestFunctions(W)

# Coupled bilinear form
a = (inner(grad(u_re), grad(v_re)) - k_background**2 * u_re * v_re) * dx + k_background * u_im * v_re * ds_outer + \
    (inner(grad(u_im), grad(v_im)) - k_background**2 * u_im * v_im) * dx - k_background * u_re * v_im * ds_outer

# Right-hand side (only on outer boundary)
L = inner(g_real, v_re) * ds_outer + inner(g_imag, v_im) * ds_outer

# Dirichlet boundary conditions on circle boundary (u = 0)
bc_circle_re = DirichletBC(W.sub(0), Constant(0.0), boundary_markers, 2)
bc_circle_im = DirichletBC(W.sub(1), Constant(0.0), boundary_markers, 2)
bcs = [bc_circle_re, bc_circle_im]

# Solve the coupled system
w = Function(W)
solve(a == L, w, bcs)

# Extract real and imaginary parts
u_sol_re, u_sol_im = w.split()

u_sol_re_proj = project(u_sol_re, V)
u_sol_im_proj = project(u_sol_im, V)

u_mag = Function(V)
u_mag.vector()[:] = np.sqrt(u_sol_re_proj.vector().get_local()**2 + u_sol_im_proj.vector().get_local()**2)

# Output to file
file_re = File("u_real.pvd")
file_im = File("u_imag.pvd")
file_mag = File("u_magnitude.pvd")
file_re << u_sol_re_proj
file_im << u_sol_im_proj
file_mag << u_mag

import matplotlib.pyplot as plt

# Plot the mesh with hole
plt.figure()
plot(mesh)
plt.title("Mesh with circular hole")
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
plt.show()

# Plot imaginary part of scattered field
plt.figure()
p = plot(u_sol_im_proj, title="Imaginary part of scattered field", cmap="plasma")
plt.colorbar(p)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# Plot absolute value (magnitude) of scattered field
plt.figure()
p = plot(u_mag, title="Magnitude of scattered field", cmap="hot")
plt.colorbar(p)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()