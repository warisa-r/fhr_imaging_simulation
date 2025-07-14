from dolfin import *
import numpy as np
from scipy.special import hankel1

# Define mesh and function space
mesh = UnitSquareMesh(64, 64)
V_element = FiniteElement("Lagrange", mesh.ufl_cell(), 6)  # Scalar element
V = FunctionSpace(mesh, V_element)  # Scalar function space

#TODO: Use the mesh from previous HH sim
#TODO: Calculate actual k
# Parameters
k_background = 20      #
k_circle = 5000              # wavenumber inside circle
circle_center = np.array([0.5, 0.5])  # circle center
circle_radius = 0.2          # circle radius
x0 = np.array([0.5, -0.05])    # source location (moved away from circle)

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

# Define spatially varying wavenumber
class WavenumberFunction(UserExpression):
    def eval(self, values, x):
        r_from_center = np.linalg.norm(x - circle_center)
        if r_from_center <= circle_radius:
            values[0] = k_circle
        else:
            values[0] = k_background
    def value_shape(self):
        return ()

# Instantiate expressions
u_inc_re = project(HankelReal(degree=2), V)
u_inc_im = project(HankelImag(degree=2), V)
k_func = project(WavenumberFunction(degree=0), FunctionSpace(mesh, "DG", 0))

# Define the outward unit normal vector
n = FacetNormal(mesh)

# Define g, the RHS of the robin boundary condition
g_real = (dot(grad(u_inc_re), n) + k_func * u_inc_im)
g_imag = (dot(grad(u_inc_im), n) - k_func * u_inc_re)

# Define mixed function space using elements
W = FunctionSpace(mesh, V_element * V_element)
(u_re, u_im) = TrialFunctions(W)
(v_re, v_im) = TestFunctions(W)

# Coupled bilinear form
a = (inner(grad(u_re), grad(v_re)) - k_func**2 * u_re * v_re) * dx + k_func * u_im * v_re * ds + \
    (inner(grad(u_im), grad(v_im)) - k_func**2 * u_im * v_im) * dx - k_func * u_re * v_im * ds

# Right-hand side
L = inner(g_real, v_re) * ds + inner(g_imag, v_im) * ds

# Solve the coupled system
w = Function(W)
solve(a == L, w)

# Extract real and imaginary parts
u_sol_re, u_sol_im = w.split()

u_sol_re_proj = project(u_sol_re, V)
u_sol_im_proj = project(u_sol_im, V)

u_mag = Function(V)
u_mag.vector()[:] = np.sqrt(u_sol_re_proj.vector().get_local()**2 + u_sol_im_proj.vector().get_local()**2)

# Output to file (e.g., Paraview)
file_re = File("u_real.pvd")
file_im = File("u_imag.pvd")
file_k = File("wavenumber.pvd")
file_re << u_sol_re
file_im << u_sol_im
file_k << k_func

import matplotlib.pyplot as plt

# Plot the wavenumber distribution
plt.figure()
p = plot(k_func, title="Wavenumber distribution", cmap="coolwarm")
plt.colorbar(p)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# Plot the mesh
plt.figure()
plot(mesh)
plt.title("Mesh")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# Plot real part of total field
plt.figure()
p = plot(u_sol_re, title="Real part of total field", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# Plot imaginary part of total field
plt.figure()
p = plot(u_sol_im, title="Imaginary part of total field", cmap="plasma")
plt.colorbar(p)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# Plot absolute value (magnitude) of total field
plt.figure()
p = plot(u_mag, title="Magnitude of total field", cmap="hot")
plt.colorbar(p)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()