from mpi4py import MPI
import sys

import numpy as np
import scipy.io
from scipy.special import hankel1

import dolfinx.fem.petsc
import ufl
import basix.ufl
from petsc4py import PETSc
from dolfinx.io import gmshio
from dolfinx.plot import vtk_mesh
import matplotlib.pyplot as plt
import pyvista
from dolfinx.fem import Function
from dolfinx import fem

from mesh_generation import generate_mesh
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export_function import export_function

pathname_calib = r'./data/'

# Load the frequency vector
freq_vec_init = scipy.io.loadmat(pathname_calib + 'freq_vec.mat')['freq_vec'].flatten()
freq_vec = np.flip(freq_vec_init) # Flip to get up-chirp

# Load the antenna position
ant_pos = np.loadtxt(pathname_calib + 'ant_pos.txt')
ant_pos_2D = ant_pos[:, [0, 2, 3, 5]] # Take receiver position in x and z dims only

transmitter_pos = ant_pos_2D[0, 0:2]
receiver_pos = ant_pos_2D[0, 2:4]
# Define some constants
c = 299792458
interested_f = freq_vec.min()
lam0 = c / interested_f

# wavenumber in free space (air)
k0 = 2* np.pi * interested_f / c

# Polynomial degree
degree = 6

# Mesh order
mesh_order = 2

comm = MPI.COMM_WORLD

generate_mesh("2d_RT_helmholtz/domain.msh", lam0, mesh_order)
file_name = "2d_RT_helmholtz/domain.msh"
mesh, cell_tags, _ = gmshio.read_from_msh(file_name, comm, rank=0, gdim=2)

# Assigning material parameter
W = dolfinx.fem.functionspace(mesh, ("DG", 0))
k_real = dolfinx.fem.Function(W)
k_real.x.array[:] = k0
# Set refractive index for different geometry within the computational domain
n_refractive = 1.5 # Refractive index for the refraction geometry
n_refractive_metal = 500 # Refractive index for the metal inlets
k_real.x.array[cell_tags.find(1)] = n_refractive_metal * k0 # Refraction geometry
k_real.x.array[cell_tags.find(2)] = n_refractive_metal * k0

# Define source term at boundary
n = ufl.FacetNormal(mesh)

# Split incident field into real and imaginary parts
def hankel_incident_real_eval(x):
    r = np.sqrt((x[0] - transmitter_pos[0])**2 + (x[1] - transmitter_pos[1])**2)
    r = np.where(r < 1e-12, 1e-12, r)
    return np.real(hankel1(0, float(k0) * r))

def hankel_incident_imag_eval(x):
    r = np.sqrt((x[0] - transmitter_pos[0])**2 + (x[1] - transmitter_pos[1])**2)
    r = np.where(r < 1e-12, 1e-12, r)
    return np.imag(hankel1(0, float(k0) * r))

V_scalar = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
uinc_real = fem.Function(V_scalar)
uinc_imag = fem.Function(V_scalar)
uinc_real.interpolate(lambda x: hankel_incident_real_eval(x))
uinc_imag.interpolate(lambda x: hankel_incident_imag_eval(x))

# Boundary source terms (split complex g = g_real + i*g_imag)
# g = ufl.dot(ufl.grad(uinc), n) - 1j * k * uinc

g_real = (ufl.dot(ufl.grad(uinc_real), n) + k_real * uinc_imag)
g_imag = (ufl.dot(ufl.grad(uinc_imag), n) - k_real * uinc_real)

# Define the weak form for the real system
# Original: (-∇²u + k²u - ik u|∂Ω) = g
# Split: u = u_real + i*u_imag
# Real part: -∇²u_real + k²u_real + k u_imag|∂Ω = g_real
# Imag part: -∇²u_imag + k²u_imag - k u_real|∂Ω = g_imag

element = basix.ufl.element("Lagrange", mesh.topology.cell_name(), degree)
V = dolfinx.fem.functionspace(mesh, element)

# Create mixed function space for (u_real, u_imag)
V_mixed = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([element, element]))

# Define trial and test functions
u_mixed = ufl.TrialFunction(V_mixed)
v_mixed = ufl.TestFunction(V_mixed)

# Split into components
u_real, u_imag = ufl.split(u_mixed)
v_real, v_imag = ufl.split(v_mixed)

# Bilinear form (LHS)
a = (
    # Real equation: -∇²u_real + k²u_real + k u_imag|∂Ω = 0
    (-ufl.inner(ufl.grad(u_real), ufl.grad(v_real)) * ufl.dx
    + k_real**2 * ufl.inner(u_real, v_real) * ufl.dx
    - k_real * ufl.inner(u_imag, v_real) * ufl.ds)
    
    # Imaginary equation: -∇²u_imag + k²u_imag - k u_real|∂Ω = 0
    + (-ufl.inner(ufl.grad(u_imag), ufl.grad(v_imag)) * ufl.dx
    + k_real**2 * ufl.inner(u_imag, v_imag) * ufl.dx
    + k_real * ufl.inner(u_real, v_imag) * ufl.ds)
)

# Linear form (RHS)
L = -(ufl.inner(g_real, v_real) * ufl.ds - ufl.inner(g_imag, v_imag) * ufl.ds)

# Solve the system
opt = {
    "ksp_type": "preonly",
    "pc_type": "lu"
}
problem = dolfinx.fem.petsc.LinearProblem(a, L, petsc_options=opt)
uh_mixed = problem.solve()

uh_real = fem.Function(V)
uh_imag = fem.Function(V)
# Extract real and imaginary parts
#uh_real, uh_imag = uh_mixed.split()
uh_real.x.array[:] = uh_mixed.x.array[0::2]  # Every other starting from 0 (real parts)
uh_imag.x.array[:] = uh_mixed.x.array[1::2]  # Every other starting from 1 (imag parts)

uh_real.name = "u_real"
uh_imag.name = "u_imag"

# Reconstruct complex solution for output
def get_complex_solution_at_point(point):
    # Find closest mesh point
    dist = np.linalg.norm(mesh.geometry.x[:, :2] - point, axis=1)
    closest_node = np.argmin(dist)
    
    real_part = uh_real.x.array[closest_node]
    imag_part = uh_imag.x.array[closest_node]
    
    return complex(real_part, imag_part)

# Visualize the mesh and wavenumber
pyvista.set_plot_theme("paraview")
sargs = dict(
    title_font_size=25,
    label_font_size=20,
    fmt="%.2e",
    color="black",
    position_x=0.1,
    position_y=0.8,
    width=0.8,
    height=0.1,
)

grid = pyvista.UnstructuredGrid(*vtk_mesh(V))
grid.cell_data["wavenumber"] = k_real.x.array.real
export_function(mesh, grid, "wavenumber", show_mesh=True, tessellate=True)

# Visualize the solution using the scalar function space V
topology, cells, geometry = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cells, geometry)

# Project DG k_real to continuous space for visualization
k_continuous = fem.Function(V)
k_continuous.interpolate(k_real)
grid.point_data["wavenumber"] = k_continuous.x.array.real

export_function(mesh, grid, "wavenumber", show_mesh=True, tessellate=True)

# Now use the scalar functions for solution visualization
grid.point_data["Abs(u)"] = np.sqrt(np.square(uh_real.x.array) + np.square(uh_imag.x.array))
grid.point_data["Re(u)"] = uh_real.x.array
grid.point_data["Im(u)"] = uh_imag.x.array

# Export visualizations
export_function(mesh, grid, "Abs(u)", show_mesh=False, tessellate=True)
