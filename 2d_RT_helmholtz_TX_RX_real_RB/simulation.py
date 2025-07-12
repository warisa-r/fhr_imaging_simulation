import pandas as pd
import numpy as np
import scipy.io
from scipy.special import hankel1
from mpi4py import MPI

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


comm = MPI.COMM_WORLD

# Load data
pathname_calib = r'./data/'

# Load the frequency vector and take maximum frequency
freq_vec = scipy.io.loadmat(pathname_calib + 'freq_vec.mat')['freq_vec'].flatten()
max_freq = np.max(freq_vec)

# Load the antenna position and take first antenna
ant_pos = np.loadtxt(pathname_calib + 'ant_pos.txt')
ant_pos_2D = ant_pos[:, [0, 2, 3, 5]]  # Take transmitter and receiver positions in x and z dims
first_antenna = ant_pos_2D[0]  # Take first antenna

transmitter_pos = first_antenna[0:2]
receiver_pos = first_antenna[2:4]

print(f"Using frequency: {max_freq:.2e} Hz")
print(f"Transmitter position: {transmitter_pos}")
print(f"Receiver position: {receiver_pos}")

# Define constants
c = 299792458
degree = 6
mesh_order = 2

def run_simulation(freq, transmitter_pos, receiver_pos, visualize=True):
    
    lam0 = c / freq
    print(f"Wavelength: {lam0:.4f} m")

    # wavenumber in free space (air)
    k0 = 2 * np.pi * freq / c

    # Generate mesh
    file_name = "domain.msh"
    generate_mesh(file_name, lam0, mesh_order, receiver_pos)
    mesh, cell_tags, _ = gmshio.read_from_msh(file_name, comm, rank=0, gdim=2)

    # Assigning material parameters
    W = dolfinx.fem.functionspace(mesh, ("DG", 0))
    k_real = dolfinx.fem.Function(W)
    k_real.x.array[:] = k0

    # Set refractive index for different geometry within the computational domain
    n_refractive = 1.5  # Refractive index for the refraction geometry
    n_refractive_metal = 500  # Refractive index for the metal inlets
    k_real.x.array[cell_tags.find(1)] = n_refractive_metal * k0  # Refraction geometry
    k_real.x.array[cell_tags.find(2)] = n_refractive_metal * k0

    # Define source term at boundary
    n = ufl.FacetNormal(mesh)

    # Split incident field into real and imaginary parts
    def u_inc_green_real(x):
        r = np.sqrt((x[0] - transmitter_pos[0])**2 + (x[1] - transmitter_pos[1])**2)
        r = np.where(r < 1e-12, 1e-12, r)
        return np.real(1j * 0.25 *hankel1(0, float(k0) * r))

    def u_inc_green_imag(x):
        r = np.sqrt((x[0] - transmitter_pos[0])**2 + (x[1] - transmitter_pos[1])**2)
        r = np.where(r < 1e-12, 1e-12, r)
        return np.imag(1j * 0.25 * hankel1(0, float(k0) * r))

    # Create function spaces
    element = basix.ufl.element("Lagrange", mesh.topology.cell_name(), degree)
    V = dolfinx.fem.functionspace(mesh, element)
    V_mixed = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([element, element]))

    # Create incident field functions (using same degree as solution)
    uinc_real = fem.Function(V)
    uinc_imag = fem.Function(V)
    uinc_real.interpolate(lambda x: u_inc_green_real(x))
    uinc_imag.interpolate(lambda x: u_inc_green_imag(x))

    # Boundary source terms (split complex g = g_real + i*g_imag)
    g_real = (ufl.dot(ufl.grad(uinc_real), n) + k_real * uinc_imag)
    g_imag = (ufl.dot(ufl.grad(uinc_imag), n) - k_real * uinc_real)

    # Define trial and test functions
    u_mixed = ufl.TrialFunction(V_mixed)
    v_mixed = ufl.TestFunction(V_mixed)

    # Split into components
    u_real, u_imag = ufl.split(u_mixed)
    v_real, v_imag = ufl.split(v_mixed)

    # Bilinear form (LHS)
    a = (
        (ufl.inner(ufl.grad(u_real), ufl.grad(v_real)) * ufl.dx
        - k_real**2 * ufl.inner(u_real, v_real) * ufl.dx
        + k_real * ufl.inner(u_imag, v_real) * ufl.ds)
        
        + (ufl.inner(ufl.grad(u_imag), ufl.grad(v_imag)) * ufl.dx
        - k_real**2 * ufl.inner(u_imag, v_imag) * ufl.dx
        - k_real * ufl.inner(u_real, v_imag) * ufl.ds) 
    )
    # Linear form (RHS)
    L = (ufl.inner(g_real, v_real) * ufl.ds + ufl.inner(g_imag, v_imag) * ufl.ds)

    # Solve the system
    opt = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = dolfinx.fem.petsc.LinearProblem(a, L, petsc_options=opt)
    uh_mixed = problem.solve()

    # Extract real and imaginary parts
    uh_real = fem.Function(V)
    uh_imag = fem.Function(V)
    uh_real.x.array[:] = uh_mixed.x.array[0::2]  # Every other starting from 0 (real parts)
    uh_imag.x.array[:] = uh_mixed.x.array[1::2]  # Every other starting from 1 (imag parts)
    
    uh_real.name = "u_real"
    uh_imag.name = "u_imag"

    # Get solution at receiver point
    def get_complex_solution_at_point(point):
        # Find closest mesh point
        dist = np.linalg.norm(mesh.geometry.x[:, :2] - point, axis=1)
        closest_node = np.argmin(dist)
        
        real_part = uh_real.x.array[closest_node]
        imag_part = uh_imag.x.array[closest_node]
        
        return complex(real_part, imag_part)

    signal = get_complex_solution_at_point(receiver_pos)

    # Visualization
    if visualize:
        pyvista.set_plot_theme("paraview")
        
        # Visualize the solution
        topology, cells, geometry = vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cells, geometry)
        
        # Project DG k_real to continuous space for visualization
        k_continuous = fem.Function(V)
        k_continuous.interpolate(k_real)
        grid.point_data["wavenumber"] = k_continuous.x.array.real
        
        # Add solution data
        grid.point_data["Abs(u)"] = np.sqrt(np.square(uh_real.x.array) + np.square(uh_imag.x.array))
        grid.point_data["Re(u)"] = uh_real.x.array
        grid.point_data["Im(u)"] = uh_imag.x.array
        
        # Mark receiver position
        dist_to_receiver = np.linalg.norm(grid.points[:, :2] - receiver_pos, axis=1)
        closest_idx = np.argmin(dist_to_receiver)
        grid.point_data["receiver_pos"] = np.zeros(len(grid.points))
        grid.point_data["receiver_pos"][closest_idx] = 1.0
        
        # Export visualizations
        export_function(mesh, grid, "wavenumber", show_mesh=True, tessellate=True)
        export_function(mesh, grid, "Abs(u)", show_mesh=False, tessellate=True)
        export_function(mesh, grid, "Re(u)", show_mesh=False, tessellate=True)
        export_function(mesh, grid, "Im(u)", show_mesh=False, tessellate=True)

        # Print information
        print(f"Receiver position: {receiver_pos}")
        print(f"Closest mesh point: {grid.points[closest_idx, :2]}")
        print(f"Distance to receiver: {dist_to_receiver[closest_idx]:.6f}")
        print(f"Signal at receiver: {signal:.6f}")
        print(f"Signal magnitude: {abs(signal):.6f}")
    
    return signal

# Run the simulation
signal = run_simulation(max_freq, transmitter_pos, receiver_pos, visualize=True)