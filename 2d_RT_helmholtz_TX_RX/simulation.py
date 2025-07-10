from mpi4py import MPI
import sys

import pandas as pd
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

# MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Only rank 0 loads data and broadcasts
if rank == 0:
    pathname_calib = r'./data/'
    
    # Load the frequency vector
    freq_vec = scipy.io.loadmat(pathname_calib + 'freq_vec.mat')['freq_vec'].flatten()

    print(freq_vec)
    # Load the antenna position
    ant_pos = np.loadtxt(pathname_calib + 'ant_pos.txt')
    ant_pos_2D = ant_pos[:, [0, 2, 3, 5]] # Take the transmitter and receiver positions in x and z dims only
    
    print(f"Loaded {len(freq_vec)} frequencies and {len(ant_pos_2D)} antennas")
else:
    freq_vec = None
    ant_pos_2D = None

# Broadcast data to all processes
freq_vec = comm.bcast(freq_vec, root=0)
ant_pos_2D = comm.bcast(ant_pos_2D, root=0)

# Define some constants
c = 299792458

# Polynomial degree
degree = 6

# Mesh order
mesh_order = 2

def run_sim(freq, transmitter_pos, receiver_pos, visualize = False):
    
    lam0 = c / freq
    print(lam0)

    # wavenumber in free space (air)
    k0 = 2* np.pi * freq / c

    file_name = f"domain_rank{rank}.msh"
    generate_mesh(file_name, lam0, mesh_order, receiver_pos)
    mesh, cell_tags, _ = gmshio.read_from_msh(file_name, comm, rank=0, gdim=2)

    # Assigning material parameter
    W = dolfinx.fem.functionspace(mesh, ("DG", 0))
    k = dolfinx.fem.Function(W)
    k.x.array[:] = k0

    # Set refractive index for different geometry within the computational domain
    n_refractive = 1.5 # Refractive index for the refraction geometry
    n_refractive_metal = 500 # Refractive index for the metal inlets
    k.x.array[cell_tags.find(1)] = n_refractive * k0 # Refraction geometry
    k.x.array[cell_tags.find(2)] = n_refractive_metal * k0

    # Define source term at boundary
    n = ufl.FacetNormal(mesh)

    # UFL does not support hankel1, so use a Function for boundary data
    def hankel_incident_eval(x):
        r = np.sqrt((x[0] - transmitter_pos[0])**2 + (x[1] - transmitter_pos[1])**2)
        r = np.where(r < 1e-12, 1e-12, r)
        return hankel1(0, float(k0) * r)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    uinc = fem.Function(V)
    uinc.interpolate(lambda x: hankel_incident_eval(x))
    g = ufl.dot(ufl.grad(uinc), n) - 1j * k * uinc

    # Define the weak form of the problem
    element = basix.ufl.element("Lagrange", mesh.topology.cell_name(), degree)
    V = dolfinx.fem.functionspace(mesh, element)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    # LHS
    a = (
        -ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + k**2 * ufl.inner(u, v) * ufl.dx
        - 1j * k * ufl.inner(u, v) * ufl.ds
    )
    L = ufl.inner(g, v) * ufl.ds # RHS

    opt = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = dolfinx.fem.petsc.LinearProblem(a, L, petsc_options=opt)
    uh = problem.solve()
    uh.name = "u"

    # Only visualize on rank 0 and only first simulation
    if visualize and rank == 0:
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

        grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh))
        grid.cell_data["wavenumber"] = k.x.array.real
        export_function(mesh, grid, "wavenumber", show_mesh=True, tessellate=True)

        # Visualize the solution
        topology, cells, geometry = vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cells, geometry)
        grid.point_data["Abs(u)"] = np.abs(uh.x.array)
        
        # Add receiver position as a point
        receiver_point = np.array([receiver_pos[0], receiver_pos[1], 0.0]).reshape(1, -1)
        grid.point_data["receiver_pos"] = np.zeros(len(grid.points))
        
        # Find closest point to receiver and mark it
        dist_to_receiver = np.linalg.norm(grid.points - receiver_point, axis=1)
        closest_idx = np.argmin(dist_to_receiver)
        grid.point_data["receiver_pos"][closest_idx] = 1.0
        
        export_function(mesh, grid, "Abs(u)", show_mesh=False, tessellate=True)

        grid.point_data["Re(u)"] = np.real(uh.x.array)
        export_function(mesh, grid, "Re(u)", show_mesh=False, tessellate=True)

        # Print receiver position info
        print(f"Receiver position: {receiver_pos}")
        print(f"Closest mesh point: {grid.points[closest_idx]}")
        print(f"Distance to receiver: {dist_to_receiver[closest_idx]:.6f}")

    # Find the signal strength at the receiver
    dist = np.linalg.norm(mesh.geometry.x[:, :2] - receiver_pos, axis=1)
    closest_node = np.argmin(dist)
    signal = uh.x.array[closest_node]
    return signal

# Initialize results matrix
num_frequencies = len(freq_vec)
num_antennas = len(ant_pos_2D)
results_matrix = np.zeros((num_frequencies, num_antennas), dtype=complex)

if rank == 0:
    print(f"Running {num_frequencies} frequencies × {num_antennas} antennas on {size} processes")

# Distribute work across MPI processes
# Each process handles a subset of antenna positions
antennas_per_process = num_antennas // size
remainder = num_antennas % size

# Calculate start and end indices for this process
start_antenna = rank * antennas_per_process + min(rank, remainder)
end_antenna = start_antenna + antennas_per_process + (1 if rank < remainder else 0)

# Process assigned antennas
for j in range(start_antenna, end_antenna):
    ant_pos_i = ant_pos_2D[j]
    receiver_pos = ant_pos_i[2:4]
    transmitter_pos = ant_pos_i[0:2]
    
    if rank == 0:
        print(f"Process {rank}: Antenna {j+1}/{num_antennas}: Tx={transmitter_pos}, Rx={receiver_pos}")
    
    for i, freq in enumerate(freq_vec):
        # Only visualize first frequency and first antenna on rank 0
        visualize = (i == 0 and j == 0 and rank == 0)
        
        signal = run_sim(freq, transmitter_pos, receiver_pos, visualize=visualize)
        results_matrix[i, j] = signal  # Store complex signal
        
        if i % 10 == 0 and rank == 0:  # Print progress every 10 frequencies
            print(f"  Process {rank}: Frequency {i+1}/{num_frequencies}: {abs(signal):.6f}")
    
    if rank == 0:
        print(f"  Process {rank}: Completed antenna {j+1}")

# Gather results from all processes
if rank == 0:
    # Receive results from other processes
    for r in range(1, size):
        recv_data = comm.recv(source=r, tag=r)
        for (i, j), signal in recv_data.items():
            results_matrix[i, j] = signal
else:
    # Send results to rank 0
    send_data = {}
    for j in range(start_antenna, end_antenna):
        for i in range(num_frequencies):
            send_data[(i, j)] = results_matrix[i, j]
    comm.send(send_data, dest=0, tag=rank)

# Only rank 0 saves the results
if rank == 0:
    # Save complex values to CSV (real and imaginary parts)
    df_real = pd.DataFrame(np.real(results_matrix))
    df_imag = pd.DataFrame(np.imag(results_matrix))

    df_real.index.name = 'Frequency_Index'
    df_real.columns = [f'Antenna_{i+1}' for i in range(num_antennas)]

    df_imag.index.name = 'Frequency_Index'
    df_imag.columns = [f'Antenna_{i+1}' for i in range(num_antennas)]

    df_real.to_csv('simulation_results_real.csv')
    df_imag.to_csv('simulation_results_imag.csv')
    
    print("Results saved:")
    print("- Real parts: 'simulation_results_real.csv'")
    print("- Imaginary parts: 'simulation_results_imag.csv'")