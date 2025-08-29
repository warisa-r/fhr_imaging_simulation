from dolfin import *
from dolfin_adjoint import *
import numpy as np
import pandas as pd

## CONSTANTS ##

LIGHT_SPEED = 299792458

AMP = 1
def plane_wave(x, k_background):
    return AMP * np.exp(1j * k_background * x[1])

def plane_wave_angle(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    direction_x = np.cos(angle_rad)
    direction_y = np.sin(angle_rad)

    def wave_func(x, k_background):
        return AMP * np.exp(1j * k_background * (x[0] * direction_x + x[1] * direction_y))
    
    return wave_func

class HelmholtzSetup:
    def __init__(self, frequency, incident_field_func, obstacle_stiffness = 50):
        self.frequency = frequency
        self.k_background = 2* np.pi * frequency / LIGHT_SPEED
        self.set_incident_field(incident_field_func)
        self.obstacle_stiffness = obstacle_stiffness

    def set_incident_field(self, incident_field_func):
        # Any incident field function that works has to take in 2 arguments: x and k_background
        k_background = self.k_background

        class IncidentReal(UserExpression):
            def eval(self, values, x):
                values[0] = np.real(incident_field_func(x, k_background))
            def value_shape(self):
                return ()

        class IncidentImag(UserExpression):
            def eval(self, values, x):
                values[0] = np.imag(incident_field_func(x, k_background))
            def value_shape(self):
                return ()

        self.u_inc_re = IncidentReal(degree = 2)
        self.u_inc_im = IncidentImag(degree = 2)

def mesh_deformation(h_vol, mesh, markers, obstacle_marker, side_wall_marker, bottom_wall_marker, obstacle_opt_marker, obstacle_stiffness):
    # Create scalar function space for material properties
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = -inner(grad(u), grad(v)) * dx
    L0 = Constant(0.0) * v * dx
    
    # Set material properties via boundary conditions
    if obstacle_opt_marker is not None:
        bcs0 = [
            DirichletBC(V, Constant(1.0), markers, side_wall_marker),
            DirichletBC(V, Constant(1.0), markers, bottom_wall_marker),
            DirichletBC(V, Constant(1.0), markers, obstacle_marker),
            DirichletBC(V, Constant(obstacle_stiffness), markers, obstacle_opt_marker),
        ]

    else:
        bcs0 = [
            DirichletBC(V, Constant(1.0), markers, side_wall_marker),
            DirichletBC(V, Constant(1.0), markers, bottom_wall_marker),
            DirichletBC(V, Constant(obstacle_stiffness), markers, obstacle_marker),
        ]
    
    # Solve for material distribution
    mu = Function(V, name="mu")
    solve(a == L0, mu, bcs0)

    # Create vector function space for displacement
    S = VectorFunctionSpace(mesh, "CG", 1)
    u_vec, v_vec = TrialFunction(S), TestFunction(S)
    
    # Define measure for obstacle boundary
    if obstacle_opt_marker is not None:
        dObs = Measure("ds",
            domain=mesh,
            subdomain_data=markers,
            subdomain_id=obstacle_opt_marker
        )
    else:
        dObs = Measure("ds",
            domain=mesh,
            subdomain_data=markers,
            subdomain_id=obstacle_marker
        )

    # Define strain and stress tensors
    def ε(w): return sym(grad(w))
    def σ(w): return 2 * mu * ε(w)

    # Elastic variational problem
    a_el = inner(σ(u_vec), grad(v_vec)) * dx
    L_el = inner(h_vol, v_vec) * dObs

    # Boundary conditions: fix bottom and side walls
    bc_el = [
        DirichletBC(S, Constant((0.0, 0.0)), markers, bottom_wall_marker),
        DirichletBC(S, Constant((0.0, 0.0)), markers, side_wall_marker)
    ]

    if obstacle_opt_marker is not None:
        bc_el.append(DirichletBC(S, Constant((0.0, 0.0)), markers, obstacle_marker))
    
    # Solve for displacement field
    s = Function(S, name="deformation")
    solve(a_el == L_el, s, bc_el)

    return s

def preprocess_reference_data(V_CG5, forward_sim_result_file_path, frequency = None, angle = None):
    
    # Only rank 0 reads the CSV file
    if MPI.comm_world.rank == 0:
        df = pd.read_csv(forward_sim_result_file_path)
        if frequency is not None:
            df = df.loc[df['frequency'] == frequency]
        if angle is not None:
            df = df.loc[df['angle'] == angle]
        points = df[["x", "y"]].values
        values = df["u"].values
    else:
        points = None
        values = None
    
    # Broadcast data to all processes
    points = MPI.comm_world.bcast(points, root=0)
    values = MPI.comm_world.bcast(values, root=0)

    mesh = V_CG5.mesh()
    tree = mesh.bounding_box_tree()
    
    point_value_map = {}
    tolerance = 1e-10
    
    for (x, y), val in zip(points, values):
        point = Point(x, y)
        try:
            cell_id = tree.compute_first_entity_collision(point)
            
            if cell_id < mesh.num_cells():
                dofs_coords = V_CG5.tabulate_dof_coordinates()
                distances = np.sqrt((dofs_coords[:, 0] - x)**2 + (dofs_coords[:, 1] - y)**2)
                closest_dofs = np.where(distances < tolerance)[0]
                
                if len(closest_dofs) > 0:
                    dof_idx = closest_dofs[0]
                    point_value_map[dof_idx] = val
                    
        except RuntimeError:
            # Point not found in this rank's mesh partition, skip silently
            pass
            
    return point_value_map

def assign_reference_data(V_CG5, point_value_map):
    u_ref = Function(V_CG5)
    
    # Handle the case where this process has no data points
    if len(point_value_map) == 0:
        # Still need to participate in collective operations
        u_ref.vector().apply("insert")
        return u_ref
    
    # Get local vector and assign known values
    u_vec = u_ref.vector().get_local()
    for dof_idx, value in point_value_map.items():
        if dof_idx < len(u_vec):
            u_vec[dof_idx] = value
    
    # Update the function - this is a collective operation
    u_ref.vector().set_local(u_vec)
    u_ref.vector().apply("insert")
    
    return u_ref
def helmholtz_solve(mesh_copy, markers_copy, h_control, hh_setup, 
                   obstacle_marker, side_wall_marker, bottom_wall_marker, data_all_side = False, obstacle_opt_marker = None):

    # Perform mesh deformation
    h_vol = transfer_from_boundary(h_control, mesh_copy)
    s = mesh_deformation(h_vol, mesh_copy, markers_copy, 
                        obstacle_marker, side_wall_marker, bottom_wall_marker, obstacle_opt_marker,
                        obstacle_stiffness=hh_setup.obstacle_stiffness)
    ALE.move(mesh_copy, s)

    # Create function space and project incident fields
    V = FunctionSpace(mesh_copy, "CG", 5)
    u_inc_re = project(hh_setup.u_inc_re, V)
    u_inc_im = project(hh_setup.u_inc_im, V)

    # Define boundary measures
    ds_bottom = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=bottom_wall_marker)
    ds_sides = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=side_wall_marker)
    if obstacle_opt_marker is not None:
        ds_obstacle = (Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=obstacle_marker) +
                        Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=obstacle_opt_marker))
    else:
        ds_obstacle = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, subdomain_id=obstacle_marker)
    ds_outer = ds_bottom + ds_sides

    # Create mixed function space for complex-valued problem
    W = FunctionSpace(mesh_copy, MixedElement([V.ufl_element(), V.ufl_element()]))
    (u_re, u_im), (v_re, v_im) = TrialFunctions(W), TestFunctions(W)

    # Define variational form
    a = (inner(grad(u_re), grad(v_re)) - hh_setup.k_background**2*u_re*v_re)*dx \
        + hh_setup.k_background*u_im*v_re*ds_outer \
        + (inner(grad(u_im), grad(v_im)) - hh_setup.k_background**2*u_im*v_im)*dx \
        - hh_setup.k_background*u_re*v_im*ds_outer

    L = Constant(0.0)*(v_re + v_im)*dx

    # Boundary conditions: u_scattered = -u_incident on obstacle
    uinc_re_neg = Function(V)
    uinc_re_neg.vector()[:] = -u_inc_re.vector()[:]
    uinc_im_neg = Function(V)
    uinc_im_neg.vector()[:] = -u_inc_im.vector()[:]

    bcs = [
        DirichletBC(W.sub(0), uinc_re_neg, markers_copy, obstacle_marker),
        DirichletBC(W.sub(1), uinc_im_neg, markers_copy, obstacle_marker),
    ]

    if obstacle_opt_marker is not None:
        bcs.append(DirichletBC(W.sub(0), uinc_re_neg, markers_copy, obstacle_opt_marker))
        bcs.append(DirichletBC(W.sub(1), uinc_im_neg, markers_copy, obstacle_opt_marker))

    # Solve the system
    w = Function(W)
    solve(a == L, w, bcs)
    
    # Extract solutions
    u_sol_re, u_sol_im = w.split()

    # Compute total fields
    u_tot_re = u_inc_re + u_sol_re
    u_tot_im = u_inc_im + u_sol_im

    # Compute magnitude
    u_tot_mag = sqrt(u_tot_re**2 + u_tot_im**2)

    # Create measure for bottom boundary with appropriate quadrature
    ds_bottom = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, 
                    subdomain_id=bottom_wall_marker)
    
    if data_all_side == True:
        ds_side_wall = Measure("ds", domain=mesh_copy, subdomain_data=markers_copy, 
                    subdomain_id=side_wall_marker)
        ds = ds_bottom + ds_side_wall
    else:
        # Normally (for the simple non entire object scan, the data is available at ds_bottom)
        ds = ds_bottom

    u_tot_mag_projected = project(u_tot_mag, V)
    
    return u_tot_mag_projected, ds, V