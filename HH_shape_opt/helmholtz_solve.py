from dolfin import *
from dolfin_adjoint import *
import numpy as np
import pandas as pd
from scipy.special import hankel1
import os
import meshio

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
        return AMP * np.exp(1j * k_background *
                            (x[0] * direction_x + x[1] * direction_y))

    return wave_func


def cylindrical_wave(amp, source):
    def cylindrical_wave_func(x, k_background):
        R = np.linalg.norm(np.array(x) - np.array(source))
        return amp * 1j / 4 * hankel1(0, k_background * R)
    return cylindrical_wave_func


class IncidentWaveSetup:
    def __init__(self, frequency, incident_field_func):
        self.frequency = frequency
        self.k_background = 2 * np.pi * frequency / LIGHT_SPEED
        self.set_incident_field(incident_field_func)

    def set_incident_field(self, incident_field_func):
        # Any incident field function that works has to take in 2 arguments: x
        # and k_background
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

        self.u_inc_re = IncidentReal()
        self.u_inc_im = IncidentImag()


def mesh_deformation(h_vol, mesh, markers, obstacle_marker, side_wall_marker,
                     receiver_edge_marker, obstacle_opt_marker, obstacle_stiffness):
    # Create scalar function space for material properties
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = -inner(grad(u), grad(v)) * dx
    L0 = Constant(0.0) * v * dx

    # Set material properties via boundary conditions
    if obstacle_opt_marker is not None:
        bcs0 = [
            DirichletBC(V, Constant(1.0), markers, side_wall_marker),
            DirichletBC(V, Constant(1.0), markers, receiver_edge_marker),
            DirichletBC(V, Constant(1.0), markers, obstacle_marker),
            DirichletBC(V, Constant(obstacle_stiffness),
                        markers, obstacle_opt_marker),
        ]

    else:
        bcs0 = [
            DirichletBC(V, Constant(1.0), markers, side_wall_marker),
            DirichletBC(V, Constant(1.0), markers, receiver_edge_marker),
            DirichletBC(V, Constant(obstacle_stiffness),
                        markers, obstacle_marker),
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
        DirichletBC(S, Constant((0.0, 0.0)), markers, receiver_edge_marker),
        DirichletBC(S, Constant((0.0, 0.0)), markers, side_wall_marker)
    ]

    if obstacle_opt_marker is not None:
        bc_el.append(DirichletBC(S, Constant(
            (0.0, 0.0)), markers, obstacle_marker))

    # Solve for displacement field
    s = Function(S, name="deformation")
    solve(a_el == L_el, s, bc_el)

    return s

def mesh_deformation_refraction(h_vol, mesh, markers, domain_markers, obstacle_marker, side_wall_marker,
                     receiver_edge_marker, obstacle_domain_marker, obstacle_stiffness):
    # Create scalar function space for material properties
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = -inner(grad(u), grad(v)) * dx
    L0 = Constant(0.0) * v * dx

    bcs0 = [
        DirichletBC(V, Constant(1.0), markers, side_wall_marker),
        DirichletBC(V, Constant(1.0), markers, receiver_edge_marker),
        DirichletBC(V, Constant(obstacle_stiffness),
                    markers, obstacle_marker),
    ]

    # Solve for material distribution
    mu = Function(V, name="mu")
    solve(a == L0, mu, bcs0)

    # Create vector function space for displacement
    S = VectorFunctionSpace(mesh, "CG", 1)
    u_vec, v_vec = TrialFunction(S), TestFunction(S)

    # Define measure for obstacle boundary
    dObs = Measure("dS",
                       domain=mesh,
                       subdomain_data=domain_markers,
                       subdomain_id=obstacle_domain_marker
                       )

    # Define strain and stress tensors
    def ε(w): return sym(grad(w))
    def σ(w): return 2 * mu * ε(w)

    # Elastic variational problem
    a_el = inner(σ(u_vec), grad(v_vec)) * dx
    L_el = inner(h_vol('+'), v_vec('+')) * dObs

    # Boundary conditions: fix bottom and side walls
    bc_el = [
        DirichletBC(S, Constant((0.0, 0.0)), markers, receiver_edge_marker),
        DirichletBC(S, Constant((0.0, 0.0)), markers, side_wall_marker)
    ]

    # Solve for displacement field
    s = Function(S, name="deformation")
    solve(a_el == L_el, s, bc_el)

    return s

def load_forward_simulation_data_bottomwall(
        measurement_data_file_path, V_ref, projection_degree=0):
    df = pd.read_csv(measurement_data_file_path)
    points = df[["x", "y"]].values
    num_data_points = len(points)
    values = df["u"].values

    # Set up the assignment
    u_ref = Function(V_ref)
    mesh = V_ref.mesh()
    tree = mesh.bounding_box_tree()
    dofmap = V_ref.dofmap()
    u_vec = u_ref.vector().get_local()

    if projection_degree == 0:
        # Logic for DG0: one DOF per cell.
        assigned = np.zeros(mesh.num_cells(), dtype=bool)
        for (x, y), val in zip(points, values):
            point = Point(x, y)
            cell_id = tree.compute_first_entity_collision(point)
            if cell_id < mesh.num_cells() and not assigned[cell_id]:
                dof_idx = dofmap.cell_dofs(cell_id)[0]
                u_vec[dof_idx] = val
                assigned[cell_id] = True
            elif cell_id < mesh.num_cells() and assigned[cell_id]:
                print(
                    f"Warning: cell {cell_id} already assigned, skipping duplicate point.")
    else:
        # Logic for CG > 0 or DG > 0: find the closest DOF within the cell.
        dof_coords = V_ref.tabulate_dof_coordinates()
        assigned_dofs = set()
        for (x, y), val in zip(points, values):
            point = Point(x, y)
            cell_id = tree.compute_first_entity_collision(point)
            if cell_id < mesh.num_cells():
                cell_dofs = dofmap.cell_dofs(cell_id)
                cell_dof_coords = dof_coords[cell_dofs]

                # Find the closest DOF in this cell to the point
                distances = np.linalg.norm(
                    cell_dof_coords - np.array([x, y]), axis=1)
                closest_local_dof_idx = np.argmin(distances)
                closest_global_dof = cell_dofs[closest_local_dof_idx]

                if closest_global_dof not in assigned_dofs:
                    u_vec[closest_global_dof] = val
                    assigned_dofs.add(closest_global_dof)
                else:
                    # This can happen if a DOF is shared by multiple cells that
                    # contain points
                    pass

    # Push the updated values into the Function
    u_ref.vector().set_local(u_vec)
    u_ref.vector().apply("insert")

    return u_ref, num_data_points


def forward_solve(h_control, inc_wave_setup, initial_guess_mesh_util,
                  return_u_scat=False, projection_degree=0):
    # Get mesh and markers from the MeshUtil object
    mesh, markers = initial_guess_mesh_util.get_mesh_and_markers(True)

    # Extract the number of the marker of each object in the simulation
    obstacle_marker = initial_guess_mesh_util.markers_dict["obstacle"]
    side_wall_marker = initial_guess_mesh_util.markers_dict["side_wall"]
    receiver_edge_marker = initial_guess_mesh_util.markers_dict["bottom_wall"]
    obstacle_opt_marker = initial_guess_mesh_util.markers_dict["obstacle_opt"]

    obstacle_stiffness = initial_guess_mesh_util.obstacle_stiffness

    # Transfer h → volume and deform the copy since we want to preserve always
    # the original
    h_vol = transfer_from_boundary(h_control, mesh)
    s = mesh_deformation(h_vol, mesh, markers, obstacle_marker, side_wall_marker,
                         receiver_edge_marker, obstacle_opt_marker, obstacle_stiffness)
    ALE.move(mesh, s)

    V = FunctionSpace(mesh, "CG", 5)
    u_inc_re = project(inc_wave_setup.u_inc_re, V)
    u_inc_im = project(inc_wave_setup.u_inc_im, V)

    ds_receiver = Measure("ds", domain=mesh, subdomain_data=markers,
                          subdomain_id=receiver_edge_marker)
    ds_sides = Measure("ds", domain=mesh, subdomain_data=markers,
                       subdomain_id=side_wall_marker)
    ds_obstacle = Measure(
        "ds", domain=mesh, subdomain_data=markers, subdomain_id=obstacle_marker)

    if obstacle_opt_marker is not None:
        # Since obstacle_marker excludes the to-be-optimized outline of the obstacle
        # we need to add the to-be-optimized outline to ds_obstacle
        ds_obstacle = ds_obstacle + \
            Measure("ds", domain=mesh, subdomain_data=markers,
                    subdomain_id=obstacle_opt_marker)

    ds_outer = ds_receiver + ds_sides

    W = FunctionSpace(mesh, MixedElement([V.ufl_element(),
                                          V.ufl_element()]))
    (u_re, u_im), (v_re, v_im) = TrialFunctions(W), TestFunctions(W)

    k_background = inc_wave_setup.k_background

    a = (inner(grad(u_re), grad(v_re)) - k_background**2 * u_re * v_re) * dx \
        + k_background * u_im * v_re * ds_outer \
        + (inner(grad(u_im), grad(v_im)) - k_background**2 * u_im * v_im) * dx \
        - k_background * u_re * v_im * ds_outer

    L = Constant(0.0) * (v_re + v_im) * dx

    # Dirichlet BCs on the obstacle u_s = - u_in on the reflective surface
    uinc_re_neg = project(-inc_wave_setup.u_inc_re, V)  # VERY IMPORTANT CHANGE
    uinc_im_neg = project(-inc_wave_setup.u_inc_im, V)  # VERY IMPORTANT CHANGE

    bcs = [
        DirichletBC(W.sub(0), uinc_re_neg, markers, obstacle_marker),
        DirichletBC(W.sub(1), uinc_im_neg, markers, obstacle_marker),

    ]

    if obstacle_opt_marker is not None:
        # Since obstacle_marker excludes the to-be-optimized outline of the obstacle
        # we need to add the to-be-optimized outline to ds_obstacle
        bcs.append(DirichletBC(W.sub(0), uinc_re_neg,
                   markers, obstacle_opt_marker))
        bcs.append(DirichletBC(W.sub(1), uinc_im_neg,
                   markers, obstacle_opt_marker))

    w = Function(W)
    solve(a == L, w, bcs)

    # Extract solutions
    u_sol_re, u_sol_im = w.split()

    # Total field expressions
    u_tot_re = u_inc_re + u_sol_re
    u_tot_im = u_inc_im + u_sol_im

    # Calculate the magnitude of scattered wave and total wave
    u_sol_mag = sqrt(u_sol_re**2 + u_sol_im**2)
    u_tot_mag = sqrt(u_tot_re**2 + u_tot_im**2)

    if projection_degree == 0:
        V_projection = FunctionSpace(mesh, "DG", 0)
    else:
        V_projection = FunctionSpace(mesh, "CG", projection_degree)

    ds_receiver = Measure("ds", domain=mesh, subdomain_data=markers,
                          subdomain_id=receiver_edge_marker)

    if return_u_scat:
        u_scat_mag_projected = project(u_sol_mag, V_projection)
        # For final signal phase comparison
        u_scat_re_projected = project(u_sol_re, V_projection)
        u_scat_im_projected = project(u_sol_im, V_projection)
        return u_scat_mag_projected, u_scat_re_projected, u_scat_im_projected, ds_receiver, V_projection

    else:
        u_tot_mag_projected = project(u_tot_mag, V_projection)
        # For final signal phase comparison
        u_tot_re_projected = project(u_tot_re, V_projection)
        u_tot_im_projected = project(u_tot_im, V_projection)
        return u_tot_mag_projected, u_tot_re_projected, u_tot_im_projected, ds_receiver, V_projection


def forward_solve_refraction(
        h_control, inc_wave_setup, initial_guess_mesh_util, return_u_scat=True, projection_degree=0):
    # Get mesh and markers from the MeshUtil object
    mesh, markers, domain_markers = initial_guess_mesh_util.get_mesh_and_markers(True)

    # Extract the number of the marker of each object in the simulation
    obstacle_marker = initial_guess_mesh_util.markers_dict["obstacle"]
    side_wall_marker = initial_guess_mesh_util.markers_dict["side_wall"]
    receiver_edge_marker = initial_guess_mesh_util.markers_dict["bottom_wall"]
    obstacle_opt_marker = initial_guess_mesh_util.markers_dict["obstacle_opt"]
    domain_marker = initial_guess_mesh_util.markers_dict["domain_marker"]
    obstacle_domain_marker = initial_guess_mesh_util.markers_dict["obstacle_domain_marker"]

    obstacle_stiffness = initial_guess_mesh_util.obstacle_stiffness

    #TODO: Not usable
    s = mesh_deformation_refraction(h_control, mesh, markers, domain_markers, obstacle_marker, side_wall_marker,
                     receiver_edge_marker, obstacle_domain_marker, obstacle_stiffness)
    ALE.move(mesh, s)

    V = FunctionSpace(mesh, "CG", 5)
    u_inc_re = project(inc_wave_setup.u_inc_re, V)
    u_inc_im = project(inc_wave_setup.u_inc_im, V)

    ds_receiver = Measure("ds", domain=mesh, subdomain_data=markers,
                          subdomain_id=receiver_edge_marker)
    ds_sides = Measure("ds", domain=mesh, subdomain_data=markers,
                       subdomain_id=side_wall_marker)
    ds_outer = ds_receiver + ds_sides

    W = FunctionSpace(mesh, MixedElement([V.ufl_element(),
                                          V.ufl_element()]))
    (u_re, u_im), (v_re, v_im) = TrialFunctions(W), TestFunctions(W)

    # TODO: Read from dic instead
    k_background = inc_wave_setup.k_background
    k_obst_fac = 3  # TODO: Dependent on permissivity and n
    k_obstacle = k_background * k_obst_fac

    # Define function g in the right hand side
    # We use k_background here since obviously we
    # will integrate g over domain boundary which won't intersect with
    # obstacle domain
    n = FacetNormal(mesh)
    g_re = dot(grad(u_inc_re), n) + k_background * u_inc_im
    g_im = dot(grad(u_inc_im), n) - k_background * u_inc_re

    dx_domain = Measure(
        "dx", domain=mesh, subdomain_data=domain_markers, subdomain_id=domain_marker)
    dx_obstacle = Measure(
        "dx", domain=mesh, subdomain_data=domain_markers, subdomain_id=obstacle_domain_marker)

    a = (inner(grad(u_re), grad(v_re)) - k_background**2 * u_re * v_re) * dx_domain \
        + (inner(grad(u_re), grad(v_re)) - k_obstacle**2 * u_re * v_re) * dx_obstacle \
        + k_background * u_im * v_re * ds_outer \
        + (inner(grad(u_im), grad(v_im)) - k_background**2 * u_im * v_im) * dx_domain \
        + (inner(grad(u_im), grad(v_im)) - k_obstacle**2 * u_im * v_im) * dx_obstacle \
        - k_background * u_re * v_im * ds_outer

    L = (g_re * v_re + g_im * v_im) * ds_outer

    bcs = []  # No Dirichlet boundary condition
    w = Function(W)
    solve(a == L, w, bcs)

    # Extract solutions
    u_tot_re, u_tot_im = w.split()

    # Scatter field expressions
    u_scat_re = u_tot_re - u_inc_re
    u_scat_im = u_tot_im - u_inc_im

    # Calculate the magnitude of scattered wave and total wave
    u_scat_mag = sqrt(u_scat_re**2 + u_scat_im**2)
    u_tot_mag = sqrt(u_tot_re**2 + u_tot_im**2)

    if projection_degree == 0:
        V_projection = FunctionSpace(mesh, "DG", 0)
    else:
        V_projection = FunctionSpace(mesh, "CG", projection_degree)

    ds_receiver = Measure("ds", domain=mesh, subdomain_data=markers,
                          subdomain_id=receiver_edge_marker)

    if return_u_scat:
        u_scat_mag_projected = project(u_scat_mag, V_projection)
        # For final signal phase comparison
        u_scat_re_projected = project(u_scat_re, V_projection)
        u_scat_im_projected = project(u_scat_im, V_projection)
        return u_scat_mag_projected, u_scat_re_projected, u_scat_im_projected, ds_receiver, V_projection

    else:
        u_tot_mag_projected = project(u_tot_mag, V_projection)
        # For final signal phase comparison
        u_tot_re_projected = project(u_tot_re, V_projection)
        u_tot_im_projected = project(u_tot_im, V_projection)
        return u_tot_mag_projected, u_tot_re_projected, u_tot_im_projected, ds_receiver, V_projection
