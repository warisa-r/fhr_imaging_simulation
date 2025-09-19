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

class IncidentWaveSetup:
    def __init__(self, frequency, incident_field_func, obstacle_stiffness = 25):
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

        self.u_inc_re = IncidentReal()
        self.u_inc_im = IncidentImag()

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

def load_forward_simulation_data_bottomwall(measurement_data_file_path, V_ref, projection_degree=0):
    df = pd.read_csv(measurement_data_file_path)
    points = df[["x", "y"]].values
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
                print(f"Warning: cell {cell_id} already assigned, skipping duplicate point.")
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
                distances = np.linalg.norm(cell_dof_coords - np.array([x, y]), axis=1)
                closest_local_dof_idx = np.argmin(distances)
                closest_global_dof = cell_dofs[closest_local_dof_idx]

                if closest_global_dof not in assigned_dofs:
                    u_vec[closest_global_dof] = val
                    assigned_dofs.add(closest_global_dof)
                else:
                    # This can happen if a DOF is shared by multiple cells that contain points
                    pass

    # Push the updated values into the Function
    u_ref.vector().set_local(u_vec)
    u_ref.vector().apply("insert")

    return u_ref

def forward_solve(h_control, inc_wave_setup, initial_guess_mesh_util, projection_degree = 0):
    # Get mesh and markers from the MeshUtil object
    mesh, markers = initial_guess_mesh_util.get_mesh_and_markers()
    
    # Extract the number of the marker of each object in the simulation
    obstacle_marker = initial_guess_mesh_util.markers_dict["obstacle"]
    side_wall_marker = initial_guess_mesh_util.markers_dict["side_wall"]
    bottom_wall_marker = initial_guess_mesh_util.markers_dict["bottom_wall"]
    obstacle_opt_marker = initial_guess_mesh_util.markers_dict["obstacle_opt"]

    # Transfer h → volume and deform the copy since we want to preserve always the original
    h_vol = transfer_from_boundary(h_control, mesh)
    s = mesh_deformation(h_vol, mesh, markers, obstacle_marker, side_wall_marker, bottom_wall_marker, obstacle_opt_marker, inc_wave_setup.obstacle_stiffness)
    ALE.move(mesh, s)

    V = FunctionSpace(mesh, "CG", 5)
    u_inc_re = project(inc_wave_setup.u_inc_re, V)
    u_inc_im = project(inc_wave_setup.u_inc_im, V)

    ds_bottom = Measure("ds", domain=mesh, subdomain_data=markers, subdomain_id=bottom_wall_marker)
    ds_sides = Measure("ds", domain=mesh, subdomain_data=markers, subdomain_id=side_wall_marker)
    ds_obstacle = Measure("ds", domain=mesh, subdomain_data=markers, subdomain_id=obstacle_marker)

    if obstacle_opt_marker != None:
        # Since obstacle_marker excludes the to-be-optimized outline of the obstacle
        # we need to add the to-be-optimized outline to ds_obstacle
        ds_obstacle = ds_obstacle + Measure("ds", domain=mesh, subdomain_data=markers, subdomain_id=obstacle_opt_marker)

    ds_outer = ds_bottom + ds_sides

    W = FunctionSpace(mesh, MixedElement([V.ufl_element(),
                                               V.ufl_element()]))
    (u_re, u_im), (v_re, v_im) = TrialFunctions(W), TestFunctions(W)

    k_background = inc_wave_setup.k_background

    a = (inner(grad(u_re), grad(v_re)) - k_background**2*u_re*v_re)*dx \
        + k_background*u_im*v_re*ds_outer \
        + (inner(grad(u_im), grad(v_im)) - k_background**2*u_im*v_im)*dx \
        - k_background*u_re*v_im*ds_outer

    L = Constant(0.0)*(v_re + v_im)*dx

    # Dirichlet BCs on the obstacle u_s = - u_in on the reflective surface
    uinc_re_neg = Function(V); uinc_re_neg.vector()[:] = -u_inc_re.vector()[:]
    uinc_im_neg = Function(V); uinc_im_neg.vector()[:] = -u_inc_im.vector()[:]

    bcs = [
      DirichletBC(W.sub(0), uinc_re_neg, markers, obstacle_marker),
      DirichletBC(W.sub(1), uinc_im_neg, markers, obstacle_marker),
      
    ]

    if obstacle_opt_marker != None:
        # Since obstacle_marker excludes the to-be-optimized outline of the obstacle
        # we need to add the to-be-optimized outline to ds_obstacle
        bcs.append(DirichletBC(W.sub(0), uinc_re_neg, markers, obstacle_opt_marker))
        bcs.append(DirichletBC(W.sub(1), uinc_im_neg, markers, obstacle_opt_marker))


    w = Function(W)
    solve(a == L, w, bcs)
    
    # Extract solutions
    u_sol_re, u_sol_im = w.split()

    # Total field expressions
    u_tot_re = u_inc_re + u_sol_re
    u_tot_im = u_inc_im + u_sol_im

    u_tot_mag = sqrt(u_tot_re**2 + u_tot_im**2)

    if projection_degree == 0:
        V_projection = FunctionSpace(mesh, "DG", 0)
    else:
        V_projection = FunctionSpace(mesh, "CG", projection_degree)
    
    u_tot_mag_projected = project(u_tot_mag, V_projection)
    
    ds_bottom = Measure("ds", domain=mesh, subdomain_data=markers, subdomain_id=bottom_wall_marker)

    return u_tot_mag_projected, ds_bottom, V_projection