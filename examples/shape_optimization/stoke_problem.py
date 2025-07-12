
# Copyright (C) 2023 Jorgen S. Dokken
#
# Solve shape optimization problem for Navier-Stokes with DOLFINx with higher order geometry
#
# SPDX-License-Identifier:    MIT

import gmsh
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl
from pathlib import Path

order = 2  # Order of mesh geometry
res = 0.2  # Mesh resolution

if MPI.COMM_WORLD.rank == 0:

    gmsh.initialize()
    upper_points = [[20, 7, 0], [15, 7, 0], [10, 7, 0],
                    [7, 6, 0], [4, 1, 0], [2, 1, 0], [0, 1, 0]]
    upper_points_tags = []
    for point in upper_points:
        upper_points_tags.append(gmsh.model.occ.addPoint(*point, res))

    lower_points = [[0, 0, 0], [2, 0, 0], [4, 0, 0], [
        7.5, 5, 0], [10, 6, 0], [15, 6, 0], [20, 6, 0]]
    lower_points_tags = []
    for point in lower_points:
        lower_points_tags.append(gmsh.model.occ.addPoint(*point, res))
    gmsh.model.occ.synchronize()

    lines = []
    lines.append(gmsh.model.occ.add_line(
        lower_points_tags[0], lower_points_tags[1]))
    lines.append(gmsh.model.occ.add_bspline(lower_points_tags[1:-1]))  # Spline
    lines.append(gmsh.model.occ.add_line(*lower_points_tags[-2:]))
    # Connection between lower and upper part
    lines.append(gmsh.model.occ.add_line(
        lower_points_tags[-1], upper_points_tags[0]))
    lines.append(gmsh.model.occ.add_line(
        upper_points_tags[0], upper_points_tags[1]))
    lines.append(gmsh.model.occ.add_bspline(upper_points_tags[1:-1]))  # Spline
    lines.append(gmsh.model.occ.add_line(*upper_points_tags[-2:]))
    # Connection between upper and lower
    lines.append(gmsh.model.occ.add_line(
        upper_points_tags[-1], lower_points_tags[0]))
    curve_loop = gmsh.model.occ.add_curve_loop(lines)
    surface = gmsh.model.occ.add_plane_surface([curve_loop])
    gmsh.model.occ.synchronize()

    lines = np.asarray(lines)
    gmsh.model.add_physical_group(1, [lines[3]], tag=11, name="Inflow")
    gmsh.model.add_physical_group(1, [lines[-1]], tag=10, name="Outflow")
    gmsh.model.add_physical_group(
        1, lines[[0, 2, 4, 6]], tag=12, name="WallFixed")
    gmsh.model.add_physical_group(1, lines[[1, 5]], tag=13, name="WallFree")
    gmsh.model.add_physical_group(2, [surface], tag=1, name="Fluid")
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)


mesh, _, ft = dolfinx.io.gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()

folder = Path("output")
folder.mkdir(exist_ok=True, parents=True)

cmap = mesh.geometry.cmaps[0]
c_el = basix.ufl.element("Lagrange", mesh.topology.cell_name(
), cmap.degree, cmap.variant, gdim=2, shape=(2,))
X = ufl.SpatialCoordinate(mesh)
W = dolfinx.fem.VectorFunctionSpace(mesh, c_el)
gradJ = dolfinx.fem.Function(W)

u_degree = max(2, order)
u_el = ufl.VectorElement("Lagrange", mesh.ufl_cell(), u_degree)
p_el = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), u_degree-1)
Z = dolfinx.fem.FunctionSpace(mesh, ufl.MixedElement([u_el, p_el]))
z = dolfinx.fem.Function(Z)
z_adjoint = dolfinx.fem.Function(Z)

u, p = ufl.split(z)
test = ufl.TestFunction(Z)
v, q = ufl.split(test)

nu = 1./400
e = nu*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx - p*ufl.div(v)*ufl.dx \
    + ufl.inner(ufl.dot(ufl.grad(u), u), v)*ufl.dx + ufl.div(u)*q*ufl.dx


def u_in(x):
    return (6*x[1] * (1-x[1], np.zeros(x.shape[1], dtype=np.float64)))


Z0, _ = Z.sub(0).collapse()
u_bc = dolfinx.fem.Function(Z0)
u_bc.interpolate(u_in)
inlet_dofs = dolfinx.fem.locate_dofs_topological(
    (Z.sub(0), Z0), ft.dim, ft.find(10))
bc_inlet = dolfinx.fem.dirichletbc(u_bc, inlet_dofs, Z.sub(0))
u_adjoint = dolfinx.fem.Function(Z0)
u_adjoint.x.array[:] = 0
bc_inlet_adjoint = dolfinx.fem.dirichletbc(u_adjoint, inlet_dofs, Z.sub(0))

wall_facets = np.sort(np.hstack([ft.find(tag) for tag in [12, 13]]))
wall_dofs = dolfinx.fem.locate_dofs_topological(
    (Z.sub(0), Z0), ft.dim, wall_facets)
u_walls = dolfinx.fem.Function(Z0)
u_walls.x.array[:] = 0
bc_walls = dolfinx.fem.dirichletbc(u_walls, wall_dofs, Z.sub(0))
bc_walls_adjoint = dolfinx.fem.dirichletbc(u_adjoint, wall_dofs, Z.sub(0))
bcs = [bc_inlet, bc_walls]
bcs_adjoint = [bc_inlet_adjoint, bc_walls_adjoint]

J = nu*ufl.inner(ufl.grad(u), ufl.grad(u))*ufl.dx
Jf = dolfinx.fem.form(J)
volume = dolfinx.fem.Constant(mesh, 1.)*ufl.dx
volume_form = dolfinx.fem.form(volume)
target_volume = mesh.comm.allreduce(
    dolfinx.fem.assemble_scalar(volume_form), op=MPI.SUM)
dvol = dolfinx.fem.form(ufl.derivative(volume, X, ufl.TestFunction(W)))


forward_problem = dolfinx.fem.petsc.NonlinearProblem(e, z, bcs=bcs)
forward_solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm, forward_problem)
forward_solver.convergence_criterion = "residual"
forward_solver.rtol = 1e-6
forward_solver.report = True
forward_solver.max_it = 25
forward_ksp = forward_solver.krylov_solver
forward_ksp.setType("preonly")
forward_ksp.getPC().setType("lu")
forward_solver.error_on_nonconvergence = True
# Adjoint problem
L = ufl.replace(e, {test: z_adjoint}) + J
all_bc_facets = np.sort(np.hstack([wall_facets, ft.find(10)]))
a_adjoint, L_adjoint = ufl.system(ufl.replace(
    ufl.derivative(L, z), {z_adjoint: ufl.TrialFunction(Z)}))
adjoint_solver = dolfinx.fem.petsc.LinearProblem(a_adjoint, L_adjoint, u=z_adjoint,
                                                 bcs=bcs_adjoint,
                                                 petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                                "pc_factor_mat_solver_type": "mumps"})


z_out = z.sub(0).collapse()


def solve_state_and_adjoint(i: int, verbose: bool = False):
    current_log_level = dolfinx.log.get_log_level()
    if verbose:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    forward_solver.solve(z)
    adjoint_solver.solve()
    z_out.x.array[:] = z.sub(0).collapse().x.array[:]
    if i % 1 == 0:
        vtx_u = dolfinx.io.VTXWriter(
            mesh.comm, folder / f"u_{i}.bp", [z_out], engine="BP4")
        vtx_u.write(0)
        vtx_u.close()
    dolfinx.log.set_log_level(current_log_level)


# Derivative of functional with "Lagrange" multiplier
dL = ufl.derivative(L, X, ufl.TestFunction(W))
dLf = dolfinx.fem.form(dL)
solve_state_and_adjoint(0)
print(0, mesh.comm.allreduce(dolfinx.fem.assemble_scalar(Jf), op=MPI.SUM))

dL_func = dolfinx.fem.Function(W)
dvol_func = dolfinx.fem.Function(W)


# Riesz representation variational problem
phi, psi = ufl.TrialFunction(W), ufl.TestFunction(W)
a_riesz = dolfinx.fem.form(ufl.inner(ufl.grad(phi), ufl.grad(psi))*ufl.dx)
facets_non_move = np.sort(np.hstack([ft.find(tag) for tag in [10, 11, 12]]))
dofs_non_move = dolfinx.fem.locate_dofs_topological(W, ft.dim, facets_non_move)
zero_bc = dolfinx.fem.Function(W)
zero_bc.x.array[:] = 0
bc_riesz = [dolfinx.fem.dirichletbc(zero_bc, dofs_non_move)]
A_riesz = dolfinx.fem.petsc.assemble_matrix(a_riesz, bc_riesz)
A_riesz.assemble()
solver_riesz = PETSc.KSP().create(mesh.comm)
solver_riesz.setOperators(A_riesz)
solver_riesz.setType(PETSc.KSP.Type.PREONLY)
solver_riesz.getPC().setType(PETSc.PC.Type.LU)

gradJ = dolfinx.fem.Function(W)

c = 0.05
for i in range(1, 51):
    dL_func.x.array[:] = 0
    dvol_func.x.array[:] = 0
    dolfinx.fem.petsc.assemble_vector(dL_func.vector, dLf)
    dolfinx.fem.petsc.assemble_vector(dvol_func.vector, dvol)
    dL_func.x.scatter_reverse(dolfinx.la.InsertMode.add)
    dvol_func.x.scatter_reverse(dolfinx.la.InsertMode.add)
    current_vol = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(volume_form), op=MPI.SUM)
    dL_func.x.array[:] += 2*c*dvol_func.x.array*(current_vol-target_volume)

    dolfinx.fem.petsc.set_bc(dL_func.vector, bc_riesz)
    solver_riesz.solve(dL_func.vector, gradJ.vector)
    mesh.geometry.x[:, :2] -= 0.5*gradJ.x.array.reshape(-1, 2)
    solve_state_and_adjoint(i)
    print(i, mesh.comm.allreduce(dolfinx.fem.assemble_scalar(Jf), op=MPI.SUM))

A_riesz.destroy()
solver_riesz.destroy()