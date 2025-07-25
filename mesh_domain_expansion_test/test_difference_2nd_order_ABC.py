    mesh = Mesh(f"mesh_test_{mesh_radius}.xml")
    boundary_markers = MeshFunction("size_t", mesh, f"mesh_test_{mesh_radius}_facet_region.xml")

    # Define function space
    V_element = FiniteElement("Lagrange", mesh.ufl_cell(), 5)
    V = FunctionSpace(mesh, V_element)

    # Instantiate expressions
    u_inc_re = project(HankelReal(degree=2), V)
    u_inc_im = project(HankelImag(degree=2), V)

    # Define the outward unit normal vector
    n = FacetNormal(mesh)

    # Define boundary measures
    ds_outer = Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=1)
    ds_circle = Measure("ds", domain=mesh, subdomain_data=boundary_markers, subdomain_id=2)

    # Define mixed function space
    W = FunctionSpace(mesh, V_element * V_element)
    (u_re, u_im) = TrialFunctions(W)
    (v_re, v_im) = TestFunctions(W)

    # Second-order ABC for circular boundary
    # The exact form for circular boundary at radius R is:
    # ∂u/∂r + iku + (1/2ikR)∂²u/∂θ² + (1/2ikR)u = 0
    # where θ is the angular coordinate
    
    # For circular domain, we can use the fact that:
    # - Curvature κ = 1/R
    # - Tangential derivative can be expressed using surface gradient
    
    R = mesh_radius  # Radius of outer boundary
    
    # First-order terms: ∂u/∂n + iku
    abc_1st_re = k_background * u_im * v_re * ds_outer
    abc_1st_im = -k_background * u_re * v_im * ds_outer
    
    # Second-order correction for circular boundary:
    # The tangential Laplacian on a circle of radius R is: ∂²u/∂θ² = R²∇_t²u
    # where ∇_t is the surface gradient operator
    
    # Surface gradient: ∇_t u = ∇u - (∇u·n)n
    # For the weak form, we need: ∫(∇_t u · ∇_t v) ds
    
    # This can be computed as: ∫(∇u·∇v - (∇u·n)(∇v·n)) ds
    grad_u_re = grad(u_re)
    grad_u_im = grad(u_im)
    grad_v_re = grad(v_re)
    grad_v_im = grad(v_im)
    
    # Normal derivatives
    du_dn_re = dot(grad_u_re, n)
    du_dn_im = dot(grad_u_im, n)
    dv_dn_re = dot(grad_v_re, n)
    dv_dn_im = dot(grad_v_im, n)
    
    # Surface gradient terms
    surf_grad_re = dot(grad_u_re, grad_v_re) - du_dn_re * dv_dn_re
    surf_grad_im = dot(grad_u_im, grad_v_im) - du_dn_im * dv_dn_im
    
    # Second-order correction: -(1/2ik)(1/R²)∫∇_t u · ∇_t v ds + (1/2ikR)∫u·v ds
    abc_2nd_tangential_re = -(1.0/(2.0*k_background*R*R)) * surf_grad_re * ds_outer
    abc_2nd_tangential_im = (1.0/(2.0*k_background*R*R)) * surf_grad_im * ds_outer
    
    abc_2nd_curvature_re = (1.0/(2.0*k_background*R)) * u_im * v_re * ds_outer
    abc_2nd_curvature_im = -(1.0/(2.0*k_background*R)) * u_re * v_im * ds_outer

    # Complete bilinear form
    a = (inner(grad(u_re), grad(v_re)) - k_background**2 * u_re * v_re) * dx + \
        abc_1st_re + abc_2nd_tangential_re + abc_2nd_curvature_re + \
        (inner(grad(u_im), grad(v_im)) - k_background**2 * u_im * v_im) * dx + \
        abc_1st_im + abc_2nd_tangential_im + abc_2nd_curvature_im

    # Homogeneous RHS
    L = Constant(0.0) * (v_re + v_im) * dx

    # Dirichlet boundary conditions on circle boundary (u = -u_inc)
    u_inc_re_neg = Function(V)
    u_inc_im_neg = Function(V)
    u_inc_re_neg.vector()[:] = -u_inc_re.vector()[:]
    u_inc_im_neg.vector()[:] = -u_inc_im.vector()[:]

    bc_circle_re = DirichletBC(W.sub(0), u_inc_re_neg, boundary_markers, 2)
    bc_circle_im = DirichletBC(W.sub(1), u_inc_im_neg, boundary_markers, 2)
    bcs = [bc_circle_re, bc_circle_im]

    # Solve the coupled system
    w = Function(W)
    solve(a == L, w, bcs)