# 2D Stokes, Hdiv-P0 MG preconditioned CG solver
# condensed mixed Hdiv-P0 equivalent to CR scheme
from ngsolve import *
from netgen.geom2d import SplineGeometry, unit_square
import time as timeit
from netgen.csg import *
from ngsolve.krylovspace import CGSolver, MinResSolver, GMResSolver
from prol import *
from mymg import *
from ngsolve.la import EigenValues_Preconditioner

maxdofs = 1e5
c_low = 10
epsilon = 1e-8

mesh = Mesh(unit_square.GenerateMesh(maxh=1/4))

# exact solution
u_exact1 = x ** 2 * (x - 1) ** 2 * 2 * y * (1 - y) * (2 * y - 1)
u_exact2 = y ** 2 * (y - 1) ** 2 * 2 * x * (x - 1) * (2 * x - 1)
u_exact = CF((u_exact1, u_exact2))

L_exactXX = (2 * x * (x - 1) ** 2 * 2 * y * (1 - y) * (2 * y - 1)
                        + x ** 2 * 2 * (x - 1) * 2 * y * (1 - y) * (2 * y - 1))
L_exactXY = (x ** 2 * (x - 1) ** 2 * 2 * (1 - y) * (2 * y - 1)
                        - x ** 2 * (x - 1) ** 2 * 2 * y * (2 * y - 1)
                        + 2 * x ** 2 * (x - 1) ** 2 * 2 * y * (1 - y))
L_exactYX = (y ** 2 * (y - 1) ** 2 * 2 * (x - 1) * (2 * x - 1)
                        + y ** 2 * (y - 1) ** 2 * 2 * x * (2 * x - 1)
                        + 2 * y ** 2 * (y - 1) ** 2 * 2 * x * (x - 1))
L_exactYY = (2 * y * (y - 1) ** 2 * 2 * x * (x - 1) * (2 * x - 1)
                        + y ** 2 * 2 * (y - 1) * 2 * x * (x - 1) * (2 * x - 1))
L_exact = CF((L_exactXX, L_exactXY, L_exactYX, L_exactYY), dims=(2, 2))

p_exact = x * (1 - x) * (1 - y) - 1 / 12


# ========= Hdiv-P0 scheme =========
V = MatrixValued(L2(mesh, order=0), mesh.dim, False)
W = HDiv(mesh, order=0, RT=True, dirichlet=".*")
M = HCurl(mesh, order=0, type1=True, dirichlet=".*")
Q = L2(mesh, order=0, lowest_order_wb=False)
fes = V * W * M * Q
(L, u, uhat, p), (G, v, vhat, q) = fes.TnT()
gfu = GridFunction(fes)
Lh, uh, uhath, ph = gfu.components

n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size
def tang(v):
        return v - (v*n)*n
# bilinear form and rhs linear form
a = BilinearForm(fes, symmetric=False, condense=True)
a += epsilon * p * q * dx # one-time Augmented Lagrangian Uzawa method
a += (InnerProduct(L, G) + c_low*u*v - p*div(v) - div(u)*q) * dx
a += (-G*n * ((u*n)*n + tang(uhat)) + L*n * ((v*n)*n + tang(vhat))) * dx(element_boundary=True)

f = LinearForm(fes)
f += (-(4 * y * (1 - y) * (2 * y - 1) * ((1 - 2 * x) ** 2 - 2 * x * (1 - x))
                + 12 * x ** 2 * (1 - x) ** 2 * (1 - 2 * y))
        + (1 - 2 * x) * (1 - y)) * v[0] * dx
f += (-(4 * x * (1 - x) * (1 - 2 * x) * ((1 - 2 * y) ** 2 - 2 * y * (1 - y))
                + 12 * y ** 2 * (1 - y) ** 2 * (2 * x - 1))
        - x * (1 - x)) * v[1] * dx
f += c_low * u_exact * v * dx

# ========= Crouzeix-Raviart scheme =========
V_cr = FESpace('nonconforming', mesh, dirichlet='.*')
fes_cr = V_cr * V_cr
(ux_cr, uy_cr), (vx_cr, vy_cr) = fes_cr.TnT()

u_cr = CF((ux_cr, uy_cr))
v_cr = CF((vx_cr, vy_cr))
GradU_cr = CF((grad(ux_cr), grad(uy_cr)))
GradV_cr = CF((grad(vx_cr), grad(vy_cr)))
divU_cr = grad(ux_cr)[0] + grad(uy_cr)[1]
divV_cr = grad(vx_cr)[0] + grad(vy_cr)[1]
# bilinear form, equivalent to condensed Hdiv-P0
a_cr = BilinearForm(fes_cr)
a_cr += (InnerProduct(GradU_cr, GradV_cr) 
         + c_low * Interpolate(u_cr, W) * Interpolate(v_cr, W)
         + 1/epsilon * divU_cr * divV_cr) * dx
# CR MG initialization
et = meshTopology(mesh, mesh.dim)
et.Update()
prolVcr = FacetProlongationTrig2(mesh, et)
a_cr.Assemble()
MG_cr = MultiGrid(a_cr.mat, prolVcr, nc=V_cr.ndof,
                    coarsedofs=fes_cr.FreeDofs(), w1=0.8,
                    nsmooth=4, sm="gs", var=True,
                    he=True, dim=mesh.dim, wcycle=False)

# L2 projection from fes_cr to fes
# one-to-one corresponding DOFs
mixmass_cr = BilinearForm(trialspace=fes_cr, testspace=fes)
mixmass_cr += (u_cr*n) * (v*n) * dx(element_boundary=True)
mixmass_cr += tang(u_cr) * tang(vhat) * dx(element_boundary=True)

fesMass = BilinearForm(fes)
fesMass += (u*n) * (v*n) * dx(element_boundary=True)
fesMass += tang(uhat) * tang(vhat) * dx(element_boundary=True)

def SolveBVP_CR(level, drawResult=False):
    t0 = timeit.time()
    fes.Update(); fes_cr.Update()
    gfu.Update()
    a.Assemble(); a_cr.Assemble()
    f.Assemble()
    mixmass_cr.Assemble(); fesMass.Assemble()

    udofs = BitArray(fes.ndof)
    udofs[:] = 0
    cur_ndof = V.ndof
    udofs[cur_ndof: cur_ndof+W.ndof] = W.FreeDofs(True)
    cur_ndof += W.ndof
    udofs[cur_ndof: cur_ndof+M.ndof] = M.FreeDofs(True)
    # global dofs mass mat => fesMass is diagonal
    # when dim = 2 !!!!!
    fesM_inv = fesMass.mat.CreateSmoother(udofs)
    E = fesM_inv @ mixmass_cr.mat # E: fes_cr => fes
    ET = mixmass_cr.mat.T @ fesM_inv

    # CR MG update
    if level > 0:
        et.Update()
        pp = [fes_cr.FreeDofs()]
        pp.append(V_cr.ndof)
        pdofs = BitArray(fes_cr.ndof)
        pdofs[:] = 0
        inner = prolVcr.GetInnerDofs(level)
        for j in range(mesh.dim):
            pdofs[j * V_cr.ndof:(j + 1) * V_cr.ndof] = inner
        # he_prol
        pp.append(a_cr.mat.Inverse(pdofs, inverse="sparsecholesky"))
        # bk smoother
        # TODO: WHY THIS IS WRONG???!!!
        # bjac = et.CreateSmoother(a_cr, {"blocktype": "vertexpatch"})
        # pp.append(bjac)
        pp.append(VertexPatchBlocks(mesh, fes_cr))
        MG_cr.Update(a_cr.mat, pp)
    
    # inv_cr = a_cr.mat.Inverse(fes_cr.FreeDofs(), inverse='umfpack')
    inv_cr = MG_cr

    pre = E @ inv_cr @ ET
    t1 = timeit.time()

    # dirichlet BC
    # homogeneous Dirichlet assumed
    f.vec.data += a.harmonic_extension_trans * f.vec
    # gfu.vec.data += E @ inv_cr @ ET * f.vec
    inv_fes = CGSolver(a.mat, pre, printrates=False, tol=1e-8, maxiter=50)

    gfu.vec.data += inv_fes * f.vec
    gfu.vec.data += a.harmonic_extension * gfu.vec
    gfu.vec.data += a.inner_solve * f.vec
        
    it = inv_fes.iterations
    # gfu_cr.vec.data += a_cr.mat.Inverse(fes_cr.FreeDofs(True), inverse='umfpack') * f_cr.vec
    t2 = timeit.time()

    lams = EigenValues_Preconditioner(mat=a.mat, pre=pre)
    print(f"Assemble & Update: {t1-t0:.2e}, Solve: {t2-t1:.2e}")
    print(f"IT: {it}, COND: {max(lams)/min(lams):.2E}, MAX PREC LAM: {max(lams):.1E}; MIN PREC LAM: {min(lams):.1E}")
    if drawResult:
        Draw(uh, mesh)
        # Draw(uh_cr, mesh)

drawResult = False
SolveBVP = SolveBVP_CR

print(f'===== DIM: {mesh.dim}, c_low: {c_low}, eps: {epsilon} =====')
SolveBVP(0, drawResult)
print(f'level: {0}')
L2_uErr = sqrt(Integrate((uh - u_exact) * (uh - u_exact), mesh))
L2_LErr = sqrt(Integrate(InnerProduct((Lh - L_exact), (Lh - L_exact)), mesh))
L2_divErr = sqrt(Integrate(div(uh) * div(uh), mesh))
print(f"uh L2-error: {L2_uErr:.3E}")
print(f"Lh L2-error: {L2_LErr:.3E}")
print(f'uh divErr: {L2_divErr:.1E}')
print('==============================')
prev_uErr = L2_uErr
prev_LErr = L2_LErr
u_rate, L_rate = 0, 0
level = 1
while True:
    with TaskManager():
        mesh.ngmesh.Refine()
        # exit if total global dofs exceed a tol
        M.Update()
        if (M.ndof * mesh.dim > maxdofs):
            print(M.ndof * mesh.dim)
            break
        print(f'level: {level}')
        SolveBVP(level, drawResult)
        # ===== convergence check =====
        meshRate = 2
        L2_uErr = sqrt(Integrate((uh - u_exact) * (uh - u_exact), mesh))
        L2_LErr = sqrt(Integrate(InnerProduct((Lh - L_exact), (Lh - L_exact)), mesh))
        L2_divErr = sqrt(Integrate(div(uh) * div(uh), mesh))
        u_rate = log(prev_uErr / L2_uErr) / log(meshRate)
        L_rate = log(prev_LErr / L2_LErr) / log(meshRate)
        print(f"uh L2-error: {L2_uErr:.3E}, uh conv rate: {u_rate:.2E}")
        print(f"Lh L2-error: {L2_LErr:.3E}, Lh conv rate: {L_rate:.2E}")
        print(f'uh divErr: {L2_divErr:.1E}')
        print('==============================')
        prev_uErr = L2_uErr
        prev_LErr = L2_LErr
        level += 1