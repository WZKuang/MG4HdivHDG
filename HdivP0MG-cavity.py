# Stokes, Hdiv-P0 MG preconditioned CG solver
# One time Lagrangian Augmented Uzawa iteration
# condensed mixed Hdiv-P0 equivalent to CR scheme
# Test case - lid-driven cavity problem
from distutils.log import error
from ngsolve import *
from netgen.geom2d import SplineGeometry, unit_square
import time as timeit
from netgen.csg import *
from ngsolve.krylovspace import CGSolver, CG
from prol import *
from mymg import *
from ngsolve.la import EigenValues_Preconditioner

import sys
if len(sys.argv) < 4:
    print('not enough input args: dim + c_low + nSmooth'); exit(1)
dim = int(sys.argv[1])
c_low = int(sys.argv[2])
nSmooth = int(sys.argv[3])

iniN = 4
maxdofs = 1e6
epsilon = 1e-6

if dim == 2:
    mesh = Mesh(unit_square.GenerateMesh(maxh=1/iniN))
    # top side dirichlet bd
    utop = CoefficientFunction((4*x*(1-x),0))
elif dim == 3:
    mesh = Mesh(unit_cube.GenerateMesh(maxh=1/iniN))
    utop = CoefficientFunction((16*x*(1-x)*y*(1-y),0,0))
else:
    error('WRONG DIMENSION!'); exit()


# ========= Hdiv-P0 scheme =========
V = MatrixValued(L2(mesh, order=0), mesh.dim, False)
if dim == 2:
    W = HDiv(mesh, order=0, RT=True, dirichlet=".*")
elif dim == 3:
    W = HDiv(mesh, order=0, dirichlet=".*") # inconsistent option
M = TangentialFacetFESpace(mesh, order=0, dirichlet=".*")
fes = V * W * M 
(L, u, uhat), (G, v, vhat) = fes.TnT()
gfu = GridFunction(fes)
Lh, uh, uhath = gfu.components

print(f'V ndof:{V.ndof}, W ndof:{W.ndof}, M ndof:{M.ndof}')

n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size
def tang(v):
        return v - (v*n)*n
# bilinear form and rhs linear form
a = BilinearForm(fes, symmetric=False, condense=True)
a += 1/epsilon * div(u) * div(v) * dx # one-time Augmented Lagrangian Uzawa method
a += (InnerProduct(L, G) + c_low*u*v) * dx
a += (-G*n * ((u*n)*n + tang(uhat)) + L*n * ((v*n)*n + tang(vhat))) * dx(element_boundary=True)

f = LinearForm(fes)

# ========= Crouzeix-Raviart scheme =========
V_cr = FESpace('nonconforming', mesh, dirichlet='.*')
if dim == 2:
    fes_cr = V_cr * V_cr
    (ux_cr, uy_cr), (vx_cr, vy_cr) = fes_cr.TnT()

    u_cr = CF((ux_cr, uy_cr))
    v_cr = CF((vx_cr, vy_cr))
    GradU_cr = CF((grad(ux_cr), grad(uy_cr)))
    GradV_cr = CF((grad(vx_cr), grad(vy_cr)))
    divU_cr = grad(ux_cr)[0] + grad(uy_cr)[1]
    divV_cr = grad(vx_cr)[0] + grad(vy_cr)[1]
elif dim == 3:
    fes_cr = V_cr * V_cr * V_cr
    (ux_cr, uy_cr, uz_cr), (vx_cr, vy_cr, vz_cr) = fes_cr.TnT()

    u_cr = CF((ux_cr, uy_cr, uz_cr))
    v_cr = CF((vx_cr, vy_cr, vz_cr))
    GradU_cr = CF((grad(ux_cr), grad(uy_cr), grad(uz_cr)))
    GradV_cr = CF((grad(vx_cr), grad(vy_cr), grad(vz_cr)))
    divU_cr = grad(ux_cr)[0] + grad(uy_cr)[1] + grad(uz_cr)[2]
    divV_cr = grad(vx_cr)[0] + grad(vy_cr)[1] + grad(vz_cr)[2]
# bilinear form, equivalent to condensed Hdiv-P0
a_cr = BilinearForm(fes_cr)
a_cr += (InnerProduct(GradU_cr, GradV_cr) 
         + c_low * Interpolate(u_cr, W) * Interpolate(v_cr, W)
        # + c_low * u_cr * v_cr
        + 1/epsilon * divU_cr * divV_cr) * dx

# ========== CR MG initialization
et = meshTopology(mesh, mesh.dim)
et.Update()
prolVcr = FacetProlongationTrig2(mesh, et) if dim==2 else FacetProlongationTet2(mesh, et)
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

    # global dofs mass mat => fesMass is diagonal
    # but only when dim = 2 !!!!!
    if dim == 2:
        fesM_inv = fesMass.mat.CreateSmoother(fes.FreeDofs(True))
    elif dim == 3:
        fesM_inv = fesMass.mat.CreateBlockSmoother(FacetPatchBlocks(mesh, fes))
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
    
    # inv_cr = a_cr.mat.Inverse(fes_cr.FreeDofs(), inverse='sparsecholesky')
    inv_cr = MG_cr

    pre = E @ inv_cr @ ET
    t1 = timeit.time()

    # dirichlet BC
    uhath.Set(utop, definedon=mesh.Boundaries("top"))
    f.vec.data = -a.mat * gfu.vec
    f.vec.data += a.harmonic_extension_trans * f.vec
    # gfu.vec.data += E @ inv_cr @ ET * f.vec
    inv_fes = CGSolver(a.mat, pre, printrates=False, tol=1e-8, maxiter=30)
    gfu.vec.data += inv_fes * f.vec
    gfu.vec.data += a.harmonic_extension * gfu.vec
    gfu.vec.data += a.inner_solve * f.vec
        
    it = inv_fes.iterations
    # gfu_cr.vec.data += a_cr.mat.Inverse(fes_cr.FreeDofs(True), inverse='umfpack') * f_cr.vec
    t2 = timeit.time()

    lams = EigenValues_Preconditioner(mat=a_cr.mat, pre=MG_cr)
    print(f"==> Assemble & Update: {t1-t0:.2e}, Solve: {t2-t1:.2e}")
    print(f"==> IT: {it}, N_smooth: {nSmooth}, COND: {max(lams)/min(lams):.2E}")
    if drawResult:
        Draw(uh, mesh, 'sol')
        input('continue')


drawResult = False
if drawResult:  import netgen.gui
SolveBVP = SolveBVP_CR
def ecrCheck(level):
    print(f'LEVEL: {level}, ALL DOFS: {fes.ndof}, GLOBAL DOFS: {W.ndof + M.ndof}')
    L2_divErr = sqrt(Integrate(div(uh) * div(uh), mesh))
    print(f'uh divErr: {L2_divErr:.1E}')
    print('==============================')

print(f'===== DIM: {mesh.dim}, c_low: {c_low}, eps: {epsilon} =====')
SolveBVP(0, drawResult)
ecrCheck(0)
level = 1
while True:
    with TaskManager():
        # uniform refinement
        mesh.ngmesh.Refine()
        # exit if total global dofs exceed a tol
        M.Update(); W.Update()
        if (W.ndof + M.ndof > maxdofs):
            print(W.ndof + M.ndof)
            break
        SolveBVP(level, drawResult)
        ecrCheck(level)
        level += 1

       