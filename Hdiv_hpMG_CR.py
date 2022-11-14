# Stokes, hp-MG preconditioned CG solver
# One time Lagrangian Augmented Uzawa iteration
from ngsolve import *
import time as timeit
from ngsolve.krylovspace import CGSolver
from ngsolve.la import EigenValues_Preconditioner
# geometry
from ngsolve.meshes import MakeStructured2DMesh
from netgen.csg import unit_cube
# customized functions
from prol import meshTopology, FacetProlongationTrig2, FacetProlongationTet2
from mymg import MultiGrid
from mySmoother import VertexPatchBlocks, FacetPatchBlocks, EdgePatchBlocks, SymmetricGS
from myStokesHelper import stokesInit
from myASP import MultiASP

import sys
if len(sys.argv) < 4:
    print('not enough input args: dim + c_low + nSmooth'); exit(1)
dim = int(sys.argv[1])
c_low = int(sys.argv[2])
nSmooth = int(sys.argv[3])

if dim != 2 and dim != 3:
    print('WRONG DIMENSION!'); exit(1)

iniN = 4
maxdofs = 1e5
epsilon = 1e-8
order = 6
drawResult = False

# ========== START of MESH and EXACT SOLUTION ==========
mesh = MakeStructured2DMesh(quads=False, nx=iniN) if dim == 2 \
       else Mesh(unit_cube.GenerateMesh(maxh=1/iniN))
helper = stokesInit(dim)
u_exact, _, _ = helper.getExactSol()
# ========== END of MESH and EXACT SOLUTION ==========





n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size
def tang(v):
        return v - (v*n)*n
# ========== START of CR MG SETUP for P-MG ==========
# # ========= Crouzeix-Raviart scheme =========
W0 = HDiv(mesh, order=0, RT=True, dirichlet=".*") if mesh.dim == 2 \
     else HDiv(mesh, order=0, dirichlet=".*")
V_cr = FESpace('nonconforming', mesh, dirichlet='.*')
if mesh.dim == 2:
    fes_cr = V_cr * V_cr
    (ux_cr, uy_cr), (vx_cr, vy_cr) = fes_cr.TnT()

    u_cr = CF((ux_cr, uy_cr))
    v_cr = CF((vx_cr, vy_cr))
    GradU_cr = CF((grad(ux_cr), grad(uy_cr)))
    GradV_cr = CF((grad(vx_cr), grad(vy_cr)))
    divU_cr = grad(ux_cr)[0] + grad(uy_cr)[1]
    divV_cr = grad(vx_cr)[0] + grad(vy_cr)[1]
elif mesh.dim == 3:
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
        + c_low * Interpolate(u_cr, W0) * Interpolate(v_cr, W0)
        + 1/epsilon * divU_cr * divV_cr) * dx
# # ========== CR MG initialization
et = meshTopology(mesh, mesh.dim)
et.Update()
prolVcr = FacetProlongationTrig2(mesh, et) if dim==2 \
          else FacetProlongationTet2(mesh, et)
with TaskManager():
    a_cr.Assemble()
    MG_cr = MultiGrid(a_cr.mat, prolVcr, nc=V_cr.ndof,
                        coarsedofs=fes_cr.FreeDofs(), w1=0.8,
                        nsmooth=nSmooth, sm="gs", var=True,
                        he=True, dim=mesh.dim, wcycle=False)

# ========== END of CR MG SETUP for P-MG ==========








# ========= START of HIGHER ORDER Hdiv-HDG SCHEME TO BE SOLVED ==========
# ========= mixed-HidvHDG scheme =========
V = MatrixValued(L2(mesh, order=order), mesh.dim, False)
if mesh.dim == 2:
    W = HDiv(mesh, order=order, RT=True, dirichlet=".*")
elif mesh.dim == 3:
    W = HDiv(mesh, order=order, dirichlet=".*") # inconsistent option in NGSolve
M = TangentialFacetFESpace(mesh, order=order, dirichlet=".*")

fes = V * W * M 
(L, u,uhat),  (G, v, vhat) = fes.TnT()

gfu = GridFunction (fes)
Lh, uh, uhath = gfu.components

# gradient by row
gradv, gradu = Grad(v), Grad(u)

# bilinear form of SIP-HdivHDG
a = BilinearForm(fes, symmetric=False, condense=True)
# volume term
a += (1/epsilon * div(u) * div(v)) * dx
a += (InnerProduct(L, G) + c_low * u * v
      -InnerProduct(gradu, G) + InnerProduct(L, gradv)) * dx
a += (tang(u-uhat) * tang(G*n) - tang(L*n) * tang(v-vhat))*dx(element_boundary=True)

# L2 projection from fes0 to fes
mixmass = BilinearForm(trialspace=fes_cr, testspace=fes)
# tangential part
mixmass += tang(u_cr) * tang(vhat) * dx(element_boundary=True)
# normal part
mixmass += (u_cr*n) * (v*n) * dx(element_boundary=True)

fesMass = BilinearForm(fes)
fesMass += tang(uhat) * tang(vhat) * dx(element_boundary=True)
fesMass += (u*n) * (v*n) * dx(element_boundary=True)
# ========= END of HIGHER ORDER Hdiv-HDG SCHEME TO BE SOLVED ==========











# ========= START of HP-MG for HIGHER ORDER Hdiv-HDG ==========

def SolveBVP_CR(level, drawResult=False):
    with TaskManager():
        t0 = timeit.time()
        fes.Update(); fes_cr.Update(); W0.Update()
        gfu.Update()
        a.Assemble(); a_cr.Assemble()
        # rhs linear form
        f = LinearForm(fes)
        f.vec.data = helper.getRhs(fes, c_low, testV=v)

        mixmass.Assemble()
        fesMass.Assemble()
        # # ========== CR MG update
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
            pp.append(VertexPatchBlocks(mesh, fes_cr))
            MG_cr.Update(a_cr.mat, pp)



        # ========== PRECONDITIONER SETUP START
        if dim == 2:
            fesM_inv = fesMass.mat.CreateSmoother(fes.FreeDofs(True))
        elif dim == 3:
            # TODO: facetPatchBlocks is not correct here. WHY????
            # TODO: facetPatchBlocks is not correct here. WHY????
            # fesM_inv = fesMass.mat.CreateBlockSmoother(FacetPatchBlocks(mesh, fes))
            fesM_inv = fesMass.mat.Inverse(fes.FreeDofs(True), inverse='sparsecholesky')
        
        E = fesM_inv @ mixmass.mat # E: fes0 => fes
        ET = mixmass.mat.T @ fesM_inv
        vBlocks = VertexPatchBlocks(mesh, fes)
        R = SymmetricGS(a.mat.CreateBlockSmoother(vBlocks)) # vertex block GS for p-MG smoothing
        
        # inv_cr = a_cr.mat.Inverse(fes_cr.FreeDofs(), inverse='sparsecholesky')
        inv_cr = MG_cr
        coarse = E @ inv_cr @ ET

        pre = MultiASP(a.mat, fes.FreeDofs(True), coarse, smoother=a.mat.CreateBlockSmoother(vBlocks), nSm=1)
        # pre = R + R - R @ a.mat @ R \
        #     + coarse - R @ a.mat @ coarse - coarse @ a.mat @ R + R @ a.mat @ coarse @ a.mat @ R # Multiplicative ASP
        # pre = R + E @ inv_cr @ ET # additive ASP
        t1 = timeit.time()




        # ========== HDG STATIC CONDENSATION and SOLVED
        # dirichlet BC
        f.vec.data -= a.mat * gfu.vec
        f.vec.data += a.harmonic_extension_trans * f.vec
        # inv_fes = pre
        inv_fes = CGSolver(a.mat, pre, printrates=False, tol=1e-8, maxiter=500)
        gfu.vec.data += inv_fes * f.vec
        gfu.vec.data += a.harmonic_extension * gfu.vec
        gfu.vec.data += a.inner_solve * f.vec



        # ========= PRINT RESULTS
        t2 = timeit.time()
        # it = 1  
        # lams = [1, 1]
        it = inv_fes.iterations
        lams = EigenValues_Preconditioner(mat=a.mat, pre=pre)
        print(f"==> Assemble & Update: {t1-t0:.2e}, Solve: {t2-t1:.2e}")
        print(f"==> IT: {it}, N_smooth: {nSmooth}, MAX LAM: {max(lams):.2e}, MIN LAM: {min(lams):.2e}, COND: {max(lams)/min(lams):.2E}")
        if drawResult:
            import netgen.gui
            Draw(Norm(uh), mesh, 'sol')
            input('continue?')
            # Draw(uh_cr, mesh)

# ========= END of HP-MG for HIGHER ORDER Hdiv-HDG ==========







SolveBVP = SolveBVP_CR
print(f'===== DIM: {mesh.dim}, ORDER:{ order} c_low: {c_low}, eps: {epsilon} =====')
SolveBVP(0, drawResult)
prev_uErr, prev_LErr = helper.ecrCheck(0, fes, mesh, uh, Lh)
level = 1
while True:
    # uniform refinement used, meshRate=2 in ecrCheck
    if mesh.dim == 2: mesh.ngmesh.Refine(); meshrate = 2
    else:   mesh.Refine(onlyonce = True); meshrate = sqrt(2)
    # exit if total global dofs exceed a0 tol
    M.Update(); W.Update()
    if (sum(W.FreeDofs(True)) + sum(W.FreeDofs(True)) > maxdofs):
        print(f'# global DOFS {sum(W.FreeDofs(True)) + sum(W.FreeDofs(True))} exceed MAX')
        break
    SolveBVP(level, drawResult)
    prev_uErr, prev_LErr = helper.ecrCheck(level, fes, mesh, uh, Lh, meshRate = meshrate, 
                                           prev_uErr=prev_uErr, prev_LErr=prev_LErr)
    level += 1

        