# mixed Hdiv-HDG for Stokes, hp-MG preconditioned CG solver
# Augmented Lagrangian Uzawa iteration for outer iteration
# Knonw solution for checking order of convergence

from ngsolve import *
import time as timeit
from ngsolve.krylovspace import CGSolver
from ngsolve.la import EigenValues_Preconditioner
# geometry
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh
# from netgen.csg import unit_cube
# customized functions
from prol import meshTopology, FacetProlongationTrig2, FacetProlongationTet2
from auxPyFiles.myMG import MultiGrid
from auxPyFiles.mySmoother import VertexPatchBlocks, EdgePatchBlocks, FacetBlocks, SymmetricGS
from auxPyFiles.myStokesHelper import stokesHelper
from auxPyFiles.myASP import MultiASP
from auxPyFiles.myIterSolver import IterSolver

import sys
if len(sys.argv) < 6:
    print('not enough input args: dim + c_low + nMGSmooth + order + precond?'); exit(1)
dim = int(sys.argv[1])
c_low = int(sys.argv[2])
nMGSmooth = int(sys.argv[3])
order = int(sys.argv[4])
precond = bool(int(sys.argv[5]))

if dim != 2 and dim != 3:
    print('WRONG DIMENSION!'); exit(1)

iniN = 2
maxdofs = 5e7
maxLevel = 6
epsilon = 1e-4
uzawaIt = 2
drawResult = False

# ========== START of MESH and EXACT SOLUTION ==========
dirichBDs = ".*"
mesh = MakeStructured2DMesh(quads=False, nx=iniN, ny=iniN) if dim == 2 \
       else MakeStructured3DMesh(hexes=False, nx=iniN, ny=iniN, nz=iniN)
helper = stokesHelper(dim)
helper.squareSolInit()
u_exact, _, _ = helper.getExactSol()
# ========== END of MESH and EXACT SOLUTION ==========





n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size
def tang(v):
        return v - (v*n)*n
# ========== START of CR MG SETUP for P-MG ==========
# # ========= Crouzeix-Raviart scheme =========
W0 = HDiv(mesh, order=0, RT=True, dirichlet=dirichBDs) if mesh.dim == 2 \
     else HDiv(mesh, order=0, dirichlet=dirichBDs)
V_cr = FESpace('nonconforming', mesh, dirichlet=dirichBDs)
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
                        nsmooth=nMGSmooth, sm="gs", var=True,
                        he=True, dim=mesh.dim, wcycle=False)

# ========== END of CR MG SETUP for P-MG ==========



# ========= START of HIGHER ORDER Hdiv-HDG SCHEME TO BE SOLVED ==========
# ========= mixed-HidvHDG scheme =========
V = MatrixValued(L2(mesh, order=order), mesh.dim, False)
if mesh.dim == 2:
    W = HDiv(mesh, order=order, RT=True, dirichlet=dirichBDs)
elif mesh.dim == 3:
    W = HDiv(mesh, order=order, 
             RT=True if order>=1 else False, dirichlet=dirichBDs) # inconsistent option when lowest order
M = TangentialFacetFESpace(mesh, order=order, dirichlet=dirichBDs)

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
# rhs linear form
f = helper.getRhs(fes, c_low, v)
# L2 projection from fes0 to fes
mixmass = BilinearForm(trialspace=fes_cr, testspace=fes)
# tangential part
mixmass += tang(u_cr) * tang(vhat) * dx(element_boundary=True)
# normal part
mixmass += (u_cr*n) * (v*n) * dx(element_boundary=True)

fesMass = BilinearForm(fes)
fesMass += tang(uhat) * tang(vhat) * dx(element_boundary=True)
fesMass += (u*n) * (v*n) * dx(element_boundary=True)


# ========== secondary variable operator for AL uzawa method
Q = L2(mesh, order=order)
p, q = Q.TnT()
b = BilinearForm(trialspace=Q, testspace=fes)
b += - p * div(v) * dx
pMass = BilinearForm(Q)
pMass += p * q * dx
# ========= END of HIGHER ORDER Hdiv-HDG SCHEME TO BE SOLVED ==========



# ========= START of HP-MG for HIGHER ORDER Hdiv-HDG ==========
def SolveBVP_CR(level, drawResult=False):
    with TaskManager():
    # with TaskManager(pajetrace=10**8):
        t0 = timeit.time()
        fes.Update(); fes_cr.Update(); W0.Update()
        gfu.Update()
        a.Assemble()
        a_cr.Assemble()
        f.Assemble()

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
            if dim == 2:
                # vertex block smoothers, if no hacker made to ngsolve source file,
                # use the following line instead
                # pp.append(VertexPatchBlocks(mesh, fes_cr))
                pp.append(fes_cr.CreateSmoothBlocks(vertex=True, globalDofs=False))
            elif dim == 3:
                # edge block smoothers in 3D to save memory, if no hacker made to ngsolve source file,
                # use the following line instead
                # pp.append(EdgePatchBlocks(mesh, fes_cr))
                pp.append(fes_cr.CreateSmoothBlocks(vertex=False, globalDofs=False))
            MG_cr.Update(a_cr.mat, pp)


 
        # ========== PRECONDITIONER SETUP START
        if dim == 2:
            fesM_inv = fesMass.mat.CreateSmoother(fes.FreeDofs(True))
        elif dim == 3:
            # facet blocks for mass mat inverse, if no hacker made to ngsolve source file,
            # use the following line instead
            # fesM_inv = fesMass.mat.CreateBlockSmoother(FacetBlocks(mesh, fes))
            fesM_inv = fesMass.mat.CreateBlockSmoother(fes.CreateFacetBlocks(globalDofs=True))

        E = fesM_inv @ mixmass.mat # E: fes0 => fes
        ET = mixmass.mat.T @ fesM_inv

        # block smoothers, if no hacker made to ngsolve source file,
        # use the following line instead
        # blocks = VertexPatchBlocks(mesh, fes) if mesh.dim == 2 else EdgePatchBlocks(mesh, fes)
        blocks = fes.CreateSmoothBlocks(vertex=True, globalDofs=True) if mesh.dim == 2 \
                 else fes.CreateSmoothBlocks(vertex=False, globalDofs=True)
        
        # inv_cr = a_cr.mat.Inverse(fes_cr.FreeDofs(), inverse='sparsecholesky')
        inv_cr = MG_cr
        coarse = E @ inv_cr @ ET

        pre = MultiASP(a.mat, fes.FreeDofs(True), coarse, 
                       smoother=a.mat.CreateBlockSmoother(blocks), 
                       nSm=0 if order==0 else 1)
        # R = SymmetricGS(a.mat.CreateBlockSmoother(vblocks)) # block GS for p-MG smoothing
        # pre = R + E @ inv_cr @ ET # additive ASP
        t1 = timeit.time()



        # ========== HDG STATIC CONDENSATION and SOLVED by AL uzawa
        p_prev = GridFunction(Q)
        p_prev.vec.data[:] = 0
        Q.Update()
        b.Assemble() #b += - p * div(v) * dx
        pMass.Assemble()
        # p mass diagonal in both 2D and 3D cases
        pMass_inv= pMass.mat.CreateSmoother(Q.FreeDofs())
        it = 0
        for _ in range(uzawaIt):
            # homo dirichlet BC
            gfu.vec.data[:] = 0
            rhs = f.vec.CreateVector()
            rhs.data = f.vec - a.mat * gfu.vec
            rhs.data += -b.mat * p_prev.vec
            # update L and u
            rhs.data += a.harmonic_extension_trans * rhs

            if precond:
                inv_fes = CGSolver(a.mat, pre, printrates=False, tol=1e-8, maxiter=500)
            else:
                inv_fes = IterSolver(mat=a.mat, pre=pre, printrates=False, tol=1e-8, atol=5e-7,
                                     maxiter=200, 
                                     freedofs=fes.FreeDofs(True))
            gfu.vec.data += inv_fes * rhs
            gfu.vec.data += a.harmonic_extension * gfu.vec
            gfu.vec.data += a.inner_solve * rhs
            # update pressure
            p_prev.vec.data += 1/epsilon * (pMass_inv @ b.mat.T * gfu.vec.data)
            it += inv_fes.iterations


        # ========= PRINT RESULTS
        t2 = timeit.time()
        it //= uzawaIt
        # lams = EigenValues_Preconditioner(mat=a.mat, pre=pre)
        print(f"==> MG in PCG: {precond}")
        print(f"==> Assemble & Update: {t1-t0:.2e}, Solve: {t2-t1:.2e}")
        print(f"==> AVG IT: {it}, N_smooth: {nMGSmooth}") #, MAX LAM: {max(lams):.2e}, MIN LAM: {min(lams):.2e}, COND: {max(lams)/min(lams):.2E}")
        if drawResult:
            import netgen.gui
            Draw(Norm(uh), mesh, 'sol')
            input('continue?')

# ========= END of HP-MG for HIGHER ORDER Hdiv-HDG ==========




# ========= START of CONVERGENCE ORDER CHECK ==========
SolveBVP = SolveBVP_CR
print(f'===== DIM: {mesh.dim}, ORDER:{ order} c_low: {c_low}, eps: {epsilon} =====')
SolveBVP(0, drawResult)
prev_uErr, prev_LErr = helper.ecrCheck(0, fes, mesh, uh, Lh)
level = 1
while True:
    # uniform refinement used, meshRate=2 in ecrCheck
    mesh.ngmesh.Refine(); meshrate = 2
    # if mesh.dim == 2:
    #     mesh.ngmesh.Refine(); meshrate = 2
    # else:
    #     mesh.Refine(onlyonce = True); meshrate = sqrt(2)
        
    # exit if total global dofs exceed a0 tol
    M.Update(); W.Update(); fes.Update()
    globalDofs = sum(W.FreeDofs(True)) + sum(M.FreeDofs(True))
    if globalDofs > maxdofs or level > maxLevel:
        print(f'# totalDofs: {fes.ndof} # global DOFS {globalDofs}')
        break
    print(f'===== LEVEL {level} =====')
    print(f'# totalDofs: {fes.ndof} # global DOFS {globalDofs}')
    SolveBVP(level, drawResult)
    prev_uErr, prev_LErr = helper.ecrCheck(level, fes, mesh, uh, Lh, meshRate = meshrate, 
                                           prev_uErr=prev_uErr, prev_LErr=prev_LErr)
    level += 1

# ========= END of CONVERGENCE ORDER CHECK ==========
