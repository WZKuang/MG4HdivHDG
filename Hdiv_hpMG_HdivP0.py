# Stokes, Hdiv-P0 MG preconditioned CG solver
# One time Lagrangian Augmented Uzawa iteration
# condensed mixed Hdiv-P0 equivalent to CR scheme
from distutils.log import error
from ngsolve import *
from ngsolve.meshes import *
from netgen.geom2d import SplineGeometry, unit_square
import time as timeit
from netgen.csg import *
from ngsolve.krylovspace import CGSolver, GMResSolver, MinResSolver
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
maxdofs = 3e5
epsilon = 1e-6
order = 3
drawResult = True

# ========== START of MESH and EXACT SOLUTION ==========

if dim == 2:
    mesh = MakeStructured2DMesh(quads=False, nx=iniN)
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
elif dim == 3:
    mesh = Mesh(unit_cube.GenerateMesh(maxh=1/iniN))
    # exact solution
    u_exact1 = x ** 2 * (x - 1) ** 2 * (2 * y - 6 * y ** 2 + 4 * y ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
    u_exact2 = y ** 2 * (y - 1) ** 2 * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
    u_exact3 = -2 * z ** 2 * (z - 1) ** 2 * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * y - 6 * y ** 2 + 4 * y ** 3)
    u_exact = CF((u_exact1, u_exact2, u_exact3))

    L_exactXX = (2 * x * (x - 1) ** 2 * (2 * y - 6 * y ** 2 + 4 * y ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
                            + x ** 2 * 2 * (x - 1) * (2 * y - 6 * y ** 2 + 4 * y ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3))
    L_exactXY = (x ** 2 * (x - 1) ** 2 * (2 - 12 * y + 12 * y ** 2) * (2 * z - 6 * z ** 2 + 4 * z ** 3))
    L_exactXZ = (x ** 2 * (x - 1) ** 2 * (2 - 12 * z + 12 * z ** 2) * (2 * y - 6 * y ** 2 + 4 * y ** 3))
    L_exactYX = (y ** 2 * (y - 1) ** 2 * (2 - 12 * x + 12 * x ** 2) * (2 * z - 6 * z ** 2 + 4 * z ** 3))
    L_exactYY = (2 * y * (y - 1) ** 2 * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
                            + y ** 2 * 2 * (y - 1) * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3))
    L_exactYZ = (y ** 2 * (y - 1) ** 2 * (2 - 12 * z + 12 * z ** 2) * (2 * x - 6 * x ** 2 + 4 * x ** 3))
    L_exactZX = (-2 * z ** 2 * (z - 1) ** 2 * (2 - 12 * x + 12 * x ** 2) * (2 * y - 6 * y ** 2 + 4 * y ** 3))
    L_exactZY = (-2 * z ** 2 * (z - 1) ** 2 * (2 - 12 * y + 12 * y ** 2) * (2 * x - 6 * x ** 2 + 4 * x ** 3))
    L_exactZZ = (
                -2 * 2 * z * (z - 1) ** 2 * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * y - 6 * y ** 2 + 4 * y ** 3)
                - 2 * z ** 2 * 2 * (z - 1) * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * y - 6 * y ** 2 + 4 * y ** 3))
    L_exact = CF((L_exactXX, L_exactXY, L_exactXZ,
                    L_exactYX, L_exactYY, L_exactYZ,
                    L_exactZX, L_exactZY, L_exactZZ), dims=(3, 3))

    p_exact = x * (1 - x) * (1 - y) * (1 - z) - 1 / 24
else:
    error('WRONG DIMENSION!'); exit()

# ========== END of MESH and EXACT SOLUTION ==========










n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size
def tang(v):
        return v - (v*n)*n
# ========== START of Hdiv-P0 MG SETUP for P-MG ==========

# ========== Hdiv-P0 scheme ==========
V0 = MatrixValued(L2(mesh, order=0), mesh.dim, False)
if mesh.dim == 2:
    W0 = HDiv(mesh, order=0, RT=True, dirichlet=".*")
elif mesh.dim == 3:
    W0 = HDiv(mesh, order=0, dirichlet=".*") # inconsistent option in NGSolve
M0 = TangentialFacetFESpace(mesh, order=0, dirichlet=".*")
fes0 = V0 * W0 * M0 
(L0, u0, uhat0), (G0, v0, vhat0) = fes0.TnT()
a0 = BilinearForm(fes0, symmetric=False, condense=True)
a0 += 1/epsilon * div(u0) * div(v0) * dx # one-time Augmented Lagrangian Uzawa method
a0 += (InnerProduct(L0, G0) + c_low*u0*v0) * dx
a0 += (-G0*n * ((u0*n)*n + tang(uhat0)) + L0*n * ((v0*n)*n + tang(vhat0))) * dx(element_boundary=True)

# ========= Crouzeix-Raviart scheme =========
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
# ========== CR MG initialization
et = meshTopology(mesh, mesh.dim)
et.Update()
prolVcr = FacetProlongationTrig2(mesh, et) if dim==2 else FacetProlongationTet2(mesh, et)
a_cr.Assemble()
MG_cr = MultiGrid(a_cr.mat, prolVcr, nc=V_cr.ndof,
                    coarsedofs=fes_cr.FreeDofs(), w1=0.8,
                    nsmooth=nSmooth, sm="gs", var=True,
                    he=True, dim=mesh.dim, wcycle=False)
# L2 projection from fes_cr to fes0
# one-to-one corresponding DOFs
mixmass_cr = BilinearForm(trialspace=fes_cr, testspace=fes0)
mixmass_cr += (u_cr*n) * (v0*n) * dx(element_boundary=True)
mixmass_cr += tang(u_cr) * tang(vhat0) * dx(element_boundary=True)

fesMass0 = BilinearForm(fes0)
fesMass0 += (u0*n) * (v0*n) * dx(element_boundary=True)
fesMass0 += tang(uhat0) * tang(vhat0) * dx(element_boundary=True)

# ========== END of Hdiv-P0 MG SETUP for P-MG ==========










# ========= START of HIGHER ORDER Hdiv-HDG SCHEME TO BE SOLVED ==========
# ========= mixed-HidvHDG scheme =========
V = MatrixValued(L2(mesh, order=order), mesh.dim, False)
if mesh.dim == 2:
    W = HDiv(mesh, order=order, RT=True, dirichlet=".*")
elif mesh.dim == 3:
    W = HDiv(mesh, order=order, dirichlet=".*") # inconsistent option in NGSolve
M = TangentialFacetFESpace(mesh, order=order, dirichlet=".*")

fes = V * W * M 
(L, u,uhat),  (G, v,vhat) = fes.TnT()

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

# linear form 
if dim == 2:
    f = LinearForm(fes)
    f += (-(4 * y * (1 - y) * (2 * y - 1) * ((1 - 2 * x) ** 2 - 2 * x * (1 - x))
                    + 12 * x ** 2 * (1 - x) ** 2 * (1 - 2 * y))
            + (1 - 2 * x) * (1 - y)) * v[0] * dx
    f += (-(4 * x * (1 - x) * (1 - 2 * x) * ((1 - 2 * y) ** 2 - 2 * y * (1 - y))
                    + 12 * y ** 2 * (1 - y) ** 2 * (2 * x - 1))
            - x * (1 - x)) * v[1] * dx
    f += c_low * u_exact * v * dx
elif dim == 3:
    f = LinearForm(fes)
    f += (-((2 - 12 * x + 12 * x ** 2) * (2 * y - 6 * y ** 2 + 4 * y ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
                        + (x ** 2 - 2 * x ** 3 + x ** 4) * (-12 + 24 * y) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
                        + (x ** 2 - 2 * x ** 3 + x ** 4) * (-12 + 24 * z) * (2 * y - 6 * y ** 2 + 4 * y ** 3))
            + (1 - 2 * x) * (1 - y) * (1 - z)
            ) * v[0] * dx
    f += (-((2 - 12 * y + 12 * y ** 2) * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
                    + (y ** 2 - 2 * y ** 3 + y ** 4) * (-12 + 24 * x) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
                    + (y ** 2 - 2 * y ** 3 + y ** 4) * (-12 + 24 * z) * (2 * x - 6 * x ** 2 + 4 * x ** 3))
            - x * (1 - x) * (1 - z)
            ) * v[1] * dx
    f += (2 * (
                (2 - 12 * z + 12 * z ** 2) * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * y - 6 * y ** 2 + 4 * y ** 3)
                + (z ** 2 - 2 * z ** 3 + z ** 4) * (-12 + 24 * x) * (2 * y - 6 * y ** 2 + 4 * y ** 3)
                + (z ** 2 - 2 * z ** 3 + z ** 4) * (-12 + 24 * y) * (2 * x - 6 * x ** 2 + 4 * x ** 3))
            - x * (1 - x) * (1 - y)
            ) * v[2] * dx
    f += c_low * u_exact * v * dx


# L2 projection from fes0 to fes, more like Embedding
mixmass = BilinearForm(trialspace=fes0, testspace=fes)
# tangential part
mixmass += tang(uhat0) * tang(vhat) * dx(element_boundary=True)
# normal part
mixmass += (u0*n) * (v*n) * dx(element_boundary=True)

fesMass = BilinearForm(fes)
fesMass += tang(uhat) * tang(vhat) * dx(element_boundary=True)
fesMass += (u*n) * (v*n) * dx(element_boundary=True)

# ========= END of HIGHER ORDER Hdiv-HDG SCHEME TO BE SOLVED ==========












# ========= START of HP-MG for HIGHER ORDER Hdiv-HDG ==========
def SolveBVP_CR(level, drawResult=False):
    t0 = timeit.time()
    fes.Update(); fes0.Update(); fes_cr.Update()
    gfu.Update()
    a.Assemble(); a0.Assemble(); a_cr.Assemble()
    f.Assemble()
    mixmass.Assemble(); mixmass_cr.Assemble(); 
    fesMass.Assemble(); fesMass0.Assemble()
    # ========== CR MG update
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
        fesM0_inv = fesMass0.mat.CreateSmoother(fes0.FreeDofs(True))
    elif dim == 3:
        fesM_inv = fesMass.mat.CreateBlockSmoother(FacetPatchBlocks(mesh, fes))
        fesM0_inv = fesMass0.mat.CreateBlockSmoother(FacetPatchBlocks(mesh, fes0))
    
    E = fesM_inv @ mixmass.mat # E: fes0 => fes
    ET = mixmass.mat.T @ fesM_inv
    vBlocks = VertexPatchBlocks(mesh, fes)
    R = SymmetricGS(a.mat.CreateBlockSmoother(vBlocks)) # vertex block GS for p-MG smoothing
    
    E0 = fesM0_inv @ mixmass_cr.mat # E0: fes_cr => fes0
    ET0 = mixmass_cr.mat.T @ fesM0_inv

    # inv_cr = a_cr.mat.Inverse(fes_cr.FreeDofs(), inverse='sparsecholesky')
    inv_cr = MG_cr

    # pre0 = E0 @ inv_cr @ ET0
    # pre = R + E @ pre0 @ ET

    pre = R + E @ a0.mat.Inverse(fes0.FreeDofs(True)) @ ET
    # pre = E @ a0.mat.Inverse(fes0.FreeDofs(True)) @ ET
    # pre = R
    t1 = timeit.time()




    # ========== HDG STATIC CONDENSATION and SOLVED
    # dirichlet BC
    f.vec.data -= a.mat * gfu.vec
    f.vec.data += a.harmonic_extension_trans * f.vec
    # inv_fes = a.mat.Inverse(fes.FreeDofs(True))
    inv_fes = CGSolver(a.mat, pre, printrates=False, tol=1e-8, maxiter=500)
    gfu.vec.data += inv_fes * f.vec
    # gfu.vec.data += E @ a0.mat.Inverse(fes0.FreeDofs(True)) @ ET * f.vec
    # gfu.vec.data += E @ E0 @ a_cr.mat.Inverse(fes_cr.FreeDofs()) @ ET0 @ ET * f.vec
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







def ecrCheck(level, meshRate=2, prev_uErr=0, prev_LErr=0):
    print(f'LEVEL: {level}, ALL DOFS: {fes.ndof}, GLOBAL DOFS: {W.ndof + M.ndof}')
    L2_uErr = sqrt(Integrate((uh - u_exact) * (uh - u_exact), mesh))
    L2_LErr = sqrt(Integrate(InnerProduct((Lh - L_exact), (Lh - L_exact)), mesh))
    L2_divErr = sqrt(Integrate(div(uh) * div(uh), mesh))
    if level > 0:
        u_rate = log(prev_uErr / L2_uErr) / log(meshRate)
        print(f"uh L2-error: {L2_uErr:.3E}, uh conv rate: {u_rate:.2E}")
        L_rate = log(prev_LErr / L2_LErr) / log(meshRate)
        print(f"Lh L2-error: {L2_LErr:.3E}, Lh conv rate: {L_rate:.2E}")
    else:
        print(f"uh L2-error: {L2_uErr:.3E}")
        print(f"Lh L2-error: {L2_LErr:.3E}")
    print(f'uh divErr: {L2_divErr:.1E}')
    print('==============================')
    return (L2_uErr, L2_LErr)


SolveBVP = SolveBVP_CR
print(f'===== DIM: {mesh.dim}, ORDER:{ order} c_low: {c_low}, eps: {epsilon} =====')
SolveBVP(0, drawResult)
prev_uErr, prev_LErr = ecrCheck(0)
level = 1
while True:
    with TaskManager():
        # uniform refinement used, meshRate=2 in ecrCheck
        mesh.ngmesh.Refine()
        # exit if total global dofs exceed a0 tol
        M.Update(); W.Update()
        if (sum(W.FreeDofs(True)) + sum(W.FreeDofs(True)) > maxdofs):
            print(sum(W.FreeDofs(True)) + sum(W.FreeDofs(True)))
            break
        SolveBVP(level, drawResult)
        prev_uErr, prev_LErr = ecrCheck(level, prev_uErr=prev_uErr, prev_LErr=prev_LErr)
        level += 1

        