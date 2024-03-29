# mixed Hdiv-HDG for 2D benchmark Oseen problms, hp-MG preconditioned GMRES solver
# Augmented Lagrangian Uzawa iteration for outer iteration
# Lid-driven cavity problem

from ngsolve import *
import time as timeit
from ngsolve.krylovspace import CGSolver, GMResSolver, GMRes
# from ngsolve.la import EigenValues_Preconditioner
# geometry
from ngsolve.meshes import MakeStructured2DMesh
from netgen.geom2d import unit_square
# from netgen.csg import unit_cube
# customized functions
from prol import meshTopology, FacetProlongationTrig2 #, FacetProlongationTet2
# from auxPyFiles.mySmoother import VertexPatchBlocks, EdgePatchBlocks, FacetBlocks, SymmetricGS
from auxPyFiles.mySolvers import MultiASP, MultiGrid


def HdivHDGOseen(dim=2, iniN=4, nu=1e-3, wind=CF((1, 0)), c_low=0, initialSol=None, epsilon=1e-6,
                 order=0, nMGSmooth=2, aspSm=4, drawResult=False, maxLevel=7, printIter=True):
                 
    maxdofs = 5e7
    if drawResult:
        import netgen.gui
    ''' TODO: ratio between nu/1/epsilon??????
        NOTE: when c_low == 0:
              1. When epsilon is large (O(10)*nu), not necessarily need harmonic extension
              2. The advection-dominated advection-diffusion term alone (epsilon->infty)
                 is not well-preconditioned by conventional CR MG (averaging) with/without
                 harmonic extension.
              3. p-MG can explode when nu -> 0.
                 
    ''' 
    uzawaIt = 1
    # ========== START of MESH ==========
    dirichBDs = ".*"
    mesh = MakeStructured2DMesh(quads=False, nx=iniN, ny=iniN)
    # mesh = Mesh(unit_square.GenerateMesh(maxh=1/iniN))
    # top side dirichlet bd
    utop = CoefficientFunction((4*x*(1-x),0))
    # ========== END of MESH ==========



    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size
    def tang(v):
            return v - (v*n)*n



    # ========= START of P0 Hdiv-HDG SCHEME TO BE SOLVED ==========
    # ========= mixed-HidvHDG scheme =========
    V0 = MatrixValued(L2(mesh, order=0), mesh.dim, False)
    W0 = HDiv(mesh, order=0, RT=True, dirichlet=dirichBDs)
    M0 = TangentialFacetFESpace(mesh, order=0, dirichlet=dirichBDs)

    fes0 = V0 * W0 * M0 
    (L0, u0,uhat0),  (G0, v0, vhat0) = fes0.TnT()

    # gradient by row
    gradv0, gradu0 = Grad(v0), Grad(u0)

    # bilinear form of SIP-HdivHDG
    a0 = BilinearForm(fes0, symmetric=False, condense=True)
    # volume term
    a0 += (1/epsilon * div(u0) * div(v0)) * dx
    a0 += (nu * InnerProduct(L0, G0) + c_low * u0 * v0
        -nu * InnerProduct(gradu0, G0) + nu * InnerProduct(L0, gradv0)) * dx
    a0 += (nu * tang(u0-uhat0) * tang(G0*n) - nu * tang(L0*n) * tang(v0-vhat0))*dx(element_boundary=True)
    # === convection part
    uhatup0 = IfPos(wind*n, tang(u0), tang(uhat0))    
    a0 += -gradv0 * wind * u0 *dx(bonus_intorder=3)
    a0 += wind*n * (uhatup0 * tang(v0-vhat0)) * dx(element_boundary=True, bonus_intorder=3)   


    # ========= START of HIGHER ORDER Hdiv-HDG SCHEME TO BE SOLVED ==========
    # ========= mixed-HidvHDG scheme =========
    V = MatrixValued(L2(mesh, order=order), mesh.dim, False)
    W = HDiv(mesh, order=order, RT=True, dirichlet=dirichBDs)
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
    a += (nu * InnerProduct(L, G) + c_low * u * v
        -nu * InnerProduct(gradu, G) + nu * InnerProduct(L, gradv)) * dx
    a += (nu * tang(u-uhat) * tang(G*n) - nu * tang(L*n) * tang(v-vhat))*dx(element_boundary=True)
    # === convection part
    uhatup = IfPos(wind*n, tang(u), tang(uhat))    
    a += -gradv * wind * u * dx(bonus_intorder=3)
    a += wind*n * (uhatup * tang(v-vhat)) * dx(element_boundary=True, bonus_intorder=3)   
    
    f = LinearForm(fes)


    # ============== TRANSFER OPERATORS
    # === L2 projection from CR to fes0, for prolongation operator
    V_cr1 = FESpace('nonconforming', mesh, dirichlet=dirichBDs)
    V_cr2 = FESpace('nonconforming', mesh, dirichlet=dirichBDs)
    fes_cr = FESpace([V_cr1, V_cr2])
    (ux_cr, uy_cr), (vx_cr, vy_cr) = fes_cr.TnT()
    u_cr, v_cr = CF((ux_cr, uy_cr)), CF((vx_cr, vy_cr))

    mixmass0 = BilinearForm(trialspace=fes_cr, testspace=fes0)
    # tangential part
    mixmass0 += tang(u_cr) * tang(vhat0) * dx(element_boundary=True)
    # normal part
    mixmass0 += (u_cr*n) * (v0*n) * dx(element_boundary=True)

    fesMass0 = BilinearForm(fes0)
    fesMass0 += tang(uhat0) * tang(vhat0) * dx(element_boundary=True)
    fesMass0 += (u0*n) * (v0*n) * dx(element_boundary=True)
    
    # === L2 projection from fes0 to CR, for prolongation operator
    ir = IntegrationRule(SEGM, 1)
    mixmass_cr = BilinearForm(trialspace=fes0, testspace=fes_cr)
    mixmass_cr += tang(uhat0) * tang(v_cr) * dx(element_boundary=True, intrules={SEGM: ir})
    mixmass_cr += u0*n * v_cr*n * dx(element_boundary=True, intrules={SEGM: ir})

    fesMass_cr = BilinearForm(fes_cr)
    fesMass_cr += tang(u_cr) * tang(v_cr) * dx(element_boundary=True, intrules={SEGM: ir})
    fesMass_cr += u_cr*n * v_cr*n * dx(element_boundary=True, intrules={SEGM: ir})

    # === L2 projection from fes0 to fes
    mixmass = BilinearForm(trialspace=fes0, testspace=fes)
    # tangential part
    mixmass += tang(uhat0) * tang(vhat) * dx(element_boundary=True)
    # normal part
    mixmass += (u0*n) * (v*n) * dx(element_boundary=True)

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

    et = meshTopology(mesh, mesh.dim)
    et.Update() 
    prol = FacetProlongationTrig2(mesh, et) # for CR element

    
    gfuList = []
    # ========= START of HP-MG for HIGHER ORDER Hdiv-HDG ==========
    def SolveBVP_CR(level, drawResult=False, MG0=None):
        with TaskManager():
        # with TaskManager(pajetrace=10**8):
            t0 = timeit.time()
            fes.Update(); fes0.Update(); fes_cr.Update()
            gfu.Update()
            # wind_proj.Update(); wind_proj.Set(wind)
            a.Assemble(); a0.Assemble()
            # rhs linear form
            f.Assemble()

            fesMass.Assemble(); mixmass.Assemble()
            fesMass0.Assemble(); mixmass0.Assemble()
            fesMass_cr.Assemble(); mixmass_cr.Assemble()

            fesM_inv = fesMass.mat.CreateSmoother(fes.FreeDofs(True))
            fesM0_inv = fesMass0.mat.CreateSmoother(fes0.FreeDofs(True))
            fesMass_cr_inv = fesMass_cr.mat.CreateSmoother(fes_cr.FreeDofs())

            E = fesM_inv @ mixmass.mat # E: fes0 => fes
            ET = mixmass.mat.T @ fesM_inv

            cr2fes0 = fesM0_inv @ mixmass0.mat # cr2fes0: fes_cr => fes0
            fes02cr = fesMass_cr_inv @ mixmass_cr.mat


            # ========== MG initialize and update
            if level == 0:
                MG0 = MultiGrid(a0.mat, prol, nc=V_cr1.ndof,
                    coarsedofs=fes0.FreeDofs(True), w1=0.8,
                    nsmooth=nMGSmooth, sm='gs',#'jc', 
                    he=True, dim=mesh.dim, wcycle=False, var=True,
                    mProject = [cr2fes0, fes02cr])
            else:
                if MG0 is None: 
                    print("WRONG MG0 INPUT"); exit(1)
                et.Update()
                pp = [fes0.FreeDofs(True)]
                pp.append(V_cr1.ndof)
                pdofs = BitArray(fes0.ndof)
                pdofs[:] = 0
                inner = prol.GetInnerDofs(level)
                pdofs[V0.ndof: V0.ndof+W0.ndof] = inner
                pdofs[V0.ndof+W0.ndof: ] = inner
                # he_prol
                pp.append(a0.mat.Inverse(pdofs, inverse="umfpack"))
                pp.append([cr2fes0, fes02cr])
                # block smoothers, if no hacker made to ngsolve source file,
                # use the following line instead
                # pp.append(VertexPatchBlocks(mesh, fes_cr))
                
                fes0Blocks = fes0.CreateSmoothBlocks(vertex=True, globalDofs=True)
                # === block GS as MG smoother, not good with nu/1/epsilon ratio
                pp.append(fes0Blocks)
                # === GMRES (pre=block jacobi) as multi-ASP smoother, as in Farrell etc., SJSC(2019)
                # === NOTE: needs to change MG0 "sm='jc'"
                # === NOTE: not good results
                # pp.append(GMResSolver(a0.mat, a0.mat.CreateBlockSmoother(fes0Blocks), printrates=False, maxiter=6))
                MG0.Update(a0.mat, pp)



            # ========== PRECONDITIONER SETUP START

            # block smoothers, if no hacker made to ngsolve source file,
            # use the following line instead
            # blocks = VertexPatchBlocks(mesh, fes) if mesh.dim == 2 else EdgePatchBlocks(mesh, fes)
            fesBlocks = fes.CreateSmoothBlocks(vertex=True, globalDofs=True)
            
            # inv0 = a0.mat.Inverse(fes0.FreeDofs(True))
            inv0 = MG0
            lowOrderSolver = E @ inv0 @ ET
            
            # === block GS smoother as multi-ASP smoother, not good with nu/1/epsilon ratio
            aspSmoother = a.mat.CreateBlockSmoother(fesBlocks)
            pre = MultiASP(a.mat, fes.FreeDofs(True), lowOrderSolver, 
                        smoother=aspSmoother,
                        nSm=0 if order==0 else aspSm)
            # === GMRES (pre=block jacobi) as multi-ASP smoother, as in Farrell etc., SJSC(2019)
            # === NOTE: seems to not work for higher-order
            # aspSmoother = GMResSolver(a.mat, a.mat.CreateBlockSmoother(fesBlocks), printrates=False, maxiter=6)
            # pre = MultiASP(a.mat, fes.FreeDofs(True), lowOrderSolver, 
            #             smoother=aspSmoother,
            #             nSm=0 if order==0 else aspSm, smType="jc", damp=1)

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
            # homo dirichlet BC
            gfu.vec.data[:] = 0
            uhath.Set(utop, definedon=mesh.Boundaries("top"))
            solTmp = gfu.vec.CreateVector()
            for it_u in range(uzawaIt):
                solTmp.data = Projector(fes.FreeDofs(True), True) * solTmp
                rhs = f.vec.CreateVector()
                rhs.data = f.vec - a.mat * gfu.vec
                rhs.data += -b.mat * p_prev.vec
                # update L and u
                rhs.data += a.harmonic_extension_trans * rhs
                inv_fes = GMResSolver(a.mat, pre, printrates=False, 
                                      tol=1e-8, maxiter=500)
                init = True if initialSol is None else False
                inv_fes.Solve(rhs=rhs, sol=solTmp, initialize=init if it_u == 0 else False)
                it += inv_fes.iterations
                # inv_fes = GMResSolver(a.mat, pre, printrates=False, tol=1e-8, maxiter=500)

                solTmp.data += a.harmonic_extension * solTmp
                solTmp.data += a.inner_solve * rhs
                # update pressure
                p_prev.vec.data += 1/epsilon * (pMass_inv @ b.mat.T * solTmp.data)
            gfu.vec.data += Projector(fes.FreeDofs(True), True) * solTmp

            # ========= PRINT RESULTS
            t2 = timeit.time()
            it //= uzawaIt
            # lams = EigenValues_Preconditioner(mat=a.mat, pre=pre)
            globalSol = gfu.vec.CreateVector()
            globalSol.data = gfu.vec.data
            gfuList.append(globalSol.data)

            if printIter:
                print(f"==> Assemble & Update: {t1-t0:.2e}, Solve: {t2-t1:.2e}")
                print(f"==> AVG MG IT: {it}, Uzawa It: {uzawaIt}, MG_smooth: {nMGSmooth}, ASP_smooth: {aspSm}")
                    #, MAX LAM: {max(lams):.2e}, MIN LAM: {min(lams):.2e}, COND: {max(lams)/min(lams):.2E}")
                L2_divErr = sqrt(Integrate(div(uh) * div(uh), mesh))
                print(f'==> uh divErr: {L2_divErr:.1E}')
            if drawResult:
                Draw(Norm(uh), mesh, 'sol')
                input('continue?')
            return MG0

    # ========= END of HP-MG for HIGHER ORDER Hdiv-HDG ==========




    # ========= START of OUTPUT ==========
    print(f'===== DIM: {mesh.dim}, ORDER:{order}, nu: {nu:.1e} c_low: {c_low}, eps: {epsilon:.1e} =====')
    MG0 = SolveBVP_CR(0, drawResult)
    level = 1
    while True:
        # uniform refinement used
        mesh.ngmesh.Refine()
            
        # exit if total global dofs exceed a0 tol
        M.Update(); W.Update(); fes.Update()
        globalDofs = sum(fes.FreeDofs(True))
        totalDofs = sum(fes.FreeDofs())
        if globalDofs > maxdofs or level > maxLevel:
            print(f'# totalDofs: {totalDofs} # global DOFS {globalDofs}')
            break
        print(f'===== LEVEL {level} =====')
        print(f'# totalDofs: {totalDofs} # global DOFS {globalDofs}')
        MG0 = SolveBVP_CR(level, drawResult, MG0)
        print(f'======================')
        level += 1
    # ========= END of OUTPUT ==========
    # return results to be used as initial guess
    return gfuList



import sys
if len(sys.argv) < 6:
    print('not enough input args: dim + c_low + nMGSmooth + nASPSmooth + order'); exit(1)
    
dim = int(sys.argv[1])
c_low = int(sys.argv[2])
nMGSmooth = int(sys.argv[3])
aspSm = int(sys.argv[4])
order = int(sys.argv[5])

if dim != 2:
    print('WRONG DIMENSION! 2D ONLY!!!'); exit(1)

# wind = CF((1, 0))
# wind = CF((0, 0))
wind = CF((4*(2*y-1)*(1-x)*x, -4*(2*x-1)*(1-y)*y))
initialSol = False
maxLevel = 6
iniN = 2
# nuList = [1e-2, 5e-3, 1e-3, 5e-4] # visocity
nuList = [1e-3] # visocity

for nu in nuList:
    initialGuess = None
    epsilon = 1/nu * 1e-6
    if initialSol:
    # stokes first, to get initial guess at each level
        initialGuess = HdivHDGOseen(dim=dim, iniN=iniN, nu=nu*5, wind=wind, c_low=c_low, epsilon=epsilon,
                    order=order, nMGSmooth=nMGSmooth, aspSm=aspSm, drawResult=False, maxLevel=maxLevel, printIter=False)
        print("=============INITIAL STOKES SOLs READY==================")
        print("==========================================================")
    print("=================OSEEN MG SOLs START HERE===================")
    # oseenEps = 1/nu * 5e-2
    # oseenEps = 1e-6
    HdivHDGOseen(dim=dim, iniN=iniN, nu=nu, wind=wind, c_low=c_low, initialSol=initialGuess, epsilon=epsilon,  
                order=order, nMGSmooth=nMGSmooth, aspSm=aspSm, drawResult=False, maxLevel=maxLevel)
    print("================================================================")
    print("================================================================")
    print("================================================================")