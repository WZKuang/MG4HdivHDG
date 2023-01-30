# mixed Hdiv-HDG for 2D benchmark Oseen problms, hp-MG preconditioned GMRES solver
# upper triangle block preconditioner as in Benzi, Olshanskii, SJSC(2006)
# Lid-driven cavity problem

from ngsolve import *
import time as timeit
from ngsolve.krylovspace import CGSolver, GMResSolver
# from ngsolve.la import EigenValues_Preconditioner
# geometry
from ngsolve.meshes import MakeStructured2DMesh
from netgen.geom2d import unit_square
# from netgen.csg import unit_cube
# customized functions
from prol import meshTopology, FacetProlongationTrig2 #, FacetProlongationTet2
from auxPyFiles.myMG import MultiGrid
# from auxPyFiles.mySmoother import VertexPatchBlocks, EdgePatchBlocks, FacetBlocks, SymmetricGS
from auxPyFiles.myASP import MultiASP


def HdivHDGOseenSaddle(dim=2, nu=1e-3, wind=CF((1, 0)), c_low=0, 
                 order=0, nMGSmooth=2, aspSm=4, drawResult=False, maxLevel=7):
                 
    maxdofs = 5e7

    # epsilon = 1/nu * 5e-2
    epsilon = 1e-5
    iniN = 1

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



    # ========= START of P0 Hdiv-HDG for the primal variable ==========
    # ========= mixed-HidvHDG scheme =========
    V0 = MatrixValued(L2(mesh, order=0), mesh.dim, False)
    W0 = HDiv(mesh, order=0, RT=True, dirichlet=dirichBDs)
    M0 = TangentialFacetFESpace(mesh, order=0, dirichlet=dirichBDs)

    primalFE = V0 * W0 * M0
    (L0, u0, uhat0),  (G0, v0, vhat0) = primalFE.TnT()

    # gradient by row
    gradv0, gradu0 = Grad(v0), Grad(u0)

    # bilinear form of SIP-HdivHDG
    a0 = BilinearForm(primalFE, symmetric=False, condense=True)
    # volume term
    a0 += (1/epsilon * div(u0) * div(v0)) * dx
    a0 += (nu * InnerProduct(L0, G0) + c_low * u0 * v0
        -nu * InnerProduct(gradu0, G0) + nu * InnerProduct(L0, gradv0)) * dx
    a0 += (nu * tang(u0-uhat0) * tang(G0*n) - nu * tang(L0*n) * tang(v0-vhat0))*dx(element_boundary=True)
    # === convection part
    uhatup0 = IfPos(wind*n, tang(u0), tang(uhat0))    
    a0 += -gradv0 * wind * u0 *dx(bonus_intorder=3)
    a0 += wind*n * (uhatup0 * tang(v0-vhat0)) * dx(element_boundary=True, bonus_intorder=3)   

    # ========= START of HIGHER ORDER Hdiv-HDG, saddle point problem ==========
    # ========= mixed-HidvHDG scheme =========
    V = MatrixValued(L2(mesh, order=order), mesh.dim, False)
    W = HDiv(mesh, order=order, RT=True, dirichlet=dirichBDs)
    M = TangentialFacetFESpace(mesh, order=order, dirichlet=dirichBDs)
    Q = L2(mesh, order=order, lowest_order_wb=True)

    fes = V * W * M * Q
    (L, u, uhat, p),  (G, v, vhat, q) = fes.TnT()

    gfu = GridFunction (fes)
    Lh, uh, uhath, ph = gfu.components

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
    # === saddle point (pressure) part
    a += (-p * div(v) - q * div(u)) * dx
    
    # TODO: WHY 1e-16*p*q needed??????????
    a += 1e-16 * p * q * dx

    f = LinearForm(fes)




    # ============== TRANSFER OPERATORS
    # === L2 projection from CR to primalFE, for prolongation operator
    V_cr1 = FESpace('nonconforming', mesh, dirichlet=dirichBDs)
    V_cr2 = FESpace('nonconforming', mesh, dirichlet=dirichBDs)
    fes_cr = FESpace([V_cr1, V_cr2])
    (ux_cr, uy_cr), (vx_cr, vy_cr) = fes_cr.TnT()
    u_cr, v_cr = CF((ux_cr, uy_cr)), CF((vx_cr, vy_cr))

    mixmass0 = BilinearForm(trialspace=fes_cr, testspace=primalFE)
    # tangential part
    mixmass0 += tang(u_cr) * tang(vhat0) * dx(element_boundary=True)
    # normal part
    mixmass0 += (u_cr*n) * (v0*n) * dx(element_boundary=True)

    fesMass0 = BilinearForm(primalFE)
    fesMass0 += tang(uhat0) * tang(vhat0) * dx(element_boundary=True)
    fesMass0 += (u0*n) * (v0*n) * dx(element_boundary=True)
    
    # === L2 projection from primalFE to CR, for prolongation operator
    ir = IntegrationRule(SEGM, 1)
    mixmass_cr = BilinearForm(trialspace=primalFE, testspace=fes_cr)
    mixmass_cr += tang(uhat0) * tang(v_cr) * dx(element_boundary=True, intrules={SEGM: ir})
    mixmass_cr += u0*n * v_cr*n * dx(element_boundary=True, intrules={SEGM: ir})

    fesMass_cr = BilinearForm(fes_cr)
    fesMass_cr += tang(u_cr) * tang(v_cr) * dx(element_boundary=True, intrules={SEGM: ir})
    fesMass_cr += u_cr*n * v_cr*n * dx(element_boundary=True, intrules={SEGM: ir})

    # === L2 projection from primalFE to fes
    mixmass = BilinearForm(trialspace=primalFE, testspace=fes)
    # tangential part
    mixmass += tang(uhat0) * tang(vhat) * dx(element_boundary=True)
    # normal part
    mixmass += (u0*n) * (v*n) * dx(element_boundary=True)

    fesMass = BilinearForm(fes)
    fesMass += tang(uhat) * tang(vhat) * dx(element_boundary=True)
    fesMass += (u*n) * (v*n) * dx(element_boundary=True)

    # ========== Schur complement approximation
    Q0 = L2(mesh, order=0)
    p0, q0 = Q0.TnT()
    b = BilinearForm(trialspace=Q0, testspace=fes)
    b += - p0 * div(v) * dx
    p0Mass = BilinearForm(Q0)
    p0Mass += p0 * q0 * dx
    # === L2 projection from Q0 to fes
    mixmass_p = BilinearForm(trialspace=Q0, testspace=fes)
    mixmass_p += p0 * q * dx
    fesMass_p = BilinearForm(fes)
    fesMass_p += p * q * dx


    # ========= END of HIGHER ORDER Hdiv-HDG SCHEME TO BE SOLVED ==========

    et = meshTopology(mesh, mesh.dim)
    et.Update() 
    prol = FacetProlongationTrig2(mesh, et) # for CR element

    


    # ========= START of HP-MG for HIGHER ORDER Hdiv-HDG ==========
    def SolveBVP_CR(level, drawResult=False, MG0=None):
        with TaskManager():
        # with TaskManager(pajetrace=10**8):
            t0 = timeit.time()
            fes.Update(); primalFE.Update(); fes_cr.Update()
            Q0.Update()
            gfu.Update()
            # wind_proj.Update(); wind_proj.Set(wind)
            a.Assemble()
            a0.Assemble()
            b.Assemble()
            # rhs linear form
            f.Assemble()

            fes_pdofs = BitArray(fes.ndof)
            fes_pdofs[:] = 0
            fes_pdofs[-Q.ndof:] = Q.FreeDofs(True)

            fes_globalUdofs = BitArray(fes.ndof)
            fes_globalUdofs[:] = 0
            fes_globalUdofs[V.ndof: V.ndof+W.ndof] = W.FreeDofs(True)
            fes_globalUdofs[V.ndof+W.ndof: V.ndof+W.ndof+M.ndof] = M.FreeDofs(True)

            fesMass.Assemble(); mixmass.Assemble()
            fesMass0.Assemble(); mixmass0.Assemble()
            fesMass_cr.Assemble(); mixmass_cr.Assemble()
            fesMass_p.Assemble(); mixmass_p.Assemble()

            fesM_inv = fesMass.mat.CreateSmoother(fes_globalUdofs)
            fesM0_inv = fesMass0.mat.CreateSmoother(primalFE.FreeDofs(True))
            fesMass_cr_inv = fesMass_cr.mat.CreateSmoother(fes_cr.FreeDofs())
            fesMp_inv = fesMass_p.mat.CreateSmoother(fes_pdofs)
            

            E = fesM_inv @ mixmass.mat # E: primalFE => fes
            ET = mixmass.mat.T @ fesM_inv

            cr2fes0 = fesM0_inv @ mixmass0.mat # cr2fes0: fes_cr => primalFE
            fes02cr = fesMass_cr_inv @ mixmass_cr.mat

            Ep = fesMp_inv @ mixmass_p.mat # Q0 -> fes
            p0Mass.Assemble()
            p0M_inv = p0Mass.mat.CreateSmoother(Q0.FreeDofs())
            prc_Schur = Ep @ (-nu * p0M_inv - 1/epsilon * p0M_inv) @ Ep.T


            # ========== MG for primal variable initialize and update
            if level == 0:
                MG0 = MultiGrid(a0.mat, prol, nc=V_cr1.ndof,
                    coarsedofs=primalFE.FreeDofs(True), w1=0.8,
                    nsmooth=nMGSmooth, sm='gs',#'jc', 
                    he=True, dim=mesh.dim, wcycle=False, var=True,
                    mProject = [cr2fes0, fes02cr])
            else:
                if MG0 is None: 
                    print("WRONG MG0 INPUT"); exit(1)
                et.Update()
                pp = [primalFE.FreeDofs(True)]
                pp.append(V_cr1.ndof)
                innerDofs = BitArray(primalFE.ndof)
                innerDofs[:] = 0
                inner = prol.GetInnerDofs(level)
                innerDofs[V0.ndof: V0.ndof+W0.ndof] = inner
                innerDofs[V0.ndof+W0.ndof: ] = inner
                # he_prol
                pp.append(a0.mat.Inverse(innerDofs, inverse="umfpack"))
                pp.append([cr2fes0, fes02cr])
                # block smoothers, if no hacker made to ngsolve source file,
                # use the following line instead
                # pp.append(VertexPatchBlocks(mesh, fes_cr))
                
                fes0Blocks = primalFE.CreateSmoothBlocks(vertex=True, globalDofs=True)
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
            # fesBlocks = fes.CreateSmoothBlocks(vertex=True, globalDofs=True)
            
            # primal_inv = a0.mat.Inverse(primalFE.FreeDofs(True), inverse='umfpack')
            primal_inv = MG0
            # lowOrderSolver = E @ inv0 @ ET

            # === block GS smoother as multi-ASP smoother, not good with nu/1/epsilon ratio
            # aspSmoother = a.mat.CreateBlockSmoother(fesBlocks)
            # pre = MultiASP(a.mat, fes.FreeDofs(True), lowOrderSolver, 
            #             smoother=aspSmoother,
            #             nSm=0 if order==0 else aspSm)
            # === GMRES (pre=block jacobi) as multi-ASP smoother, as in Farrell etc., SJSC(2019)
            # === NOTE: seems to not work for higher-order
            # aspSmoother = GMResSolver(a.mat, a.mat.CreateBlockSmoother(fesBlocks), printrates=False, maxiter=6)
            # pre = MultiASP(a.mat, fes.FreeDofs(True), lowOrderSolver, 
            #             smoother=aspSmoother,
            #             nSm=0 if order==0 else aspSm, smType="jc", damp=1)

            # R = SymmetricGS(a.mat.CreateBlockSmoother(vblocks)) # block GS for p-MG smoothing
            # pre = R + E @ inv_cr @ ET # additive ASP
            t1 = timeit.time()

            pre = (E @ primal_inv @ ET + Projector(fes_pdofs, True)) \
                 @ (Projector(fes_globalUdofs, True) - Projector(fes_pdofs, True)
                    + b.mat @ Ep.T) \
                 @ (Projector(fes_globalUdofs, True) - prc_Schur)
            # pre = E @ primal_inv @ ET - Ep @ p0M_inv @ Ep.T
            # pre = a.mat.Inverse(fes.FreeDofs(True), inverse='umfpack')


            # ========== HDG STATIC CONDENSATION and SOLVED by AL uzawa
            # homo dirichlet BC
            gfu.vec.data[:] = 0
            uhath.Set(utop, definedon=mesh.Boundaries("top"))
            rhs = f.vec.CreateVector()
            rhs.data = f.vec - a.mat * gfu.vec
            # static condensation of HDG
            rhs.data += a.harmonic_extension_trans * rhs

            inv_fes = GMResSolver(a.mat, pre, printrates=False, tol=1e-8, maxiter=500)
            # inv_fes = GMResSolver(a.mat, a.mat.Inverse(fes.FreeDofs(True), inverse='umfpack'), 
                                #   printrates=False, tol=1e-8, maxiter=500)
            # inv_fes = a.mat.Inverse(fes.FreeDofs(True), inverse='umfpack')
            gfu.vec.data += inv_fes * rhs
            gfu.vec.data += a.harmonic_extension * gfu.vec
            gfu.vec.data += a.inner_solve * rhs
            it = inv_fes.iterations
            # it = 1


            # ========= PRINT RESULTS
            t2 = timeit.time()
            # lams = EigenValues_Preconditioner(mat=a.mat, pre=pre)
            print(f"==> Assemble & Update: {t1-t0:.2e}, Solve: {t2-t1:.2e}")
            print(f"==> AVG MG IT: {it}, MG_smooth: {nMGSmooth}, ASP_smooth: {aspSm}")
                  #, MAX LAM: {max(lams):.2e}, MIN LAM: {min(lams):.2e}, COND: {max(lams)/min(lams):.2E}")
            L2_divErr = sqrt(Integrate(div(uh) * div(uh), mesh))
            print(f'==> uh divErr: {L2_divErr:.1E}')
            if drawResult:
                import netgen.gui
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
        globalDofs = sum(W.FreeDofs(True)) + sum(M.FreeDofs(True))
        if globalDofs > maxdofs or level > maxLevel:
            print(f'# totalDofs: {fes.ndof} # global DOFS {globalDofs}')
            break
        print(f'===== LEVEL {level} =====')
        print(f'# totalDofs: {fes.ndof} # global DOFS {globalDofs}')
        MG0 = SolveBVP_CR(level, drawResult, MG0)
        print(f'======================')
        level += 1

    # ========= END of OUTPUT ==========



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
# nuList = [1e-2, 5e-3, 1e-3, 5e-4] # visocity
nuList = [1e-4] # visocity

for nu in nuList:
    HdivHDGOseenSaddle(dim=dim, nu=nu, wind=wind, c_low=c_low, 
                order=order, nMGSmooth=nMGSmooth, aspSm=aspSm, drawResult=False, maxLevel=7)
    print("================================================================")
    print("================================================================")
    print("================================================================")