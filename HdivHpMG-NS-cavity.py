# mixed Hdiv-HDG for the Navier-Stokes, hp-MG preconditioned GMRES solver
# Augmented Lagrangian Uzawa iteration for outer iteration
# Lid-driven cavity problem

from ngsolve import *
import time as timeit
from ngsolve.krylovspace import GMResSolver, GMRes
# geometry
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh
# from netgen.geom2d import unit_square
# from netgen.csg import unit_cube
# customized functions
from prol import meshTopology, FacetProlongationTrig2, FacetProlongationTet2
from auxPyFiles.myMG import MultiGrid
# from auxPyFiles.mySmoother import VertexPatchBlocks, EdgePatchBlocks, FacetBlocks, SymmetricGS
from auxPyFiles.myASP import MultiASP
import math

# For each linearized Oseen problem, needs nested MG preconditioner 
# for the lowest order case.
def OseenOperators(dim:int=2, iniN:int=4, nu:float=1e-3, wind=None, 
                   c_low:int=0, epsilon:float=1e-6, pseudo_timeinv:float=0.0,
                   order:int=0, nMGSmooth:int=2, aspSm:int=4, maxLevel:int=7):
    ''' TODO: Take out assembling parts that repeat in each picard iteration!!!
    '''
    # ========== START of MESH ==========
    dirichBDs = ".*"
    if dim==2:
        mesh = MakeStructured2DMesh(quads=False, nx=iniN, ny=iniN)
        vertexBlock = True
    elif dim==3:
        mesh = MakeStructured3DMesh(hexes=False, nx=iniN, ny=iniN, nz=iniN)
        vertexBlock = False # edge-patched blocks in 3D to save memory
    # ========== END of MESH ==========

    if wind is None:
        wind = CF((0, 0)) if dim == 2 else CF((0, 0, 0))
    n = specialcf.normal(mesh.dim)
    def tang(v):
        return v - (v*n)*n


    # ========= START of HIGHER ORDER Hdiv-HDG SCHEME TO BE SOLVED by p-MG==========
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
    a += -gradv * wind * u * dx(bonus_intorder=order)
    a += wind*n * (uhatup * tang(v-vhat)) * dx(element_boundary=True, bonus_intorder=order)   
    
    
    f = LinearForm(fes)
    # === pseudo time marching for relaxation
    if pseudo_timeinv > 1e-16:
        a += pseudo_timeinv * u * v * dx
        f += pseudo_timeinv * wind * v * dx



    # ========= START of P0 Hdiv-HDG SCHEME TO BE SOLVED by h-MG==========
    # ========= mixed-HidvHDG scheme =========
    V0 = MatrixValued(L2(mesh, order=0), mesh.dim, False)
    W0 = HDiv(mesh, order=0, RT=True, dirichlet=dirichBDs) if mesh.dim == 2 \
         else HDiv(mesh, order=0, RT=False, dirichlet=dirichBDs)
    M0 = TangentialFacetFESpace(mesh, order=0, dirichlet=dirichBDs)
    

    # Interpolate the wind
    ''' NOTE: The Set function of RT0 GridFunction in 3D could result in NaN!!!
    '''
    windW = HDiv(mesh, order=max(1, order), RT=True)
    wind_h0 = GridFunction(windW)
    wind_h0.Set(wind)

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
    uhatup0 = IfPos(wind_h0*n, tang(u0), tang(uhat0))  
    a0 += -gradv0 * wind_h0 * u0 *dx(bonus_intorder=order)
    a0 += wind_h0*n * uhatup0 * (tang(v0)-tang(vhat0)) * dx(element_boundary=True, bonus_intorder=order)   
    # === pseudo time marching for relaxation
    if pseudo_timeinv > 1e-16:
        a0 += pseudo_timeinv * u0 * v0 * dx




    # ============== TRANSFER OPERATORS
    # === L2 projection from CR to fes0, for prolongation operator
    V_cr = FESpace('nonconforming', mesh, dirichlet=dirichBDs)
    if dim == 2:
        fes_cr = V_cr * V_cr
        (ux_cr, uy_cr), (vx_cr, vy_cr) = fes_cr.TnT()
        u_cr, v_cr = CF((ux_cr, uy_cr)), CF((vx_cr, vy_cr))
    else:
        fes_cr = V_cr * V_cr * V_cr
        (ux_cr, uy_cr, uz_cr), (vx_cr, vy_cr, vz_cr) = fes_cr.TnT()
        u_cr, v_cr = CF((ux_cr, uy_cr, uz_cr)), CF((vx_cr, vy_cr, vz_cr))

    mixmass0 = BilinearForm(trialspace=fes_cr, testspace=fes0)
    # tangential part
    mixmass0 += tang(u_cr) * tang(vhat0) * dx(element_boundary=True)
    # normal part
    mixmass0 += (u_cr*n) * (v0*n) * dx(element_boundary=True)

    fesMass0 = BilinearForm(fes0)
    fesMass0 += tang(uhat0) * tang(vhat0) * dx(element_boundary=True)
    fesMass0 += (u0*n) * (v0*n) * dx(element_boundary=True)
    
    # === L2 projection from fes0 to CR, for prolongation operator
    if dim == 2:
        ir = IntegrationRule(SEGM, 1)
        mixmass_cr = BilinearForm(trialspace=fes0, testspace=fes_cr)
        mixmass_cr += tang(uhat0) * tang(v_cr) * dx(element_boundary=True, intrules={SEGM: ir})
        mixmass_cr += u0*n * v_cr*n * dx(element_boundary=True, intrules={SEGM: ir})

        fesMass_cr = BilinearForm(fes_cr)
        fesMass_cr += tang(u_cr) * tang(v_cr) * dx(element_boundary=True, intrules={SEGM: ir})
        fesMass_cr += u_cr*n * v_cr*n * dx(element_boundary=True, intrules={SEGM: ir})
    else:
        ir = IntegrationRule(TRIG, 1)
        mixmass_cr = BilinearForm(trialspace=fes0, testspace=fes_cr)
        mixmass_cr += tang(uhat0) * tang(v_cr) * dx(element_boundary=True, intrules={TRIG: ir})
        mixmass_cr += u0*n * v_cr*n * dx(element_boundary=True, intrules={TRIG: ir})

        fesMass_cr = BilinearForm(fes_cr)
        fesMass_cr += tang(u_cr) * tang(v_cr) * dx(element_boundary=True, intrules={TRIG: ir})
        fesMass_cr += u_cr*n * v_cr*n * dx(element_boundary=True, intrules={TRIG: ir})

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

    # ========= prolongation operators for CR
    et = meshTopology(mesh, mesh.dim)
    et.Update() 
    prol = FacetProlongationTrig2(mesh, et) if dim==2 \
           else FacetProlongationTet2(mesh, et) # for CR element

    MG0 = None
    # ========= START of Operators Assembling ==========
    with TaskManager():
    # with TaskManager(pajetrace=10**8):
        for level in range(maxLevel+1):
            fes0.Update(); fes_cr.Update()
            windW.Update(); wind_h0.Update(); wind_h0.Set(wind)
            a0.Assemble()

            fesMass0.Assemble(); mixmass0.Assemble()
            fesMass_cr.Assemble(); mixmass_cr.Assemble()
            if dim == 2:
                fesM0_inv = fesMass0.mat.CreateSmoother(fes0.FreeDofs(True))
                fesMass_cr_inv = fesMass_cr.mat.CreateSmoother(fes_cr.FreeDofs())
            else:
                fesM0_inv = fesMass0.mat.CreateBlockSmoother(fes0.CreateFacetBlocks(globalDofs=True))
                fesMass_cr_inv = fesMass_cr.mat.CreateBlockSmoother(fes_cr.CreateFacetBlocks(globalDofs=False))
            cr2fes0 = fesM0_inv @ mixmass0.mat # cr2fes0: fes_cr => fes0
            fes02cr = fesMass_cr_inv @ mixmass_cr.mat

            # ========== MG initialize and update
            if level == 0:
                MG0 = MultiGrid(a0.mat, prol, nc=V_cr.ndof,
                    coarsedofs=fes0.FreeDofs(True), w1=0.8,
                    nsmooth=nMGSmooth, sm='gs',#'jc', 
                    he=True, dim=mesh.dim, wcycle=False, var=True,
                    mProject = [cr2fes0, fes02cr])
            else:
                et.Update()
                pp = [fes0.FreeDofs(True)]
                pp.append(V_cr.ndof)
                pdofs = BitArray(fes0.ndof)
                pdofs[:] = 0
                innerFaces = prol.GetInnerDofs(level)
                # Hdiv Inner dofs
                pdofs[V0.ndof: V0.ndof+W0.ndof] = innerFaces
                ''' NOTE: TangentialFacetFESpace in 3D has 2 dofs on each facet,
                          different from 2D cases!!!!
                '''
                # Hcurl Inner dofs
                if dim == 2:
                    pdofs[V0.ndof+W0.ndof: ] = innerFaces
                else:
                    for i in range(mesh.nface):
                        if innerFaces[i]:
                            pdofs.Set(V0.ndof+W0.ndof+ 2*i)
                            pdofs.Set(V0.ndof+W0.ndof+ 2*i +1)
                # he_prol
                pp.append(a0.mat.Inverse(pdofs, inverse="umfpack"))
                pp.append([cr2fes0, fes02cr])
                # block smoothers, if no hacker made to ngsolve source file,
                # use the following line instead
                # pp.append(VertexPatchBlocks(mesh, fes_cr)) 
                fes0Blocks = fes0.CreateSmoothBlocks(vertex=vertexBlock, globalDofs=True)
                # === block GS as MG smoother, not good with nu/1/epsilon ratio
                pp.append(fes0Blocks)
                MG0.Update(a0.mat, pp)
            
            if level < maxLevel:
                # mesh.ngmesh.Refine()
                if mesh.dim == 2:
                    mesh.ngmesh.Refine()
                else:
                    mesh.Refine(onlyonce = True)

            
        
        # ==== Update of high order Hdiv-HDG
        fes.Update()
        a.Assemble(); f.Assemble()
        
        fesMass.Assemble(); mixmass.Assemble()
        if dim == 2:
            fesM_inv = fesMass.mat.CreateSmoother(fes.FreeDofs(True))
        else:
            fesM_inv = fesMass.mat.CreateBlockSmoother(fes.CreateFacetBlocks(globalDofs=True))
        E = fesM_inv @ mixmass.mat # E: fes0 => fes
        ET = mixmass.mat.T @ fesM_inv

        # inv0 = a0.mat.Inverse(fes0.FreeDofs(True))
        inv0 = MG0
        lowOrderSolver = E @ inv0 @ ET
        # ========== Multi-ASP operator
        # block smoothers, if no hacker made to ngsolve source file,
        # use the following line instead
        # blocks = VertexPatchBlocks(mesh, fes) if mesh.dim == 2 else EdgePatchBlocks(mesh, fes)
        fesBlocks = fes.CreateSmoothBlocks(vertex=vertexBlock, globalDofs=True)   
        # === block GS smoother as multi-ASP smoother, not good with nu/1/epsilon ratio
        aspSmoother = a.mat.CreateBlockSmoother(fesBlocks)
        pre_ASP = MultiASP(a.mat, fes.FreeDofs(True), lowOrderSolver, 
                            smoother=aspSmoother,
                            nSm=0 if order==0 else aspSm)
        # ========== Secondary Operators for Uzawa
        Q.Update()
        b.Assemble() #b += - p * div(v) * dx
        pMass.Assemble()
        # p mass diagonal in both 2D and 3D cases
        pMass_inv= pMass.mat.CreateSmoother(Q.FreeDofs())
    
    # ========= END of Operators Assembling ==========
    # NOTE: MeshTopology needed to be returned together with
    #       MG0, otherwise segmentation error
    return mesh, et, fes, a, f, pre_ASP, b, pMass_inv



    


def nsSolver(dim:int=2, iniN:int=4, nu:float=1e-3, div_penalty:float=1e6,
             order:int=0, nMGSmooth:int=2, aspSm:int=4, maxLevel:int=7,
             pseudo_timeinv:float=0.0, rtol:float=1e-8, drawResult:bool=False):
    
    epsilon = 1/nu/div_penalty
    uzawaIt = 8//int(math.log10(div_penalty))
    # if drawResult:
    #     import netgen.gui
    print("#########################################################################")
    print(f"#  DIM: {dim}, order: {order}, uzawaIt: {uzawaIt}"), 
    print(f"#  h_corase: 1/{iniN*2}, h_fine: 1/{2*iniN*2**maxLevel}, maxLevel: {maxLevel}")
    print(f"#  MG nSm: {nMGSmooth}, Multi-ASP nSm: {aspSm}")
    print(f"#  viscosity: {nu:.1e}, c_div: {div_penalty:.1e}, epsilon: {epsilon:.1e}")
    print("#########################################################################")
    
    
    def oneOseenSolver(aMesh, aFes, aA, aAPrc, aB, aPm_inv, aF, bdSol, prevSol=None, init:bool=True):
        # HDG STATIC CONDENSATION and SOLVED by AL uzawa
        Q = L2(aMesh, order=order)
        p_prev = GridFunction(Q)
        p_prev.vec.data[:] = 0
        solTmp = bdSol.vec.CreateVector()
        ''' NOTE: GMResSolver calculate the delta based on the current provided (global) sol.
                  It will deal with rhs with rhs -= a.mat * solTmp,
                  So in this static condensation process, solTmp should not contain BD data,
                  otherwise cause wrong sol near BDs.
        '''
        it = 0
        with TaskManager():
            if not init:
                solTmp.data = Projector(aFes.FreeDofs(True), True) * prevSol.vec
            for it_u in range(uzawaIt):
                solTmp.data = Projector(aFes.FreeDofs(True), True) * solTmp
                # ====== static condensation and solve
                rhs = aF.vec.CreateVector()
                rhs.data = aF.vec - aA.mat * bdSol.vec
                rhs.data += -aB.mat * p_prev.vec
                # # static condensation
                rhs.data += aA.harmonic_extension_trans * rhs
                inv_fes = GMResSolver(aA.mat, aAPrc, printrates=False, 
                                      tol=1e-8, atol=1e-10, maxiter=100)
                # use prev uzawa sol as initial guess
                inv_fes.Solve(rhs=rhs, sol=solTmp, initialize=init if it_u==0 else False)
                it += inv_fes.iterations

                solTmp.data += aA.harmonic_extension * solTmp
                solTmp.data += aA.inner_solve * rhs
                # update pressure
                p_prev.vec.data += 1/epsilon * (aPm_inv @ aB.mat.T * solTmp.data)
        
        it //= uzawaIt
        return bdSol.vec.data+Projector(fes.FreeDofs(True), True)*solTmp.data, it
        


    # ===================== START OF NS SOLVING ====================
    with TaskManager():
        # ==== Upper BD
        if dim==2:
            utop = CoefficientFunction((4*x*(1-x),0))
        elif dim==3:
            utop = CoefficientFunction((16*x*(1-x)*y*(1-y),0,0))

        # ====== 1. Stokes solver to get initial
        t0 = timeit.time()
        mesh, et, fes, a, f, pre_ASP, b, pMass_inv = \
                OseenOperators(dim=dim, iniN=iniN, nu=nu, wind=None, 
                                c_low=0, epsilon=epsilon,order=order, 
                                nMGSmooth=nMGSmooth, aspSm=aspSm, maxLevel=maxLevel)
        # mesh, et, fes, b, pMass_inv => the same during the NS solving process
        t1 = timeit.time()
        gfu = GridFunction(fes)
        Lh, uh, uhath = gfu.components
        gfu_bd = GridFunction(fes)
        _, _, uhath_bd = gfu_bd.components
        # homo dirichlet BC
        uhath.Set(utop, definedon=mesh.Boundaries("top"))
        uhath_bd.Set(utop, definedon=mesh.Boundaries("top"))
        
        gfu.vec.data, _ = oneOseenSolver(mesh, fes, a, pre_ASP, b, pMass_inv, f, gfu_bd)
        t2 = timeit.time()
        uNorm0 = sqrt(Integrate(uh**2, mesh))
        print(f"Stokes initial finished Assem {t1-t0:.1e} Cal {t2-t1:.1e}",
              f"uh init norm: {uNorm0:.1e} atol: {uNorm0*rtol:.1e}")
        print("#########################################################################")
        print("#############################  PICARD IT  ###############################")
        # if drawResult:
        #     Draw(Norm(uh), mesh, "velNorm")
        #     input('init Stokes')


        # ====== 2. Picard Iteration
        uh_prev = uh.vec.CreateVector()
        atol = max(uNorm0 * rtol, 1e-10) # set lower bound for absolute tol
        diffNorm = uNorm0
        avg_picardIt = 0
        picardCnt = 1
        MAXCNT = 80
        while diffNorm > atol:
            if picardCnt > MAXCNT:
                print("MAX Picard Iter reached!!! Not converged!!!")
                break
            t0 = timeit.time()
            mesh, et, fes, a, f, pre_ASP, b, pMass_inv = \
                OseenOperators(dim=dim, iniN=iniN, nu=nu, wind=uh,
                                c_low=0, epsilon=epsilon,order=order, 
                                nMGSmooth=nMGSmooth, aspSm=aspSm, maxLevel=maxLevel,
                                pseudo_timeinv=pseudo_timeinv)
            t1 = timeit.time()
            uh_prev.data = uh.vec
            gfu.vec.data, it = oneOseenSolver(mesh, fes, a, pre_ASP, b, pMass_inv, f, 
                                              gfu_bd, gfu, init=False) 
            t2 = timeit.time()
            uh.vec.data -= uh_prev
            diffNorm = sqrt(Integrate(uh**2, mesh))
            uh.vec.data += uh_prev
            L2_divErr = sqrt(Integrate(div(uh)**2, mesh))
            print(f"#{picardCnt:>2}, pseudo_timeinv: {pseudo_timeinv:.1e}, GMRes_it: {it:>2},", 
                  f"diff_norm = {diffNorm:.1e}, uh divErr: {L2_divErr:.1E},",
                  f"t_assem: {t1-t0:.1e}, t_cal: {t2-t1:.1e}")
            
            ''' TODO: Adaptivity of pseudo_timeinv!!!
            '''
            avg_picardIt = avg_picardIt * (picardCnt-1) / picardCnt + it / picardCnt
            picardCnt += 1
        
        if drawResult:
            import netgen.gui
            Draw(Norm(uh), mesh, "velNorm")
            input("picard")  
        print(f"Picard Avg It: {avg_picardIt:.1f}")
        print("###########################  PICARD IT END  #############################")
        print("#########################################################################")

        
        
        ''' TODO!!!!!
        '''
        # ====== 3. Newton Iteration
        

                

if __name__ == '__main__':
    # nuList = [1e-2, 1e-3, 5e-4]
    nuList = [1e-2]
    orderList = [0]
    for aNu in nuList:
        for aOrder in orderList:
            for maxLevel in [5]:
                nsSolver(dim=3, iniN=4, nu=aNu, div_penalty=1e6,
                        order=aOrder, nMGSmooth=2, aspSm=2, maxLevel=maxLevel, 
                        pseudo_timeinv=0.0, rtol=1e-8, drawResult=False)
