# mixed Hdiv-HDG for the Navier-Stokes, hp-MG preconditioned GMRES solver
# Augmented Lagrangian Uzawa iteration for outer iteration
# Lid-driven cavity problem

from ngsolve import *
import time as timeit
from ngsolve.krylovspace import GMResSolver, GMRes, CGSolver
# from ngsolve.la import EigenValues_Preconditioner
# geometry
from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh
# from netgen.geom2d import unit_square
# from netgen.csg import unit_cube
# customized functions
from prol import meshTopology, FacetProlongationTrig2 #, FacetProlongationTet2
from auxPyFiles.myMG import MultiGrid
# from auxPyFiles.mySmoother import VertexPatchBlocks, EdgePatchBlocks, FacetBlocks, SymmetricGS
from auxPyFiles.myASP import MultiASP



# For each linearized Oseen problem, needs nested MG preconditioner 
# for the lowest order case.
def OseenOperators(dim:int=2, iniN:int=4, nu:float=1e-3, wind=CF((0, 0)), 
                   c_low:int=0, epsilon:float=1e-6,
                   order:int=0, nMGSmooth:int=2, aspSm:int=4, maxLevel:int=7):
    # ========== START of MESH ==========
    dirichBDs = ".*"
    if dim==2:
        mesh = MakeStructured2DMesh(quads=False, nx=iniN, ny=iniN)
    elif dim==3:
        mesh = MakeStructured3DMesh(hexes=False, nx=iniN, ny=iniN, nz=iniN)
    # ========== END of MESH ==========


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
    a += -gradv * wind * u * dx(bonus_intorder=3)
    a += wind*n * (uhatup * tang(v-vhat)) * dx(element_boundary=True, bonus_intorder=3)   
    
    f = LinearForm(fes)



    # ========= START of P0 Hdiv-HDG SCHEME TO BE SOLVED by h-MG==========
    # ========= mixed-HidvHDG scheme =========
    V0 = MatrixValued(L2(mesh, order=0), mesh.dim, False)
    W0 = HDiv(mesh, order=0, RT=True, dirichlet=dirichBDs) if mesh.dim == 2 \
         else HDiv(mesh, order=0, dirichlet=dirichBDs)
    M0 = TangentialFacetFESpace(mesh, order=0, dirichlet=dirichBDs)
    
    # Interpolate the wind
    wind_h0 = GridFunction(W0)
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
    a0 += wind_h0*n * (uhatup0 * tang(v0-vhat0)) * dx(element_boundary=True, bonus_intorder=order)   




    # ============== TRANSFER OPERATORS
    # === L2 projection from CR to fes0, for prolongation operator
    V_cr = FESpace('nonconforming', mesh, dirichlet=dirichBDs)
    fes_cr = V_cr * V_cr
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

    # ========= prolongation operators for CR
    et = meshTopology(mesh, mesh.dim)
    et.Update() 
    prol = FacetProlongationTrig2(mesh, et) # for CR element

    t0 = timeit.time()
    MG0 = None
    # ========= START of Operators Assembling ==========
    with TaskManager():
    # with TaskManager(pajetrace=10**8):
        for level in range(maxLevel+1):
            fes0.Update(); fes_cr.Update()
            wind_h0.Update(); wind_h0.Set(wind)
            a0.Assemble()

            fesMass0.Assemble(); mixmass0.Assemble()
            fesMass_cr.Assemble(); mixmass_cr.Assemble()

            fesM0_inv = fesMass0.mat.CreateSmoother(fes0.FreeDofs(True))
            fesMass_cr_inv = fesMass_cr.mat.CreateSmoother(fes_cr.FreeDofs())

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
                MG0.Update(a0.mat, pp)
            
            if level < maxLevel:
                mesh.ngmesh.Refine()

            
        
        # ==== Update of high order Hdiv-HDG
        fes.Update()
        a.Assemble(); f.Assemble()
        
        fesMass.Assemble(); mixmass.Assemble()
        fesM_inv = fesMass.mat.CreateSmoother(fes.FreeDofs(True))
        E = fesM_inv @ mixmass.mat # E: fes0 => fes
        ET = mixmass.mat.T @ fesM_inv

        # inv0 = a0.mat.Inverse(fes0.FreeDofs(True))
        inv0 = MG0
        lowOrderSolver = E @ inv0 @ ET
        # ========== Multi-ASP operator
        # block smoothers, if no hacker made to ngsolve source file,
        # use the following line instead
        # blocks = VertexPatchBlocks(mesh, fes) if mesh.dim == 2 else EdgePatchBlocks(mesh, fes)
        fesBlocks = fes.CreateSmoothBlocks(vertex=True, globalDofs=True)   
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
    t1 = timeit.time()
    print(f"===> Oseen Operator Finished: {t1-t0:.2e}")

    return mesh, et, fes, a, f, pre_ASP, b, pMass_inv



    
def nsSolver(dim:int=2, iniN:int=4, nu:float=1e-3, div_penalty:float=1e6,
             order:int=0, nMGSmooth:int=2, aspSm:int=4, maxLevel:int=7,
             drawResult:bool=False):
    
    epsilon = 1/nu/div_penalty
    # ==== Upper BD
    if dim==2:
        utop = CoefficientFunction((4*x*(1-x),0))
    elif dim==3:
        utop = CoefficientFunction((16*x*(1-x)*y*(1-y),0,0))

    if drawResult:
        import netgen.gui
    print("#########################################################################")
    print(f"DIM: {dim}, order: {order}")
    print(f"h_corase: 1/{iniN*2}, h_fine: 1/{epsilon:.1e}, maxLevel: {maxLevel}")
    print(f"viscosity: {nu:.1e}, c_div: {div_penalty:.1e}, epsilon: {1/nu/div_penalty:.1e}")
    print("#########################################################################")
    
    
    wind = CF((4*(2*y-1)*(1-x)*x, -4*(2*x-1)*(1-y)*y))
    # wind = CF((0, 0))
    mesh, et, fes, a, f, pre_ASP, b, pMass_inv = \
                OseenOperators(dim=dim, iniN=iniN, nu=nu, wind=wind, 
                                c_low=0, epsilon=epsilon,order=order, 
                                nMGSmooth=nMGSmooth, aspSm=aspSm, maxLevel=maxLevel)

    # ========== HDG STATIC CONDENSATION and SOLVED by AL uzawa
    Q = L2(mesh, order=order)
    p_prev = GridFunction(Q)
    p_prev.vec.data[:] = 0
    # homo dirichlet BC
    gfu = GridFunction(fes)
    Lh, uh, uhath = gfu.components
    gfu.vec.data[:] = 0
    uhath.Set(utop, definedon=mesh.Boundaries("top"))

    t0 = timeit.time()
    # ====== static condensation and solve
    rhs = f.vec.CreateVector()
    rhs.data = f.vec - a.mat * gfu.vec
    rhs.data += -b.mat * p_prev.vec
    # # static condensation
    rhs.data += a.harmonic_extension_trans * rhs
    inv_fes = GMResSolver(a.mat, pre_ASP, printrates=False, tol=1e-8, maxiter=200)
    gfu.vec.data += inv_fes * rhs
    it = inv_fes.iterations

    gfu.vec.data += a.harmonic_extension * gfu.vec
    gfu.vec.data += a.inner_solve * rhs
    # update pressure
    p_prev.vec.data += 1/epsilon * (pMass_inv @ b.mat.T * gfu.vec.data)
    
    t1 = timeit.time()

    L2_divErr = sqrt(Integrate(div(uh) * div(uh), mesh))
    print(f"===> one step Oseen solved: {t1-t0:.2e}, IT: {it}, uh divErr: {L2_divErr:.1E}")
    if drawResult:
        Draw(Norm(uh), mesh, "velNorm")
        input('continue?')


nsSolver(dim=2, iniN=1, nu=1e-3, div_penalty=1e6,
         order=0, nMGSmooth=2, aspSm=2, maxLevel=6, drawResult=False)
