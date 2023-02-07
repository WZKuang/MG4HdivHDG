# Head files for blocks and GS smoother
# if no hacker made to ngsolve source file, use functions here
from ngsolve import *
from ngsolve.la import BaseMatrix


def mixedHDGblockGenerator(dim:int=2, mesh=None, dirichBDs=None, iniN:int=4, bisec3D:bool=True,
                           order:int=0, maxLevel:int=7):
    from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh
    # Generate pre-assembled blocks to save time, especially in 3D cases
    # result[0] -> fes0 facet/edge-patched blocks for smoothing
    # result[1] -> fes0 facet blocks for mass mat inverse
    # result[2] -> fes_cr facet blocks for mass mat inverse
    # result[3] -> fes facet/edge-patched blocks for smoothing
    # result[4] -> fes facet blocks for mass mat inverse
    # ========== START of MESH ==========
    # unit_square/cube by default
    if mesh is None:
        dirichBDs = ".*"
        if dim==2:
            mesh = MakeStructured2DMesh(quads=False, nx=iniN, ny=iniN)
        elif dim==3:
            mesh = MakeStructured3DMesh(hexes=False, nx=iniN, ny=iniN, nz=iniN)
    vertexBlock = True if dim == 2 else False # edge-patched blocks in 3D to save memory
    # ========== END of MESH ==========

    V = MatrixValued(L2(mesh, order=order), mesh.dim, False)
    if mesh.dim == 2:
        W = HDiv(mesh, order=order, RT=True, dirichlet=dirichBDs)
    elif mesh.dim == 3:
        W = HDiv(mesh, order=order, 
                 RT=True if order>=1 else False, dirichlet=dirichBDs) # inconsistent option when lowest order
    M = TangentialFacetFESpace(mesh, order=order, dirichlet=dirichBDs)
    fes = V * W * M 



    V0 = MatrixValued(L2(mesh, order=0), mesh.dim, False)
    W0 = HDiv(mesh, order=0, RT=True, dirichlet=dirichBDs) if mesh.dim == 2 \
         else HDiv(mesh, order=0, RT=False, dirichlet=dirichBDs)
    M0 = TangentialFacetFESpace(mesh, order=0, dirichlet=dirichBDs)
    fes0 = V0 * W0 * M0 



    V_cr = FESpace('nonconforming', mesh, dirichlet=dirichBDs)
    if dim == 2:
        fes_cr = V_cr * V_cr
    else:
        fes_cr = V_cr * V_cr * V_cr
    
    fes0patchBlocks = []
    fes0facetBlocks = []
    fesCrFacetBlocks = []
    # ========= START of Operators Assembling ==========
    with TaskManager():
        for level in range(maxLevel+1):
            fes0.Update(); fes_cr.Update()

            if dim == 3:
                fes0facetBlocks.append(fes0.CreateFacetBlocks(globalDofs=True))
                fesCrFacetBlocks.append(fes_cr.CreateFacetBlocks(globalDofs=False))

            # ========== MG initialize and update
            if level == 0:
                fes0patchBlocks.append(None)
            else:
                fes0patchBlocks.append(fes0.CreateSmoothBlocks(vertex=vertexBlock, globalDofs=True))
            
            if level < maxLevel:
                # mesh.ngmesh.Refine()
                if mesh.dim == 3 and bisec3D:
                    mesh.Refine(onlyonce = True)
                else:
                    mesh.ngmesh.Refine()         
        
        result = []
        result.append(fes0patchBlocks)
        result.append(fes0facetBlocks)
        result.append(fesCrFacetBlocks)
        # ==== Update of high order Hdiv-HDG
        fes.Update()
        result.append(fes.CreateSmoothBlocks(vertex=vertexBlock, globalDofs=True)  ) 
        if dim == 3:
            result.append(fes.CreateFacetBlocks(globalDofs=True))
        else:
            result.append(None)
    
    return result


# ========================================
# Other blocks to be used without hacking the NGSolve source code
# could be much slower
# ========================================
def VertexPatchBlocks(mesh, fes):
    blocks = []
    freedofs = fes.FreeDofs(True)
    for v in mesh.vertices:
        vdofs = set()
        for f in mesh[v].elements:
            vdofs |= set(d for d in fes.GetDofNrs(f) if freedofs[d])
        blocks.append(vdofs)
    return blocks

def EdgePatchBlocks(mesh, fes):
    blocks = []
    freedofs = fes.FreeDofs(True)
    for v in mesh.edges:
        vdofs = set()
        for el in mesh[v].elements:
            vdofs |= set(d for d in fes.GetDofNrs(el) if freedofs[d])
        blocks.append(vdofs)
    return blocks

def FacetPatchBlocks(mesh, fes):
    blocks = []
    freedofs = fes.FreeDofs(True)
    for f in mesh.faces:
        fdofs = set()
        for el in mesh[f].elements:
            fdofs |= set(d for d in fes.GetDofNrs(el) if freedofs[d])
        blocks.append(fdofs)
    return blocks

def FacetBlocks(mesh, fes):
    blocks = []
    for f in mesh.faces:
        blocks.append(set(d for d in fes.GetDofNrs(f) if fes.FreeDofs(True)[d]))
    return blocks

class SymmetricGS(BaseMatrix):
    def __init__ (self, smoother):
        super(SymmetricGS, self).__init__()
        self.smoother = smoother
    def Mult (self, x, y):
        y[:] = 0.0
        self.smoother.Smooth(y, x)
        self.smoother.SmoothBack(y, x)
    def Height (self):
        return self.smoother.height
    def Width (self):
        return self.smoother.height