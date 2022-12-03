# Head files for blocks and GS smoother
# if no hacker made to ngsolve source file, use functions here

from ngsolve.la import BaseMatrix

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