from ngsolve.la import BaseMatrix
from ngsolve import BitArray, CreateVVector, Norm, sqrt, Projector, Vector, \
        Matrix
import numpy as np

# DEFINE AUXILIARY FUNCTIONS
def VertexPatchBlocks(mesh, fes):
    blocks = []
    freedofs = fes.FreeDofs()
    for v in mesh.vertices:
        vdofs = set()
        if mesh.dim==2:
          for f in mesh[v].edges:              
            if v in mesh[f].vertices:
                vdofs |= set(d for d in fes.GetDofNrs(f) if freedofs[d])
        else:
          for f in mesh[v].faces:
            if v in mesh[f].vertices:
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

def FacetBlocks(mesh, fes):
    blocks = []
    freedofs = BitArray(fes.ndof)
    freedofs[:] = 0
    freedofs[fes.components[0].ndof:] = fes.components[1].FreeDofs()
    for v in mesh.facets:
        vdofs = set(d for d in fes.GetDofNrs(v) if freedofs[d])
        blocks.append(vdofs)
    return blocks


class SymmetricGS(BaseMatrix):
    def __init__ (self, smoother):
        super(SymmetricGS, self).__init__()
        self.smoother = smoother
    def Mult (self, x, y):
        y[:] = 0.0
        self.smoother.Smooth(y, x)
        self.smoother.SmoothBack(y,x)
    def Height (self):
        return self.smoother.height
    def Width (self):
        return self.smoother.height

# multigrid with variable smoothing
class MultiGrid(BaseMatrix):
    def __init__ (self, mat, prol, coarsedofs, nc, w1=0.9, sm="gs", var=False,
            nsmooth=1, wcycle=False, he=False, dim = 1, js=False, he0 = ()):
        super(MultiGrid, self).__init__()
        self.mats = [mat]
        self.smoothers = [ () ]
        self.activeDofs = [ coarsedofs ]
        self.he_prol = [ he0 ]
        self.he = he
        self.w1 = w1
        self.js = js
        self.sm = sm
        self.var = var
        self.wcycle = wcycle
        self.nsmooth = nsmooth
        self.prol = prol
        self.dim = dim
        self.nactive = [nc]
        self.nlevels = 0
        #self.invcoarseproblem = mat.Inverse(coarsedofs, inverse="sparsecholesky")
        self.invcoarseproblem = mat.Inverse(coarsedofs, inverse="umfpack")

    def Update(self, mat, pp):
        self.mats.append (mat)
        dofs = pp[-1]
        if type(dofs)==list: # block smoother
            self.smoothers.append(mat.CreateBlockSmoother( dofs ) )
        elif type(dofs)==BitArray: # point smoother
            self.smoothers.append(mat.CreateSmoother( dofs ) )
        else: # already a smoother
            self.smoothers.append(dofs)
        self.nlevels = len(self.mats)
        # hacker: only works for point smoother
        self.activeDofs.append(pp[0])
        self.nactive.append(pp[1])
        if self.he == True:
            self.he_prol.append(pp[2])
    
    def Height(self):
        return self.mats[-1].height
    def Width(self):
        return self.mats[-1].width
    def Mult(self, b, x):
        self.MGM(len(self.mats)-1, b, x)

    def MGM(self, level, b, rho):
        # initialize to zero
        rho[:] = 0
        if self.var: # variable V cycle
            ms = self.nsmooth*2**(self.nlevels-1-level) # number of smoothers
        else:
            ms = self.nsmooth
        # set-zero V dofs in b??? TODO
        if level > 0:
            nc = self.mats[level-1].height
            nmc = self.nactive[level-1] # number of active dofs in M
            nmf = self.nactive[level] # number of active dofs in M
            prolType = type(nmf)
            tmp = b.CreateVector()
            tmp0 = b.CreateVector()
            
            # coarse grid vector
            tmpC = CreateVVector(nc) 
            rhoC = CreateVVector(nc) 
            rhoC0 = CreateVVector(nc) 
            tmpC[:] = 0
            rhoC[:] = 0
            # pre-smoothing: m-steps GS/JAC
            if self.sm =="gs": 
                for i in range(ms):
                    self.smoothers[level].Smooth(rho, b)
                tmp.data = b - self.mats[level] * rho
            else:
                tmp.data = b
                for i in range(ms):
                    rho.data += self.w1*self.smoothers[level]*tmp # jacobi
                    tmp.data = b - self.mats[level] * rho
            
            if self.wcycle>-1:
               # harmonic extension part: update residual???
               if self.he:
                   if prolType == list: # pressure transform needed
                       print('WARNING: MIXED MG STILL BUGGY. '
                             'CORRECT PRESSURE HARMONIC EXTENSION NEEDED.')
                       uHe_prol = self.he_prol[level][0]
                       pHe_transf_f = self.he_prol[level][1]
                       pHe_transf_c = self.he_prol[level - 1][1]
                   else:
                       uHe_prol = self.he_prol[level]
                   # here tmp is the residual after smoothing
                   # tmp0 is the harmonic extension of tmp (of primal variable u)
                   tmp0.data = uHe_prol * tmp
                   tmp.data -= self.mats[level] * tmp0

               # FIXME: Projector project out inactive dofs on fine level
               tmp.data = Projector(self.activeDofs[level], True) * tmp
               if prolType==int: # only one prolongation
                   # coarse grid correction
                   for i in range(self.dim):
                       self.prol.Restrict(level, tmp[i*nmf:(i+1)*nmf])
                   #  convert tmp --> to coarse grid vector
                   for i in range(self.dim):
                       tmpC[i*nmc:(i+1)*nmc].data = tmp[i*nmf:i*nmf+nmc]
               else: # prolType==list
                   # coarse grid correction (mixed form)
                   for i in range(self.dim):
                       self.prol[0].Restrict(level, tmp[i*nmf[0]:(i+1)*nmf[0]])
                   self.prol[1].Restrict(level, tmp[self.dim*nmf[0]:
                       self.dim*nmf[0]+nmf[1]])
                   #  convert tmp --> to coarse grid vector
                   for i in range(self.dim):
                       tmpC[i*nmc[0]:(i+1)*nmc[0]].data = tmp[i*nmf[0]:
                               i*nmf[0]+nmc[0]]
                   tmpC[self.dim*nmc[0]:self.dim*nmc[0]+nmc[1]].data = \
                           tmp[self.dim*nmf[0]:self.dim*nmf[0]+nmc[1]]

               self.MGM(level-1, tmpC , rhoC)

               # W-cycle
               if level>1 and self.wcycle==1:
                  # W cycle (FIXME)
                  tmpC.data -= self.mats[level-1]*rhoC
                  self.MGM(level-1, tmpC , rhoC0)
                  rhoC.data += rhoC0 #update

               if prolType==int: # only one prolongation
                   #  convert rhoC --> to fine grid vector
                   for i in range(self.dim):
                       tmp[i*nmf:i*nmf+nmc].data = rhoC[i*nmc:(i+1)*nmc]
                   
                   for i in range(self.dim):
                       self.prol.Prolongate(level, tmp[i*nmf:(i+1)*nmf])
               else:
                   #  convert tmp --> to coarse grid vector
                   for i in range(self.dim):
                       tmp[i*nmf[0]:i*nmf[0]+nmc[0]].data = \
                               rhoC[i*nmc[0]:(i+1)*nmc[0]]
                   tmp[self.dim*nmf[0]:self.dim*nmf[0]+nmc[1]].data = \
                           rhoC[self.dim*nmc[0]:self.dim*nmc[0]+nmc[1]]
                   for i in range(self.dim):
                       self.prol[0].Prolongate(level, tmp[i*nmf[0]:(i+1)*nmf[0]])
                   self.prol[1].Prolongate(level, tmp[self.dim*nmf[0]:
                       self.dim*nmf[0]+nmf[1]])

               
               # here tmp is the naively prolongated correction of rho
               # update rho
               #rho.data += tmp
               # FIXME: Projector project out inactive dofs on fine level
               tmp.data = Projector(self.activeDofs[level], True) * tmp
               
               # harmonic extension part
               if self.he:
                   tmp0.data = self.mats[level] * tmp
                   tmp.data -= uHe_prol * tmp0
                   # he has been performed on u_l
                   # print('new prol working')

               rho.data += tmp 
        
            # post-smoothing: m-steps GS/damped Jac
            if self.sm =="gs":
                for i in range(ms):
                    self.smoothers[level].SmoothBack(rho, b)
            else:
                for i in range(ms):
                    tmp.data = b - self.mats[level] * rho            
                    rho.data += self.w1*self.smoothers[level]*tmp # jacobi
        else:
            # p1 problem on the coarsest mesh
            rho.data = self.invcoarseproblem * b

