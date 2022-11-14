# Head file for two level multiplicative ASP preconditioner
# by Wenzheng Kuang, 11/12/2022

from ngsolve.la import BaseMatrix
from ngsolve import *
import numpy as np

# multigrid with variable smoothing
class MultiASP(BaseMatrix):
    def __init__ (self, mat, activeDOFs, coarseSolver, smoother, nSm=1):
        super().__init__()
        self.mat = mat
        self.activeDOFs = activeDOFs
        self.coarseSol = coarseSolver
        self.sm = smoother # assumed GS
        self.nSm = nSm

    def Height(self):
        return self.mat.height
    def Width(self):
        return self.mat.width

    def Mult(self, b, rho): # b => rhs; rho => sol
        # initialize to zero
        rho[:] = 0
        # set-zero V dofs in b??? TODO
        residual = b.CreateVector()
        correction = b.CreateVector() # auxiliary space correction

        # ========== pre-smoothing: GS assumed
        for i in range(self.nSm):
            self.sm.Smooth(rho, b)
        residual.data = b - self.mat * rho

        # ========== coarse grid correction
        # Projector project out inactive dofs on fine level
        residual.data = Projector(self.activeDOFs, True) * residual
        correction.data = self.coarseSol * residual
        ### Project out inactive DOFs
        correction.data = Projector(self.activeDOFs, True) * correction
        
        rho.data += correction
    
        # ========== post-smoothing: GS assumed
        for i in range(self.nSm):
            self.sm.SmoothBack(rho, b)