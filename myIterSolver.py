from ngsolve import *
from ngsolve.krylovspace import LinearSolver

class IterSolver(LinearSolver):
    def __init__(self, *args, freedofs, **kargs):
        super().__init__(*args, **kargs)
        self.freedofs = freedofs

    def _SolveImpl(self, rhs: BaseVector, sol: BaseVector):
        proj = Projector(self.freedofs, True)
        r = rhs.CreateVector()
        d = rhs.CreateVector()
        r.data = rhs - self.mat * sol
        d.data = proj * r
        # L2 norm for residual checking here
        res_norm = sqrt(InnerProduct(d, d))
        if self.CheckResidual(res_norm):
            return
        
        while True:
            # self.pre => approximation of A inverse
            sol.data += self.pre * r

            r.data = rhs - self.mat * sol
            d.data = proj * r
            prev_res_norm = res_norm
            res_norm = sqrt(InnerProduct(d, d))
            # if res_norm > prev_res_norm:
            #     print('Iterative solver NOT CONVERGING!!! STOPPED!!!')
            #     return
            # else:
            #     # print(f'Converge rate: {res_norm / prev_res_norm}')
            #     pass
            if self.CheckResidual(res_norm):
                return



