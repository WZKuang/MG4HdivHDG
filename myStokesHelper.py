# exact solutions and corresponding helpers for Stokes equations, domain = [1 x 1]^d
# by Wenzheng Kuang, 11/13/2022
from ngsolve import *

class stokesHelper:
    def __init__(self, dim):
        self.dim = dim

    # ========== exact sol in a unit square/cube
    def squareSolInit(self):
        if self.dim == 2:
            # exact solution
            u_exact1 = x ** 2 * (x - 1) ** 2 * 2 * y * (1 - y) * (2 * y - 1)
            u_exact2 = y ** 2 * (y - 1) ** 2 * 2 * x * (x - 1) * (2 * x - 1)
            self.u_exact = CF((u_exact1, u_exact2))

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
            self.L_exact = CF((L_exactXX, L_exactXY, L_exactYX, L_exactYY), dims=(2, 2))

            self.p_exact = x * (1 - x) * (1 - y) - 1 / 12
        elif self.dim == 3:
            # exact solution
            u_exact1 = x ** 2 * (x - 1) ** 2 * (2 * y - 6 * y ** 2 + 4 * y ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
            u_exact2 = y ** 2 * (y - 1) ** 2 * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
            u_exact3 = -2 * z ** 2 * (z - 1) ** 2 * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * y - 6 * y ** 2 + 4 * y ** 3)
            self.u_exact = CF((u_exact1, u_exact2, u_exact3))

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
            self.L_exact = CF((L_exactXX, L_exactXY, L_exactXZ,
                            L_exactYX, L_exactYY, L_exactYZ,
                            L_exactZX, L_exactZY, L_exactZZ), dims=(3, 3))

            self.p_exact = x * (1 - x) * (1 - y) * (1 - z) - 1 / 24
    
    def getExactSol(self):
        return (self.u_exact, self.L_exact, self.p_exact)


    # ========== rhs corresponding to the exact sol
    def getRhs(self, fes, c_low, testV):
        f = LinearForm(fes)
        if self.dim == 2:
            f += (-(4 * y * (1 - y) * (2 * y - 1) * ((1 - 2 * x) ** 2 - 2 * x * (1 - x))
                            + 12 * x ** 2 * (1 - x) ** 2 * (1 - 2 * y))
                    + (1 - 2 * x) * (1 - y)) * testV[0] * dx
            f += (-(4 * x * (1 - x) * (1 - 2 * x) * ((1 - 2 * y) ** 2 - 2 * y * (1 - y))
                            + 12 * y ** 2 * (1 - y) ** 2 * (2 * x - 1))
                    - x * (1 - x)) * testV[1] * dx
            f += c_low * self.u_exact * testV * dx
        elif self.dim == 3:
            f += (-((2 - 12 * x + 12 * x ** 2) * (2 * y - 6 * y ** 2 + 4 * y ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
                                + (x ** 2 - 2 * x ** 3 + x ** 4) * (-12 + 24 * y) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
                                + (x ** 2 - 2 * x ** 3 + x ** 4) * (-12 + 24 * z) * (2 * y - 6 * y ** 2 + 4 * y ** 3))
                    + (1 - 2 * x) * (1 - y) * (1 - z)
                    ) * testV[0] * dx
            f += (-((2 - 12 * y + 12 * y ** 2) * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
                            + (y ** 2 - 2 * y ** 3 + y ** 4) * (-12 + 24 * x) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
                            + (y ** 2 - 2 * y ** 3 + y ** 4) * (-12 + 24 * z) * (2 * x - 6 * x ** 2 + 4 * x ** 3))
                    - x * (1 - x) * (1 - z)
                    ) * testV[1] * dx
            f += (2 * (
                        (2 - 12 * z + 12 * z ** 2) * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * y - 6 * y ** 2 + 4 * y ** 3)
                        + (z ** 2 - 2 * z ** 3 + z ** 4) * (-12 + 24 * x) * (2 * y - 6 * y ** 2 + 4 * y ** 3)
                        + (z ** 2 - 2 * z ** 3 + z ** 4) * (-12 + 24 * y) * (2 * x - 6 * x ** 2 + 4 * x ** 3))
                    - x * (1 - x) * (1 - y)
                    ) * testV[2] * dx
            f += c_low * self.u_exact * testV * dx
        
        with TaskManager():
            f.Assemble()
            return f.vec.data

    # ========== convergence order check with respect to the exact sol
    def ecrCheck(self, level, fes, mesh, uh, Lh, meshRate=2, prev_uErr=0, prev_LErr=0):
        print(f'LEVEL: {level}, ALL DOFS: {fes.ndof}, GLOBAL DOFS: {sum(fes.FreeDofs(True))}')
        L2_uErr = sqrt(Integrate((uh - self.u_exact) * (uh - self.u_exact), mesh))
        L2_LErr = sqrt(Integrate(InnerProduct((Lh - self.L_exact), (Lh - self.L_exact)), mesh))
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



