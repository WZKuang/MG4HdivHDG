# exact solutions and corresponding helpers for the
# Stokes equations and the NS equation (Kovasznay flow)

from ngsolve import *

# ================ Stokes and generalized Stokes
# domain = [1 x 1]^d
class stokesHelper:
    def __init__(self, dim):
        self.dim = dim

        # ========== exact sol
        # L := Grad(u) here
        u1 = x ** 2 * (x - 1) ** 2 * 2 * y * (1 - y) * (2 * y - 1) if dim == 2 else\
             x ** 2 * (x - 1) ** 2 * (2 * y - 6 * y ** 2 + 4 * y ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
        u2 = y ** 2 * (y - 1) ** 2 * 2 * x * (x - 1) * (2 * x - 1) if dim == 2 else\
             y ** 2 * (y - 1) ** 2 * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
        u3 = -2 * z ** 2 * (z - 1) ** 2 * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * y - 6 * y ** 2 + 4 * y ** 3)

        du1 = CF((u1.Diff(x), u1.Diff(y))) if self.dim == 2 else CF((u1.Diff(x), u1.Diff(y), u1.Diff(z)))
        du2 = CF((u2.Diff(x), u2.Diff(y))) if self.dim == 2 else CF((u2.Diff(x), u2.Diff(y), u2.Diff(z)))
        du3 = CF((u3.Diff(x), u3.Diff(y), u3.Diff(z)))

        d2u1 = du1[0].Diff(x)+du1[1].Diff(y) if self.dim == 2 else du1[0].Diff(x)+du1[1].Diff(y)+du1[2].Diff(z)
        d2u2 = du2[0].Diff(x)+du2[1].Diff(y) if self.dim == 2 else du2[0].Diff(x)+du2[1].Diff(y)+du2[2].Diff(z)
        d2u3 = du3[0].Diff(x)+du3[1].Diff(y)+du3[2].Diff(z)
        if self.dim == 2:
            # exact solution
            self.u_exact = CF((u1 , u2))
            self.L_exact = CF((du1, du2), dims=(2, 2))
            self.p_exact = x * (1 - x) * (1 - y) - 1 / 12
            self.source = CF((-d2u1 + self.p_exact.Diff(x),
                              -d2u2 + self.p_exact.Diff(y)))

        elif self.dim == 3:
            # exact solution
            self.u_exact = CF((u1 , u2, u3))
            self.L_exact = CF((du1, du2, du3), dims=(3,3))
            self.p_exact = x * (1 - x) * (1 - y) * (1 - z) - 1 / 24
            self.source = CF((-d2u1 + self.p_exact.Diff(x),
                              -d2u2 + self.p_exact.Diff(y),
                              -d2u3 + self.p_exact.Diff(z)))

    
    def getExactSol(self):
        return (self.u_exact, self.L_exact, self.p_exact)


    # ========== rhs corresponding to the exact sol
    def getRhs(self, fes, c_low, testV):
        f = LinearForm(fes)
        f += self.source * testV * dx
        f += c_low * self.u_exact * testV * dx
        return f

    # ========== convergence order check with respect to the exact sol
    def ecrCheck(self, level, fes, mesh, uh, Lh, meshRate=2, prev_uErr=0, prev_LErr=0):
        print(f'LEVEL: {level}, ALL DOFS: {sum(fes.FreeDofs())}, GLOBAL DOFS: {sum(fes.FreeDofs(True))}')
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




class nsHelper:
    # domain: unit square/cube
    def __init__(self, dim, nu):
        self.dim = dim
        # ========== exact sol
        # L := Grad(u) here
        u1 = x ** 2 * (x - 1) ** 2 * 2 * y * (1 - y) * (2 * y - 1) if dim == 2 else\
             x ** 2 * (x - 1) ** 2 * (2 * y - 6 * y ** 2 + 4 * y ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
        u2 = y ** 2 * (y - 1) ** 2 * 2 * x * (x - 1) * (2 * x - 1) if dim == 2 else\
             y ** 2 * (y - 1) ** 2 * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * z - 6 * z ** 2 + 4 * z ** 3)
        u3 = -2 * z ** 2 * (z - 1) ** 2 * (2 * x - 6 * x ** 2 + 4 * x ** 3) * (2 * y - 6 * y ** 2 + 4 * y ** 3)

        du1 = CF((u1.Diff(x), u1.Diff(y))) if self.dim == 2 else CF((u1.Diff(x), u1.Diff(y), u1.Diff(z)))
        du2 = CF((u2.Diff(x), u2.Diff(y))) if self.dim == 2 else CF((u2.Diff(x), u2.Diff(y), u2.Diff(z)))
        du3 = CF((u3.Diff(x), u3.Diff(y), u3.Diff(z)))

        d2u1 = du1[0].Diff(x)+du1[1].Diff(y) if self.dim == 2 else du1[0].Diff(x)+du1[1].Diff(y)+du1[2].Diff(z)
        d2u2 = du2[0].Diff(x)+du2[1].Diff(y) if self.dim == 2 else du2[0].Diff(x)+du2[1].Diff(y)+du2[2].Diff(z)
        d2u3 = du3[0].Diff(x)+du3[1].Diff(y)+du3[2].Diff(z)
        if self.dim == 2:
            # exact solution
            self.u_exact = CF((u1 , u2))
            self.L_exact = CF((du1, du2), dims=(2, 2))
            self.p_exact = x * (1 - x) * (1 - y) - 1 / 12

            self.source = CF((-nu*d2u1 + self.p_exact.Diff(x) + self.u_exact * du1,
                              -nu*d2u2 + self.p_exact.Diff(y) + self.u_exact * du2))

        elif self.dim == 3:
            # exact solution
            self.u_exact = CF((u1 , u2, u3))
            self.L_exact = CF((du1, du2, du3), dims=(3,3))
            self.p_exact = x * (1 - x) * (1 - y) * (1 - z) - 1 / 24

            self.source = CF((-nu*d2u1 + self.p_exact.Diff(x) + self.u_exact * du1,
                              -nu*d2u2 + self.p_exact.Diff(y) + self.u_exact * du2,
                              -nu*d2u3 + self.p_exact.Diff(z) + self.u_exact * du3))

    
    def getExactSol(self):
        return (self.u_exact, self.L_exact, self.p_exact)


    # ========== rhs corresponding to the exact sol
    def getRhs(self, fes, testV):
        f = LinearForm(fes)
        f += self.source * testV * dx
        return f

    # ========== convergence order check with respect to the exact sol
    def ecrCheck(self, level, mesh, uh, Lh, meshRate=2, prev_uErr=0.0, prev_LErr=0.0):
        L2_uErr = sqrt(Integrate((uh - self.u_exact) * (uh - self.u_exact), mesh))
        L2_LErr = sqrt(Integrate(InnerProduct((Lh - self.L_exact), (Lh - self.L_exact)), mesh))
        L2_divErr = sqrt(Integrate(div(uh) * div(uh), mesh))
        if level > 0:
            u_rate = log(prev_uErr / L2_uErr) / log(meshRate)
            print(f"uh L2-error: {L2_uErr:.1E}, uh conv rate: {u_rate:.1E}")
            L_rate = log(prev_LErr / L2_LErr) / log(meshRate)
            print(f"Grad(uh) L2-error: {L2_LErr:.1E}, Grad(uh) conv rate: {L_rate:.1E}")
        else:
            print(f"uh L2-error: {L2_uErr:.1E}")
            print(f"Grad(uh) L2-error: {L2_LErr:.1E}")
        print(f'uh divErr: {L2_divErr:.1E}')
        print('==============================')
        return (L2_uErr, L2_LErr)



