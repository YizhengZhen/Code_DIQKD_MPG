# -*- coding: utf-8 -*-
"""
Author: Zhen YZ
Date: Mar 29, 2023

This is the class to calculate H(A|E) using Brown-Fawzi-Fawzi quasi-entropy method for the MPG-based protocol.
See their paper: https://arxiv.org/abs/2106.13692
    or their project: https://github.com/peterjbrown519/DI-rates
Note: The ncpol2sdpa package >= 1.12.3 is required to run this code:
    https://github.com/peterjbrown519/ncpol2sdpa
"""
import numpy as np
import ncpol2sdpa as ncp
from itertools import product
from sympy.physics.quantum.dagger import Dagger
import chaospy


class MSG_SDP:
    def __init__(self, m, verbose=False, parallel=False):
        """
        :param m: the number of quadrature points
        :param verbose: True or False
        :param parallel: True or False
        """
        # Generate quadrature points and weights
        tt, ww = chaospy.quad_gauss_radau(m, chaospy.Uniform(0, 1), 1)
        self.M = 2 * m
        self.T = tt[0][:m * 2 - 1]
        self.CK = [ww[i] / (tt[0][i] * np.log(2)) for i in range(m * 2 - 1)]

        # Generate the Alice and Bob's measurements
        self.A = [Ai for Ai in ncp.generate_measurements([4, 4, 4], 'A')]
        self.B = [Bj for Bj in ncp.generate_measurements([4, 4, 4], 'B')]

        # Generate the operators used to compute the quasi-entropy
        self.Z = ncp.generate_operators('Z', 2, hermitian=False)

        # Generate the SDP relaxation
        self.SDP = ncp.SdpRelaxation(ncp.flatten([self.A, self.B, self.Z]),
                                     verbose=verbose, normalized=True, parallel=parallel)

        # Generate the projection operators corresponding to outcome 0 for each entry of magic square
        self.projA = [[ma[0] + ma[1], ma[0] + ma[2], 1 - ma[1] - ma[2]]
                      for ma in self.A]
        self.projB = [[mb[0] + mb[1], mb[0] + mb[2], mb[1] + mb[2]]
                      for mb in self.B]

    # The objective function for the SDP relaxation
    def msg_obj_j(self, j, x, y):
        mas = [self.projA[x][y], 1 - self.projA[x][y]]
        return sum(ma * (mz + Dagger(mz) + (1 - self.T[j]) * Dagger(mz) * mz)
                   + self.T[j] * mz * Dagger(mz)
                   for ma, mz in zip(mas, self.Z[j])
                   )
    """
    def obj_j(self, j, x):
        mas = self.A[x] + [1 - sum(self.A[x][_] for _ in range(3))]
        return sum(ma * (mz + Dagger(mz) + (1 - self.T[j]) * Dagger(mz) * mz)
                   + self.T[j] * mz * Dagger(mz)
                   for ma, mz in zip(mas, self.Z)
                   )"""

    def get_subs(self):
        subs = {}
        subs.update(ncp.projective_measurement_constraints(self.A, self.B))
        for a, z in product(ncp.flatten([self.A, self.B]), ncp.flatten(self.Z)):
            subs.update({z * a: a * z, Dagger(z) * a: a * Dagger(z)})

        return subs

    def extra_monomials(self):
        monos = []
        z_all = ncp.flatten(self.Z)
        for a, z in product(self.A[0], z_all):
            monos += [a * Dagger(z) * z]
        """
        a_all = ncp.flatten(self.A)
        b_all = ncp.flatten(self.B)
        for a, b, z in product(a_all, b_all, z_all):
            monos += [a * b * z, a * b * Dagger(z)]"""

        return monos

    def oper_ineq(self, j):
        op_ineq = []
        alpj = max(1 / self.T[j], 1 / (1 - self.T[j])) * 3 / 2
        for z in self.Z:
            op_ineq += [alpj - z * Dagger(z), alpj - Dagger(z) * z]

        return op_ineq

    # Constraint on full probabilities
    def constr_prob_full(self, probabilities):
        cons = []
        for x, y in product(range(3), range(3)):
            for a, b in product(range(3), range(3)):
                cons += [self.A[x][a] * self.B[y][b] - probabilities[x][y][a][b]]

        return cons

    # Constraint on the probabilities of all entries of magic square
    def constr_prob(self, probabilities):
        cons = []
        for x, y in product(range(3), range(3)):
            cons += [self.projA[x][y] * self.projB[y][x] - probabilities[x][y][0],
                     self.projA[x][y] * (1 - self.projB[y][x]) - probabilities[x][y][1],
                     (1 - self.projA[x][y]) * self.projB[y][x] - probabilities[x][y][2]]

        return cons

    # Constraint on the winning probability for each entry of magic square
    def constr_threshold(self, threshold):
        cons = []
        for proja, projb in product(self.projA, self.projB):
            cons += [((2 * proja - 1) * (2 * projb - 1) + 1) / 2 - threshold]

        return cons

    # Constraint on the average pay-off for the magic square
    def constr_ineq(self, value):

        ob_a = [[2 * projs[0] - 1, 2 * projs[1] - 1, 2 * projs[2] - 1]
                for projs in self.projA]
        ob_b = [[2 * projs[0] - 1, 2 * projs[1] - 1, 2 * projs[2] - 1]
                for projs in self.projB]

        ineq = ob_a[0][0] * ob_b[0][0] + ob_a[0][1] * ob_b[1][0] + ob_a[0][2] * ob_b[2][0] \
            + ob_a[1][0] * ob_b[0][1] + ob_a[1][1] * ob_b[1][1] + ob_a[1][2] * ob_b[2][1] \
            + ob_a[2][0] * ob_b[0][2] + ob_a[2][1] * ob_b[1][2] + ob_a[2][2] * ob_b[2][2]

        return [ineq / 9 - value]

    # Initialization of the SDP relaxation
    def init(self):
        self.SDP.get_relaxation(level=2,
                                equalities=[],
                                inequalities=[],
                                momentequalities=[],
                                momentinequalities=self.constr_ineq(1),
                                objective=self.msg_obj_j(0, 0, 0),
                                substitutions=self.get_subs(),
                                extramonomials=self.extra_monomials())
        print(f'SDP is initialized.')

    # Solve the SDP relaxation and return the lower bound on H(A|E)
    def get_hage(self, x, y):
        hage = sum(self.CK)
        for j in range(self.M - 1):
            self.SDP.set_objective(self.msg_obj_j(j, x, y))
            self.SDP.solve(solver='mosek')
            if self.SDP.status == 'optimal':
                hage += self.CK[j] * self.SDP.dual
                print(j, self.SDP.status, self.SDP.primal, self.SDP.dual, hage)
            else:
                hage = 0.
                print(j, self.SDP.status, self.SDP.primal, self.SDP.dual, hage)
                break

        return hage


if __name__ == '__main__':

    import time

    Time = time.time()
    Model = MSG_SDP(m=3, verbose=True, parallel=True)
    Model.init()
    Hage = Model.get_hage(0, 0)

    print(f'Time: {(time.time() - Time) / 60: 2f} min.\n\n')
    print(Hage)
