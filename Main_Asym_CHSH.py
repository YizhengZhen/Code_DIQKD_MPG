# -*- coding: utf-8 -*-
"""
Author: Zhen YZ
Date: Mar 29, 2023

This is the class to calculate H(A|E) using Brown-Fawzi-Fawzi quasi-entropy method for the CHSH-based protocol.
See their paper: https://arxiv.org/abs/2106.13692
    or their project: https://github.com/peterjbrown519/DI-rates
Note: The ncpol2sdpa package >= 1.12.3 is required to run this code:
    https://github.com/peterjbrown519/ncpol2sdpa
"""

import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import chaospy
from Mod_Func import *


class aCHSH_SDP:
    def __init__(self, m, verbose=False, parallel=False):
        tt, ww = chaospy.quad_gauss_radau(m, chaospy.Uniform(0, 1), 1)
        self.M = 2 * m
        self.T = tt[0][:m * 2 - 1]
        self.CK = [ww[i] / (tt[0][i] * np.log(2)) for i in range(m * 2 - 1)]

        self.A = [Ai for Ai in ncp.generate_measurements([2, 2], 'A')]
        self.B = [Bj for Bj in ncp.generate_measurements([2, 2], 'B')]

        self.Z = ncp.generate_operators('Z', 2, hermitian=False)

        self.SDP = ncp.SdpRelaxation(ncp.flatten([self.A, self.B, self.Z]),
                                     verbose=verbose, normalized=True, parallel=parallel)

    def obj_j(self, j):
        mas = [self.A[0][0],  1 - self.A[0][0]]

        return sum(ma * (mz + Dagger(mz) + (1 - self.T[j]) * Dagger(mz) * mz)
                   + self.T[j] * mz * Dagger(mz)
                   for ma, mz in zip(mas, self.Z)
                   )

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

    def constr_ineq(self, val, eps):
        ma0 = 2 * self.A[0][0] - 1
        ma1 = 2 * self.A[1][0] - 1
        mb0 = 2 * self.B[0][0] - 1
        mb1 = 2 * self.B[1][0] - 1

        return [(1 - eps) * ma0 * (mb0 + mb1) / 2 + eps * ma1 * (mb0 - mb1) / 2 - val]

    def init(self):
        self.SDP.get_relaxation(level=2,
                                equalities=[],
                                inequalities=[],
                                momentequalities=[],
                                momentinequalities=self.constr_ineq(0, 0.5),
                                objective=self.obj_j(0),
                                substitutions=self.get_subs(),
                                extramonomials=self.extra_monomials())
        print(f'SDP is initialized.')

    def get_hage(self):
        hage = sum(self.CK)
        for j in range(self.M - 1):
            self.SDP.set_objective(self.obj_j(j))
            self.SDP.solve(solver='mosek')
            if self.SDP.status == 'optimal':
                hage += self.CK[j] * self.SDP.dual
                print(j, self.SDP.status, self.SDP.primal, self.SDP.dual, hage)
            else:
                hage = 0.
                print(j, self.SDP.status, self.SDP.primal, self.SDP.dual, hage)
                break

        return hage


def get_hages(eps, model):

    vals = np.linspace(np.sqrt(1 - 2 * eps + 2 * eps**2), 1 - eps, 50)
    hages = []
    for val in vals:
        model.SDP.process_constraints(equalities=[],
                                      inequalities=[],
                                      momentequalities=[],
                                      momentinequalities=model.constr_ineq(val, eps))
        hages += [model.get_hage()]

    return hages, vals


def get_data_achsh(eps, model):
    hages, vals = get_hages(eps=eps, model=model)
    nus, etas, hagbs_nu, hagbs_eta = [], [], [], []
    mu = np.arctan(eps/(1-eps))

    def _fun_nu(nu, val):
        correl = correl_2qubit(np.pi/4, [0., np.pi/2], [mu, -mu], 1., nu[0])
        return (1-eps)/2 * (correl[0] + correl[1]) + eps/2 * (correl[2] - correl[3]) - val

    def _fun_eta(eta, val):
        correl = correl_2qubit(np.pi / 4, [0., np.pi / 2], [mu, -mu], eta[0], 1.)
        return (1 - eps) / 2 * (correl[0] + correl[1]) + eps / 2 * (correl[2] - correl[3]) - val

    for _val in vals:
        nu_best = fsolve(_fun_nu, x0=np.array([1.]), args=(_val,), xtol=1e-6)[0]
        nus += [nu_best]
        pabs_nu = (np.ones(4) + np.array([nu_best, -nu_best, -nu_best, nu_best])) / 4
        hagbs_nu += [cond_entropy(pabs_nu, [pabs_nu[0] + pabs_nu[2], pabs_nu[1] + pabs_nu[3]])]

        eta_best = fsolve(_fun_eta, x0=np.array([1.]), args=(_val,), xtol=1e-6)[0]
        etas += [eta_best]
        pabs_eta = eta_best**2 * np.array([0.5, 0., 0., 0.5]) \
            + eta_best*(1-eta_best) * np.array([0.5, 0., 0.5, 0.]) \
            + (1-eta_best)*eta_best * np.array([0.5, 0.5, 0., 0.]) \
            + (1-eta_best)**2 * np.array([1., 0., 0., 0.])
        hagbs_eta += [cond_entropy(pabs_eta,
                                   [pabs_eta[0] + pabs_eta[2], pabs_eta[1] + pabs_eta[3]])]
    keys_nu = np.array(hages) - np.array(hagbs_nu)
    keys_nu[keys_nu < 0] = 0.
    keys_eta = np.array(hages) - np.array(hagbs_eta)
    keys_eta[keys_eta < 0] = 0.

    data = np.vstack((vals, hages, nus, hagbs_nu, keys_nu, etas, hagbs_eta, keys_eta)).T

    return data


if __name__ == '__main__':

    Model = aCHSH_SDP(8, verbose=False, parallel=True)
    Model.init()

    Data01 = get_data_achsh(0.01, Model)
    np.savetxt('aCHSH_eps01.csv', Data01, delimiter=',')

    Data1 = get_data_achsh(0.1, Model)
    np.savetxt('aCHSH_eps1.csv', Data1, delimiter=',')

    Data3 = get_data_achsh(0.3, Model)
    np.savetxt('aCHSH_eps3.csv', Data3, delimiter=',')

    Data5 = get_data_achsh(0.5, Model)
    np.savetxt('aCHSH_eps5.csv', Data5, delimiter=',')
