import numpy as np
import qutip as qtp
import ncpol2sdpa as ncp
from itertools import product
from sympy.physics.quantum.dagger import Dagger
import csv
import time
import chaospy


class BFFnew:
    def __init__(self, m, level=2):
        tt, ww = chaospy.quad_gauss_radau(m, chaospy.Uniform(0, 1), 1)
        self.T = tt[0][:m * 2 - 1]
        self.CK = [ww[i] / (tt[0][i] * np.log(2)) for i in range(m * 2 - 1)]

        self.A = [Ai for Ai in ncp.generate_measurements([4, 4, 4], 'A')]
        self.B = [Bj for Bj in ncp.generate_measurements([4, 4, 4], 'B')]
        self.LEVEL = level

        self.M = 2 * m
        self.Z = [[ncp.generate_operators('Z'+str(xx)+str(mm), 4, hermitian=False)
                   for mm in range(self.M - 1)]
                  for xx in range(3)]

        self.SDP = ncp.SdpRelaxation(ncp.flatten([self.A, self.B, self.Z]),
                                     verbose=0, normalized=True, parallel=1)

    def obj_j(self, j, x):
        expr = 0
        for ma, mz in zip([self.A[x][0], self.A[x][1], self.A[x][2],
                           1 - self.A[x][0] - self.A[x][1] - self.A[x][2]],
                          self.Z[x][j]):
            expr += ma * (mz + Dagger(mz) + (1 - self.T[j]) * Dagger(mz) * mz) \
                + self.T[j] * mz * Dagger(mz)
        return expr


def get_subs():
    subs = {}
    # Get Alice and Bob's projective measurement constraints
    subs.update(ncp.projective_measurement_constraints(A, B))
    # Finally we note that Alice and Bob's operators should All commute with Eve's ops
    for a, z in product(ncp.flatten([A, B]), Z):
        subs.update({z * a: a * z, Dagger(z) * a: a * Dagger(z)})

    return subs


def extra_monomials():
    monos = []
    # Add ABZ
    zz = Z + [Dagger(z) for z in Z]
    aa = ncp.flatten(A)
    bb = ncp.flatten(A)

    for a, z in product(aa + bb, zz):
        monos += [a * z]
        monos += [a * Dagger(z) * z]
    for a, b, z in product(aa, bb, zz):
        monos += [a * b * z]
    # Add monos appearing in objective function
    for z in Z:
        monos += [Dagger(z) * z]
        monos += [z * Dagger(z)]

    return monos


def oper_ineq():
    op_ineq = []
    zz = [Z[:2 * M - 1], Z[2 * M - 1:]]
    for i in range(2 * M - 1):
        alpha = max(1 / T[i], 1 / (1 - T[i])) * 3 / 2
        for a in range(2):
            op_ineq += [alpha - zz[a][i] * Dagger(zz[a][i])]
            op_ineq += [alpha - Dagger(zz[a][i]) * zz[a][i]]

    return op_ineq


SYS = [np.pi / 4, 0., np.pi / 2, np.pi / 4, -np.pi / 4, 0.]

SDP.get_relaxation(level=LEVEL,
                   equalities=[],
                   inequalities=oper_ineq(),
                   momentequalities=constraints(SYS, 1., 1.),
                   momentinequalities=[],
                   objective=obj_j(0),
                   substitutions=get_subs(),
                   extramonomials=extra_monomials())

"""
FN = 'BFFnew_CHSH_LEVEL' + str(LEVEL) + '.csv'
FIELDS = ['Success', 'eta', 'q', 'KeyRate', 'HAgE', 'HAgB', 'Time']
with open(FN, 'w') as o:
    csv.writer(o).writerow(FIELDS)"""

FN = 'BFFnew_tCHSH_LEVEL' + str(LEVEL) + '.csv'
FIELDS = ['Succ', 'x0', 'KeyRate', 'HAgE', 'HAgB', 'b2', 'Succ', 'Time']
with open(FN, 'w') as o:
    csv.writer(o).writerow(FIELDS)

DATAs = []
X0 = 0.25
# ETA, Q = 1., 1.
# FLAG = 0
while True:
    T0 = time.time()

    THETA = X0 * np.pi
    AX, BY = Fun.t_chsh_measurement(THETA)
    SYS0 = [THETA, AX[0], AX[1], BY[0], BY[1], 0.]

    SDP.process_constraints(equalities=[],
                            inequalities=oper_ineq(),
                            momentequalities=constraints(SYS0, 1., 1.),   # ETA, Q),
                            momentinequalities=[])
    SUCC = True
    HAGE = np.sum(CK)
    for J in range(2 * M - 1):
        SDP.set_objective(obj_j(J))
        SDP.solve('mosek')
        # print(self.sdp.dual, self.sdp.status)
        if SDP.status == 'optimal':
            HAGE += CK[J] * SDP.dual
        else:
            HAGE = 0.
            SUCC = False
            break
    # HAGB = Fun.hagb_val(SYS[0], SYS[1], SYS[-1], ETA, Q)
    HAGB, B2, FG = Fun.hagb_min(THETA, AX[0], AX[0], 1., 1.)
    KEY = max(0., HAGE - HAGB)
    T1 = time.time()
    DATA = [str(SUCC), f'{X0:.3f}', f'{KEY:.3f}', f'{HAGE:.3f}', f'{HAGB:.3f}',
            f'{B2:.3f}', str(FG), f'{(T1 - T0) / 3600:.3f}']

    print(DATA)

    with open(FN, 'a') as o:
        csv.writer(o).writerow(DATA)
    DATAs += [DATA]

    if X0 < 5e-3:
        break
    X0 = X0 - 0.01

    """
    if FLAG == 0 and HAGE > 5e-5:
        ETA = ETA - 0.002
    elif FLAG == 1 and HAGE > 5e-5:
        Q = Q - 0.002
    elif FLAG == 1 and HAGE < 5e-5:
        break
    else:
        ETA, Q = 1, 0.998
        FLAG = 1"""
