# -*- coding: utf-8 -*-
"""
Author: Zhen YZ
Date: Mar 29, 2023

"""
import numpy as np
import qutip as qtp
from itertools import product
from qutip.qip.operations import swap
from scipy.optimize import fsolve


def entropy(pp):
    """ entropy of a distribution pp.
    """
    ent = 0.
    for p in pp:
        if 0. < p < 1.:
            ent += -p * np.log2(p)

    return ent


def cond_entropy(pabs, pbs):
    return entropy(pabs) - entropy(pbs)


def get_state(theta, nv):
    psi = np.cos(theta) * qtp.ket('00') + np.sin(theta) * qtp.ket('11')
    rho = qtp.tensor(nv * qtp.ket2dm(psi) + (1 - nv) * qtp.qeye([2, 2]) / 4,
                     nv * qtp.ket2dm(psi) + (1 - nv) * qtp.qeye([2, 2]) / 4)
    return swap(N=4, targets=[1, 2]) * rho * swap(N=4, targets=[1, 2]).dag()


def get_proj_meas():
    """ Get the projective measurement operators """
    sx, sz = qtp.sigmax(), qtp.sigmaz()
    ketp, ketm = (qtp.ket('0') + qtp.ket('1')) / np.sqrt(2), (qtp.ket('0') - qtp.ket('1')) / np.sqrt(2)
    pj_x0 = lambda a: qtp.tensor(sx ** a[1], sx ** a[0]) * qtp.ket('00')
    pj_x1 = lambda a: qtp.tensor(sz ** a[0], sz ** a[1]) * qtp.tensor(ketp, ketp)
    pj_x2 = lambda a: qtp.tensor(sz ** a[0], sz ** a[1]) * (qtp.tensor(ketp, qtp.ket('1'))
                                                            - qtp.tensor(ketm, qtp.ket('0'))) / np.sqrt(2)
    pj_y0 = lambda b: qtp.tensor(sz ** b[1], sx ** b[0]) * qtp.tensor(ketp, qtp.ket('0'))
    pj_y1 = lambda b: qtp.tensor(sx ** b[0], sz ** b[1]) * qtp.tensor(qtp.ket('0'), ketp)
    pj_y2 = lambda b: qtp.tensor(sx ** b[0], sz ** b[1]) * (qtp.ket('00') + qtp.ket('11')) / np.sqrt(2)

    pjAs = [[qtp.ket2dm(pj_x0([0, 0])), qtp.ket2dm(pj_x0([0, 1])), qtp.ket2dm(pj_x0([1, 0]))],
            [qtp.ket2dm(pj_x1([0, 0])), qtp.ket2dm(pj_x1([0, 1])), qtp.ket2dm(pj_x1([1, 0]))],
            [qtp.ket2dm(pj_x2([0, 0])), qtp.ket2dm(pj_x2([0, 1])), qtp.ket2dm(pj_x2([1, 0]))]]
    pjBs = [[qtp.ket2dm(pj_y0([0, 0])), qtp.ket2dm(pj_y0([0, 1])), qtp.ket2dm(pj_y0([1, 0]))],
            [qtp.ket2dm(pj_y1([0, 0])), qtp.ket2dm(pj_y1([0, 1])), qtp.ket2dm(pj_y1([1, 0]))],
            [qtp.ket2dm(pj_y2([0, 0])), qtp.ket2dm(pj_y2([0, 1])), qtp.ket2dm(pj_y2([1, 0]))]]

    return pjAs, pjBs


def get_full_probs(theta, nv):
    rho = get_state(theta, nv)
    pjAs, pjBs = get_proj_meas()

    probab = [[[[(rho * qtp.tensor(pjAs[x][a], pjBs[y][b])).tr().real
                 for b in range(3)]
                for a in range(3)]
               for y in range(3)]
              for x in range(3)]
    proba = [[(rho * qtp.tensor(pjAs[x][a], qtp.qeye([2, 2]))).tr().real
              for a in range(3)]
             for x in range(3)]
    probb = [[(rho * qtp.tensor(qtp.qeye([2, 2]), pjBs[y][b])).tr().real
              for b in range(3)]
             for y in range(3)]

    return probab, proba, probb


def get_meas():
    si, sx, sy, sz = qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()
    ma = [[qtp.tensor(si, sz), qtp.tensor(sz, si), qtp.tensor(sz, sz)],
          [qtp.tensor(sx, si), qtp.tensor(si, sx), qtp.tensor(sx, sx)],
          [-qtp.tensor(sx, sz), -qtp.tensor(sz, sx), qtp.tensor(sy, sy)]]
    mb = [[ma[0][y], ma[1][y], ma[2][y]] for y in range(3)]

    return ma, mb


def get_probabilities(theta, nv):
    """ Get the probabilities of MPG
    """
    rho = get_state(theta, nv)
    ma, mb = get_meas()

    ii = qtp.qeye([2, 2])
    probs = [[[(rho * qtp.tensor((ii + a * ma[x][y]) / 2, (ii + b * mb[y][x]) / 2)).tr().real
               for a, b in product([+1, -1], [+1, -1])]
              for y in range(3)]
             for x in range(3)]
    means = [[(rho * qtp.tensor(ma[x][y], mb[y][x])).tr().real
              for y in range(3)]
             for x in range(3)]

    return probs, means


def get_hagbs(probs):
    hagbs = [[cond_entropy(probs[x][y],
                           [probs[x][y][0] + probs[x][y][2], probs[x][y][1] + probs[x][y][3]])
              for y in range(3)]
             for x in range(3)]

    return hagbs


def get_hagb_nu(val):
    def _fun(nu):
        _, means = get_probabilities(np.pi / 4, nu[0])

        return np.sum(means) / 9 - val

    nu_best = fsolve(_fun, x0=np.array([1.]))[0]
    probs, _ = get_probabilities(np.pi / 4, nu_best)

    return nu_best, np.sum(get_hagbs(probs)) / 9


def get_hagb_eta(val):
    # Wrong!:
    # _fun = lambda eta: eta[0]**2 * 1 + eta[0]*(1-eta[0]) * 0.5 * 2 + (1-eta[0])**2 * 8/9 - val
    _fun = lambda eta: eta[0] ** 2 * 1 + (1 - eta[0]) ** 2 * (7 / 9) - val
    eta_best = fsolve(_fun, x0=np.array([1.]))[0]
    probs = eta_best ** 2 * np.array([[[0.5, 0., 0., 0.5] for _ in range(3)] for _ in range(3)]) \
        + eta_best * (1 - eta_best) * np.array([[[0.5, 0., 0.5, 0.] for _ in range(3)],
                                                [[0.5, 0., 0.5, 0.] for _ in range(3)],
                                                [[0., 0.5, 0., 0.5] for _ in range(3)]]) \
        + (1 - eta_best) * eta_best * np.array([[[0.5, 0.5, 0., 0.] for _ in range(3)],
                                                [[0.5, 0.5, 0., 0.] for _ in range(3)],
                                                [[0., 0., 0.5, 0.5], [0., 0., 0.5, 0.5], [0.5, 0.5, 0., 0.]]]) \
        + (1 - eta_best) ** 2 * np.array([[[1., 0., 0., 0.] for _ in range(3)],
                                          [[1., 0., 0., 0.] for _ in range(3)],
                                          [[1., 0., 0., 0.], [1., 0., 0., 0.], [0., 1., 0., 0.]]])
    """probs = eta_best ** 2 * np.array([[[0.5, 0., 0., 0.5] for _ in range(3)] for _ in range(3)]) \
            + eta_best * (1 - eta_best) * np.array([[[0.5, 0., 0.5, 0.] for _ in range(3)],
                                                    [[0.5, 0., 0.5, 0.] for _ in range(3)],
                                                    [[0.5, 0., 0.5, 0.], [0.5, 0., 0.5, 0.], [0., 0.5, 0., 0.5]]]) \
            + (1 - eta_best) * eta_best * np.array([[[0.5, 0., 0.5, 0.] for _ in range(3)]
                                                    for _ in range(3)]) \
            + (1 - eta_best) ** 2 * np.array([[[1., 0., 0., 0.] for _ in range(3)],
                                              [[1., 0., 0., 0.] for _ in range(3)],
                                              [[1., 0., 0., 0.], [1., 0., 0., 0.], [0., 1., 0., 0.]]])"""

    return eta_best, np.sum(get_hagbs(probs)) / 9


def correl_2qubit(theta, axs, bys, eta, nv):
    """ the correlations <AxBy>, <Ax>, <By>,
        in the order [A0B0, A0B1, A1B0, A1B1, A0, A1, B0, B1].
    """
    psi = np.cos(theta) * qtp.ket('00') + np.sin(theta) * qtp.ket('11')
    dm = nv * qtp.ket2dm(psi) + (1. - nv) * qtp.qeye([2, 2]) / 4.

    ma, mb = [], []
    for _a in axs:
        ma += [eta * (np.cos(_a) * qtp.sigmaz() + np.sin(_a) * qtp.sigmax())
               - (1. - eta) * qtp.qeye(2)]
    for _b in bys:
        mb += [eta * (np.cos(_b) * qtp.sigmaz() + np.sin(_b) * qtp.sigmax())
               - (1. - eta) * qtp.qeye(2)]

    mean_vals = []
    for _ma, _mb in product(ma, mb):
        mean_vals += [(dm * qtp.tensor(_ma, _mb)).tr().real]
    for _ma in ma:
        mean_vals += [(dm * qtp.tensor(_ma, qtp.qeye(2))).tr().real]
    for _mb in mb:
        mean_vals += [(dm * qtp.tensor(qtp.qeye(2), _mb)).tr().real]

    return np.array(mean_vals)


def key_chsh(eta, nv):
    """ The upper and lower bounds of key rate based on the CHSH values.
    """
    correl = correl_2qubit(np.pi / 4, [0., np.pi / 2], [np.pi / 4, -np.pi / 4], eta, nv)
    ss = correl[0] + correl[1] + correl[2] - correl[3]  # CHSH value
    qq = (1 - nv * eta ** 2 - (1 - eta) ** 2) / 2  # bit error rate
    if ss > 2.:
        tmp = np.sqrt((ss / 2.) ** 2 - 1)
        key_l = 1. - entropy([qq, 1 - qq]) - entropy([(1 + tmp) / 2., (1 - tmp) / 2.])
        key_l = max(0., key_l)
    else:
        key_l = 0

    return key_l


def get_msg_data():
    data_m8 = np.genfromtxt('Data/DataFinal_m8.csv', delimiter=',', skip_header=1)
    wexps = data_m8[:, 0]
    hages = np.sum(data_m8[:, 2:], 1) / data_m8[:, 2:].shape[1]
    nvs, hagbs_nv, etas, hagbs_eta = [], [], [], []
    for val in wexps:
        nv, hagb_nv = get_hagb_nu(val)
        nvs += [nv]
        hagbs_nv += [hagb_nv]
        eta, hagb_eta = get_hagb_eta(val)
        etas += [eta]
        hagbs_eta += [hagb_eta]

    keyrate_nv = hages - np.array(hagbs_nv)
    keyrate_nv[keyrate_nv < 0] = 0
    keyrate_eta = hages - np.array(hagbs_eta)
    keyrate_eta[keyrate_eta < 0] = 0

    data = np.vstack((wexps, hages, nvs, hagbs_nv, keyrate_nv, etas, hagbs_eta, keyrate_eta)).T

    return data


def get_msg_full_data():
    datas = [np.genfromtxt(f'HAgE_FullStatistics_x{x}y{y}.csv', delimiter=',', skip_header=1)
             for x, y in product(range(1), range(2))]
    num = min(data.shape[0] for data in datas)
    nvs = datas[0][:num, 0]
    hagbs, hages, means = [], [], []
    for _ in range(num):
        probs, mean = get_probabilities(np.pi / 4, nvs[_])
        means += [np.sum(mean) / 9]
        hagbs += [np.sum(get_hagbs(probs)) / 9]
        hages += [np.sum([data[_, 2] for data in datas]) / len(datas)]
    hages, hagbs = np.array(hages), np.array(hagbs)
    keyrates = hages - hagbs
    keyrates[keyrates < 0] = 0
    new_data = np.vstack((nvs, keyrates, hages, hagbs, means)).T

    return new_data


if __name__ == '__main__':
    # Data = get_msg_full_data()
    # np.savetxt('MSG_full.csv', Data, delimiter=',')
    Data = np.genfromtxt('Data/MSG.csv', delimiter=',', skip_header=1)
    Data_new = Data[:, :5]
    Data_add = []
    for _K in range(Data.shape[0]):
        Eta, Hagb = get_hagb_eta(Data[_K, 0])
        Data_add += [[Eta, Hagb, max(Data[_K, 1] - Hagb, 0)]]
    DAta_new = np.hstack((Data_new, np.array(Data_add)))
    np.savetxt('MSG_corrected.csv', DAta_new, delimiter=',',
               header='Val,HAgE,nu,HAgB_nu,key_nu,eta,HAgB_eta,key_eta')

    # Test the optimized nu and eta
    Data_c = np.genfromtxt('MSG_corrected.csv', delimiter=',', skip_header=1)
    Diff_nv, Diff_eta = 0., 0.
    for _K in range(Data.shape[0]):
        _, Mean_nv = get_probabilities(np.pi / 4, Data_c[_K, 2])
        Diff_nv += abs(Data_c[_K, 0] - np.sum(Mean_nv) / 9)

        Probs = Data_c[_K, 5] ** 2 * np.array([[[0.5, 0., 0., 0.5] for _ in range(3)] for _ in range(3)]) \
            + Data_c[_K, 5] * (1 - Data_c[_K, 5]) * np.array([[[0.5, 0., 0.5, 0.] for _ in range(3)],
                                                              [[0.5, 0., 0.5, 0.] for _ in range(3)],
                                                              [[0., 0.5, 0., 0.5] for _ in range(3)]]) \
            + (1 - Data_c[_K, 5]) * Data_c[_K, 5] * np.array([[[0.5, 0.5, 0., 0.] for _ in range(3)],
                                                              [[0.5, 0.5, 0., 0.] for _ in range(3)],
                                                              [[0., 0., 0.5, 0.5], [0., 0., 0.5, 0.5], [0.5, 0.5, 0., 0.]]]) \
            + (1 - Data_c[_K, 5]) ** 2 * np.array([[[1., 0., 0., 0.] for _ in range(3)],
                                                  [[1., 0., 0., 0.] for _ in range(3)],
                                                  [[1., 0., 0., 0.], [1., 0., 0., 0.], [0., 1., 0., 0.]]])
        Mean_eta = [[Probs[x][y][0] - Probs[x][y][1] - Probs[x][y][2] + Probs[x][y][3]
                     for y in range(3)]
                    for x in range(3)]
        Diff_eta += abs(Data_c[_K, 0] - np.sum(Mean_eta) / 9)

    print(Diff_nv, Diff_eta)
