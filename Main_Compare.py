# -*- coding: utf-8 -*-
"""
Author: Zhen YZ
Date:


"""

from Mod_Func import *
import matplotlib.pyplot as plt


def Fig1_basic():
    datas = np.genfromtxt('Data/DataFinal_m8.csv', skip_header=1, delimiter=',')
    hages = np.sum(datas[:, 2:], 1) / 3
    krs = []
    for kk in range(datas.shape[0]):
        # Wrong!: data[kk, 0] is the average value of observable inequality expression.
        # pab = [datas[kk, 0] / 2, (1 - datas[kk, 0]) / 2, (1 - datas[kk, 0]) / 2, datas[kk, 0] / 2]
        pab = [(1 + datas[kk, 0]) / 4, (1 - datas[kk, 0]) / 4,
               (1 - datas[kk, 0]) / 4, (1 + datas[kk, 0]) / 4]
        krs += [max(0, hages[kk] - cond_entropy(pab, [pab[0] + pab[2], pab[1] + pab[3]]))]

    fig, axl = plt.subplots(1, 1, figsize=(10, 5))
    axr = axl.twinx()
    for _ in ['left', 'right', 'bottom', 'top']:
        axr.spines[_].set_linewidth(2)
    axr.spines['left'].set_edgecolor('tab:blue')
    axr.spines['right'].set_edgecolor('tab:red')

    axl.plot(datas[:, 0], hages, linewidth=2.5, marker='o', color='tab:blue', alpha=0.8)
    axr.plot(datas[:, 0], krs, linewidth=2.5, marker='o', color='tab:red', alpha=0.8)

    axl.set_xlim((8 / 9, 1))
    axl.set_xlabel(r'$\omega_{\rm exp}$ [$Q$]', fontsize=25)
    # Expectation = Val, Winning probability = (Val + 1) / 2, Q = (1 - Val) / 2
    axl.set_xticks([0.89, 0.92, 0.95, 0.98, 1.00],
                   ['0.945 [5.5%]', '0.960 [4.0%]', '0.975 [2.5%]', '0.990 [1.0%]', '1 [0%]'], fontsize=18)
    axl.tick_params(axis='x', which='major', pad=10)
    axl.set_ylim((0, 1))
    axl.set_ylabel(r'$H({\bf A}|E)_{\tau_{xy}}$', fontsize=22, color='tab:blue')
    axl.set_yticks([0, 0.5, 1], ['0.0', '0.5', '1.0'], fontsize=20, color='tab:blue')
    axr.set_ylim((0, 1))
    axr.set_ylabel(r'key rate $R$', fontsize=22, color='tab:red')
    axr.set_yticks([1, 0.5, 0], [r'$\gamma$', r'$\frac{\gamma}{2}$', '0'],
                   fontsize=20, color='tab:red')

    # axr.set_xticks([0, 0.02, 0.04, 0.06, 0.08], ['0', '2', '4', '6', '8'], fontsize=20)

    plt.tight_layout()
    plt.show()
    return fig


def msg_chsh():
    data_msg = np.genfromtxt('MSG_corrected.csv', delimiter=',', skip_header=1)
    data_achsh1 = np.genfromtxt('Data/aCHSH_eps1.csv', delimiter=',', skip_header=1)
    data_achsh2 = np.genfromtxt('Data/aCHSH_eps2.csv', delimiter=',', skip_header=1)
    data_achsh3 = np.genfromtxt('Data/aCHSH_eps3.csv', delimiter=',', skip_header=1)
    data_achsh5 = np.genfromtxt('Data/aCHSH_eps5.csv', delimiter=',', skip_header=1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    ax[0].plot(data_achsh1[:, 2], data_achsh1[:, 4] * 0.9, label=r'CHSH, $\epsilon=0.1$',
               linewidth=2, color='deepskyblue', linestyle='dashed', alpha=0.9)
    ax[0].plot(data_achsh2[:, 2], data_achsh2[:, 4] * 0.8, label=r'CHSH, $\epsilon=0.2$',
               linewidth=2, color='tab:blue', linestyle='dashed', alpha=0.9)
    ax[0].plot(data_achsh3[:, 2], data_achsh3[:, 4] * 0.7, label=r'CHSH, $\epsilon=0.3$',
               linewidth=2, color='royalblue', linestyle='dashed', alpha=0.9)
    ax[0].plot(data_achsh5[:, 2], data_achsh5[:, 4] * 0.5, label=r'CHSH, $\epsilon=0.5$',
               linewidth=2, color='navy', linestyle='dashed', alpha=0.9)
    ax[0].plot(data_msg[:, 2], data_msg[:, 4], label='MSG',
               linewidth=2, color='tab:red', marker='o', alpha=0.9)

    ax[1].plot(data_achsh1[:, 5], data_achsh1[:, 7] * 0.9, label=r'CHSH, $\epsilon=0.1$',
               linewidth=2, color='deepskyblue', linestyle='dashed', alpha=0.9)
    ax[1].plot(data_achsh2[:, 5], data_achsh2[:, 7] * 0.8, label=r'CHSH, $\epsilon=0.2$',
               linewidth=2, color='tab:blue', linestyle='dashed', alpha=0.9)
    ax[1].plot(data_achsh3[:, 5], data_achsh3[:, 7] * 0.7, label=r'CHSH, $\epsilon=0.3$',
               linewidth=2, color='royalblue', linestyle='dashed', alpha=0.9)
    ax[1].plot(data_achsh5[:, 5], data_achsh5[:, 7] * 0.5, label=r'CHSH, $\epsilon=0.5$',
               linewidth=2, color='navy', linestyle='dashed', alpha=0.9)
    ax[1].plot(data_msg[:, 5], data_msg[:, 7], label='MPG',
               linewidth=2, color='tab:red', marker='o', alpha=0.9)

    hds, lbs = ax[1].get_legend_handles_labels()
    ax[1].legend(hds[::-1], lbs[::-1], fontsize=20, fancybox=True)
    ax[0].set_xlabel(r'State visibility $\nu$', fontsize=25)
    ax[1].set_xlabel(r'Detection efficiency $\eta$', fontsize=25)
    ax[0].set_ylabel(r'Key rate $R$', fontsize=25)
    ax[0].set_xlim([0.94, 1.0003])
    ax[1].set_xlim([0.94, 1.0003])
    ax[0].set_ylim([-0.02, 1.01])
    ax[0].set_yticks([0, 0.5, 1], ['0', r'$\frac{\gamma}{2}$', r'$\gamma$'], fontsize=20)
    ax[0].set_xticks([0.94, 0.96, 0.98, 1], ['0.94', '0.96', '0.98', '1.00'], fontsize=20)
    ax[1].set_xticks([0.94, 0.96, 0.98, 1], ['0.94', '0.96', '0.98', '1.00'], fontsize=20)
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot(data_achsh1[:, 5], data_achsh1[:, 7] * 0.9, label=r'CHSH, $\epsilon=0.1$',
               linewidth=2, color='deepskyblue', linestyle='dashed', alpha=0.9)
    ax.plot(data_achsh2[:, 5], data_achsh2[:, 7] * 0.8, label=r'CHSH, $\epsilon=0.2$',
               linewidth=2, color='tab:blue', linestyle='dashed', alpha=0.9)
    ax.plot(data_achsh3[:, 5], data_achsh3[:, 7] * 0.7, label=r'CHSH, $\epsilon=0.3$',
               linewidth=2, color='royalblue', linestyle='dashed', alpha=0.9)
    ax.plot(data_achsh5[:, 5], data_achsh5[:, 7] * 0.5, label=r'CHSH, $\epsilon=0.5$',
               linewidth=2, color='navy', linestyle='dashed', alpha=0.9)
    ax.plot(data_msg[:, 5], data_msg[:, 7], label='MPG',
               linewidth=2, color='tab:red', marker='o', alpha=0.9)

    hds, lbs = ax.get_legend_handles_labels()
    ax.legend(hds[::-1], lbs[::-1], fontsize=20, fancybox=True)
    ax.set_xlabel(r'Detection efficiency $\eta$', fontsize=25)
    ax.set_ylabel(r'Key rate $R$', fontsize=25)
    ax.set_xlim([0.94, 1.0003])
    ax.set_ylim([-0.02, 1.01])
    ax.set_xticks([0.94, 0.96, 0.98, 1], ['0.94', '0.96', '0.98', '1.00'], fontsize=20)
    """

    plt.tight_layout()
    plt.show()

    return fig


def msg_ineq_vs_full():
    data_msg_full = np.genfromtxt('Data/MSG_full.csv', delimiter=',', skip_header=1)
    data_msg_ineq = np.genfromtxt('Data/MSG.csv', delimiter=',', skip_header=1)
    data_chsh5 = np.genfromtxt('Data/aCHSH_eps5.csv', delimiter=',', skip_header=1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(data_msg_full[:, -1], data_msg_full[:, 2], label='MPG full-statistic',
               linewidth=2, color='tab:green', alpha=0.9)
    ax[0].plot(data_msg_ineq[:, 0], data_msg_ineq[:, 1], label='MPG inequality',
               linewidth=2, color='tab:red', alpha=0.9)

    ax[1].plot(data_msg_full[:, 0], data_msg_full[:, 1], label='MPG full-statistic',
               linewidth=2, color='tab:green', alpha=0.9)
    ax[1].plot(data_msg_ineq[:, 2], data_msg_ineq[:, 4], label='MPG inequality',
               linewidth=2, color='tab:red', alpha=0.9)
    ax[1].plot(data_chsh5[:, 2], data_chsh5[:, 4] * 0.5, label=r'CHSH, $\epsilon=0.5$',
               linewidth=2, color='tab:blue', linestyle='dashed', alpha=0.9)

    ax[0].set_xlabel(r'Winning probability $\omega$', fontsize=20)
    ax[0].set_ylabel(r'$H(A|E)$', fontsize=20)
    ax[0].legend(fontsize=16)
    ax[0].set_xlim([0.88, 1.0003])
    ax[0].set_ylim([-0.01, 1.0003])
    ax[0].set_xticks([0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00],
                     ['0.94', '0.95', '0.96', '0.97', '0.98', '0.99', '1.00'], fontsize=16)
    ax[0].set_yticks([0, 0.5, 1], ['0.0', '0.5', '1.0'], fontsize=16)

    ax[1].set_xlabel(r'State visibility $\nu$', fontsize=20)
    ax[1].set_ylabel(r'Key rate $R$', fontsize=20)
    ax[1].legend(fontsize=16)
    ax[1].set_xlim([0.94, 1.0003])
    ax[1].set_ylim([-0.01, 1.0003])
    ax[1].set_yticks([0, 0.5, 1], ['0', r'$\frac{\gamma}{2}$', r'$\gamma$'], fontsize=16)
    ax[1].set_xticks([0.94, 0.96, 0.98, 1], ['0.94', '0.96', '0.98', '1.00'], fontsize=16)
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    ax.plot(data_msg_full[:, -1], data_msg_full[:, 2], label='MPG full-statistic',
            linewidth=2, color='tab:green', alpha=0.9)
    ax.plot(data_msg_ineq[:, 0], data_msg_ineq[:, 1], label='MPG inequality',
            linewidth=2, color='tab:red', alpha=0.9)
    ax.set_xlabel(r'Winning probability $\omega$', fontsize=20)
    ax.set_ylabel(r'$H(A|E)$', fontsize=20)
    ax.legend(fontsize=16)
    ax.set_xlim([0.88, 1.0003])
    ax.set_ylim([-0.01, 1.0003])
    ax.set_xticks([0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00],
                  ['0.94', '0.95', '0.96', '0.97', '0.98', '0.99', '1.00'], fontsize=16)
    ax.set_yticks([0, 0.5, 1], ['0.0', '0.5', '1.0'], fontsize=16)
    """
    fig.tight_layout()
    fig.show()
    return fig


if __name__ == '__main__':
    # Fig = msg_chsh()
    # Fig = Fig1_basic()
    Fig = msg_ineq_vs_full()
    # Fig.savefig('Fig_main_corrected.pdf')
    Fig.savefig('Fig_MPG_full_vs_ineq_corrected_panel1.pdf')
