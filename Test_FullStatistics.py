# -*- coding: utf-8 -*-
"""
Author: Zhen YZ
Date: Mar 29, 2023

Compute the key rate of MPG-based protocol.
"""

from Mod_SDP import *
from Mod_Func import *
import csv
import time

Time = time.time()

""" 
pp = 0.125
ProbAB = [[[[pp, pp, 0, 0], [pp, pp, 0, 0], [0, 0, pp, pp], [0, 0, pp, pp]],  # P(ab|x=0,y=0)
           [[pp, pp, 0, 0], [0, 0, pp, pp], [pp, pp, 0, 0], [0, 0, pp, pp]],  # P(ab|x=0,y=1)
           [[pp, pp, 0, 0], [0, 0, pp, pp], [0, 0, pp, pp], [pp, pp, 0, 0]]  # P(ab|x=0,y=2)
           ],
          [[[pp, 0, pp, 0], [pp, 0, pp, 0], [0, pp, 0, pp], [0, pp, 0, pp]],  # P(ab|x=1,y=0)
           [[pp, 0, pp, 0], [0, pp, 0, pp], [pp, 0, pp, 0], [0, pp, 0, pp]],  # P(ab|x=1,y=1)
           [[pp, 0, pp, 0], [0, pp, 0, pp], [0, pp, 0, pp], [pp, 0, pp, 0]]  # P(ab|x=1,y=2)
           ],
          [[[0, pp, pp, 0], [0, pp, pp, 0], [pp, 0, 0, pp], [pp, 0, 0, pp]],  # P(ab|x=2,y=0)
           [[0, pp, pp, 0], [pp, 0, 0, pp], [0, pp, pp, 0], [pp, 0, 0, pp]],  # P(ab|x=2,y=1)
           [[0, pp, pp, 0], [pp, 0, 0, pp], [pp, 0, 0, pp], [0, pp, pp, 0]]  # P(ab|x=2,y=2)
           ]
          ]
ProbA = ProbB = [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
"""

Nv = 1
ProbAB, ProbA, ProbB = get_full_probs(np.pi/4, Nv)

Model = MSG_SDP(m=8, verbose=False, parallel=True)
Model.SDP.get_relaxation(level=2,
                         equalities=[],
                         inequalities=[],
                         momentequalities=Model.constr_prob_full(ProbAB, ProbA, ProbB),
                         momentinequalities=[],
                         objective=Model.msg_obj_j(0, 0, 0),
                         substitutions=Model.get_subs(),
                         extramonomials=Model.extra_monomials())

print(f'Time: {(time.time() - Time) / 60: 2f} min')

for x, y in product(range(3), range(3)):
    FN = f'HAgE_FullStatistics_x{x}y{y}.csv'
    with open(FN, 'w') as o:
        csv.writer(o).writerow(['Nv', 'Time', f'HAgE_{x}{y}'])

    Nv = 1
    while True:
        Time = time.time()
        ProbAB, ProbA, ProbB = get_full_probs(np.pi / 4, Nv)
        Model.SDP.process_constraints(equalities=[],
                                      inequalities=[],
                                      momentequalities=Model.constr_prob_full(ProbAB, ProbA, ProbB),
                                      momentinequalities=[])
        Hage = Model.get_hage(x, y)

        Time = round((time.time() - Time) / 3600, 2)

        print(f'x: {x}, y: {y}, nv: {Nv}, HAgE: {Hage}, Time: {Time} min')
        with open(FN, 'a') as o:
            csv.writer(o).writerow([Nv, Time, Hage])

        if Hage < 1e-6:
            break

        Nv -= 0.005


Data_mpg_ineq = np.genfromtxt('Data/MSG.csv', delimiter=',', skip_header=1)
Data_mpg_full = np.genfromtxt('Data/MSG_full.csv', delimiter=',', skip_header=1)
Data_chsh_ineq = np.genfromtxt('Data/aCHSH_eps5.csv', delimiter=',', skip_header=1)


