# -*- coding: utf-8 -*-
"""
Author: Zhen YZ
Date:
"""

from Mod_SDP import *
import csv
import time

Time = time.time()

Model = MSG_SDP(m=8, verbose=False, parallel=True)
Model.init()

print(f'Time: {(time.time() - Time) / 60: 2f} min')

for x, y in product(range(3), range(3)):
    FN = f'HAgE_Ineq_x{x}y{y}.csv'
    with open(FN, 'w') as o:
        csv.writer(o).writerow(['Val', 'Time', f'HAgE_{x}{y}'])

    Val = 1.
    while True:
        Time = time.time()
        Model.SDP.process_constraints(equalities=[],
                                      inequalities=[],
                                      momentequalities=[],
                                      momentinequalities=Model.constr_ineq(Val))
        Hage = Model.get_hage(x, y)

        Time = round((time.time() - Time) / 3600, 2)

        print(f'x: {x}, y: {y}, Value: {Val}, HAgE: {Hage}, Time: {Time} min')
        with open(FN, 'a') as o:
            csv.writer(o).writerow([Val, Time, Hage])

        if Hage < 1e-6:
            break

        Val -= 0.005
