# Computation details
___

## Feasibility of different levels

|  level   |  m  | SDP variables | constrs | value  | Time  |
|:--------:|:---:|:-------------:|:-------:|:------:|:-----:|
|    2     |  5  |     90099     |  29133  | 1.9892 | 78min |
| 2+$A_0Z$ |  3  |     95265     |  32655  | 1.9699 | 47min |
| 2+$ABZ$  |  3  |    575127     | 209277  |   x    |   x   |


* MOSEK does not have automatic dualizer for the problem; the primal problem is solved.
* The bounded and unbounded seem have no difference, and the bounded cost more time.
* Maruto and SCC have the same performance.
* Preferred choice: m=8, Level=2+A0Z, bounded=F.


