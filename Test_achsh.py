
from Mod_Func import *

Data = np.genfromtxt('Data/aCHSH_eps5.csv', delimiter=',', skip_header=1)
Eps = 0.5
Mu = np.arctan(Eps/(1-Eps))
Diff_nu, Diff_eta = 0., 0.
for _K in range(Data.shape[0]):
    Corr_nu = correl_2qubit(np.pi/4, [0., np.pi/2], [Mu, -Mu], 1., Data[_K, 2])
    Diff_nu += abs((1 - Eps) / 2 * (Corr_nu[0] + Corr_nu[1]) + Eps / 2 * (Corr_nu[2] - Corr_nu[3]) - Data[_K, 0])
    Corr_eta = correl_2qubit(np.pi / 4, [0., np.pi / 2], [Mu, -Mu], Data[_K, 5], 1.)
    Diff_eta += abs((1 - Eps) / 2 * (Corr_eta[0] + Corr_eta[1]) + Eps / 2 * (Corr_eta[2] - Corr_eta[3]) - Data[_K, 0])

print(Diff_nu, Diff_eta)