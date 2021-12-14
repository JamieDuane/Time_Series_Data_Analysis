from initualizing_process import ADMMRelaxation
import numpy as np
import json, math
from parameter import PARAMETER

class CSOAP_algo(PARAMETER):
    def __init__(self, U, Xt):
        super().__init__(Xt)
        self.U = U
        with open('data/data.json', 'r') as f:
            data = json.load(f)
            self.Xt = np.array(data)
        self.T = len(self.Xt)-1

    def truncnate(self, V_dcmp, S_HAT):
        dim = len(V_dcmp)
        U = np.zeros(V_dcmp.shape)
        for i in range(dim):
            v = V_dcmp[i, :]
            idx = v.argsort()[::-1]
            for j in range(S_HAT):
                index = idx[j]
                U[i, index] = v[index]
        return U

    def thin_QR(self, V):
        v, R = np.linalg.qr(V)
        return v, R

    def soap(self, S, R, S_HAT):
        for t in range(R):
            V = np.dot(S, self.U)
            V_dcmp, R1 = self.thin_QR(V)
            U_hat = self.truncnate(V_dcmp, S_HAT)
            self.U, R2 = self.thin_QR(U_hat)

    def csoap(self, N, RHO, BETA, C, k, S_HAT):
        pi = ADMMRelaxation(cov=self.S(0, C), rho=RHO, beta=BETA, k=k)
        Uinit = pi.initialize(self.T)
        U_hat0 = self.truncnate(Uinit, S_HAT)
        Uinit = self.thin_QR(U_hat0)
        U = []
        U.append(self.soap(self.S(0, C), Uinit, S_HAT))
        for i in range(1, N):
            Ui_N = self.soap(self.S(k/N, C), U[-1], S_HAT)
            U.append(Ui_N)
        return U
