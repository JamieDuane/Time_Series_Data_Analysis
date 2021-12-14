import math
import numpy as np
from scipy.optimize import Bounds,

class PARAMETER(object):
    def __init__(self, Xt, q):
        self.T = len(Xt[0,:])-1
        self.Xt = Xt
        self.d = len(Xt)
        self.q = q

    def RHO_BETA(self, C, q):
        '''
        :param C: some constant defined somewhere
        :param T: period
        :param d: dimension of time series data
        :param q: compression dimension
        :return: rho and beta for ADMM initialization
        '''
        rho = C*(math.log(self.T, 10)**1.5)*(math.log(self.d, 10)/self.T)**0.5
        beta = rho*self.d/(q**0.5)
        return rho, beta

    def S(self, w):
        M = self.C() * int(math.log(self.T, 10))
        res = 0
        for t in range(-M, M + 1):
            if t < 0:
                t = -t
            NtT = int((self.T - t) / (t + 1))
            R_hat = 0
            for k in range(NtT + 1):
                R_hat += np.dot(self.Xt[:, (k + 1) * t + k].reshape((self.T+1, 1)), self.Xt[:, k * t + k].reshape((1, self.T+1)))
            R_hat = R_hat / (NtT + 1)
            res += R_hat * complex(math.cos(2 * math.pi * w * t), -math.sin(2 * math.pi * w * t))
        return res

    def YITA(self):
        bounds = Bounds([0], [1])


    def rosen(self, w):
        S = self.S(w)
        e, v = np.linalg.eig(S)
        idx = e.argsort()[::-1]
        evq = e[idx[self.q-1]]
        evq_1 = e[idx[self.q]]
        return -(3*evq_1+evq)/(evq_1+3*evq)

    def S_HAT(self, C, s, q):
        yita = self.YITA()
        s_hat = C * max(4*q/(yita**(-0.5)-1)**(-2), 1)*s
        return s_hat

    def R_HAT(self, C, q, s):
        yita = self.YITA()
        A = min(((2*yita)**0.5)/4, (q*yita*(1-yita**0.5)/2)**0.5)
        r = ((q*(self.d**2)*(math.log(self.T, 10)**3)*math.log(self.d, 10)/self.T)**0.5) * (A-C*(math.log(self.T, 10)**1.5)*(math.log(self.d, 10)/self.T)**0.5)**(-1)
        r_hat = 4*(math.log(1/yita, 10)**(-1))*math.log(C*((yita/8)**0.5)*(math.log(self.T, 10))**1.5*(s*math.log(self.d, 10)/self.T)**0.5, 10)
        return r, r_hat

    def N(self, C, s):
        n = C*math.log(self.T, 10)**(-1.5)*(s*math.log(self.d, 10)/self.T)**(-0.5)
        return n

    def C(self):
        return 0
