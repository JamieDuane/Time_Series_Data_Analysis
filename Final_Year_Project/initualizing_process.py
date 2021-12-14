import numpy as np

class ADMMRelaxation(object):
    def __init__(self, cov, rho, beta, k):
        self.cov = cov
        self.d = len(cov)
        self.phi = np.zeros((self.d,self.d))
        self.pi = np.zeros((self.d, self.d))
        self.shita = np.zeros((self.d,self.d))
        self.rho = rho
        self.beta = beta
        self.k = k

    def get_ei_val_vec(self, mtrx):
        e, v = np.linalg.eig(mtrx)
        idx = e.argsort()[::-1]
        return e[idx], v[idx]

    def piforward(self): # Projection
        mtrx=self.phi+np.true_divide(self.shita, self.beta)+np.true_divide(self.cov, self.beta)
        ei_val, Q = self.get_ei_val_vec(mtrx)
        ei_dcmp = np.diag(ei_val)
        d = len(mtrx)
        # minimize 1/2 xTPx + px
        # subject to Gx <= h; Ax = b; lb <= x <= ub
        from qpsolvers import solve_qp
        Pc = np.diag([2.]*d)
        pl = np.dot(np.array([-2.]), ei_val.reshape((1,2))).reshape((d,))
        G = np.array([0.]*d)
        h = np.array([0.])
        A = np.array([1.]*d)
        b = np.array([self.k])
        lb = np.array([0.]*d)
        ub = np.array([1.]*d)
        sol = solve_qp(Pc, pl, G, h, A, b, lb=lb, ub=ub)
        # Retrive Q NewEig Q-1:
        self.pi = np.dot(np.dot(Q, np.diag(sol)), np.linalg.inv(Q))

    #phit = np.array([[1,3],[2,6]])
    #shitat = np.array([[2,7],[3,8]])
    #cov = np.array([[2,3],[8,20]])
    #beta = 2
    #print(piforward(phit, shitat, cov, 2))

    def phiforward(self): # Soft Thresholding
        for i in range(self.d):
            for j in range(self.d):
                check = self.pi[i, j]-self.shita[i, j]/self.beta
                if abs(check) <= self.rho/self.beta:
                    self.phi[i, j] = 0
                else:
                    sign = 1 if check > 0 else -1 if check < 0 else 0
                    self.phi[i, j] = sign * (abs(check)-self.rho/self.beta)

    def shitaforward(self):
        self.shita = np.substract(self.shita, self.rho * np.subtract(self.pi, self.phi))

    def initialize(self, T):
        res = np.zeros((self.d, self.d))
        for _ in range(T):
            self.piforward()
            self.phiforward()
            self.shitaforward()
            res = np.add(res, self.pi)
        pi = np.true_divide(res, T)
        ei_val, ei_vec = np.linalg.eig(pi)
        return ei_vec[:, :self.k]