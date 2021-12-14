import json
import numpy as np

with open('data/data.json', 'r') as f:
    data = json.load(f)
    Xt = np.array(data)

R = []
X0 = Xt[:, 0]
for i in range(200):
    Xi = Xt[:, i]
    Ri = np.cov(X0, Xi)
    R.append(Ri[0][1])

# sample cross-covarance matrice
R_hat = []
for t in range(200):
    NtT = int((199-t)/(t+1))
    s = 0
    for k in range(NtT+1):
        #print(t, k, NtT)
        s += np.dot(Xt[:,(k+1)*t+k].reshape((1000,1)), Xt[:, k*t+k].reshape((1, 1000)))
    R_hat.append(s/(NtT+1))
print(R_hat)

import math
def get_S(Xt, w, C, T):
    M = C* int(math.log(T, 10))
    res = 0
    for t in range(-M, M+1):
        if t < 0:
            t = -t
        NtT = int((T-t)/(t+1))
        R_hat = 0
        for k in range(NtT+1):
            R_hat += np.dot(Xt[:, (k+1)*t+k].reshape((1000, 1)), Xt[:, k*t+k].reshape((1,1000)))
        R_hat = R_hat/(NtT+1)
        res += R_hat*complex(math.cos(2*math.pi*w*t), -math.sin(2*math.pi*w*t))
    return res

