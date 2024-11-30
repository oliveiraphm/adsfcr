import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
tr = np.array([ [0.93, 0.05, 0.02],
                [0.05, 0.80, 0.15],
                [0.00, 0.0, 1.00]])

w = np.array([300, 80, 20])

print( np.average(tr[:, 2][:-1], weights = w[:-1])
)

target = 0.055

def zs_opt(zs, tr, w, target):

    tr_cs = np.cumsum(tr, axis=1)
    tr_in = norm.ppf(tr_cs)
    tr_in[tr_in == np.inf] = 15
    tr_in[tr_in == -np.inf] = -15
    tr_zs = tr_in + zs
    tr_cs = norm.cdf(tr_zs)
    tr_sc = np.column_stack((tr_cs[:,0], np.diff(tr_cs, axis =1 )))
    w_dr = np.average(tr_sc[:,2][:-1], weights = w[:-1])
    mm =(w_dr-target) ** 2
    
    return mm

zs =minimize(   fun= zs_opt,
                x0 = 0,
                args = (tr, w, target),
                method = "BFGS")

print(zs.x)

def tr_scaled(tr,zs):

    tr_cs = np.cumsum(tr, axis=1)
    tr_in = norm.ppf(tr_cs)
    tr_in[tr_in == np.inf] = 15
    tr_in[tr_in ==-np.inf] = -15
    tr_zs = tr_in + zs
    tr_cs = norm.cdf(tr_zs)
    tr_sc = np.column_stack( (tr_cs[:, 0], np.diff(tr_cs, axis = 1)))

    return tr_sc

tr_opt = tr_scaled(tr = tr, zs = zs.x)

print(np.round(tr_opt, 4))

print(np.average(tr_opt[:, 2][:-1], weights = w[:-1]))