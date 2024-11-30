import numpy as np
from scipy.optimize import root_scalar

amount = 10e3
ir_y = 0.0599
ir_m = ir_y / 12
maturity = 60 
cost = 150

annuity = amount * ir_m / (1 - (1 + ir_m) ** (-maturity))
print(annuity)

print(
np.sum(np.repeat(annuity, maturity) / 
       np.cumprod(1+np.repeat(ir_m, maturity)))
)

def eir_opt(ir, amount, annuity, maturity, cost):

    cf = np.repeat(annuity, maturity)
    df = np.cumprod(1 + np.repeat(ir, maturity))

    return np.sum(cf / df) - (amount - cost)

eir = root_scalar(f = eir_opt, 
                  args = (amount, annuity, maturity, cost),
                  bracket = [0,1],
                  x0 = ir_m).root

print(eir)

print(eir * 12)

print(np.sum(np.repeat(annuity, maturity) / np.cumprod(1 + np.repeat(eir, maturity))))