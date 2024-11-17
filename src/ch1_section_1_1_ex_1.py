import numpy as np

annuity = 292.30
maturity = 18
ir_y = 0.0649
ir_m = ir_y / 12
npv = sum(annuity / (1+ir_m) ** np.arange(1, maturity+1))
print(npv)