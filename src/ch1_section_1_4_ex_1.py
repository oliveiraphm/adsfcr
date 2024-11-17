import numpy as np

amount = 50000
maturity = 96
ir_y = 0.0499
ir_m = ir_y / 12

annuity_init = amount * ir_m / (1 - (1 + ir_m)**-(maturity))
print(annuity_init)

mr = 25

rp = amount - sum( (annuity_init - amount * ir_m) * (( 1 + ir_m)**(np.arange(1,mr)-1)))
print(rp)

annuity_new = annuity_init * (1-0.15)
print(annuity_new)

cf = np.repeat(a=annuity_new, repeats=maturity)

npv_cf = cf / np.cumprod(1 + np.repeat( a= ir_m, repeats = maturity))

npv_cf_cs = np.cumsum(npv_cf)

period_indx = np.where(npv_cf_cs <= rp)
maturity_new = period_indx[0][len(period_indx[0])-1]
print(maturity_new+1)
print(maturity_new - (maturity - mr - 1) + 1)

npv_maturity_new = sum(cf[0:(maturity_new+1)] / np.cumprod(1+np.repeat(a = ir_m, repeats = maturity_new+1)))
print(npv_maturity_new)