import numpy as np

amount = 8000
maturity = 8 * 12
ir_y = 0.0799
ir_m = ir_y / 12
annuity_init = amount * ir_m / (1 - (1 + ir_m) ** (-maturity))
annuity_init

print(annuity_init)

mm = 25
ml = 3

rp = amount - sum( (annuity_init - amount * ir_m) * ((1+ir_m)**(np.arange(1,mm)-1)) )
print(rp) 

ai = rp * ((1+ ir_m) ** ml-1)
print(ai) 

np = rp + ai 
print(np)

annuity_new = np * ir_m / (1 - (1+ir_m) ** -(maturity - mm + 1))
print(annuity_new)
print(annuity_new - annuity_init)