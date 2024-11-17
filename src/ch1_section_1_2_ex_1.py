import pandas as pd
import numpy as np

amount = 5000
maturity = 18
ir_y = 0.0649
ir_m = ir_y / 12

def create_repayment_plan(p, r, m):

    annuity = p * r / (1 - (1+r) ** (-m))
    rp = pd.DataFrame({
                           "month": [*range(1, m+1)],
                           "remaining_principal":[None]*m,
                           "monthly_principal":[None]*m,
                           "monthly_interest":[None]*m,
                           "annuity":[annuity]*m

    })
    rp.loc[0, "remaining_principal"] = p
    for i in rp.index:
        if (i == rp.index[-1]):
            rp.loc[i, "monthly_principal"] = rp.remaining_principal[i]
            rp.loc[i, "monthly_interest"] = rp.annuity[i] - rp.monthly_principal[i]
        else:
            rp.loc[i, "monthly_interest"] = rp.remaining_principal[i] * r
            rp.loc[i, "monthly_principal"] = rp.annuity[i] - rp.monthly_interest[i]
            rp.loc[i+1, "remaining_principal"] = rp.remaining_principal[i] - rp.monthly_principal[i]
    return rp
    
rp = create_repayment_plan(p = amount, r = ir_m, m = maturity)
print(rp) 

npv = sum(rp["annuity"] / (1 + ir_m) ** np.arange(1, maturity + 1))
print(npv)

annuity_checking = amount * ir_m / (1 - (1+ir_m) ** (-maturity))
print(annuity_checking)

n = 5
print((annuity_checking - amount * ir_m ) * ((1 + ir_m) ** (n-1)))
print(annuity_checking - (annuity_checking - amount * ir_m) * ((1+ir_m) ** (n-1)))
print(amount - sum((annuity_checking - amount * ir_m) * ((1 + ir_m) **(np.arange(1, n) - 1))))

annuity = amount * ir_m / (1 - (1 + ir_m) ** (-maturity))
monthly_principal = (annuity_checking - amount * ir_m) * ((1+ir_m)**(np.arange(1,maturity+1)-1))
remaining_principal = amount - np.cumsum(np.concatenate(([0], monthly_principal[:-1])))
monthly_interest = annuity_checking - monthly_principal
rp = pd.DataFrame({
                       "month": np.arange(1, maturity+1),
                       "remaining_principal": remaining_principal,
                       "monthly_principal": monthly_principal,
                       "monthly_interest": monthly_interest,
                       "annuity": np.repeat(annuity_checking, maturity)

})
print(rp)