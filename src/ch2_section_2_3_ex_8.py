import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

n = 45 
B = 30 #1000
beta1 = 0.50
beta2 = 0.30
rho = [0, 0.25, 0.50, 0.75, 0.95]

def sim_betas(n, beta1, beta2, rho):

    cm = np.array([
        [1, beta1, beta2],
        [beta1, 1, rho],
        [beta2, rho, 1]
    ])

    db = np.random.multivariate_normal(
        mean = [0,0,0],
        cov = cm,
        size = n
    )

    db = pd.DataFrame(data = db, columns = ["y", "x1", "x2"])

    ols_reg = smf.ols(formula = "y ~ x1 + x2", data = db).fit()

    betas = ols_reg.params[1:]

    return betas

res = [None] * B

for i in range(1, B+1):
    np.random.seed(i)
    sim_results = [sim_betas(n, beta1, beta2, rho_i) for rho_i in rho]
    res_i = pd.DataFrame(data = sim_results)
    res_i["simulation"] = i
    res_i["rho"] = rho
    res[i-1] = res_i

res = pd.concat(objs = res, ignore_index = True)

print(res.groupby("rho").agg(
    x1_low =("x1", lambda x:np.percentile(a=x,q =2.5)),
    x1_cnt =("x1", lambda x:np.percentile(a=x,q =50)),
    x1_upp =("x1", lambda x:np.percentile(a=x,q =97.5)),
    x2_low =("x2", lambda x:np.percentile(a=x,q =2.5)),
    x2_cnt =("x2", lambda x:np.percentile(a=x,q =50)),
    x2_upp =("x2", lambda x:np.percentile(a=x,q =97.5)),
    x1_std =("x1", lambda x:np.std(a=x,ddof =1)),
    x2_std =("x2", lambda x:np.std(a=x,ddof= 1))
 ).reset_index())