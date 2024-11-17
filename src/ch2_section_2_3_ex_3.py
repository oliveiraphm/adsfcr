import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


np.random.seed(991)
x1 = np.random.normal(loc = 5, scale = 2, size = 100)
x2 = np.random.normal(loc = 3, scale = 1, size = 100)
epsilon = np.random.normal(loc = 0, scale = 1, size = 100)

y = 0.5 * x1 - 0.7* x2 + epsilon

db = pd.DataFrame({"x1":x1, "x2":x2, "y":y})
model = sm.formula.ols(formula="y ~ x1 + x2", data=db).fit()

def bp(model, studentize):
    X = model.model.exog
    resid = model.resid

    sigma2 = np.sum(resid ** 2) / len(resid)
    n = len(resid)
    if studentize:
        f = resid ** 2 - sigma2
        het_reg = sm.OLS(f, X).fit()
        test_stat = n * np.sum(het_reg.fittedvalues ** 2) / np.sum(f ** 2)
    else:
        f = resid ** 2 / sigma2 -1
        het_reg = sm.OLS(f, X).fit()
        test_stat = 0.5 * np.sum(het_reg.fittedvalues ** 2)
    p_val = 1 - chi2.cdf(test_stat, het_reg.df_model)
    res = {"test_stat": test_stat, "p_val":p_val}
    return res

print(bp(model=model, studentize=True))

name = ["statistic", "p.val"]
test = sm.stats.het_breuschpagan(resid = model.resid, exog_het = model.model.exog, robust = True)
dict(zip(name, test[0:2]))