import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


np.random.seed(991)
x1 = np.random.normal(loc = 5, scale = 2, size = 100)
x2 = np.random.normal(loc = 3, scale = 1, size = 100)
epsilon = np.random.normal(loc = 0, scale = 1, size = 100)

y = 0.5 * x1 - 0.7* x2 + epsilon

db = pd.DataFrame({"x1":x1, "x2":x2, "y":y})
model = smf.ols(formula="y ~ x1 + x2", data=db).fit()
model_summary = pd.DataFrame({
    "coefficients" : model.params,
    "std_errors" : model.bse,
    "t-values" : model.tvalues,
    "p-values" : model.pvalues
})

print(model_summary)

resid = np.array(model.resid)

def dw_stat(resid):
    n = len(resid)
    lag = 1
    test_stat = np.sum((resid[lag:n]-resid[0:(n -lag)])**2) / np.sum(resid**2)
    res = {"test_stat":test_stat}

    return res

print(dw_stat(resid=resid))