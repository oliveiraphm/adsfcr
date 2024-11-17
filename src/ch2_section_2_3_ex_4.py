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


def vif(model):
    X = model.model.exog

    iv = model.model.exog_names[1:]
    iv_l = len(iv)
    if iv_l <= 1:
        res = pd.DataFrame({"dv":[None], "vif":[0]})
        return res
    vif_data = {"dv": [], "vif": [] }
    for i in range(iv_l):
        dv_i = iv[i]
        iv_i = [x for x in iv if x not in dv_i]
        frm = f"{dv_i} ~ {' + '.join(iv_i)}"
        model_i = smf.ols(formula = frm, data = db).fit()
        r2_i = model_i.rsquared

        vif_i = 1 / (1-r2_i)
        vif_data["dv"].append(dv_i)
        vif_data["vif"].append(vif_i)

    return pd.DataFrame(vif_data)

print(vif(model = model))

X = db[["x1", "x2"]]
X = X.assign(const = 1)
vif_res = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
print(vif_res[:1])