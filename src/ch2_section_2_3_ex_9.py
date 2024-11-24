import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Simulation parameters
n = 45  # number of observations
iv = 2  # number of independent variables

# Simulate the x
np.random.seed(123)
x = np.random.normal(size=(n, iv))
x_df = pd.DataFrame(x, columns=[f"x{i+1}" for i in range(iv)])

# True regression parameters
intercept = 0
beta1 = 0.50
beta2 = -0.45

# Simulate the target
y = intercept + beta1 * x_df["x1"] + beta2 * x_df["x2"]

# Store the inputs in a dataframe
db = pd.concat([pd.Series(y, name="y"), x_df], axis=1)

n = db.shape[0]
alpha = 0.05
phi = 0.70
B = 50 #10000

columns = ["b1", "b2", "b1.p", "b2.p", "b1.p.hac", "b2.p.hac"]

res = pd.DataFrame(index = range(1, B+1), columns = columns)
arparams = np.array([phi])
ar = np.r_[1, -arparams]

for i in range(1, B + 1):
    np.random.seed(i*3)
    error_i = sm.tsa.ArmaProcess(ar = ar).generate_sample(nsample = n)
    error_i = error_i - np.mean(error_i)

    db["y_sim"] = db["y"] + error_i

    lr = smf.ols(formula = "y_sim ~ x1 + x2", data = db).fit()

    est_p_i = lr.params.values[1:]
    p_values_i = lr.pvalues.values[1:]
    est_p = np.r_[est_p_i, p_values_i]
    res.loc[i, ["b1", "b2", "b1.p", "b2.p"]] = np.r_[est_p_i, p_values_i]
    hac = lr.get_robustcov_results(cov_type = "HAC", maxlags=1)
    res.loc[i, ["b1.p.hac", "b2.p.hac"]] = hac.pvalues[1:]

ci_bs = res[["b1", "b2"]].apply(lambda x : np.percentile( a = x, q = [alpha*100/2, 50, (1-alpha/2)*100]) )

print(ci_bs)