import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t


np.random.seed(123)
x1 = np.random.normal(loc = 5, scale = 2, size = 100)
x2 = np.random.normal(loc = 3, scale = 1, size = 100)
epsilon = np.random.normal(loc = 0, scale = 1, size = 100)

y = 0.5 * x1 - 0.7* x2 + epsilon

db = pd.DataFrame({"x1":x1, "x2":x2, "y":y})

model = sm.formula.ols(formula="y ~ x1 + x2", data=db).fit()
model_summary = pd.DataFrame({
    "coefficients" : model.params,
    "std_errors" : model.bse,
    "t-values" : model.tvalues,
    "p-values" : model.pvalues
})

print(model_summary)

X = sm.add_constant(np.column_stack((x1, x2)))

beta_coef = np.linalg.inv(X.T @ X) @ X.T @ y 
print(beta_coef)

n = len(y)
k = X.shape[1]
y_hat = np.dot(X, beta_coef)
sse = np.sum((y - y_hat) ** 2)
rse = sse / (n-k)
var_cov_matrix = rse * np.linalg.inv(X.T @ X)
beta_se = np.sqrt(np.diag(var_cov_matrix))
print(beta_se)

t_val = beta_coef / beta_se
print(t_val)

p_val = (1 - t.cdf(np.abs(t_val), n-k)) * 2
print(p_val)