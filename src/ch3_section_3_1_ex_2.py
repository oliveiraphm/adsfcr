import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

np.random.seed(123)

x1 = np.random.normal(loc = 5, scale = 2, size = 100)
x2 = np.random.normal(loc = 3, scale = 1, size = 100)

errors = np.random.normal(loc = 0, scale = 1, size = 100)

y = 0.5 * x1 - 0.7 * x2 + errors

db = pd.DataFrame({"y" : y, "x1": x1, "x2": x2})

model = sm.formula.ols(formula = 'y ~ x1 + x2', data = db).fit()

print(model.summary())

def ssq(beta_coef, X, Y):
    resid = Y - np.dot(X, beta_coef)
    opt_f = np.sum(resid ** 2)

    return opt_f

X = np.column_stack((np.repeat(1, db.shape[0]), db[["x1", "x2"]]))
Y = db["y"]
optim_const_lin = minimize(fun = ssq, 
                          x0 = np.zeros(X.shape[1]),
                          args = (X, Y),
                          bounds = ((None, None),
                                    (None, None),
                                    (0, None)),
                                    method = "L-BFGS-B"
                          )

beta_coef = optim_const_lin.x
names = ["Intercept", "x1", "x2"]
beta_coef = dict(zip(names, beta_coef))
print(beta_coef)