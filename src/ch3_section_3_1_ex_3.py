import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize
import random

np.random.seed(321)
x1 = np.random.normal(loc = 5, scale = 2, size = 100)
x2 = np.random.normal(loc = 3, scale = 1, size = 100)

y = random.choices([0, 1], k=100)

db = pd.DataFrame({"x1":x1, "x2":x2, "y":y})
model = smf.glm(  formula="y ~ x1 + x2"
                , data=db
                , family = sm.families.Binomial()
                ).fit()

print(model.summary())

def log_likelihood(beta_coef, X, Y):

    lin_pred = np.dot(X, beta_coef)
    opt_f = -np.sum(Y * (lin_pred - np.log(1 + np.exp(lin_pred))) + (1 - Y) * (-np.log(1 + np.exp(lin_pred))))

    return opt_f

X = np.column_stack((np.repeat(1, db.shape[0]), db[["x1", "x2"]]))
Y = db["y"]
optim_const_log = minimize(fun = log_likelihood, 
                          x0 = np.zeros(X.shape[1]),
                          args = (X, Y),
                          bounds = ((None, None),
                                    (None, None),
                                    (0, None)),
                                    method = "L-BFGS-B"
                          )

beta_coef = optim_const_log.x
names = ["Intercept", "x1", "x2"]
beta_coef = dict(zip(names, beta_coef))
print(beta_coef)