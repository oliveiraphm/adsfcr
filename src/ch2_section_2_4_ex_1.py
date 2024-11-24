import numpy as np
import pandas as pd
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from scipy.stats import norm

np.random.seed(321)
x1 = np.random.normal(loc = 5, scale = 2, size = 100)
x2 = np.random.normal(loc = 3, scale = 1, size = 100)

y = random.choices([0, 1], k=100)

db = pd.DataFrame({"x1":x1, "x2":x2, "y":y})
model = smf.glm(  formula="y ~ x1 + x2"
                , data=db
                , family = sm.families.Binomial()
                ).fit()
model_summary = pd.DataFrame({
    "coefficients" : model.params,
    "std_errors" : model.bse.to_list(),
    "t-values" : model.tvalues.tolist(),
    "p-values" : model.pvalues.tolist()
})

def log_likelihood(beta_coef, X, y):

    fitted = np.exp(np.dot(X, beta_coef)) / (1 + np.exp(np.dot(X, beta_coef)))
    opt_f = -np.sum(y * np.log(fitted) + (1-y) * np.log(1-fitted))

    return opt_f

X = np.column_stack((np.repeat(1, db.shape[0]), db[["x1", "x2"]]))
y = db["y"]

opt_result = minimize(fun = log_likelihood,
                      x0 = np.zeros(X.shape[1]),
                      args = (X,y),
                      method = "L-BFGS-B"
                      )
beta_coef = opt_result.x
print(beta_coef)

p_hat = np.exp(np.dot(X, beta_coef)) / ( 1 + np.exp(np.dot(X, beta_coef)))
W = np.diag(p_hat * (1 - p_hat))

beta_se = np.sqrt(np.diag(np.linalg.inv(np.dot(np.dot(X.T, W), X))))
print(beta_se)

z_val = beta_coef / beta_se
print(z_val)

p_val = 2 * (1 - norm.cdf(x = np.abs(z_val)))
print(p_val)

def auc(predictions, observed):

    df = pd.DataFrame({"predictions" : predictions,
                       "observed": observed})
    df["prediction_rank"] = df.predictions.copy().rank()
    n0 = sum(df.observed == 0)
    n1 = sum(df.observed == 1)
    u = sum(df.prediction_rank[df.observed == 0]) - n0 * (n0 + 1) / 2
    res = 1 - u / n0 / n1

    return(res)

print(auc(predictions = model.predict(), observed = db["y"]))