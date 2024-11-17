import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import jarque_bera, shapiro, chi2
import matplotlib.pyplot as plt



np.random.seed(991)
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

resid = model.resid
print(shapiro(x = resid))

print(jarque_bera(x = resid))

plt.hist(resid,bins="auto",alpha=0.7,rwidth=0.85)
plt.grid(axis="y",alpha=0.75)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of the regression residuals")
plt.show()