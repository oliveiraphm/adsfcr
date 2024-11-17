import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Simulation parameters
n = 45  # number of observations
iv = 2  # number of independent variables

# Simulate the x values
np.random.seed(123)
x = np.random.normal(size=(n, iv))  # Generate normal random numbers
x = pd.DataFrame(x, columns=[f"x{i+1}" for i in range(iv)])  # Create a DataFrame

# True regression parameters
intercept = 0
beta1 = 0.85
beta2 = -0.65

# Simulate the target (y)
y = intercept + beta1 * x["x1"] + beta2 * x["x2"]

# Store the inputs into a DataFrame
db = pd.DataFrame({"y": y, **x.to_dict(orient="list")})  # Combine y and x into a single DataFrame

#print(db)  # Display the first few rows of the DataFrame

alpha = 0.05
B = 1000
xw = np.array([0.60, 0.35])

columns = ["b1", "b2", "b1.pval", "b2.pval"]
res_unw = pd.DataFrame(index=range(B), columns = columns, dtype = float)
res_wts = res_unw.copy()

for i in range(B):

    np.random.seed((i+1) * 2)
    error_i = np.random.normal(loc=0, 
                               scale = np.exp(xw[0] * db['x1'] + xw[1]*db['x2']))
    error_i = error_i - np.mean(error_i)

    db["resid_i"] = np.log(error_i ** 2)
    h_i = np.exp(smf.ols(formula="resid_i ~ x1 + x2", data=db).fit().fittedvalues)
    weights_i = 1 / h_i

    db["y_sim"] = db["y"] + error_i
    lr_unw = smf.ols(formula="y_sim ~ x1 + x2", data = db).fit()
    lr_wts = sm.WLS.from_formula(formula = "y_sim ~ x1 + x2",
                                 weights = weights_i,
                                 data=db).fit()
    
    res_unw.loc[i, ] = lr_unw.params[1:].tolist() + lr_unw.pvalues[1:].tolist()
    res_wts.loc[i, ] = lr_wts.params[1:].tolist() + lr_wts.pvalues[1:].tolist()

{"beta1":0.85, "beta2":-0.65}
b12_ci_unw=res_unw[["b1","b2"]].quantile([alpha/ 2,0.5,1-alpha/ 2])
print(b12_ci_unw)

b12_ci_wts=res_wts[["b1","b2"]].quantile([alpha/ 2,0.5,1-alpha/ 2])
print(b12_ci_wts)

print((res_unw[["b1.pval","b2.pval"]]< alpha).mean())
print((res_wts[["b1.pval","b2.pval"]]< alpha).mean())


#git add .
#git commit -m "mensagem"
#git push