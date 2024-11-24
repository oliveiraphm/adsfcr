import numpy as np
import pandas as pd
import scipy.stats as stats
# Set the seed for reproducibility
np.random.seed(3)

# Generate random samples for x and y
x = np.random.normal(loc=2, scale=2, size=50)   # Mean=2, SD=2, n=50
y = np.random.normal(loc=1, scale=1, size=100)  # Mean=1, SD=1, n=100

# Combine x and y into a single DataFrame with a sample indicator
db = pd.DataFrame({
    'value': np.concatenate([x, y]),
    'sample': ['x'] * 50 + ['y'] * 100  # Labels for x and y
})

stats.ttest_ind(    a =db[db["sample"].isin(["x"])]["value"] ,
                    b =db[db["sample"].isin(["y"])]["value"] ,
                    alternative="two-sided",
                    equal_var =False
                )

def tt_2is(x,y,alternative,alpha):
    n_x= len(x)
    n_y = len(y)
   
    x_bar = np.mean(x)
    y_bar = np.mean(y)

    se_x = np.std(x, ddof = 1) / np.sqrt(n_x)
    se_y = np.std(y, ddof = 1) / np.sqrt(n_y)
 
    se_c = np.sqrt(se_x**2 + se_y**2)

    stat = (x_bar- y_bar) / se_c
    df = se_c**4 / ((se_x ** 4 / (n_x - 1)) + (se_y ** 4 / (n_y- 1)))

    if alternative == "less":
        p_value = stats.t.cdf(stat, df=df)
    elif alternative == "two-sided":
        p_value = 2 * (1- stats.t.cdf(np.abs(stat), df=df))
    elif alternative == "greater":
        p_value = 1- stats.t.cdf(stat, df=df)

    tr = p_value < alpha
 
    res = {
        "n_x": n_x,
        "n_y": n_y,
        "x_bar": x_bar,
        "y_bar": y_bar,
        "se_x": se_x,
        "se_y": se_y,
        "se_c": se_c,
        "stat": stat,
        "df": df,
        "p_value": p_value,
        "H1": tr
    }
 
    return res

print(
pd.DataFrame([tt_2is(x = db[db["sample"].isin(["x"])]["value"], 
                     y = db[db["sample"].isin(["y"])]["value"],
                     alternative = "two-sided",
                     alpha = 0.05)])
)