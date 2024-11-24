import numpy as np
import pandas as pd
import scipy.stats as stats


# Set the seed for reproducibility
np.random.seed(1)

# Generate random samples from a normal distribution
x = np.random.normal(loc=0, scale=1, size=300)

# Save the data to a CSV file
db = pd.DataFrame({'x': x})

stats.ttest_1samp(a = db["x"], popmean=0)

def tt_1s(x, mu, alternative, alpha):
    n = len(x)
    x_bar = np.mean(x)
    se_bar = np.std(x, ddof = 1) / np.sqrt(n)
    stat = (x_bar - mu) / se_bar

    if alternative == "less":
        p_value = stats.t.cdf(stat, df = n - 1)
    elif alternative == "two-sided":
        p_value = 2 * (1 - stats.t.cdf(np.abs(stat), df = n - 1))
    elif alternative == "greater":
        p_value = 1 - stats.t.cdf(stat, df = n - 1)

    tr = p_value < alpha

    res = {
        "n": n,
        "x_bar": x_bar,
        "se_bar": se_bar,
        "stat" : stat,
        "df": n - 1,
        "p_value": p_value,
        "H1" : tr
    }

    return res

print(pd.DataFrame([tt_1s(x=db["x"], alternative= "two-sided", mu =0, alpha =0.05)]))



np.random.seed(2)

# Generate random samples for x and y
x = np.random.normal(loc=2, scale=2, size=100)  # Mean=2, SD=2, n=100
y = np.random.normal(loc=1, scale=1, size=100)  # Mean=1, SD=1, n=100

# Combine x and y into a DataFrame and save to a CSV file
db = pd.DataFrame({'x': x, 'y': y})

print(stats.ttest_rel(a =db["x"], b =db["y"], alternative= "greater"))
print(pd.DataFrame([tt_1s(x=(db["x"]-db["y"]), alternative= "greater", mu =0, alpha =0.05)]))