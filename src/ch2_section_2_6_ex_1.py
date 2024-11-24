import numpy as np
import pandas as pd 
from statsmodels.stats.proportion import proportions_chisquare
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm

# Set the seed for reproducibility
np.random.seed(991)

# Generate the 'defaults' array
x = np.random.binomial(n=1, p=0.07, size=10000)
defaults = pd.DataFrame({"x": x})
p0 = 0.065
print(proportions_chisquare(count = sum(defaults.x),
                      nobs = len(defaults.x),
                      value = p0)[:2])

p = np.mean(defaults.x)
n = len(defaults.x)
test_stat = (p- p0) / np.sqrt(p0 * (1 - p0) / n)
print(test_stat)

p_val = 2 * norm.cdf(x = -test_stat)
print(p_val)


np.random.seed(321)
rg1 = np.random.binomial(n=1, p=0.025, size=500)
rg2 = np.random.binomial(n=1, p=0.04, size=350)

ratings = pd.concat([
    pd.DataFrame({'x': rg1, 'rg': 'RG_1'}),
    pd.DataFrame({'x': rg2, 'rg': 'RG_2'})
])


proportions_ztest(count = [ sum(ratings.x[ratings.rg.isin(["RG_2"])]), 
                            sum(ratings.x[ratings.rg.isin(["RG_1"])])], 
                            nobs=[sum(ratings.rg.isin(["RG_2"])), 
                                 sum(ratings.rg.isin(["RG_1"]))],
                            alternative="larger")

rg1 = ratings.x[ratings.rg.isin(["RG_1"])]
p1 = np.mean(rg1)
n1 = len(rg1)
rg2 = ratings.x[ratings.rg.isin(["RG_2"])]
p2 = np.mean(rg2)
n2 = len(rg2)
pp = (p1 * n1 + p2 * n2) / (n1 + n2)
se = np.sqrt( pp * (1 - pp) * ((1 / n1) + (1 / n2)))
test_stat = (p2-p1) / se
print(test_stat)

p_val = 1 - norm.cdf(x = test_stat)
print(p_val)