from scipy.stats import binom

pd = 0.025
nd = 85
n = 3000
print(1 - binom.cdf(nd-1, n, pd))