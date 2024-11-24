from scipy.stats import beta

pd = 0.025
nd = 85
n = 3000
print(beta.cdf(pd, nd + 0.5, n - nd +0.5))