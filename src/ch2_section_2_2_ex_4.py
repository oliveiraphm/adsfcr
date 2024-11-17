import numpy as np
from scipy.stats import rankdata, norm

np.random.seed(984)
x1 = np.random.normal(loc = 10, scale = 2, size = 500)
x2 = np.random.normal(loc = 20, scale = 3, size = 500)
x = np.concatenate((x1,x2))

def inverse_normal_transform(x, k = 3/8, method="average"):
    n = len(x)
    r = rankdata(a=x, method=method)
    xt = norm.ppf((r-k) / (n - 2 * k + 1))

    return xt

x_trans = inverse_normal_transform(x=x)
print([min(x_trans), np.mean(x_trans), np.median(x_trans), max(x_trans)])