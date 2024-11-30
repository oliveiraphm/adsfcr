import numpy as np
from scipy.stats import norm

def cv(x):
    k = len(x)
    x_tot = x.sum()
    cv_terms = (x / x_tot - 1 / k) ** 2
    cv = (k * cv_terms.sum()) ** 0.5

    return cv

def cv_test(x_curr, x_init):

    cv_curr = cv(x = x_curr)
    cv_init = cv(x = x_init)
    k = len(x_curr)
    z_stat = (k -1) ** 0.5 * (cv_curr - cv_init) / \
    (cv_curr**2 * (0.5 + cv_curr**2)) ** 0.5

    p_value = 1 - norm.cdf(z_stat)

    return(p_value)

x_curr = np.array([100,200,300])
x_init = np.array([95, 205, 290])

print(cv_test(x_curr = x_curr, x_init = x_init))