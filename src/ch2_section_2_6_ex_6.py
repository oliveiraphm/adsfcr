import numpy as np
import pandas as pd
from scipy.stats import f
from itertools import combinations
from statsmodels.stats.libqsturng import psturng

np.random.seed(991)
n = 300
x = np.random.uniform(low=0.0, high=1.0, size=n)
g = np.random.choice([1, 2, 3], size=n, replace=True)
db = pd.DataFrame({'x': x, 'g': g})

def welch_anova(x, g, alpha):
    K = len(np.unique(g))
    g_u = np.unique(g)
    n_i = np.array([len(x[g == group]) for group in g_u])
    xm_i = np.array([np.mean(x[g == group]) for group in g_u])
    s2_i = np.array([np.var(x[g == group], ddof=1) for group in g_u])
    w_i = n_i / s2_i
    w = np.sum(w_i)
    xm_w = np.sum(w_i * xm_i/w)
    num = np.sum(w_i * (xm_i - xm_w) ** 2) / (K - 1)
    den = 1 + (2 * (K-2) / (K**2 - 1)) * \
    np.sum((1 / (n_i - 1)) * (1 - w_i / w) ** 2)

    stat = num / den
    df1 = K - 1
    df2 = (K**2 - 1) / (3 * np.sum((1/(n_i - 1)) * (1 - w_i/w)**2))
    p_value = 1 - f.cdf(stat, df1, df2)
    tr = p_value < alpha
    res = {
        "stat" : stat,
        "df1" : df1,
        "df2" : df2,
        "p.value" : p_value,
        "H1" : tr
    }
    return res

print(pd.DataFrame([welch_anova(x = db["x"], g = db["g"], alpha=0.05)]))


def gh_test(x,g,alpha):
    g_u= np.sort(np.unique(g))
    n_i= np.array([len(x[g == group]) for group in g_u])
    xm_i= np.array([np.mean(x[g == group])for group in g_u])
    s2_i= np.array([np.var(x[g == group],ddof= 1) for group in g_u])
    pwc= list(combinations(g_u,2))
    pwc = [(x- 1, y- 1) for x, y in pwc]
    res = []
    for pair in pwc:
        md = xm_i[pair[0]] - xm_i[pair[1]]
        se = np.sqrt(s2_i[pair[0]] / n_i[pair[0]] + s2_i[pair[1]] / n_i[pair[1]])
        stat = np.abs(md) / se
        nmeans = len(g_u)
        df = (s2_i[pair[0]] / n_i[pair[0]] + s2_i[pair[1]] / n_i[pair[1]])**2 / \
             ((s2_i[pair[0]] / n_i[pair[0]])**2 / (n_i[pair[0]]) +
             (s2_i[pair[1]] / n_i[pair[1]])**2 / (n_i[pair[1]]))
        p_value = psturng( q = stat * np.sqrt(2),
                           r = nmeans,
                           v = df
                           )
        tr = p_value < alpha
        res_i = pd.DataFrame({
            "pair": [f"{pair[0] + 1}-{pair[1] + 1}"],
            "mean.difference": [md],
            "standard.error": [se],
            "stat": [stat],
            "nmeans": [nmeans],
            "df": [df],
            "p.value": p_value,
            "H1": tr
        })
    res.append(res_i)
    res = pd.concat(res, ignore_index = True)
    return res


print(gh_test(x =db["x"], g =db["g"], alpha =0.05))