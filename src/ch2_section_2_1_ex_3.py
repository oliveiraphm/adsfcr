import numpy as np

loan_amount = np.array([10e3, 8.5e3, 2.5e3, 12.7e3, 5.6e3])
q3 = np.quantile(loan_amount, q=0.75)
q1 = np.quantile(loan_amount, q=0.25)
print(q3-q1)

def wtd_quantiles(x, w, probs):
    ord = np.argsort(x)
    x = x[ord]
    w = w[ord]
    cdf = (np.cumsum(w) - 0.5 * w) /np.sum(w)
    qs = np.interp(x = probs, xp = cdf, fp = x)
    return qs

weights = np.array([10, 30, 50, 20, 40])
q25_75 = wtd_quantiles(x=loan_amount,
                       w = weights,
                       probs = np.array([0.25, 0.75])
                       )
print(q25_75[1] - q25_75[0])