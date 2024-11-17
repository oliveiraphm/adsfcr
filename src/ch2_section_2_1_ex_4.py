import numpy as np

loan_amounts = np.array([10e3, 8.5e3, 2.5e3, 12.7e3, 5.6e3])
weights = np.array([10,30,50,20,40])

xm = np.mean(loan_amounts)
n = len(loan_amounts)
print(np.sum(np.abs(loan_amounts - xm)) / (n-1))

wm = np.average(loan_amounts, weights=weights)
nw = np.sum(weights)
print(np.sum(weights * np.abs(loan_amounts - wm)) / (nw - 1))

