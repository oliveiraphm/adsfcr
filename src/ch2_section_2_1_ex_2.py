import numpy as np

loan_amounts = np.array([10e3, 8.5e3, 2.5e3, 12.7e3, 5.6e3])
weights = np.array([10,30,50,20,40])

n = len(loan_amounts)
print(np.sum((loan_amounts-np.mean(loan_amounts))**2) / (n-1))

wm = np.average(loan_amounts, weights=weights)
nf = np.sum(weights)
print(np.sum(weights*(loan_amounts - wm)**2) / (nf-1))
print(np.std(loan_amounts, ddof=1))
print(np.sqrt(np.var(loan_amounts, ddof=1)))

