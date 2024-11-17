import numpy as np

loan_amount = np.array([10e3, 8.5e3, 2.5e3, 12.7e3, 5.6e3])
print(max(loan_amount)-min(loan_amount))
print(np.ptp(loan_amount))