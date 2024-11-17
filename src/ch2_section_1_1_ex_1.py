import numpy as np
import pandas as pd

loan_amounts = np.array([10e3, 8.5e3, 2.5e3, 12.7e3, 5.6e3])
print(np.sum(loan_amounts) / len(loan_amounts))
print(np.mean(loan_amounts))

weights = np.array([10, 30, 50, 20, 40])
print(np.sum(loan_amounts*weights) / np.sum(weights))
print(np.average(loan_amounts, weights=weights))