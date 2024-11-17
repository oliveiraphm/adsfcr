import numpy as np
import pandas as pd
from scipy import stats

ir = np.array([5, 3.99, 7.99, 4.25, 7.99, 7.99])
unique_elements, counts = np.unique(ar = ir, return_counts = True)
print(unique_elements[counts == np.max(counts)])
print(stats.mode(a = ir))
weights = np.array([10, 30, 50, 20, 40, 50])
data = pd.DataFrame({"ir":ir, "weights":weights})
ir_wtbl = data.groupby("ir")["weights"].sum()
print(ir_wtbl[ir_wtbl == ir_wtbl.max()])