import numpy as np 
import pandas as pd

np.random.seed(991)

x = np.random.normal(
    loc = 50,
    scale = 3,
    size = 1000
)

x_edb = pd.cut(x=x, bins=4)

print(
   x_edb.value_counts().sort_index() 
)

x_qub = pd.qcut(x=x, q=4)

print(
   x_qub.value_counts().sort_index() 
)