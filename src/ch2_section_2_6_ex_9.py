import numpy as np
#simulate number of observations
no = np.array([100, 200, 300, 230, 80])
#calculate hhi
hhi = np.sum((no / np.sum(no))**2)
print(hhi)