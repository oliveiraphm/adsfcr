import numpy as np
ir = np.array([5, 3.99, 7.99, 4.25, 7.99, 7.99])
print(np.median(ir))

weights = np.array([10, 30, 50, 20, 40, 50])
ord = np.argsort(ir)
ir = ir[ord]
weights = weights[ord]

cdf = (np.cumsum(weights) -0.5*weights ) / np.sum(weights)
print(
    np.interp(
        x = 0.50,
        xp = cdf, 
        fp = ir
    )
)