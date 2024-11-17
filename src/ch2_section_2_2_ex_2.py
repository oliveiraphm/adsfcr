import numpy as np

np.random.seed(123)

x = np.random.normal(
    loc = 50,
    scale = 3,
    size = 1000
)

print([np.min(x), np.mean(x),np.max(x)])

def normalize(x, new_min, new_max):

    x_r = np.ptp(x)
    x_t = (x - np.min(x)) / x_r * (new_max-new_min) + new_min

    return x_t

x_n = normalize(x=x, new_min=0, new_max=10)

print([np.min(x_n), np.mean(x_n),np.max(x_n)])
