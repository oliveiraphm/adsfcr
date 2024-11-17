import numpy as np

np.random.seed(123)

x = np.random.normal(
    loc = 70, 
    scale = 5,
    size = 1000
)

print([np.mean(x), np.std(x)])

def standardize(x):
    x_s = (x - np.mean(x)) / np.std(x)
    return x_s

x_s = standardize(x = x)

print([np.mean(x_s), np.std(x_s)])