import numpy as np

def skewness(x, w):
    
    wm = np.average(x, weights=w)
    std = np.sqrt(np.sum(w * (x - wm) ** 2) / np.sum(w))
    nom = np.sum(w * ((x - wm) / std) ** 3)
    den = np.sum(w)
    res = nom / den

    return res

def kurtosis(x, w):

    wm = np.average(x, weights=w)
    std = np.sqrt(np.sum(w * (x - wm) ** 2) / np.sum(w))
    nom = np.sum(w * ((x - wm) / std) ** 4)
    den = np.sum(w)
    res = nom / den

    return res

loan_amounts = np.array([10e3, 8.5e3, 2.5e3, 12.7e3, 5.6e3])
weights = np.array([10,30,50,20,40])

print(
skewness( x = loan_amounts,
         w = np.ones(len(loan_amounts)))
)

print(
skewness( x = loan_amounts,
         w = weights)
)

print(
kurtosis( x = loan_amounts,
         w = np.ones(len(loan_amounts)))
)

print(
kurtosis( x = loan_amounts,
         w = weights)
)