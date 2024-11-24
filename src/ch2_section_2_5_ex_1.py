import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf

# Set the seed for reproducibility
np.random.seed(123)

# Number of observations
N = 50

# Generate AR(1) process for gdp
ar_gdp = np.array([1, -0.65])  # AR coefficient: 0.65
ma = np.array([1])  # MA coefficient: 0
arma_gdp = ArmaProcess(ar_gdp, ma)
gdp = arma_gdp.generate_sample(nsample=N)

# Generate unemployment variable
unemployment = 0.05 * gdp + np.random.normal(size=N)

# Generate AR(1) process for wage
ar_wage = np.array([1, -0.75])  # AR coefficient: 0.75
arma_wage = ArmaProcess(ar_wage, ma)
wage = arma_wage.generate_sample(nsample=N)

# Dependent variable (odr)
odr = -0.35 * gdp + 0.30 * unemployment - 0.10 * wage + np.random.normal(size=N)

# Store data in a DataFrame
db = pd.DataFrame({
    'odr': odr,
    'gdp': gdp,
    'unemployment': unemployment,
    'wage': wage
})


std_db = (  db.iloc[:, 1:] - np.mean(db.iloc[:, 1:], axis = 0)) /  \
            np.std(db.iloc[:, 1:], axis = 0, ddof = 1)

np.apply_along_axis(lambda x : [np.mean(x), np.std(x, ddof = 1)],
                    axis = 0,
                    arr = std_db)

pca = PCA(n_components = 3, svd_solver = "full")
pca_res = pd.DataFrame(pca.fit_transform(std_db))
pca_res.head()

cov_mat = np.cov(std_db, rowvar = False)
eigen_val, eigen_vec = np.linalg.eig(cov_mat)
sorted_indices = np.argsort(eigen_val)[::1]
eigen_val = eigen_val[sorted_indices]
eigen_vec = eigen_vec[:, sorted_indices]

var_expl = eigen_val / np.sum(eigen_val)

trans_data = pd.DataFrame(np.dot(std_db, eigen_vec))

db["pca_1"] = pca_res[0]

ols_p = smf.ols(formula = "odr ~ pca_1", data=db).fit()
print(ols_p.params)

dr_e = {"gdp": "-", "unemployment": "+", "wage": "-"}

dr_o = np.sign(ols_p.params["pca_1"] * pca.components_[0])
dr_o = np.where(dr_o == 1, "+", "-")

dr_c = pd.DataFrame(
     {    "EXPECTED": dr_e.values(),
          "OBSERVED": dr_o
     },
     index = dr_e.keys()
)

print(dr_c)