import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sim_rep = 100
iv_n = [1, 2, 3, 5, 9, 15]
n = 45
iv_l = len(iv_n)
alpha = 0.05
B = 100 #MUDAR AQUI PRA IR MAIS RAPIDO

print(np.array([n / iv for iv in iv_n]))

res = [None] * sim_rep

for z in range(sim_rep):
    res_z = [None] * iv_l

    for i in range(iv_l):
        np.random.seed((i + 1) + (z + 1))
        iv_i = iv_n[i]
        iv = np.random.standard_t(df=2, size=(n, iv_i))

        db = pd.DataFrame(iv, columns=[f"x{i}" for i in range(0, iv_i)])
        formula = f'y ~ {" + ".join(db.columns)}'

        res_i = pd.DataFrame(
            np.nan,
            index=range(0, B),
            columns=["simulation"] +
                    list(db.columns) +
                    [f"{col}.pval" for col in db.columns] +
                    ["f.test"]
        )

        for j in range(0, B):
            np.random.seed((i + 1) + (j + 1) * 10 + (z + 1))
            db["y"] = np.random.standard_t(df=2, size=n)

            lr_j = smf.ols(formula=formula, data=db).fit()
            fs_j = lr_j.fvalue
            fs_p = lr_j.f_pvalue

            res_j = [j] + list(lr_j.params[1:]) + list(lr_j.pvalues[1:]) + [fs_p]
            res_i.loc[j] = res_j

        res_i.insert(0, "iv_n", iv_i)
        res_z[i] = res_i

    res_z_mean = [np.mean(x["f.test"] < alpha) for x in res_z]
    res[z] = res_z_mean

res = pd.DataFrame(res, columns=iv_n)


print(res.mean())