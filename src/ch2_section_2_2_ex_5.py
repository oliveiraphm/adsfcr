import numpy as np
import pandas as pd

data = pd.DataFrame({"category":["A", "B", "A", "C", "B"]})
data["ordinal_encoding "] = data["category"].astype("category").cat.codes + 1
print(data)

mm = pd.get_dummies(data = data["category"],
                    prefix = "category")*1
print(mm)

def nde(categories):

    uc = np.sort(np.unique(categories))
    ndv = [pd.DataFrame()] * (len(uc) - 1)

    for i in range(1, len(uc)):
        uc_i = uc[:i]
        uc_r = uc[~np.isin(uc, uc_i)]
        ndv_i = pd.DataFrame(np.where(np.isin(categories, uc_i), 0, 1))
        ndv_i.columns = [f'category_{"".join(uc_i)}_vs_{"".join(uc_r)}']
        ndv[i - 1] = ndv_i

    ndv = pd.concat(ndv, axis = 1)

    return ndv

print(nde(categories = data["category"]))

np.random.seed(123)

data = pd.DataFrame({
    "target": np.random.normal(loc=10, scale=2, size=7),
    "category": ["A", "A", "B", "A", "C", "B", "C"]
})

data["category_avg"] = data.groupby("category")["target"].transform("mean")

print(data)

np.random.seed(123)

data = pd.DataFrame({
    "target": np.random.binomial(n=1, p=0.05, size=1000),
    "category": np.random.choice(list("ABC"), size=1000, replace=True)
})

woe_tbl = data.groupby("category").agg(
    no = ("target", "size"),
    ng = ("target", lambda x: (x==0).sum()),
    nb = ("target", "sum")
).reset_index()

woe_tbl["dr"] = woe_tbl["nb"] / woe_tbl["no"]
so = woe_tbl["no"].sum()
sg = woe_tbl["ng"].sum()
sb = woe_tbl["nb"].sum()
woe_tbl["dist_g"] = woe_tbl["ng"] / sg
woe_tbl["dist_b"] = woe_tbl["nb"] / sb
woe_tbl["woe"] = np.log(woe_tbl["dist_g"] / woe_tbl["dist_b"])

woe_vec = woe_tbl.set_index("category")["woe"].to_dict()
data["woe"] = data["category"].map(woe_vec)
print(data.head())