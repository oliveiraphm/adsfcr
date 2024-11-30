import pandas as pd
import numpy as np

np.random.seed(4321)

# Simulate development sample risk factor
rf_dev = np.random.choice(['a', 'b', 'c', 'd'], size=int(1e4), replace=True)

# Simulate application sample risk factor
rf_app = np.random.choice(rf_dev, size=int(0.3 * len(rf_dev)), replace=False)

def psi(dev, app):

    tbl_dev = pd.DataFrame(dev.value_counts(normalize = True)).reset_index()
    tbl_app = pd.DataFrame(app.value_counts(normalize = True)).reset_index()
    tbl = pd.merge(
        left = tbl_dev,
        right = tbl_app,
        left_on = "dev",
        right_on = "app",
        how = "outer",
        sort = True
    )
    psi_value = np.sum((tbl["proportion_x"] - tbl["proportion_y"]) * 
                       np.log(tbl["proportion_x"] / tbl["proportion_y"]))
    
    return psi_value

print(psi(dev = rf_dev.dev, app=rf_app.app))