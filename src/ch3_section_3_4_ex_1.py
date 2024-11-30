import pandas as pd
import numpy as np
from scipy.optimize import minimize

rs =pd.DataFrame({"rating": [f"RG_{i}" for i in range(1,9)],
 "no": [100, 250, 400, 750, 700, 300 ,100 ,50],
 "dr":[0.003, 0.01, 0.025 ,0.03 ,0.045 ,0.08 , 0.1 ,0.13]
 })
 #currentportfoliodefaultrate

print(np.average(rs["dr"],weights=rs["no"]))


ct = 0.047

def calib_opt(dr, w, ct):

    log_odds = np.log(dr / (1 -dr))

    def opt_f(x, lo, w, ct):

        a = x[0]
        b = x[1]
        df = pd.DataFrame({"lo": pd.Series(lo), "w":w})
        df.dropna(inplace = True)
        lo = df.lo.copy()
        w = df.w.copy()
        pd_inverse = np.exp(a + b * lo) / (1 + np.exp(a + b *lo))
        opt = (sum(w * pd_inverse / (sum(w))) - ct) ** 2

        return opt
    
    lo_ab = minimize(opt_f,
                     x0 = [0,1],
                     args = (log_odds, 
                             w, 
                             ct),
                     method = "BFGS"
                     )
    a = lo_ab.x[0]
    b = lo_ab.x[1]
    params = pd.DataFrame({
        "a": [a],
        "b" : [b]
    })
    pd_calib = np.exp(a + b * log_odds) / (1 + np.exp(1 + b * log_odds))
    keys = ["ab", "calibrated"]
    res = dict(zip(keys, [params, pd_calib]))

    return(res)

pd_opt = calib_opt(dr = rs["dr"], w = rs["no"], ct = ct)
print(pd_opt)

print( np.average(pd_opt.get("calibrated"), weights=rs["no"]))