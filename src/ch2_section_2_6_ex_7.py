import pandas as pd
from scipy.stats import chi2

rs =pd.DataFrame({"rating_grade":["RG1","RG2","RG3","RG4","RG5"],
                  "no":[47,95,68,53,37],
                  "nb":[3,20,17,24,28]
 })


rs["odr"] =rs["nb"]/ rs["no"]

rs["pd"]= [0.03065976,0.11608515,0.29069078,0.55144457,0.76484545]

k =len(rs)

nb =rs["nb"]

no =rs["no"]

pdp= rs["pd"]

hl =((nb-no *pdp) **2 /(no* pdp*(1-pdp))).sum()

p_value= chi2.sf(hl,df= k)
res=pd.DataFrame({"test_statistic":[hl],
                  "p_value":[p_value],
                  "df":[k]
 })
 
print(res)