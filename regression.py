from __future__ import print_function
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std


params = []
targets = []

with open("data/btc_dataset.csv", "r") as btc_dataset:
    header = btc_dataset.readline()
    params = []
    targets = []
    for line in btc_dataset.readlines():
        line = line.replace('\n', '')
        values = line.split(';')
        params.append(values[:1])
        targets.append(values[-1])

params = np.array(params, dtype=np.float32)
targets = np.array(targets, dtype=np.float32)

print("params = ", params)
print("targets = ", targets)

res = sm.OLS(targets, params).fit()
print(res.summary())

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(targets, 'o', label="data")
ax.plot(res.fittedvalues, 'r--.', label="OLS")
ax.legend(loc='best')

plt.show()