from __future__ import print_function
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.metrics import mean_squared_error
import math
import sys

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

train_interval = sys.argv[1].split(',')
train_begin = int(train_interval[0])
train_end = int(train_interval[1])

test_interval = sys.argv[2].split(',')
test_begin = int(test_interval[0])
test_end = int(test_interval[1])

REGRESSION_PLOTS_FOLDER = "plots/markov_autoregression/"

print("Train = [", train_begin, ",", train_end, "]")
print("Test = [", test_begin, ",", test_end, "]")

train = {
    "params": [],
    "targets": []
}

test = {
    "params": [],
    "targets": []
}

with open("data/btc_dataset_sample.csv", "r") as btc_dataset:
    header = btc_dataset.readline()
    count = 0
    for line in btc_dataset.readlines():
        line = line.replace('\n', '')
        values = line.split(';')
        if count >= train_begin and count <= train_end:
            train["params"].append(values[:-1])
            train["targets"].append(values[-1])
        elif count >= test_begin and count <= test_end:
            test["params"].append(values[:-1])
            test["targets"].append(values[-1])
        count += 1

params = np.array(train["params"], dtype=np.float32)
targets = np.array(train["targets"], dtype=np.float32)

# print("training params = ", params)
# print("training targets = ", targets)

res = sm.tsa.MarkovAutoregression(targets, k_regimes=2, order=4, switching_ar=False).fit()
print(res.summary())

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(targets, 'o', label="data")
ax.plot(res.fittedvalues, 'r--.', label="OLS")
ax.legend(loc='best')
fig.suptitle("training " + str(train_begin) + "-" + str(test_end))
fig.savefig(REGRESSION_PLOTS_FOLDER + "training_train_" + str(train_begin) + "_" + str(train_end) + "_" + str(test_begin) + "_" + str(train_end))

params = np.array(test["params"], dtype=np.float32)
targets = np.array(test["targets"], dtype=np.float32)

# print("testing params = ", params)
# print("testing targets = ", targets)

predicted = res.predict(params)

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(targets, 'o', label="data")
ax.plot(predicted, 'r--.', label="OLS")
ax.legend(loc='best')
fig.suptitle("testing " + str(test_begin) + "-" + str(test_end))
fig.savefig(REGRESSION_PLOTS_FOLDER + "testing_train_" + str(train_begin) + "_" + str(train_end) + "_" + str(test_begin) + "_" + str(train_end))

print("RMSE = ", rmse(predicted, targets))
errs = []
abs_errs = []
var_err = []
for i in range(len(targets)):
    errs.append((predicted[i] - targets[i]) / targets[i])
    abs_errs.append(abs(predicted[i] - targets[i])/ targets[i])
    var_err.append((predicted[i] - targets[i - 1]) - (targets[i] - targets[i - 1]) / (targets[i] - targets[i - 1]))

print("Abs Error: ", sum(abs_errs) / len(abs_errs))

err_df = pd.DataFrame(errs)
err_df.plot(title="Measurement Error " + str(test_begin) + "-" + str(test_end))
plt.savefig(REGRESSION_PLOTS_FOLDER + "measurement_error_train_" + str(train_begin) + "_" + str(train_end) + "_" + str(test_begin) + "_" + str(train_end))

abs_err_df = pd.DataFrame(abs_errs)
abs_err_df.plot(title="Absolute Measurement Error " + str(test_begin) + "-" + str(test_end))
plt.savefig(REGRESSION_PLOTS_FOLDER + "absolute_measuremente_error_train_" + str(train_begin) + "_" + str(train_end) + "_" + str(test_begin) + "_" + str(train_end))

var_err_df = pd.DataFrame(var_err)
var_err_df.plot(title="Variance Error " + str(test_begin) + "-" + str(test_end))
plt.savefig(REGRESSION_PLOTS_FOLDER + "variance_error_train_" + str(train_begin) + "_" + str(train_end) + "_" + str(test_begin) + "_" + str(train_end))

# plt.show()