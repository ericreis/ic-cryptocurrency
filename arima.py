from __future__ import print_function
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.metrics import mean_squared_error
import math
import sys
from statsmodels.tsa.arima_model import ARIMA
import datetime

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

train_interval = sys.argv[1].split(',')
train_begin = int(train_interval[0])
train_end = int(train_interval[1])

test_interval = sys.argv[2].split(',')
test_begin = int(test_interval[0])
test_end = int(test_interval[1])

print("Train = [", train_begin, ",", train_end, "]")
print("Test = [", test_begin, ",", test_end, "]")

date_train = []
train = []

date_test = []
test = [None] * 50

with open("data/bitcoin_dataset_sample.csv", "r") as btc_dataset:
    header = btc_dataset.readline()
    count = 0
    for line in btc_dataset.readlines():
        line = line.replace('\n', '')
        values = line.split(',')
        if count >= train_begin and count <= train_end:
            date_splited = values[0].split('/')
            dt = datetime.date(int(date_splited[2]) + 2000, int(date_splited[0]), int(date_splited[1]))
            date_train.append(dt)
            train.append(float(values[1]))
        elif count >= test_begin and count <= test_end:
            date_splited = values[0].split('/')
            dt = datetime.date(int(date_splited[2]) + 2000, int(date_splited[0]), int(date_splited[1]))
            date_test.append(dt)
            test.append(float(values[1]))
        count += 1

y_train = pd.Series(train, index=date_train)

print(y_train.tail())

res = sm.tsa.ARMA(train, order=(10,10)).fit()
print(res.summary())

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(train, 'o', label="data")
ax.plot(res.fittedvalues, 'r--.', label="OLS")
ax.legend(loc='best')
fig.suptitle("training " + str(train_begin) + "-" + str(test_end))

fig, ax = plt.subplots(figsize=(10,8))
start = test_begin - train_begin
end = test_end - train_begin
predicted = res.predict(start=start, end=end)
fig = res.plot_predict(start=start - 50, end=end, ax=ax)
legend = ax.legend(loc='upper left')

ax.plot(test, 'o', label="data")
# ax.plot(predicted, 'r--.', label="OLS")
# ax.legend(loc='best')
# fig.suptitle("testing " + str(test_begin) + "-" + str(test_end))

# # print("RMSE = ", rmse(predicted, targets))
# # errs = []
# # abs_errs = []
# # var_err = []
# # for i in range(len(targets)):
# #     errs.append((predicted[i] - targets[i]) / targets[i])
# #     abs_errs.append(abs(predicted[i] - targets[i])/ targets[i])
# #     var_err.append((predicted[i] - targets[i - 1]) - (targets[i] - targets[i - 1]) / (targets[i] - targets[i - 1]))

# # print("Abs Error: ", sum(abs_errs) / len(abs_errs))

# # # err_df = pd.DataFrame(errs)
# # # err_df.plot(title="Measurement Error " + str(test_begin) + "-" + str(test_end))
# # # plt.savefig(REGRESSION_PLOTS_FOLDER + "measurement_error_train_" + str(train_begin) + "_" + str(train_end) + "_" + str(test_begin) + "_" + str(train_end))

# # abs_err_df = pd.DataFrame(abs_errs)
# # abs_err_df.plot(title="Absolute Measurement Error " + str(test_begin) + "-" + str(test_end))
# # plt.savefig(REGRESSION_PLOTS_FOLDER + "absolute_measuremente_error_train_" + str(train_begin) + "_" + str(train_end) + "_" + str(test_begin) + "_" + str(train_end))

# # var_err_df = pd.DataFrame(var_err)
# # var_err_df.plot(title="Variance Error " + str(test_begin) + "-" + str(test_end))
# # plt.savefig(REGRESSION_PLOTS_FOLDER + "variance_error_train_" + str(train_begin) + "_" + str(train_end) + "_" + str(test_begin) + "_" + str(train_end))

plt.show()