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

with open("results/repeat_last/stdout_" + str(train_begin) + "_" + str(test_end) + ".txt", 'w') as f:
    sys.stdout = f

    print("Train = [", train_begin, ",", train_end, "]")
    print("Test = [", test_begin, ",", test_end, "]")

    targets = []
    predicted = []

    with open("data/bitcoin_dataset_sample.csv", "r") as btc_dataset:
        header = btc_dataset.readline()
        count = 0
        last_value = -1
        for line in btc_dataset.readlines():
            line = line.replace('\n', '')
            values = line.split(',')
            value = float(values[1])
            if count >= test_begin and count <= test_end:
                targets.append(value)
                predicted.append(last_value)
            last_value = value
            count += 1

    targets = np.array(targets, dtype=np.float32)
    predicted = np.array(predicted, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(targets, 'o', label="data")
    ax.plot(predicted, 'r--.', label="Repeat last")
    ax.legend(loc='best')
    fig.suptitle("Repeat Last Test " + str(test_begin) + "-" + str(test_end))
    fig.savefig("plots/repeat_last/test_" + str(train_begin) + "_" + str(test_end))

    # Errors
    errs = []
    var_errs = []
    for i in range(len(targets)):
        errs.append(abs(predicted[i] - targets[i])/ targets[i])
        var_errs.append((predicted[i] - targets[i - 1]) - (targets[i] - targets[i - 1]) / (targets[i] - targets[i - 1]))

    print("RMSE = ", rmse(predicted, targets))
    print("Abs Error: ", sum(errs) / len(errs))

    err_df = pd.DataFrame(errs)
    err_df.plot(title="Percentage error " + str(test_begin) + "-" + str(test_end))
    plt.savefig("plots/repeat_last/error_" + str(train_begin) + "_" + str(test_end))

    var_err_df = pd.DataFrame(var_errs)
    var_err_df.plot(title="Variance Error " + str(test_begin) + "-" + str(test_end))
    plt.savefig("plots/repeat_last/var_error_" + str(train_begin) + "_" + str(test_end))

    # plt.show()