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

REGRESSION_PLOTS_FOLDER = "plots/regression/"

with open("results/regression/stdout_" + str(train_begin) + "_" + str(test_end) + ".txt", 'w') as f:
    sys.stdout = f

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

    res = sm.OLS(targets, params).fit()
    print(res.summary())

    # Training plot
    # fig, ax = plt.subplots(figsize=(8,6))

    # ax.plot(targets, 'o', label="data")
    # ax.plot(res.fittedvalues, 'r--.', label="OLS")
    # ax.legend(loc='best')
    # fig.suptitle("training " + str(train_begin) + "-" + str(train_end))
    # fig.savefig(REGRESSION_PLOTS_FOLDER + "train" + str(train_begin) + "_" + str(train_end) + "_" + str(test_begin) + "_" + str(train_end))

    params = np.array(test["params"], dtype=np.float32)
    targets = np.array(test["targets"], dtype=np.float32)

    predicted = res.predict(params)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(targets, 'o', label="data")
    ax.plot(predicted, 'r--.', label="OLS")
    ax.legend(loc='best')
    fig.suptitle("OLS Regression Test " + str(test_begin) + "-" + str(test_end))
    fig.savefig(REGRESSION_PLOTS_FOLDER + "test_" + str(train_begin) + "_" + str(test_end))

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
    plt.savefig(REGRESSION_PLOTS_FOLDER + "error_" + str(train_begin) + "_" + str(test_end))

    var_err_df = pd.DataFrame(var_errs)
    var_err_df.plot(title="Variance Error " + str(test_begin) + "-" + str(test_end))
    plt.savefig(REGRESSION_PLOTS_FOLDER + "var_error_" + str(train_begin) + "_" + str(test_end))
