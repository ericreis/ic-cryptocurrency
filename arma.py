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

with open("results/arma/stdout_" + str(train_begin) + "_" + str(test_end) + ".txt", 'w') as f:
    sys.stdout = f

    print("Train = [", train_begin, ",", train_end, "]")
    print("Test = [", test_begin, ",", test_end, "]")

    date_train = []
    train = []

    date_test = []
    test = []

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
                train.append(math.log(float(values[1])))
            elif count >= test_begin and count <= test_end:
                date_splited = values[0].split('/')
                dt = datetime.date(int(date_splited[2]) + 2000, int(date_splited[0]), int(date_splited[1]))
                date_test.append(dt)
                test.append(math.log(float(values[1])))
            count += 1

    y_train = pd.Series(train)

    res = sm.tsa.ARMA(train, order=(10,10)).fit()
    print(res.summary())

    start = test_begin - train_begin
    end = test_end - train_begin

    predicted = res.predict(start=start, end=end)

    fig, ax = plt.subplots(figsize=(10,8))
    fig = res.plot_predict(start=start - 50, end=end, ax=ax)
    ax.plot(([None] * 50) + test, 'o', label="data")
    legend = ax.legend(loc='upper left')
    fig.suptitle("ARMA Test " + str(test_begin) + "-" + str(test_end))
    plt.savefig("plots/arma/future_test_" + str(train_begin) + "_" + str(test_end))

    period_well_predicted = 0
    future_test = []
    for i in range(len(test)):
        predicted[i] = math.exp(predicted[i])
        future_test.append(math.exp(test[i]))
        if (abs(predicted[i] - future_test[i]) / future_test[i]) > 0.05:
            break
        period_well_predicted += 1

    print("Predicted up to", period_well_predicted, "days with less than 5% error")

    predicted = []

    for i in range(end - start + 1):
        try:
            y_train = pd.Series(train, index=date_train)
            res = sm.tsa.ARMA(train, order=(10,10)).fit()
            p = res.predict(start=start + i, end=start + i)[0]
            predicted.append(math.exp(p))
        except Exception as e:
            predicted.append(None)
            print("On i =", i, ":", str(e))
        date_train.pop(0)
        date_train.append(date_test[i])
        train.pop(0)
        train.append(test[i])

    for i in range(len(test)):
        test[i] = math.exp(test[i])

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(test, 'o', label="data")
    ax.plot(predicted, 'r--.', label="ARMA")
    ax.legend(loc='best')
    fig.suptitle("ARMA Test " + str(test_begin) + "-" + str(test_end))
    fig.savefig("plots/arma/test_" + str(train_begin) + "_" + str(test_end))

    # Errors
    errs = []
    var_errs = []
    sum_errs = 0
    errs_count = 0
    mse = 0
    for i in range(len(test)):
        try:
            if math.isnan(predicted[i]) or math.isnan(test[i]):
                errs.append(None)
                var_errs.append(None)
                print("On i =", i, ": NaN")
            else:
                errs.append(abs(predicted[i] - test[i])/ test[i])
                var_errs.append((predicted[i] - test[i - 1]) - (test[i] - test[i - 1]) / (test[i] - test[i - 1]))
                sum_errs += abs(predicted[i] - test[i])/ test[i]
                mse += (predicted[i] - test[i]) ** 2
                errs_count += 1
        except Exception as e:
            errs.append(None)
            var_errs.append(None)
            print("On i =", i, ":", str(e))

    test = np.array(test, dtype=np.float32)
    predicted = np.array(predicted, dtype=np.float32)

    rmse = np.sqrt(mse/errs_count)

    print("RMSE = ", rmse)
    print("Abs Error: ", sum_errs / errs_count)

    err_df = pd.DataFrame(errs)
    err_df.plot(title="Percentage error " + str(test_begin) + "-" + str(test_end))
    plt.savefig("plots/arma/error_" + str(train_begin) + "_" + str(test_end))

    var_err_df = pd.DataFrame(var_errs)
    var_err_df.plot(title="Variance Error " + str(test_begin) + "-" + str(test_end))
    plt.savefig("plots/arma/var_error_" + str(train_begin) + "_" + str(test_end))

    # plt.show()