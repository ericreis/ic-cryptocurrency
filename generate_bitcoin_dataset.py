import sys
import ntpath

import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import math
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf


# get dataset info from arguments
dataset_path = sys.argv[1]
dataset_name = ntpath.basename(dataset_path).split(".")[0]
print("Dataset: ", dataset_name)

# import dataset as panda dataframe
df = pd.DataFrame.from_csv(dataset_path)
print("Shape: ", df.shape)
print("Columns: ", df.columns)

params = [{
        "column_id": 9,
        "column_name": "btc_hash_rate",
        "column_lag": 20
    }, {
        "column_id": 11,
        "column_name": "btc_miners_revenue",
        "column_lag": 10
    }, {
        "column_id": 22,
        "column_name": "btc_estimated_transaction_volume_usd",
        "column_lag": 10
    }]

for param in params:
    column_name = param["column_name"]
    param["values"] = df[column_name].values

print("Params: ", params)

target = {
    "column_id": 0,
    "column_name": "btc_market_price",
    "column_lag": 10
}

column_name = target["column_name"]
target["values"] = df[column_name].values

print("Target: ", target)

initial_row = 20
with open("data/btc_dataset.csv", "w") as btc_file:
    # Write header
    column_lag = target["column_lag"]
    for i in range(1, column_lag + 1):
        btc_file.write(str(target["column_name"]) + "(t-" + str(i) + ");")
    for param in params:
        column_lag = param["column_lag"]
        for i in range(1, column_lag + 1):
            btc_file.write(str(param["column_name"]) + "(t-" + str(i) + ");")
    btc_file.write(str(target["column_name"]) + "(t)\n")

    # Fill with data
    for row in range(initial_row, len(target["values"])):
        column_lag = target["column_lag"]
        for i in range(1, column_lag + 1):
            btc_file.write(str(target["values"][row - i]) + ";")
        for param in params:
            column_lag = param["column_lag"]
            for i in range(1, column_lag + 1):
                btc_file.write(str(param["values"][row - i]) + ";")
        btc_file.write(str(target["values"][row]) + "\n")
