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
df = pd.DataFrame.from_csv(dataset_path, sep=';')
print("Shape: ", df.shape)
print("Columns: ", df.columns)

x = df["btc_market_price(t)"].values

price_df = pd.DataFrame(x)
price_df.plot()

plt.show()