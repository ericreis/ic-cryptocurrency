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

column_index = 0

x = df.as_matrix(columns=df.columns[column_index:column_index + 1])

test = pd.DataFrame(x)
test.plot()

plot_acf(x)

plt.show()

# plt.savefig("plots/autocorrelation/" + dataset_name + "-" + df.columns[column_index] + ".png")