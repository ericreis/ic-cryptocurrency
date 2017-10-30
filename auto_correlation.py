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

columns_ids = list(map(lambda x: int(x), sys.argv[2].split(',')))
print("Columns: ", columns_ids)

# import dataset as panda dataframe
df = pd.DataFrame.from_csv(dataset_path)
print("Shape: ", df.shape)
print("Columns: ", df.columns)

for columns_id in columns_ids:
    # x = df.as_matrix(columns=df.columns[columns_id:columns_id + 1])
    x = df[df.columns[columns_id]].values
    fig = plot_acf(x)
    fig.suptitle(df.columns[columns_id] + " Autocorrelation")

plt.show()