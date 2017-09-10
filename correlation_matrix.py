print(__doc__)

import sys
import ntpath
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# get dataset info from arguments
dataset_path = sys.argv[1]
dataset_name = ntpath.basename(dataset_path).split(".")[0]

print("Dataset: ", dataset_name)

# import dataset as panda dataframe
df = pd.DataFrame.from_csv(dataset_path)
print("Shape: ", df.shape)
print("Columns: ", df.columns)

# rename columns
df.rename(columns=lambda x: "x" + str(np.where(df.columns == x)[0][0]), inplace=True)

corr = df.corr()
plt.figure(figsize=(12,12))
plt.matshow(corr, fignum=1)
plt.xticks(range(len(corr.columns)), corr.columns);
plt.yticks(range(len(corr.columns)), corr.columns);
plt.clim(-1, 1)
plt.colorbar()

plt.savefig("plots/correlation-matrix/" + dataset_name + ".png")
