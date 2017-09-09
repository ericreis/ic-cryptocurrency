print(__doc__)

import sys
import ntpath
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



# get dataset info from arguments
dataset_path = sys.argv[1]
dataset_name = ntpath.basename(dataset_path).split(".")[0]

print(dataset_path)
print(dataset_name)

# import dataset as panda dataframe
df = pd.DataFrame.from_csv(dataset_path)
print(df.shape)
print(df.columns)

# rename columns
df.rename(columns=lambda x: "x" + str(np.where(df.columns == x)[0][0]), inplace=True)
print(df.columns)

# create the scatter matrix matplotlib object
pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal='kde')

# save plot and show
plt.savefig("plots/scatter-matrix/" + dataset_name + ".png")
# plt.show()
