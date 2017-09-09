print(__doc__)

import sys
import ntpath
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import glob

# get folder info from arguments
folder_path = sys.argv[1]
fodler_name = ntpath.basename(folder_path).split(".")[0]

print(folder_path)
print(fodler_name)

dic = {}
headers = []

for full_filename in glob.glob(folder_path + "/" + "*_price.csv"):
    filename = ntpath.basename(full_filename).split(".")[0]
    with open(full_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        count = 0
        for row in reader:
            if count == 0:
                headers += [filename.replace("_price", "") + "_" + column.replace(" ", "_") for column in row]
                dic = {**dic, **dict.fromkeys([filename.replace("_price", "") + "_" + column.replace(" ", "_") for column in row])}
                print(dic)
            count += 1
        print()
