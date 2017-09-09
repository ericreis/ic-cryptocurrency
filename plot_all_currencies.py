print(__doc__)

import sys
import ntpath
import glob
from subprocess import call

# get folder info from arguments
folder_path = sys.argv[1]

print(folder_path)

for full_filename in glob.glob(folder_path + "/" + "*_price.csv"):
    call(["python3", "scatter_matrix.py", full_filename])
    call(["python3", "correlation_matrix.py", full_filename])