from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import matplotlib.pyplot as plt
import sys
import pickle

train_interval = sys.argv[1].split(',')
train_begin = int(train_interval[0])
train_end = int(train_interval[1])

test_interval = sys.argv[2].split(',')
test_begin = int(test_interval[0])
test_end = int(test_interval[1])

ds = SupervisedDataSet(50, 1)

test_params = []
test_targets = []

with open("data/btc_dataset_sample.csv", "r") as btc_dataset:
    header = btc_dataset.readline()
    count = 0
    for line in btc_dataset.readlines():
        line = line.replace('\n', '')
        values = line.split(';')
        if count >= train_begin and count <= train_end:    
            ds.addSample(values[:-1], values[-1])
        elif count >= test_begin and count <= test_end:
            test_params.append(values[:-1])
            test_targets.append(values[-1])
        count += 1

n = buildNetwork(ds.indim, 1000, ds.outdim, recurrent=True)
t = BackpropTrainer(n, learningrate=0.001, momentum=0, verbose=True)
t.trainUntilConvergence(ds, 10000)
t.testOnData(verbose=False)

fileObject = open('bpnn', 'w')
pickle.dump(n, fileObject)
fileObject.close()

# fileObject = open('bpnn','r')
# n = pickle.load(fileObject)

predicted = []

for i in range(len(test_params)): 
    predicted.append(n.activate(test_params[i]))
    print("Predicted:", predicted[-1])
    print("Real", test_targets[i])

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(test_targets, 'o', label="data")
ax.plot(predicted, 'r--.', label="bpnn")
ax.legend(loc='best')

plt.show()
