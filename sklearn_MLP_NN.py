from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import pickle
import sys


def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))


train_interval = sys.argv[1].split(',')
train_begin = int(train_interval[0])
train_end = int(train_interval[1])

test_interval = sys.argv[2].split(',')
test_begin = int(test_interval[0])
test_end = int(test_interval[1])

with open("results/mlp_nn/stdout_" + str(train_begin) + "_" + str(test_end) + ".txt", 'w') as f:
    sys.stdout = f

    print("Train = [", train_begin, ",", train_end, "]")
    print("Test = [", test_begin, ",", test_end, "]")

    train_params = []
    train_targets = []

    test_params = []
    test_targets = []

    with open("data/btc_dataset_sample.csv", "r") as btc_dataset:
        header = btc_dataset.readline()
        count = 0
        for line in btc_dataset.readlines():
            line = line.replace('\n', '')
            values = line.split(';')
            if count >= train_begin and count <= train_end:    
                train_params.append(values[:-1])
                train_targets.append(values[-1])
            elif count >= test_begin and count <= test_end:
                test_params.append(values[:-1])
                test_targets.append(values[-1])
            count += 1

    mlp = MLPRegressor(hidden_layer_sizes=(20), activation="relu", solver="lbfgs", alpha=0.001, batch_size="auto", learning_rate="constant", max_iter=12000)

    train_params = np.array(train_params, dtype=np.float64)
    train_targets = np.array(train_targets, dtype=np.float64)
    test_params = np.array(test_params, dtype=np.float64)
    test_targets = np.array(test_targets, dtype=np.float64)

    mlp_model = mlp.fit(train_params, train_targets)

    predicted = mlp.predict(test_params)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(test_targets, 'o', label="target")
    ax.plot(predicted, 'r--.', label="predicted")
    ax.legend(loc='best')
    fig.suptitle("MLP NN Test " + str(test_begin) + "-" + str(test_end))
    fig.savefig("plots/mlp_nn/test_" + str(train_begin) + "_" + str(test_end))

    # Errors
    errs = []
    var_errs = []
    for i in range(len(test_targets)):
        errs.append(abs(predicted[i] - test_targets[i])/ test_targets[i])
        var_errs.append((predicted[i] - test_targets[i - 1]) - (test_targets[i] - test_targets[i - 1]) / (test_targets[i] - test_targets[i - 1]))

    print("RMSE = ", rmse(predicted, test_targets))
    print("Abs Error: ", sum(errs) / len(errs))

    err_df = pd.DataFrame(errs)
    err_df.plot(title="Percentage error " + str(test_begin) + "-" + str(test_end))
    plt.savefig("plots/mlp_nn/error_" + str(train_begin) + "_" + str(test_end))

    var_err_df = pd.DataFrame(var_errs)
    var_err_df.plot(title="Variance Error " + str(test_begin) + "-" + str(test_end))
    plt.savefig("plots/mlp_nn/var_error_" + str(train_begin) + "_" + str(test_end))

    # plt.show()
