if [ "$1" == "all" ] || [ "$1" == "regression" ]; then
    # REGRESSION
    python3 regression.py 0,550 551,600
    python3 regression.py 600,1150 1151,1200
    python3 regression.py 800,1200 1201,1250

elif [ "$1" == "all" ] || [ "$1" == "mlp_nn" ]; then
    # NEURAL NETWORK
    python3 sklearn_MLP_NN.py 0,550 551,600
    python3 sklearn_MLP_NN.py 600,1150 1151,1200
    python3 sklearn_MLP_NN.py 800,1200 1201,1250

elif [ "$1" == "all" ] || [ "$1" == "arima" ]; then
    # ARIMA
    python3 arima.py 0,550 551,600
    python3 arima.py 600,1150 1151,1200
    python3 arima.py 800,1200 1201,1250
    
fi
