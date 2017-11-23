if [ "$1" == "all" ] || [ "$1" == "regression" ]; then
    # REGRESSION
    echo 'running regression for 0 to 600'
    python3 regression.py 0,550 551,600
    echo 'running regression for 600 to 1200'
    python3 regression.py 600,1150 1151,1200
    echo 'running regression for 800 to 1250'
    python3 regression.py 800,1200 1201,1250

elif [ "$1" == "all" ] || [ "$1" == "mlp_nn" ]; then
    # NEURAL NETWORK
    echo 'running mlp_nn for 0 to 600'
    python3 sklearn_MLP_NN.py 0,550 551,600
    echo 'running mlp_nn for 600 to 1200'
    python3 sklearn_MLP_NN.py 600,1150 1151,1200
    echo 'running mlp_nn for 800 to 1250'
    python3 sklearn_MLP_NN.py 800,1200 1201,1250

elif [ "$1" == "all" ] || [ "$1" == "arma" ]; then
    # ARIMA
    echo 'running arma for 0 to 600'
    python3 arma.py 0,550 551,600
    echo 'running arma for 600 to 1200'
    python3 arma.py 600,1150 1151,1200
    echo 'running arma for 800 to 1250'
    python3 arma.py 800,1200 1201,1250
    
fi
