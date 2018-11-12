# FYS-STK4155 - Applied data analysis and machine learning: Project 2

In this respotory you will find codes used for project 2 in FYS-STK4155. 

### Authors
The projects found in this repository is a results of the collaboration between

* **Maren Rasmussen** - (https://github.com/mjauren)

* **Arnlaug Høgås Skjæveland** - (https://github.com/arnlaugs)

* **Markus Leira Asprusten** - (https://github.com/maraspr)



## Description of programs

* ** NeuralNetwork.py **: 
Simple neural network for classification. 
    Input variables:
        - X_data: dataset, features
        - Y_data: classes
        - n_hidden_neurons: number neurons in the hidden layer
        - n_catogories: number of categories / neurons in the final
            output layer
        - epochs: number of times running trough training data
        - batch_size: number of datapoint in each batch for calculating
            gradient for gradient descent
        - eta: learning rate
        - lmbd: regularization parameter
        - activation_func: activation function, sigmoid is standard
        - activation_func_out: activation function for output
        - cost_func: Cost function

Used by calling:

    NN = NeuralNetwork(X_train, Y_train, ... )
    NN.train() 
    NN.predict(X_test)
