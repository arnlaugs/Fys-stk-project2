# FYS-STK4155 - Applied data analysis and machine learning: Project 2

In this respotory you will find codes used for project 2 in FYS-STK4155. 

### Authors
The projects found in this repository is a results of the collaboration between

* **Maren Rasmussen** - (https://github.com/mjauren)

* **Arnlaug Høgås Skjæveland** - (https://github.com/arnlaugs)

* **Markus Leira Asprusten** - (https://github.com/maraspr)

## Downloding data:
To run thees programs you need to downlode dataset from Mehta et.al. Found here: https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/

## Description of programs

**NeuralNetwork.py**: 

Simple neural network for classification with one hidden layer. 
Input variables:
* X_data: dataset, features
* Y_data: classes
* n_hidden_neurons: number neurons in the hidden layer
* n_catogories: number of categories / neurons in the final
            output layer
* epochs: number of times running trough training data
* batch_size: number of datapoint in each batch for calculating
            gradient for gradient descent
* eta: learning rate
* lmbd: regularization parameter
* activation_func: activation function, sigmoid is standard
* activation_func_out: activation function for output
* cost_func: Cost function
* leaky_a: Slope for negative values in Leaky ReLU

Used by calling:

    NN = NeuralNetwork(X_train, Y_train, ... )
    NN.train() 
    NN.predict(X_test)

Can also provide heatmaps illustrating which values of the learning rate, regularization parameter, and the number of hidden neurons that gives the best accuracies:

    NN = NeuralNetwork(X_train, Y_train, ... )

    NN.heatmap_eta_lambda()
    NN.heatmap_neurons_eta()
   
   
**NN_Linear_model.py**:

 Modifies the class from NeuralNetwork.py to work for a linear, non-classification case.
 Input variables
* X_data: dataset, features
* Y_data: classes
* n_hidden_neurons: number neurons in the hidden layer
* epochs: number of times running trough training data
* batch_size: number of datapoint in each batch for calculating
            gradient for gradient descent
* eta: learning rate
* lmbd: regularization parameter
* activation_func: activation function, sigmoid is standard
* activation_func_out: activation function for output
* cost_func: Cost function
* leaky_a: Slope for negative values in Leaky ReLU

Used by calling:

    NN = NeuralNetwork(X_train, Y_train, ... )
    NN.train() 
    NN.predict(X_test)

Can also provide heatmaps illustrating which values of the learning rate and the number of hidden neurons that gives the best accuracies:

    NN = NeuralNetwork(X_train, Y_train, ... )

    NN.heatmap_neurons_eta()
    
**regression.py**:
A class for linear regression using OLS, Ridge or Lasso. 
Used by calling:

            model=OLS() (or Lasso() or Ridge())
            model.fit(x,y)
            y_predicted=model.predict(x)



**linear_regression.py**:

Uses Lasso, Ridge and OLS regression to find the coupling constant J. It tests for different  λ values. It plot the MSE, R2 score. And J for different spinns.

**logistic_regression.py**:

Uses data from Mehta et.al. and clasifies if it is ordered or disorded spin configurations. It plots a heatmap of how accurate it is for different learning rates and number of iterations. 

