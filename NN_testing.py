import warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
from NeuralNetwork import NeuralNetwork

X = np.array([[0.0], [1.0]])
y = np.array([0, 2])
mlp = MLPRegressor( solver              = 'sgd',      # Stochastic gradient descent.
                    activation          = 'logistic', # Skl name for sigmoid.
                    alpha               = 0.0,        # No regularization for simplicity.
                    hidden_layer_sizes  = (3) )    # Full network is of size (1,3,1).

#mlp.out_activation_ = 'softmax'

# Force sklearn to set up all the necessary matrices by fitting a data set.
# We dont care if it converges or not, so lets ignore raised warnings.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mlp.fit(X,y)

# A single, completely random, data point which we will propagate through
# the network.
X      = np.array([[1.125982598]])
target = np.array([ 8.29289285])
mlp.predict(X)

# ==========================================================================
n_samples, n_features   = X.shape
batch_size              = n_samples
hidden_layer_sizes      = mlp.hidden_layer_sizes
if not hasattr(hidden_layer_sizes, "__iter__"):
    hidden_layer_sizes = [hidden_layer_sizes]
hidden_layer_sizes = list(hidden_layer_sizes)
layer_units = ([n_features] + hidden_layer_sizes + [mlp.n_outputs_])
activations = [X]
activations.extend(np.empty((batch_size, n_fan_out))
                   for n_fan_out in layer_units[1:])
deltas      = [np.empty_like(a_layer) for a_layer in activations]
coef_grads  = [np.empty((n_fan_in_, n_fan_out_))
               for n_fan_in_, n_fan_out_ in zip(layer_units[:-1],
                                                layer_units[1:])]
intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in layer_units[1:]]
# ==========================================================================



activations                       = mlp._forward_pass(activations)
loss, coef_grads, intercept_grads = mlp._backprop(
        X, target, activations, deltas, coef_grads, intercept_grads)


nn = NeuralNetwork( X_data = X,
                    Y_data = target,
                    n_hidden_neurons = 3,
                    n_categories = 1,
                    activation_func_out = 'sigmoid',
                    cost_func = 'MSE')

# Copy the weights and biases from the scikit-learn network to your own.
nn.hidden_weights, nn.output_weights = mlp.coefs_
nn.hidden_bias, nn.output_bias = mlp.intercepts_


# Call your own backpropagation function, and you're ready to compare with
# the scikit-learn code.
nn.feed_forward()
nn.backpropagation()

for i, a in enumerate([nn.a_h, nn.a_o]) :
    assert np.allclose(a, activations[i])

for i, derivative_bias in enumerate([nn.hidden_bias_gradient, nn.output_bias_gradient]) :
    assert np.allclose(derivative_bias, intercept_grads[i])

for i, derivative_weight in enumerate([nn.hidden_weights_gradient, nn.output_weights_gradient]) :
    assert np.allclose(derivative_weight, coef_grads[i])
