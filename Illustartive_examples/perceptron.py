from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
from matplotlib2tikz import save as tikz_save

np.random.seed(81)

# Making data to classify
X, y = make_classification(n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1, random_state=0)

# Activation funtion
def predict(X, w, b):
    activation = np.sum(np.dot(X,w) + b)
    if activation >= 0:
        return 1
    else:
        return 0

# Function to train the model
def train_weights(X, y, eta, n_epoch):
    bias = 0.01
    weight = np.random.randn(X.shape[1])

    # for epoch in range(n_epoch):
    #     error = predict(X, weight, bias)-y
    #
    #     weight_gradient = np.dot(X.T, error)
    #     bias_gradient = np.sum(error)
    #
    #     bias -= eta*bias_gradient
    #     weight -= eta*weight_gradient

    return bias, weight

# Finding the optimal bias and weights
b, w = train_weights(X, y, 0.01, 100)
x = np.linspace(-3,3,50)


w[1] += 15
b += 2

# Calculating the slope and intercept of the line
# separating the data into two classes.
slope = -(b/w[1])/(b/w[0])
intercept = -b/w[1]


# Plotting the data and the classifying line.
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25)
plt.plot(x, slope*x + intercept, c='k', linestyle='--')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axis([-2.6, 2.6, -1.6, 1.6])
tikz_save('perceptron3.tex', figureheight="\\figureheight", figurewidth="\\figurewidth")
plt.show()
