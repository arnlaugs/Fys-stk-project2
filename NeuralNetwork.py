import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns


# Ensure the same random numbers appear every time
np.random.seed(0)

# setup the feed-forward pass, subscript h = hidden layer
def sigmoid(x):
    return 1/(1 + np.exp(-x))

class NeuralNetwork:
    """
    Input variables:
        - X_data: dataset, features
        - Y_data: classes
        - n_hidden_neurons: number neurons in the hidden layer
        - n_catogories: number of categories / neurons in the final
            output layer
        - epochs: idk
        - batch_size: number of datapoint in each batch for calculating
            gradient for gradient descent
        - eta: learning rate
        - lmbd: regularization parameter
    """
    def __init__(
        self,
        X_data,
        Y_data,
        n_hidden_neurons=50,
        n_categories=10,
        epochs=10,
        batch_size=100,
        eta=0.1,
        lmbd=0.0,

    ):

        # Setting selv values
        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        """
        Initialize the weights with random numbers from the “standard
        normal” distribution. Initialize biases to be arrays with 0.01.
        """
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        """
        Used for training the network.
        1) Calculates z = W*X + b for hidden layer.
        2) Then calculates the activation function of z, giving a.
        3) Calculates z = W*X + b = W*a + b for output layer.
        4) Calculates the softmax function of the output values giving
            the probabilities.
        """
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = sigmoid(self.z_h)
        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        """
        Feed forward for output. Does the same as feed_forward, but
        does not save the variables. Returns the probabilities.
        """
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        """
        1) Computes the error of the output result compared to the acual Y-values.
        2) Computes the propagate error.
        3) Computes the gradient of the weights and the biases for the output layer
            and hidden layer.
        4) If a regularization parameter is given, the weights are multiplied with
            this before calculating the output weights and biases.
        """
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        """
        Returns the most probable class.
        """
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        """
        Returns the probabilities for the different outputs / classes
        for the given data X.
        """
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        """
        Trains the model. For each epoch, for each of the minibatches:
            1) Chose datapoints for minibatch.
            2) Make data of the chosen bathes of datapoints.
            3) Run feed forward and backpropagation
        """
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()

    def finding_nemo(self):
        sns.set()

        eta_vals = np.logspace(-5, 1, 7)
        lmbd_vals = np.logspace(-5, 1, 7)

        train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
        test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

        for i, eta in enumerate(eta_vals):
            for j, lmbd in enumerate(lmbd_vals):
                DNN = NeuralNetwork(self.X_data_full, self.Y_data_full, eta=eta,
                                    lmbd=lmbd, epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    n_hidden_neurons=self.n_hidden_neurons,
                                    n_categories=self.n_categories)
                DNN.train()

                train_pred = DNN.predict(X_train)
                test_pred = DNN.predict(X_test)

                train_accuracy[i][j] = accuracy_score(Y_train, train_pred)
                test_accuracy[i][j] = accuracy_score(Y_test, test_pred)


        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Training Accuracy")
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.show()

        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Test Accuracy")
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.show()






def load_data():
    """
    Reads in and reshapes the data file containing 16*10000 samples taken
    in T=np.arange(0.25,4.0001,0.25). Pickle reads the file and returns
    the Python object (1D array, compressed bits). It also reads in
    the labels of the samples and maps 0 state to -1 (Ising variable
    can take values +/-1).

    Returns arrays containing the data and labels for the ordered and
    disordered states.
    """

    # Reading in and reshaping
    folder = 'IsingData/'
    file_name = "Ising2DFM_reSample_L40_T=All.pkl"
    data = pickle.load(open(folder+file_name,'rb'))
    data = np.unpackbits(data).reshape(-1, 1600)
    data = data.astype('int')
    data[np.where(data==0)] = -1

    file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl"
    labels = pickle.load(open(folder+file_name,'rb'))

    # Divide data into ordered, critical and disordered
    X_ordered=data[:70000,:]
    Y_ordered=labels[:70000]

    X_critical=data[70000:100000,:]
    Y_critical=labels[70000:100000]

    X_disordered=data[100000:,:]
    Y_disordered=labels[100000:]

    # Freeing up memory by deleting old arrays
    del data,labels

    # Creating arrays, using only the 1000 first elements
    X = np.concatenate((X_ordered[:1000],X_disordered[:1000]))
    Y = np.concatenate((Y_ordered[:1000],Y_disordered[:1000]))
    return X, Y


# Pick random data points from ordered and disordered states
# to create the training and test sets.
X, Y = load_data()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8)

def to_categorical_numpy(integer_vector):
    """
    Creates bolean arrays for each category.
    """
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector

Y_train_onehot, Y_test_onehot = to_categorical_numpy(Y_train), to_categorical_numpy(Y_test)


# Defining variables need in the Neural Network
epochs = 10
batch_size = 10
eta = 0.01
lmbd = 0.01
n_hidden_neurons = 10
n_categories = 2

NN = NeuralNetwork(X_train, Y_train_onehot, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)

NN.finding_nemo()
#NN.train()
#test_predict = NN.predict(X_test)
