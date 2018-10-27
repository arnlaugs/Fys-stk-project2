import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2  #L2-norm?
from keras.optimizers import SGD   # Stochastic gradient descent optimizer.
# Ensure the same random numbers appear every time
np.random.seed(0)

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


    n_inputs = len(X)
    X = X.reshape(n_inputs, -1)
    Y = to_categorical(Y)
    return X, Y


# Pick random data points from ordered and disordered states
# to create the training and test sets.
X, Y = load_data()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8)


def create_neural_network_keras(n_neurons_layer1, n_categories, eta, lmbd):
    """
    Creates neural network with keras using n_neurons_layer1 of neurons in first layer,
    n_neurons_layer2 in second layer, and n_catogories in the last layer.
    """
    model = Sequential()
    model.add(Dense(n_neurons_layer1, activation='sigmoid', kernel_regularizer=l2(lmbd)))
    #model.add(Dense(n_neurons_layer2, activation='sigmoid', kernel_regularizer=l2(lmbd)))
    model.add(Dense(n_categories, activation='softmax'))

    sgd = SGD(lr=eta)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


epochs = 10
batch_size = 10
n_neurons_layer1 = 10
n_categories = 2

eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
DNN_keras = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        DNN = create_neural_network_keras(n_neurons_layer1, n_categories,
                                         eta=eta, lmbd=lmbd)
        DNN.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        scores = DNN.evaluate(X_test, Y_test)

        DNN_keras[i][j] = DNN

import seaborn as sns

sns.set()

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        DNN = DNN_keras[i][j]

        train_accuracy[i][j] = DNN.evaluate(X_train, Y_train)[1]
        test_accuracy[i][j] = DNN.evaluate(X_test, Y_test)[1]


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
