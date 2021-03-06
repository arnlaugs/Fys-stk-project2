import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from NeuralNetwork import NeuralNetwork as NN


class NeuralNetwork(NN):

    def __init__(
        self,
        X_data,
        Y_data,
        n_hidden_neurons=100,
        epochs=10,
        batch_size=100,
        eta=0.1,
        lmbd=0.0,
        activation_func = 'relu',
        activation_func_out = 'leaky_relu',
        cost_func = 'MSE',
        leaky_a = 0.01):

        if len(Y_data.shape) == 1:
            Y_data = np.expand_dims(Y_data, 1)

        n_categories =  Y_data.shape[1]

        NN.__init__(self, X_data, Y_data, n_hidden_neurons, n_categories, epochs, batch_size, eta, lmbd, activation_func, activation_func_out, cost_func)



    def predict(self, X):
        """
        Returns the output
        """
        return self.feed_forward_out(X)

    def heatmap_neurons_eta(self, X_train, Y_train, X_test, Y_test):
        sns.set()

        if len(Y_train.shape) == 1:
            Y_train = np.expand_dims(Y_train, 1)

        if len(Y_test.shape) == 1:
            Y_test = np.expand_dims(Y_test, 1)

        eta_vals = np.logspace(-6, -1, 6)
        neuron_vals = [1,10,100]


        train_accuracy = np.zeros((len(neuron_vals),len(eta_vals)))
        test_accuracy = np.zeros((len(neuron_vals),len(eta_vals)))

        for i, neuron in enumerate(neuron_vals):
            for j, eta in enumerate(eta_vals):
                print("training DNN with %4d neurons and SGD eta=%0.6f." %(neuron,eta) )
                DNN = NeuralNetwork(self.X_data_full, self.Y_data_full, eta=eta,
                                    lmbd=0.0, epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    n_hidden_neurons=neuron)
                DNN.train()

                train_pred = DNN.predict(X_train)
                test_pred = DNN.predict(X_test)

                train_accuracy[i][j] = r2_score(Y_train, train_pred)
                test_accuracy[i][j] = r2_score(Y_test, test_pred)

                print(train_accuracy[i][j])


        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Training Accuracy")
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")

        fig, ax = plt.subplots(figsize = (10, 10))
        sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Test Accuracy")
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.show()



if __name__ == "__main__":
    # System size
    L = 50               # Number of spins
    N = 10000            # Number of states


    # Create 10000 random Ising states with spin +1 or -1
    states = np.random.choice([-1, 1], size=(N,L))

    def ising_energies(states,L):
        """
        This function calculates the energies of the states in the nn Ising Hamiltonian
        """
        J = np.zeros((L,L),)

        # Creating J-matrix with ones on the off-diagonal
        for i in range(L):
            J[i,(i+1)%L] -= 1.0

        # Calculating the energies E for each state
        E = np.einsum('...i,ij,...j->...',states,J,states)

        return E


    energies = ising_energies(states,L)


    # Reshape Ising states by using a single index p={j,k}, S_jS_k --> X_p
    states = np.einsum('...i,...j->...ij', states, states)
    shape = states.shape
    states = states.reshape((shape[0],shape[1]*shape[2]))

    X_train, X_test, Y_train, Y_test = train_test_split(states, energies, train_size=0.8, test_size = 0.2)


    model = NeuralNetwork(X_train, Y_train, batch_size = 10, epochs = 50, n_hidden_neurons=10, eta=0.00001)
    model.train()
    print(r2_score(Y_train, model.predict(X_train)))
    print(r2_score(Y_test, model.predict(X_test)))
    print(r2_score(energies, model.predict(states)))



    model.heatmap_neurons_eta(X_train, Y_train, X_test, Y_test)


