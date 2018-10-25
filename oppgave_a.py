import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

import numpy as np
from regression import OLS, Ridge,Lasso
import matplotlib.pyplot as plt
np.random.seed(12)



# System size
L = 40             # Number of spins
N = 10000          # Number of states


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

    """
    #The Einstein sum does the same as the following nested
    #for-loops, but is much more efficient.
    E = np.zeros(N)

    for i in range(N):
        for j in range(L):
            for k in range(L):
                E[i] += J[j,k]*states[i][j]*states[i][k]
    """
    return E


energies = ising_energies(states,L)


# Reshape Ising states by using a single index p={j,k}, S_jS_k --> X_p
states = np.einsum('...i,...j->...ij', states, states)
shape = states.shape
states = states.reshape((shape[0],shape[1]*shape[2]))

# Building final data set of states and energies
data = [states, energies]

# Define number of samples
n_samples = 500


# Define train and test data sets
X_train = data[0][:n_samples]
y_train = data[1][:n_samples]
X_test = data[0][n_samples:3*n_samples//2]
y_test = data[1][n_samples:3*n_samples//2]


lmbdas = np.logspace(3, 5, 2)
cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')

for lmbda in lmbdas:
    model_OLS = OLS()
    J_OLS = (model_OLS.fit(X_train, y_train, ret=True)).reshape((L,L))

    model_R = Ridge(lmbda=lmbda)
    J_R = (model_R.fit(X_train, y_train, ret=True)).reshape((L,L))

    model_L = Lasso(alpha=lmbda)
    J_L = (model_L.fit(X_train, y_train, ret=True)).reshape((L,L))

    fig = plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(J_OLS, **cmap_args)
    plt.title('OLS')

    plt.subplot(1,3,2)
    plt.imshow(J_R, **cmap_args)
    plt.title(r'Ridge $\lambda =$%g' %lmbda)

    # P
    plt.subplot(1,3,3)
    im = plt.imshow(J_L,**cmap_args)
    plt.title(r'Lasso $\alpha =$%g' %lmbda)

    # Making colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
plt.show()
