import numpy as np
from regression import OLS, Ridge,Lasso
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
np.random.seed(12)
from sklearn import linear_model
from functions import *
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# System size
L = 40             # Number of spins
N = 1000         # Number of states


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


# Create list for storing MSE-values and R2-values
MSE_train_OLS = []
MSE_test_OLS = []
R2_train_OLS = []
R2_test_OLS = []

MSE_train_R = []
MSE_test_R = []
R2_train_R = []
R2_test_R = []

MSE_train_L = []
MSE_test_L = []
R2_train_L = []
R2_test_L = []



#lmbda_labels=[r'$10^{-4}$',r'$10^{-1}$',"1", "100"]
cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
k=5 #number of k-folds

#calculate for OLS
model_OLS = OLS()
J_OLS = (model_OLS.fit(x, y, ret=True)).reshape((L,L))
ypred = model_OLS.predict(x)
MSE_train_OLS.append(mean_squared_error(y, model_OLS.predict(x)))
R2_train_OLS.append(r2_score(y, model_OLS.predict(x)))
K_foldvalues=K_fold(data[0],data[1],k,1, model_OLS)
MSE_test_OLS.append(K_foldvalues[0])
R2_test_OLS.append(K_foldvalues[1])

fig = plt.figure()
im=plt.imshow(J_OLS, **cmap_args)
plt.title('OLS')
fig.colorbar(im)
savefigure("OLS",figure=fig)


i=1
j=0
#Set lambdas
lmbdas = np.logspace(-4, 5, 10)
for lmbda in lmbdas:
    """ Calculate and plot the J-values for Lasso and Ridge. And calculate R2 and MSE """

    model_R = Ridge(lmbda=lmbda)
    J_R = (model_R.fit(x, y, ret=True)).reshape((L,L))
    MSE_train_R.append(mean_squared_error(y, model_R.predict(x)))
    R2_train_R.append(r2_score(y, model_R.predict(x)))
    K_foldvalues=K_fold(data[0],data[1],k,lmbda, model_R)
    MSE_test_R.append(K_foldvalues[0])
    R2_test_R.append(K_foldvalues[1])



    model_L = Lasso(alpha=lmbda)
    J_L = (model_L.fit(x, y, ret=True)).reshape((L,L))
    MSE_train_L.append(mean_squared_error(y, model_L.predict(x)))
    R2_train_L.append(r2_score(y, model_L.predict(x)))
    K_foldvalues=K_fold(data[0],data[1],k,lmbda, model_L)
    MSE_test_L.append(K_foldvalues[0])
    R2_test_L.append(K_foldvalues[1])


    if i==5:
        plt.savefig("J_1_n=1000.png")
        i==1
    plt.subplot(2,2,i)
    plt.imshow(J_R, **cmap_args)
    plt.title(r'R,$\lambda$ =' +str(lmbda),fontsize=15)
    i+=1
    plt.subplot(2,2,i)
    im = plt.imshow(J_L,**cmap_args)
    plt.title(r'L, $\alpha$ =' +str(lmbda),fontsize=15)

    i+=1
    j+=1

cbar_ax = fig.add_axes([-1,-0.5, 0.5, 1])
fig.colorbar(im, cax=cbar_ax)
plt.savefig("J_2_n=1000.png")

def plot_error_estimates():
    """
    Plots the means squared error and the R2-score with
    respect to the lambda-values.
    """
    # Plotting the R2-scores
    fig=plt.figure()
    #ax=fig.gca()
    plt.semilogx([lmbdas[0],lmbdas[-1]], [R2_train_OLS,R2_train_OLS], 'b', label='OLS (train)')
    plt.semilogx([lmbdas[0],lmbdas[-1]], [R2_test_OLS,R2_test_OLS], 'b--', label='OLS (test)')
    plt.semilogx(lmbdas, R2_train_R, 'r', label='Ridge (train)')
    plt.semilogx(lmbdas, R2_test_R, 'r--', label='Ridge (test)')
    plt.semilogx(lmbdas, R2_train_L, 'g', label='Lasso (train)')
    plt.semilogx(lmbdas, R2_test_L, 'g--', label='Lasso (test)')
    plt.title('R2-score')
    plt.legend()
    plt.grid()
    savefigure("R2_score_n_1000",figure=fig)

    # Plotting the MSE-scores
    fig2=plt.figure()
    ax=fig2.gca()
    ax.semilogx([lmbdas[0],lmbdas[-1]], [MSE_train_OLS,MSE_train_OLS], 'b', label='OLS (train)')
    ax.semilogx([lmbdas[0],lmbdas[-1]], [MSE_test_OLS,MSE_test_OLS], 'b--', label='OLS (test)')
    ax.semilogx(lmbdas, MSE_train_R, 'r', label='Ridge (train)')
    ax.semilogx(lmbdas, MSE_test_R, 'r--', label='Ridge (test)')
    ax.semilogx(lmbdas, MSE_train_L, 'g', label='Lasso (train)')
    ax.semilogx(lmbdas, MSE_test_L, 'g--', label='Lasso (test)')
    ax.set_title('Mean squared error')
    plt.legend()
    plt.grid()
    plt.show()
    savefigure("MSE_n_1000",figure=fig2)
plot_error_estimates()
