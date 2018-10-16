import numpy as np
import scipy.sparse as sp
import sys
sys.path.append('../Fys-Stk-Project-1/functions/')
from regression import OLS
import matplotlib.pyplot as plt
np.random.seed(14)

import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

### define Ising model aprams
# system size
L=40

# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(400,L))

def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)

    return E
# calculate Ising energies
energies=ising_energies(states,L)

#print(energies)
states=np.einsum('...i,...j->...ij', states, states)
#print(states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))
# build final data set
Data=[states,energies]
coefs_OLS=[]
x=Data[0]
y=Data[1]

model=OLS()

coefs_OLS.append(model.fit(x,y,ret=True))

coef_OLS_plot=np.array(coefs_OLS).reshape(40,40)
for i in range(L):
    print(coef_OLS_plot[i,(i+1)%L])
cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
plt.imshow(coef_OLS_plot,**cmap_args)
plt.colorbar()

plt.show()

