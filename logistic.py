import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
def read_t(path,t):
    data = pickle.load(open(path+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
    print(data.shape)
    return np.unpackbits(data).astype(int).reshape(-1,1600)
    #return data
def sigmoid(t):
    return np.exp(t)/(1-np.exp(t))

L=40
path="IsingData/"
temp_list=[0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.75,3.00,3.25,3.50,3.75,4.00]
data=read_t(path,4.00)

print(data.shape)
plt.imshow(data[10000-1].reshape(L,L))
plt.show()
