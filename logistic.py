import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
def read_t(t):
    data = pickle.load(open('Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))
    print(data.shape)
    return np.unpackbits(data).astype(int).reshape(-1,1600)
    #return data
L=40
data=read_t(2.25)

print(data.shape)
plt.imshow(data[10000-1].reshape(L,L))
plt.show()
