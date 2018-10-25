import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
def read_t(path,t):
    data = pickle.load(open(path+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))


    return np.unpackbits(data).astype(int).reshape(-1,1600)
    #return data
def sigmoid(x,w):
    t=np.dot(x,w)

    return 1/(1+np.exp(-t))
def loss(h, y):

    return (-y * np.log(h) - (1 - y) * np.log(1 - h+1e-12)).mean()

def custom_loss(y, y_predicted):
    return -(y*np.log(y_predicted) - (1-y)*np.log(1-y_predicted)**2).mean()

def weigths_update(x,y,w,lr,itterations):
    N=len(x)
    for k in range(itterations):
        predictions=sigmoid(x,w)

        p=predictions-y
        #print(predictions.shape, y.shape, x.shape,p.shape)
        w=w-1/N*lr*np.dot(x.T,p)



    return w
L=40
path="IsingData/"
labels = pickle.load(open(path+"Ising2DFM_reSample_L40_T=All_labels.pkl",'rb'))
temp_list=[0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.75,3.00,3.25,3.50,3.75,4.00]
i=0

for t in temp_list:
    if t== 0.25:
        x=read_t(path,t)
        y=labels[:10000]
    else:
        x=np.concatenate((x,read_t(path,t)))
        y=np.concatenate((y,labels[10000*i:10000*(i+1)]))
    if t==1.75:
        i+=4
    else:
        i+=1
itterations=1000
y=y.reshape(-1,1)
#x=x[:30000,:]
#y=y[:30000]
w=np.random.randn(x.shape[1],1)


lr=0.01

w=weigths_update(x,y,w,lr,itterations)

cost=loss(sigmoid(x,w),y)
print( cost)


#plt.imshow(data[10000-1].reshape(L,L))
#plt.show()
