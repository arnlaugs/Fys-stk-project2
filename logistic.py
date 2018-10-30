import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
np.random.seed(14)





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
    for k in range(int(itterations)):
        predictions=sigmoid(x,w)

        p=predictions-y
        #print(predictions.shape, y.shape, x.shape,p.shape)
        w=w-1/N*lr*np.dot(x.T,p)
        if k%50==0:
            #cost=loss(sigmoid(x,w),y)
            #print( cost,accurasy(x,w,y),k, lr)
            print(k)

    return w

def accurasy(x,w,y):
    a=0
    #print(len(x),x[0].shape)
    for i in range(len(x)):
        if sigmoid(x[i],w)>0.5:
            t=1
        else:
            t=0
        if t==y[i]:
            a+=1
    return a/len(x)
def train_test_split(X,Y,train_size=0.8):
    """ Takes in X and Y values and split them random in to test and learn"""
    print(X.shape)
    i=np.arange(X.shape[0])
    np.random.shuffle(i)
    i=i[0:int(X.shape[0]*train_size)]
    x_test=np.delete(X,i,0)
    y_test=np.delete(Y,i,0)
    x_learn=np.take(X,i,0)
    y_learn=np.take(Y,i,0)


    return x_learn,y_learn,x_test,y_test



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
itterations=300
y=y.reshape(-1,1)
#x=x[:30000,:]
#y=y[:30000]

x_train,y_train,x_test,y_test=train_test_split(x,y,train_size=0.8)
print(x_train.shape, x_test.shape)


learning_rates=[1e-4,1e-3,1-2,1e-1,1,2]
#learning_rates=[0.01]
itterations=[10,2e2,5e2,1e3,5e2]


for lr in learning_rates:

    w=np.random.randn(x_train.shape[1],1)
    w=weigths_update(x_train,y_train,w,lr,itterations[1])
    print(lr)
    #cost=loss(sigmoid(x,w),y)
    #print( cost)
    print(accurasy(x_test,w,y_test))
    print(accurasy(x_train,w,y_train))
#plt.imshow(data[10000-1].reshape(L,L))
#plt.show()
