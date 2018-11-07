import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
np.random.seed(14)
import seaborn as sns
import pandas as pd
from functions import *


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
        #if k%50==0:
            #cost=loss(sigmoid(x,w),y)
            #print( cost,accurasy(x,w,y),k, lr)
        #    print(k)

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


#learning_rates=[1e-4,1e-3,1e-2,1e-1]
learning_rates=[0.01]
#itterations=[10,1e2,5e2]
itterations=[10,1e2]
test_accuracy= np.zeros((len(itterations),len(learning_rates)))
train_accuracy=  np.zeros((len(itterations),len(learning_rates)))

for j in range(len(itterations)):
    i=0
    for lr in learning_rates:

        w=np.random.randn(x_train.shape[1],1)
        w=weigths_update(x_train,y_train,w,lr,itterations[j])
        print(lr,j)
        #cost=loss(sigmoid(x,w),y)
        #print( cost)
        test_accuracy[j][i]=accurasy(x_test,w,y_test)
        train_accuracy[j][i]=accurasy(x_train,w,y_train)
        print (test_accuracy[j][i],itterations[j],lr )
        i+=1

sns.set()
fig, ax = plt.subplots(figsize = (len(learning_rates),len(itterations)))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis", vmin=0, vmax=1)
ax.set_title("Test Accuracy")
ax.set_ylabel("Itterations")
ax.set_xlabel("Learning rate")
plt.show()
savefigure("logistic_accurasy_test_test", figure=fig)

fig, ax = plt.subplots(figsize = (len(learning_rates),len(itterations)))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis", vmin=0, vmax=1)
ax.set_title("Training Accuracy")
ax.set_ylabel("Itterations")
ax.set_xlabel("Learning rate")
plt.show()
savefigure("logistic_accurasy_train_test",figure=fig)
#plt.imshow(data[10000-1].reshape(L,L))
#plt.show()
