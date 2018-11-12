import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
np.random.seed(14)
import seaborn as sns
import pandas as pd
from functions import *
from sklearn import linear_model


def read_t(path,t):
    """Reads data of one temperature """
    data = pickle.load(open(path+'Ising2DFM_reSample_L40_T=%.2f.pkl'%t,'rb'))


    return np.unpackbits(data).astype(int).reshape(-1,1600)
    #return data
def sigmoid(x,w):
    """Calculate the sigmoid for x with weigths. """
    t=np.dot(x,w)

    return 1/(1+np.exp(-t))
def loss(h, y):
    """Loss function, take in predicted values h and accurate values y """
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()



def weigths_update(x,y,w,lr,itterations):
    """ Calculate the weights from x,y and a input weight, set a learning rate and give number of iterations"""
    N=x.shape[0]

    for k in range(int(itterations)):
        predictions=sigmoid(x,w)

        p=predictions-y
        w=w-lr*1/N*np.dot(x.T,p)

    return w

def accurasy(x,w,y):
    """Calculate the accuracy from the data x,y and the weights w """
    a=0
    #print(len(x),x[0].shape)
    for i in range(len(x)):
        if sigmoid(x[i],w)>0.5:
            t=1
        else:
            t=0
        if t==y[i]:
            a+=1

    return a/x.shape[0]
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
    """Takes the data from spesified temperatures. """
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

y=y.reshape(-1,1)

#Splitt in test and train
x_train,y_train,x_test,y_test=train_test_split(x,y,train_size=0.8)

#Compear with sklearn
logreg=linear_model.LogisticRegression()
logreg.fit(x_train, y_train)

print(logreg.score(x_train,y_train))
print(logreg.score(x_test,y_test))

#Test for different learning rates and number of iterations
learning_rates=[1e-4,1e-3,1e-2,1e-1]
itterations=[10,1e2,5e2]
test_accuracy= np.zeros((len(itterations),len(learning_rates)))
train_accuracy=  np.zeros((len(itterations),len(learning_rates)))

for j in range(len(itterations)):
    i=0
    for lr in learning_rates:

        w=np.random.randn(x_train.shape[1],1)
        w=weigths_update(x_train,y_train,w,lr,itterations[j])
        print(lr,j)
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
