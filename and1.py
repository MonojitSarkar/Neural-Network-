import numpy as np
import matplotlib.pyplot as plt

np.random.seed(9)

n_inputs = 2
n_hidden = 2
n_output = 1
w,b = [],[]
costs = []
n_epochs = 10000
lr = 0.5

x = np.array([[0,0,1,1],[0,1,0,1]])
y = np.array([0,0,0,1])
w1 = np.random.rand(n_hidden,n_inputs)
w.append(w1)
b1 = np.random.rand(n_hidden,1)
b.append(b1)
w2 = np.random.rand(n_output,n_hidden)
w.append(w2)
b2 = np.array(n_output)
b.append(b2)
print('weights and bias before')
print(w)
print(b)


def forward_propagate(w1,b1,x):
    z = np.dot(w1,x) + b1
    output = activation(z)
    return output

def backward_propagate(y,y_prime,input,m,weights=None):

    if weights is not None:
        dcostdw = np.dot((np.dot(weights.T,(cost_derivative_y_prime(y,y_prime,m)*activation_derivative(y_prime)))*input*(1-input)),x.T)
        #dcostdw = np.dot(weights.T,(cost_derivative_y_prime(y,y_prime,m)*activation_derivative(y_prime)))*input*(1-input)
        dcostdb = np.sum(np.dot(weights.T,(cost_derivative_y_prime(y,y_prime,m)*activation_derivative(y_prime)))*input*(1-input),axis=1)
        return dcostdw,dcostdb.reshape(1,n_hidden).T
    else:
        dcostdw =  np.dot(cost_derivative_y_prime(y,y_prime,m)*activation_derivative(y_prime),input.T)
        dcostdb =  np.sum((cost_derivative_y_prime(y,y_prime,m)*activation_derivative(y_prime)),axis=1)
        return dcostdw,dcostdb

def activation(z):
    return 1.0/(1+np.exp(-z))

def cost(y,y_prime,m):
    return ((y-y_prime)**2)/(2*m)

def activation_derivative(z):
    return z*(1-z)

def cost_derivative_y_prime(y,y_prime,m):
    return -(y-y_prime)/(2*m)

for i in range(n_epochs):
    hidden_output = forward_propagate(w[0],b[0],x)
    output = forward_propagate(w[1],b[1],hidden_output)
    totalCost = np.sum(cost(y,output,x.shape[1]))
    costs.append(totalCost)
    wei,bia = backward_propagate(y,output,hidden_output,x.shape[1])
    hidden_weight,hidden_bias = backward_propagate(y,output,hidden_output,x.shape[1],weights=w[1])
    w[1] = w[1] - (lr*wei)
    b[1] = b[1] - (lr*bia)
    w[0] = w[0] - (lr*hidden_weight)
    b[0] = b[0] - (lr*hidden_bias)

print('after training')
print(w)
print(b)

hidden_predict = forward_propagate(w[0],b[0],x)
prediction = forward_propagate(w[1],b[1],hidden_predict)
print(prediction)
print(y)

from numpy import asarray
from numpy import save

data1 = []
data1.append(w)
data1.append(b)

data = asarray(np.array(data1))
save('data.npy',data)
