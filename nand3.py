import numpy as np
import matplotlib.pyplot as plt

np.random.seed(9)

n_inputs = 2
n_hidden = 2
n_output = 1
w,b = [],[]
costs = []
n_epochs =5000
lr = 0.1

x = np.array([[0,0,1,1],[0,1,0,1]])
y = np.array([1,1,1,0])
w1 = np.random.rand(n_hidden,n_inputs)
w.append(w1)
b1 = np.random.rand(n_hidden,1)
b.append(b1)
w2 = np.random.rand(n_output,n_hidden)
w.append(w2)

def forward_propagate(weight,x,bias=None):
    if bias is not None:
        z = np.dot(weight,x) + bias
        output = activation(z)
    else:
        z = np.dot(weight,x)
        output = z
    return output

def backward_propagate(y,y_prime,input,m,weights=None):

    if weights is not None:
        dcostdw = np.dot((np.dot(weights.T,(cost_derivative_y_prime(y,y_prime,m)*1))*activation_derivative(hidden_output)),x.T)
        #dcostdw = np.dot(weights.T,(cost_derivative_y_prime(y,y_prime,m)*activation_derivative(y_prime)))*input*(1-input)
        dcostdb = np.sum(np.dot(weights.T,(cost_derivative_y_prime(y,y_prime,m)*1))*activation_derivative(hidden_output),axis=1)
        return dcostdw,dcostdb.reshape(1,n_hidden).T
    else:
        dcostdw =  np.dot(cost_derivative_y_prime(y,y_prime,m)*1,input.T)
        dcostdb =  np.sum((cost_derivative_y_prime(y,y_prime,m)*1),axis=1)
        return dcostdw,dcostdb

def activation(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def cost(y,y_prime,m):
    return ((y-y_prime)**2)/(2*m)

def activation_derivative(z):
    return 1-(activation(z)**2)

def cost_derivative_y_prime(y,y_prime,m):
    return -(y-y_prime)/(2*m)

for i in range(n_epochs):
    hidden_output = forward_propagate(w[0],x,b[0])
    output = forward_propagate(w[1],hidden_output)
    totalCost = np.sum(cost(y,output,x.shape[1]))
    costs.append(totalCost)
    wei,bia = backward_propagate(y,output,hidden_output,x.shape[1])
    hidden_weight,hidden_bias = backward_propagate(y,output,hidden_output,x.shape[1],weights=w[1])
    w[1] = w[1] - (lr*wei)
    #b[1] = b[1] - (lr*bia)
    w[0] = w[0] - (lr*hidden_weight)
    b[0] = b[0] - (lr*hidden_bias)

hidden_predict = forward_propagate(w[0],x,b[0])
prediction = forward_propagate(w[1],hidden_predict)
print(prediction)
print(y)

print(costs[-2:])
