import numpy as np
import matplotlib.pyplot as plt

w,b = list(np.load('data.npy'))
print(w[0])
print(b[0])
x = np.array([[0,0,1,1],[0,1,0,1]])
y = np.array([0,1,1,1])
def forward_propagate(w1,b1,x):
    z = np.dot(w1,x) + b1
    output = activation(z)
    return output
def activation(z):
    return 1.0/(1+np.exp(-z))
x_min,x_max = x[0,:].min()-1,x[0,:].max()+1
y_min,y_max = x[1,:].min()-1,x[1,:].max()+1
steps = 500
print(x_min,x_max)

x_span = np.linspace(x_min,x_max,steps)
y_span = np.linspace(y_min,y_max,steps)

xx,yy = np.meshgrid(x_span,y_span)
xPlot = np.c_[xx.ravel(),yy.ravel()].T

hidden_plot = forward_propagate(w[0],b[0],xPlot)
prediction_plot = forward_propagate(w[1],b[1],hidden_plot)
z = prediction_plot.reshape(xx.shape)
cmap= plt.get_cmap('Paired')

plt.contourf(xx,yy,z,cmap=cmap,alpha=0.5)

plt.scatter(x[0],x[1])
#plt.plot(range(n_epochs),costs)
plt.show()
