import numpy as np
import matplotlib.pyplot as plt

w,b = list(np.load('data.npy'))

iris = datasets.load_iris()
x,y = iris.data,iris.target

def train_dataset(x,y):
    a = np.array(x[0:45,:2])
    a = np.append(a,x[50:95,:2],axis=0)
    a = np.append(a,x[100:145:,:2],axis=0)
    a[:,0] = (a[:,0]-a[:,0].min())/(a[:,0].max()-a[:,0].min())
    a[:,1] = (a[:,1]-a[:,1].min())/(a[:,1].max()-a[:,1].min())
    b = np.array(y[0:45])
    b = np.append(b,y[50:95],axis=0)
    b = np.append(b,y[100:145],axis=0)
    b = np.array(b).astype(np.float32)
    b = (b-b.min())/(b.max()-b.min())
    return a,b

def test_dataset(x,y):
    a = np.array(x[45:50,:2])
    a = np.append(a,x[95:100,:2],axis=0)
    a = np.append(a,x[145:150,:2],axis=0)
    a[:,0] = (a[:,0]-a[:,0].min())/(a[:,0].max()-a[:,0].min())
    a[:,1] = (a[:,1]-a[:,1].min())/(a[:,1].max()-a[:,1].min())
    b = np.array(y[45:50])
    b = np.append(b,y[95:100],axis=0)
    b = np.append(b,y[145:150],axis=0)
    b = np.array(b).astype(np.float32)
    b = (b-b.min())/(b.max()-b.min())
    return a,b

x_train, y_train = train_dataset(x,y)
x_train = x_train.T
x_test, y_test = test_dataset(x,y)
x_test = x_test.T
print(x_test.shape)

def forward_propagate(weight,x,bias=None):
    if bias is not None:
        z = np.dot(weight,x) + bias
        output = activation(z)
    else:
        z = np.dot(weight,x)
        output = z
    return output

def activation(z):
    return 1.0/(1+np.exp(-z))
x_min,x_max = x_train[0,:].min()-1,x_train[0,:].max()+1
y_min,y_max = x_train[1,:].min()-1,x_train[1,:].max()+1
steps = 500
print(x_min,x_max)

x_span = np.linspace(x_min,x_max,steps)
y_span = np.linspace(y_min,y_max,steps)

xx,yy = np.meshgrid(x_span,y_span)
xPlot = np.c_[xx.ravel(),yy.ravel()].T

hidden_plot = forward_propagate(w[0],xPlot,b[0])
prediction_plot = forward_propagate(w[1],hidden_plot)
z = prediction_plot.reshape(xx.shape)
cmap= plt.get_cmap('Paired')

plt.contourf(xx,yy,z,cmap=cmap,alpha=0.5)

plt.scatter(x[0],x[1])
#plt.plot(range(n_epochs),costs)
plt.show()
