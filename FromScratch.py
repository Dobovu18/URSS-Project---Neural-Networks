import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(5000)


X, y = spiral_data(120, 3)



#Activation functions
def ReLU(x):
    #ReLU
    return np.maximum(0, x)
def dReLU(x):
    #Derivative of the ReLU function. Just the Heaviside function
    if x < 0:
        return 0
    else:
        return 1

def Sigmoid(x):
    #Sigmoid Activation function
    return 1/(1 + np.exp(-x))
def dSigmoid(x):
    #Derivative of the sigmoid function
    return Sigmoid(x)*(1 - Sigmoid(x))

def Activate(x):
    return ReLU(x)
def dActivative(x):
    return dReLU(x)
#---------------------------------------

class Layer:
    #This class represents a layer in the neural network, which contains neurons, each with weights and biases.
    def __init__(self, n_inputs, n_neurons):
        #n_inputs is dimension of the feature space, or the number of neurons in the preceeding layer
        #n_neurons is the number of neurons in the layer
        self.weight = 0.1 * np.random.randn(n_inputs, n_neurons) #weights and biases initialised randomly 
        self.bias = np.random.randn(1, n_neurons)
    def forward(self, inputs):
        #forward pass
        self.z = np.dot(inputs, self.weight) + self.bias #z^L = W^L.a^[L-1] + b^L
        self.output = Activate(z) #Apply activation function
        self.d = np.diag(dActivate(z))
    def backward(self, inputs):
        self.delta = np.dot(self.d, inputs)
    def setWeight(self, new_weight):
        #Allows for weights to be changed.
            self.weight = new_weight
    def setBias(self, new_bias):
        #Allows for biases to be changed
            self.bias = new_bias

"""
class Activation:
    #This class is used for activation functions, e.g ReLU and Sigmoid
    def ReLU(self, inputs):
        self.ReLU = np.maximum(0, inputs)
        return self.ReLU
    def Sigmoid(self, inputs):
        self.Sigmoid = 1/(1 + np.exp(-inputs))
        return self.Sigmoid
"""



layer = [Layer(2,2), Layer(2,3), Layer(3,2)]

K = [1, 2, 3]
print(np.diag(K))


"""
layer1 = Layer(2,5)
activation = Activation()

layer1.forward(X)
"""
