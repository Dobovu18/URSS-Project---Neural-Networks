import numpy as np
import scipy.special
import random
import matplotlib.pyplot as plt

#Neural network designed to classify images of pictures
#Code inspired by 'Make Your Own Neural Network' by Tariq Rashid and 
#'Deep Learning: An Introduction for Applied Mathematicians' by Catherine F. Higham and Desmond J. Higham.

class neuralNetwork:
    #Multi-layer perceptron neural network
    def __init__(self, inputnodes, hiddennodes, hiddenlayers, outputnodes, learningrate):
        #Create Network Structure
        self.L = hiddenlayers + 2
        self.networkStructure = [hiddennodes] * self.L
        self.networkStructure[0] = inputnodes
        self.networkStructure[self.L - 1] = outputnodes
        #Configure Weights and Biases

        self.W = [None] * self.L
        self.b = [None] * self.L

        for i in range(1, self.L):
            self.W[i] = np.random.normal(0.0, pow(self.networkStructure[i], -0.5), (self.networkStructure[i], self.networkStructure[i - 1]))
            self.b[i] = np.random.normal(0.0, pow(self.networkStructure[i], -0.5), (self.networkStructure[i], 1))

        #Learning rate
        self.lr = learningrate

        #Activation function and derivative
        self.activation = lambda x: scipy.special.expit(x) #Sigmoid activation function
        self.d_activation = lambda x: self.activation(x) * (1 - self.activation(x))
        
        

    def train(self, inputs_list, targets_list):
        #Train the neural network on one test sample
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T

        a = [None] * self.L
        z = [None] * self.L
        d = [None] * self.L

        a[0] = inputs
        z[0] = inputs

        #Forward-pass
        for i in range(1, self.L):
            z[i] = np.dot(self.W[i], a[i - 1]) + self.b[i]
            a[i] = self.activation(z[i])
            

        #Backpropagation of errors
        d[self.L - 1] = self.d_activation(z[self.L - 1]) * (a[self.L - 1] - targets)
        for i in reversed(range(1, self.L - 1)):
            d[i] = self.d_activation(z[i]) * np.dot(self.W[i + 1].T, d[i + 1])
            

        #Gradient Descent
        for i in reversed(range(1, self.L)):
            self.W[i] -= self.lr * np.dot(d[i], np.transpose(a[i - 1]))
            self.b[i] -= self.lr * d[i]
            
        
        #output_errors = a[2] - targets #d[2]
        #hidden_errors = np.dot(self.W[2].T, output_errors) #d[1]
        #self.W[1] -= self.lr * np.dot((hidden_errors * self.d_activation(z[1])), np.transpose(a[0]))#learns slower??
        #self.W[2] -= self.lr * np.dot((output_errors * self.d_activation(z[2])), np.transpose(a[1]))
        #For some reason switching to this method for gradient descent gave more accurate results?
    

    def train_from_file(self, training_file, sample_size, epochs):
        #Train network on a training set from a given file
        training_data_file = open(training_file, 'r')
        training_data_list = training_data_file.readlines()
        training_data_file.close()

        records = training_data_list
        if (sample_size <= len(training_data_list)):
            records = random.sample(training_data_list, sample_size)

        #Train over epochs
        for e in range(epochs):
            for record in records:
                all_values = record.split(',')
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 #Preparing inputs from file, task/file specific
                targets = np.zeros(self.networkStructure[self.L - 1]) + 0.01
                targets[int(all_values[0])] = 0.99
                self.train(inputs, targets)
                
            
    

    def predict(self, inputs_list):
        #Predict the output given an input
        inputs = np.array(inputs_list, ndmin=2).T
        a = [None] * self.L
        z = [None] * self.L
        a[0] = inputs
        z[0] = inputs
        for i in range(1, self.L):
            z[i] = np.dot(self.W[i], a[i - 1]) + self.b[i]
            a[i] = self.activation(z[i])
        return a[self.L - 1]

#-----------------------------------------------------------------------------------------------------------

def neuralNetworkPerformanceTest(hidden_nodes, hidden_layers, learning_rate, training_sample_size, epochs):
    #--[Configure the Network]
    input_nodes = 784 #input_nodes should be left untouched
    output_nodes = 10 #output_nodes should be left untouched

    #These variables here can be played around with


    #--[Create and Train the Network]
    n = neuralNetwork(input_nodes, hidden_nodes, hidden_layers, output_nodes, learning_rate)
    n.train_from_file("mnist_train.csv", training_sample_size, epochs)

    #--[Test the Network]
    test_data_file = open("mnist_test.csv",'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    N = 5 #Number of repeats for the experiment

    performance = [0.0] * N #Performance for N experiments

    for i in range(N):
        scorecard = []
        n = neuralNetwork(input_nodes, hidden_nodes, hidden_layers, output_nodes, learning_rate)
        n.train_from_file("mnist_train.csv", training_sample_size, epochs)
        for record in test_data_list:
            all_values = record.split(',')
            correct_label = int(all_values[0])
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            outputs = n.predict(inputs)
            label = np.argmax(outputs)
            if (label == correct_label):
                scorecard.append(1)
            else:
                scorecard.append(0)
            
        performance[i] = sum(scorecard) / len(scorecard) * 100
    
    return performance

#-----------------------------------------------------------------------------------------------------------
hidden_nodes = 100
hidden_layers = 2
learning_rate = 0.2
training_sample_size = 60000
epochs = 3

performance = neuralNetworkPerformanceTest(hidden_nodes, hidden_layers, learning_rate, training_sample_size, epochs)
print("Performance Average: ", np.average(performance))
print("Performance Standard Deviation: ", np.std(performance))
print("Performance Raw Data: ", performance)