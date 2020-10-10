import numpy as np
import scipy.special
import random
import matplotlib.pyplot as plt
import ActivationFunctions as activation
import openpyxl as xl

#Neural network designed to classify images of pictures
#Training set of data: http://www.pjreddie.com/media/files/mnist_train.csv
#Test set of data: http://www.pjreddie.com/media/files/mnist_test.csv
#Code inspired by 'Make Your Own Neural Network' by Tariq Rashid and 
#'Deep Learning: An Introduction for Applied Mathematicians' by Catherine F. Higham and Desmond J. Higham.


class neuralNetwork:
    #Multi-layer perceptron (MLP) neural network
    def __init__(self, inputnodes = 784, hiddennodes = 200, hiddenlayers = 2, outputnodes = 10, learningrate = 1E-3, training_file = "mnist_train.csv", test_file = "mnist_test.csv", sample_size = 60000, activation_function = activation.Sigmoid, activation_derivative = activation.d_Sigmoid):
        #Create Network Structure
        self.L = hiddenlayers + 2
        self.networkStructure = [hiddennodes] * self.L
        self.networkStructure[0] = inputnodes
        self.networkStructure[self.L - 1] = outputnodes
        #Configure Weights and Biases

        self.W = [None] * self.L
        self.b = [None] * self.L
        #np.random.seed(0)
        #random.seed(0)

        #Xavier Initialisation
        for i in range(1, self.L):
            self.W[i] = np.random.normal(0.0, self.networkStructure[i]**-0.5, (self.networkStructure[i], self.networkStructure[i - 1]))
            self.b[i] = np.random.normal(0.0, self.networkStructure[i]**-0.5, (self.networkStructure[i], 1))

        #Learning rate
        self.lr = learningrate

        #Activation function and derivative
        self.activation = lambda x: activation_function(x)
        self.d_activation = lambda x: activation_derivative(x)

        #The training data:
        self.training_file = training_file
        self.training_data_file = open(self.training_file, 'r')
        self.training_data_list = self.training_data_file.readlines()
        self.training_data_file.close()
        self.sample_size = sample_size

        self.records = self.training_data_list
        if (self.sample_size <= len(self.training_data_list)):
            #random.seed(0)
            self.records = random.sample(self.training_data_list, self.sample_size)
        elif(self.sample_size > len(self.training_data_file)):
            self.records = self.training_data_list

        #Plotting cost function as neural network learns
        self.counter = 0
        self.performance_scores = []
        self.counter_values = []
        self.test_file = test_file
        #self.costval = []
    
    def reinitialise_parameters(self):
        self.counter = 0
        self.performance_scores = []
        self.counter_values = []
        #np.random.seed(0)
        for i in range(1, self.L):
            self.W[i] = np.random.normal(0.0, self.networkStructure[i]**-0.5, (self.networkStructure[i], self.networkStructure[i - 1]))
            self.b[i] = np.random.normal(0.0, self.networkStructure[i]**-0.5, (self.networkStructure[i], 1))
    
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
    
    def train_from_file(self, epochs):
        #Train over epochs
        for e in range(epochs):
            #random.seed(0)
            random.shuffle(self.records)
            for record in self.records:
                all_values = record.split(',')
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 #Preparing inputs from file, task/file specific
                targets = np.zeros(self.networkStructure[self.L - 1]) + 0.01
                targets[int(all_values[0])] = 0.99
                self.train(inputs, targets)
    
    def track_performance_and_train(self, epochs, interval):
        #Train over epochs
        for e in range(epochs):
            #random.seed(0)
            random.shuffle(self.records)
            for record in self.records:
                all_values = record.split(',')
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 #Preparing inputs from file, task/file specific
                targets = np.zeros(self.networkStructure[self.L - 1]) + 0.01
                targets[int(all_values[0])] = 0.99
                self.train(inputs, targets)
                if (self.counter in interval):
                    self.performance_scores.append(self.checkPerformance(self.test_file))
                self.counter += 1
        self.performance_scores.append(self.checkPerformance(self.test_file))  
    
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
    
    def checkPerformance(self, filename):
        test_data_file = open(filename, 'r')
        test_data_list = test_data_file.readlines()
        test_data_file.close()

        scorecard = []
        for record in test_data_list:
            all_values = record.split(',')
            correct_label = int(all_values[0])
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            outputs = self.predict(inputs)
            label = np.argmax(outputs)
            if (label == correct_label):
                scorecard.append(1)
            else:
                scorecard.append(0)
            
        performance = sum(scorecard) / len(scorecard) * 100
        return performance
    
    def getPerformanceValues(self):
        return self.performance_scores #[self.counter_values, self.performance_scores]

#-----------------------------------------------------------------------------------------------------------