import numpy as np
import scipy.special
import random
import matplotlib.pyplot as plt

#Neural network designed to classify images of pictures
#Training set of data: http://www.pjreddie.com/media/files/mnist_train.csv
#Test set of data: http://www.pjreddie.com/media/files/mnist_test.csv
#Code inspired by 'Make Your Own Neural Network' by Tariq Rashid and 
#'Deep Learning: An Introduction for Applied Mathematicians' by Catherine F. Higham and Desmond J. Higham.

class neuralNetwork:
    #Multi-layer perceptron neural network
    def __init__(self, inputnodes, hiddennodes, hiddenlayers, outputnodes, learningrate, training_file, sample_size):
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

        #The training data:
        self.training_file = training_file
        self.training_data_file = open(self.training_file, 'r')
        self.training_data_list = self.training_data_file.readlines()
        self.training_data_file.close()
        self.sample_size = sample_size

        self.records = self.training_data_list
        if (self.sample_size <= len(self.training_data_list)):
            self.records = random.sample(self.training_data_list, self.sample_size)
        elif(self.sample_size > len(self.training_data_file)):
            self.records = self.training_data_list

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
    
    def train_from_file(self, epochs):
        #Train over epochs
        for e in range(epochs):
            random.shuffle(self.records)
            for record in self.records:
                all_values = record.split(',')
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 #Preparing inputs from file, task/file specific
                targets = np.zeros(self.networkStructure[self.L - 1]) + 0.01
                targets[int(all_values[0])] = 0.99
                self.train(inputs, targets)

    def getCost(self):
        #Calculating the cost/empircial risk of the neural network
        #Train the neural network first before calculating the cost
        cost = 0
        for record in self.records:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 #Preparing inputs from file, task/file specific
            targets = np.zeros(self.networkStructure[self.L - 1]) + 0.01
            targets[int(all_values[0])] = 0.99
            err = targets - self.predict(inputs).T
            cost += pow(np.linalg.norm(err),2)
        cost /= 2 * self.sample_size
        return cost

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

def neuralNetworkPerformanceTest(hidden_nodes = 100, hidden_layers = 2, learning_rate = 0.5, training_sample_size = 20000, epochs = 1, number_of_experiments = 3, training_file = "mnist_train.csv"):
    #--[Configure the Network]
    input_nodes = 784 #input_nodes should be left untouched
    output_nodes = 10 #output_nodes should be left untouched

    #--[Test the Network]
    test_data_file = open("mnist_test.csv",'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    N = number_of_experiments #Number of repeats for the experiment

    performance = [0.0] * N #Performance for N experiments

    for i in range(N):
        scorecard = []
        n = neuralNetwork(input_nodes, hidden_nodes, hidden_layers, output_nodes, learning_rate, training_file, training_sample_size)
        n.train_from_file(epochs)
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

def neuralNetworkCostFunctionTest(hidden_nodes = 100, hidden_layers = 2, learning_rate = 0.5, training_sample_size = 20000, epochs = 1, number_of_experiments = 3, training_file = "mnist_train.csv", train = True):
    #--[Configure the Network]
    input_nodes = 784 #input_nodes should be left untouched
    output_nodes = 10 #output_nodes should be left untouched
    N = number_of_experiments

    cost = [0.0] * N

    for i in range(N):
        n = neuralNetwork(input_nodes, hidden_nodes, hidden_layers, output_nodes, learning_rate, training_file, training_sample_size)
        n.train_from_file(epochs)
        cost[i] = n.getCost()
    
    return cost

#[Variables]#\/\/\/\/\/\/\/\//\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

#hiddenLayers = np.array(range(1, 11, 1)) #✔
#hiddenNodes = np.array(range(25, 625, 25)) #✔
#learningRate = np.linspace(0, 3, 61) #✔
#sampleSize = np.array([100, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]) #✔
#epoch = np.array(range(1, 11)) #✔
#----------------------------------------------------------------------------------------------------------------

print(neuralNetworkCostFunctionTest())
print(neuralNetworkPerformanceTest())

#--[Neural Network Cost]-------------------------------------------------------------------------------------
#cost_avg = []
#cost_min_err = []
#cost_max_err = []

#for x in epoch:
#    cost = neuralNetworkCostFunctionTest(epochs = x) #change
    
#    avg = np.average(cost)
#    min_err = avg - np.min(cost)
#    max_err = np.max(cost) - avg

#    cost_avg.append(avg)
#    cost_min_err.append(min_err)
#    cost_max_err.append(max_err)

#cost_err = [cost_min_err, cost_max_err]

#--[Neural Network Performance]------------------------------------------------------------------------------
#performance_avg = []
#performance_min_err = []
#performance_max_err = []

#Processing values to plot on the graph
#for x in epoch: #Change
#        performance = neuralNetworkPerformanceTest(epochs = x) #Change

#        avg = np.average(performance) 
#        min_err = avg - np.min(performance)
#        max_err = np.max(performance) - avg

#        performance_avg.append(avg)
#        performance_min_err.append(min_err)
#        performance_max_err.append(max_err)

#performance_err = [performance_min_err, performance_max_err]

#--[Graph Creation]------------------------------------------------------------------------------------------
#legendLabel = "hidden layers = 2\nnodes per hidden layer = 100\nlearning rate = 0.5\ntraining set size = 20000" #change
#Plotting/Creating the graph
#plt.errorbar(epoch, cost_avg, #Change
#            yerr = cost_err,
#            label = legendLabel,
#            fmt = '-x',
#            ecolor = 'red',
#            capsize = 2)

#plt.title("Number Of Epochs vs Value Of Cost Function") #Change
#plt.xlabel("Number Of Epochs") #Change
#plt.ylabel("Value Of Cost Function")
#plt.legend()
#plt.grid(b=True,
#        which='major',
#        color='#666666', 
#        linestyle='-')
#plt.minorticks_on()
#plt.grid(b=True,
#        which='minor', 
#        color='#999999', 
#        linestyle='-', 
#        alpha=0.2)
#plt.show()