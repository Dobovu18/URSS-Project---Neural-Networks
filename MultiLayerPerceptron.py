import numpy as np
import scipy.special
import random
import matplotlib.pyplot as plt
import ActivationFunctions as activation

#Neural network designed to classify images of pictures
#Training set of data: http://www.pjreddie.com/media/files/mnist_train.csv
#Test set of data: http://www.pjreddie.com/media/files/mnist_test.csv
#Code inspired by 'Make Your Own Neural Network' by Tariq Rashid and 
#'Deep Learning: An Introduction for Applied Mathematicians' by Catherine F. Higham and Desmond J. Higham.


class neuralNetwork:
    #Multi-layer perceptron neural network
    def __init__(self, inputnodes, hiddennodes, hiddenlayers, outputnodes, learningrate, training_file, sample_size, activation_function, activation_derivative):
        #Create Network Structure
        self.L = hiddenlayers + 2
        self.networkStructure = [hiddennodes] * self.L
        self.networkStructure[0] = inputnodes
        self.networkStructure[self.L - 1] = outputnodes
        #Configure Weights and Biases

        self.W = [None] * self.L
        self.b = [None] * self.L

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
            self.records = random.sample(self.training_data_list, self.sample_size)
        elif(self.sample_size > len(self.training_data_file)):
            self.records = self.training_data_list

        #Plotting cost function as neural network learns
        #self.counter = 0
        #self.costval = []

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
            random.shuffle(self.records)
            for record in self.records:
                all_values = record.split(',')
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 #Preparing inputs from file, task/file specific
                targets = np.zeros(self.networkStructure[self.L - 1]) + 0.01
                targets[int(all_values[0])] = 0.99
                self.train(inputs, targets)
                #self.counter += 1
                #self.costval.append(self.getCost())
    

    def getCost(self):
        #edit... use np.mean
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

def neuralNetworkPerformanceTest(input_nodes = 784, output_nodes = 2, hidden_nodes = 16, hidden_layers = 2, learning_rate = 0.8, training_sample_size = 20000, epochs = 5, number_of_experiments = 1, training_file = "mnist_train.csv", test_file = "mnist_test.csv", activation_function = activation.Sigmoid, activation_derivative = activation.d_Sigmoid):
    #--[Test the Network]
    test_data_file = open(test_file,'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    N = number_of_experiments #Number of repeats for the experiment

    performance = [0.0] * N #Performance for N experiments

    for i in range(N):
        scorecard = []
        n = neuralNetwork(input_nodes, hidden_nodes, hidden_layers, output_nodes, learning_rate, training_file, training_sample_size, activation_function, activation_derivative)
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

def neuralNetworkCostFunctionTest(input_nodes = 784, output_nodes = 2, hidden_nodes = 16, hidden_layers = 2, learning_rate = 0.8, training_sample_size = 20000, epochs = 20, number_of_experiments = 3, training_file = "mnist_train.csv", train = True, activation_function = activation.Sigmoid, activation_derivative = activation.d_Sigmoid):
    N = number_of_experiments
    cost = [0.0] * N

    for i in range(N):
        n = neuralNetwork(input_nodes, hidden_nodes, hidden_layers, output_nodes, learning_rate, training_file, training_sample_size, activation_function, activation_derivative)
        n.train_from_file(epochs)
        cost[i] = n.getCost()
    
    return cost

#[Variables]

#hiddenLayers = np.array(range(1, 11, 1)) #✔
#hiddenNodes = np.array(range(25, 625, 25)) #✔
#learningRate = np.linspace(0, 3, 61) #✔
#sampleSize = np.array([100, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]) #✔
#epoch = np.array(range(1, 11)) #✔
#----------------------------------------------------------------------------------------------------------------

#print(neuralNetworkCostFunctionTest(training_file = "mnist_binary_train.csv", training_sample_size = 12665))
print(neuralNetworkPerformanceTest(
    input_nodes = 784,
    output_nodes = 10,
    hidden_nodes = 200,
    hidden_layers = 2,
    learning_rate = 1E-3,
    training_sample_size = 60000,
    epochs = 1,
    number_of_experiments = 3,
    training_file = 'mnist_train.csv',
    test_file = 'mnist_test.csv',
    activation_function = activation.LeakyReLU,
    activation_derivative = activation.d_LeakyReLU
))

#--[Tracking Neural Network Training Progress]---------------------------------------------------------------
#neuralNetwork_1 = neuralNetwork(784, 100, 1, 10, 0.8, "mnist_train.csv", 100)
#neuralNetwork_1.train_from_file(10)
#plt.plot(neuralNetwork_1.getTrainingInfo()[0], neuralNetwork_1.getTrainingInfo[1])



#neuralNetwork_2 = neuralNetwork(784, 600, 5, 10, 3.0, "mnist_train.csv", 20000)

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