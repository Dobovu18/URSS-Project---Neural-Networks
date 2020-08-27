import numpy as np
import scipy.special
import matplotlib.pyplot as plt

#Code was followed from the book 'Make Your Own Neural Network' by Tariq Rashid
#I will by editting to add biases, more layers and greater generality to the code


class neuralNetwork:

    #Things to add:
        #Support for adding extra hidden layers
        #Adding biases

    #initialising the neural network
    def __init__(self, input_nodes, number_of_hidden_layers, hidden_nodes, output_nodes, learning_rate):
        #initialising the layers
        self.innodes = input_nodes
        self.n_hidlayers = number_of_hidden_layers
        self.hidnodes = hidden_nodes
        self.outnodes = output_nodes

        #learning rate
        self.lr = learning_rate

        #initialising weights
        self.wih = np.random.normal(0.0, pow(self.hidnodes, -0.5), (self.hidnodes, self.innodes))
        self.who = np.random.normal(0.0, pow(self.outnodes, -0.5), (self.outnodes, self.hidnodes))
        
        #activation function and its derivative
        self.activation = lambda x: scipy.special.expit(x) #1 / (1 + np.exp(-x)) #
        self.d_activation = lambda x: self.activation(x)*(1 - self.activation(x))
        pass

    #training the network
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T

        #calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        #calculate signals leaving the hidden layer
        hidden_outputs = self.activation(hidden_inputs)

        #calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        #calculate signals leaving final output layer
        final_outputs = self.activation(final_inputs)

        #error management and backpropagation
        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.who.T, output_errors)

        #updating the weights for the links between hidden and output layers
        self.who += self.lr * np.dot((output_errors * self.d_activation(final_inputs)), np.transpose(hidden_outputs))

        pass

    #query the network
    def query(self, inputs_list):
        #Add multiple hidden layers in future

        #Convert inputs list to 2D array
        inputs = np.array(inputs_list, ndmin = 2).T

        #calculate signals entering hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        #calculating signals emerging from hidden layer
        hidden_outputs = self.activation(hidden_inputs)

        #signals entering final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        #signals emerging from final output layer
        final_outputs = self.activation(final_inputs)

        return final_outputs

    pass

#Load in training data
training_data_file = open("mnist_train.csv")
training_data_list = training_data_file.readlines()
training_data_file.close()

#Load in testing data
test_data_file = open("mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

def learning_rate_vs_accuracy(learning_rate):
    #number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    #learning rate

    #creating the neural network
    n = neuralNetwork(input_nodes, 1, hidden_nodes, output_nodes, learning_rate)

    #reading in the mnist training data 
    

    #train the neural network
    #going through all records in the training dataset for record in trainingdatalist
    for record in training_data_list:
        all_values = record.split(',') #separates the record at commas
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 #scaling and shifitng input
        #all_values[0] is the correct output, all_values[1:] gives the greyscale values for each pixel in the 28x28 img
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        #previous two lines make a vector that is 0.01 for the target everywhere but 0.99 for the correct corresnponding component

        n.train(inputs, targets)
        pass

    #now to test the network


    #Scorecard
    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])

        #scaling and shifting inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        #query the network
        outputs = n.query(inputs)

        #component of output vector with greatest value corresnponds with the value the network believes image is
        label = np.argmax(outputs)
        #print(label, "network's answer")

        #Is the label correct or incorrect?
        if (label == correct_label):
            scorecard.append(1) #add 1 to the scorecard if network is correct
        else:
            scorecard.append(0) #add 0 to the scorecard if network is incorrect
            pass
        pass
    accuracy = sum(scorecard)/len(scorecard) * 100
    
    return accuracy


X = np.linspace(0, 3, 30)
Y = []
for x in X:
    Y.append(learning_rate_vs_accuracy(x))

plt.plot(X,Y)
plt.show()