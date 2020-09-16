import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import ActivationFunctions as af
import random

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.wih = np.random.normal(0.0, self.hnodes**-0.5, (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, self.onodes**-0.5, (self.onodes, self.hnodes))

        self.lr = learningrate

        self.activation = lambda x: af.ReLU(x)
        self.d_activation = lambda x: af.d_ReLU(x)
        
        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.lr * np.dot((output_errors * self.d_activation(final_inputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * self.d_activation(hidden_inputs)), np.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation(final_inputs)

        return final_outputs

input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 1E-3
epochs = 3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

test_data_file = open("mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass
pass
performance = sum(scorecard) / len(scorecard) * 100
print("Performance: ", performance)

#record = test_data_list[random.randint(0, 9999)]
#print("Selected Test: ", record[0])
#all_values = record.split(',')
#correct_label = int(all_values[0])
#inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
##inputs = np.asfarray(all_values[1:])
#outputs = n.query(inputs)
#label = np.argmax(outputs)
#print("Activation outputs:\n", outputs)
#print("Label: ", label, " @ learning rate", learning_rate)

#img_array = np.asfarray(all_values[1:]).reshape((28,28))
#plt.imshow(img_array, cmap = 'Greys', interpolation = 'None')
#plt.show()
