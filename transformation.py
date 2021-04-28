import csv
import os

import neural_network as nn
import matplotlib.pyplot as plt
import numpy as np

# Create target Directory
path = "results/transformation"
try:
    os.makedirs(path)
    print("Directory ", path, " Created ")
except FileExistsError:
    print("Directory ", path, " already exists")

# Load TXT
filename = str("data/transformation.txt")
with open(filename, 'r') as inFile:
    data = np.loadtxt(inFile, delimiter=" ")

n_epoch = 10000
epsilon = 0.001

# Prezentacja transformacji
inputs = 4
hidden_neuron = 2
outputs = 4
bias = 1
lr = 0.02
momentum = 0.005
approximation = 0

network = nn.NeuralNetwork(inputs, hidden_neuron, outputs, bias, lr, momentum, approximation)
network.train_network(data, 50000, 0.001)
for row in data:
    print(row[:inputs])
    print(network.predict(row))


# Ponizej znajduja sie instrukcje dotyczace wykonywania wykresow i tabel

# print("Training with bias and 1 hidden neuron: ")
# network_wb_1n = nn.NeuralNetwork(4, 1, 4, 1, 0.1, 0)
# error_history_wb_1n, epoch_wb_1n = network_wb_1n.train_network(data, n_epoch, epsilon)
#
# print("Training with bias and 2 hidden neurons: ")
# network_wb_2n = nn.NeuralNetwork(4, 2, 4, 1, 0.1, 0)
# error_history_wb_2n, epoch_wb_2n = network_wb_2n.train_network(data, n_epoch, epsilon)
#
# with open('results/transformation/predictedAnswersWithBias.txt', 'w', newline='') as fileOut:
#     writer = csv.writer(fileOut, delimiter=' ',
#                         quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['output1', 'output2'])
#     for row in data:
#         network_wb_2n.predict(row)
#         writer.writerow(network_wb_2n.hidden_outputs)
#
# print("Training with bias and 3 hidden neurons: ")
# network_wb_3n = nn.NeuralNetwork(4, 3, 4, 1, 0.1, 0)
# error_history_wb_3n, epoch_wb_3n = network_wb_3n.train_network(data, n_epoch, epsilon)
#
# print("Training without bias and 1 hidden neuron: ")
# network_wob_1n = nn.NeuralNetwork(4, 1, 4, 0, 0.1, 0)
# error_history_wob_1n, epoch_wob_1n = network_wob_1n.train_network(data, n_epoch, epsilon)
#
# print("Training without bias and 2 hidden neurons: ")
# network_wob_2n = nn.NeuralNetwork(4, 2, 4, 0, 0.1, 0)
# error_history_wob_2n, epoch_wob_2n = network_wob_2n.train_network(data, n_epoch, epsilon)
#
# with open('results/transformation/predictedAnswersWithoutBias.txt', 'w', newline='') as fileOut:
#     writer = csv.writer(fileOut, delimiter=' ',
#                         quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['output1', 'output2'])
#     for row in data:
#         network_wob_2n.predict(row)
#         writer.writerow(network_wob_2n.hidden_outputs)
#
# print("Training without bias and 3 hidden neurons: ")
# network_wob_3n = nn.NeuralNetwork(4, 3, 4, 0, 0.1, 0)
# error_history_wob_3n, epoch_wob_3n = network_wob_3n.train_network(data, n_epoch, epsilon)
#
# plt.title("Error with bias")
# plt.xlabel('epoch')
# plt.ylabel('MSE')
# plt.grid(alpha=.4, linestyle='--')
# plt.plot(range(epoch_wb_1n), error_history_wb_1n, 'red', label='1 neuron')
# plt.plot(range(epoch_wb_2n), error_history_wb_2n, 'green', label='2 neurons')
# plt.plot(range(epoch_wb_3n), error_history_wb_3n, 'blue', label='3 neurons')
# plt.legend()
# plt.savefig('results/transformation/transformation_error_with_bias.png')
# plt.clf()
#
# plt.title("Error without bias")
# plt.xlabel('epoch')
# plt.ylabel('MSE')
# plt.grid(alpha=.4, linestyle='--')
# plt.plot(range(epoch_wob_1n), error_history_wob_1n, 'red', label='1 neuron')
# plt.plot(range(epoch_wob_2n), error_history_wob_2n, 'green', label='2 neurons')
# plt.plot(range(epoch_wob_3n), error_history_wob_3n, 'blue', label='3 neurons')
# plt.legend()
# plt.savefig('results/transformation/transformation_error_without_bias.png')
# plt.clf()
#
# # Write table to TXT
# iterations = []
# iterations_sum = 0
# with open('results/transformation/transformation_avgLearningTime.txt', 'w', newline='') as fileOut:
#     writer = csv.writer(fileOut, delimiter=' ',
#                         quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['mom', 'lr', 'iter'])
#     for i in range(100):  # THIS RUNS FOR A LONG TIME!!! BECAUSE ITS 100 times training the neural network
#         momentum = round(np.random.uniform(0.00001, 0.05), 4)
#         print(momentum)
#         learning_rate = round(np.random.uniform(0.01, 1), 2)
#         print(learning_rate)
#         network = nn.NeuralNetwork(4, 2, 4, 1, learning_rate, momentum, 0, 0)
#         iteration = network.train_network(data, 1000000, 0.01)[1]
#         iterations.append(iteration)
#         iterations_sum += iteration
#         print(iteration)
#         writer.writerow([momentum, learning_rate, iteration])
#     avg = iterations_sum / len(iterations)
#     standard_deviation = 0
#     for i in iterations:
#         standard_deviation += ((i - avg) ** 2)
#     standard_deviation /= len(iterations)
#     standard_deviation = np.sqrt(standard_deviation)
#     writer.writerow([standard_deviation])
