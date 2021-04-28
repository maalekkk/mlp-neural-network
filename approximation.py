import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import neural_network as nn

# Create target Directory
path = "results/approximation"
try:
    os.makedirs(path)
    print("Directory ", path, " Created ")
except FileExistsError:
    print("Directory ", path, " already exists")

# Load TXT
filename = str("data/approx_1.txt")
with open(filename, 'r') as inFile:
    train_data = np.loadtxt(inFile, delimiter=" ")

filename = str("data/approx_2.txt")
with open(filename, 'r') as inFile2:
    train_data2 = np.loadtxt(inFile2, delimiter=" ")

filename = str("data/approx_test.txt")
with open(filename, 'r') as inFile2:
    test_data = np.loadtxt(inFile2, delimiter=" ")

# Prediction of a network after learning
x = np.arange(-4, 4, 0.01)
color = ['brown', 'red', 'green', 'blue', 'orange', 'purple', 'yellow']

def predict(network):
    y = list()
    for row in x:
        y.append(network.predict([row]))
    return y

# Prezentacja aproksymacji
hidden_neuron = 10
bias = 1
n_of_inputs = 1
n_of_outputs = 1
n_epoch = 2000
epsilon = 0.01
l_rate = 0.01
m_rate = 0.005
approximation = 1

network4 = nn.NeuralNetwork(n_of_inputs, hidden_neuron, n_of_outputs, bias, l_rate, m_rate, approximation)
plt.title("Approximation Presentation\n" +
          "(training data 1)")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(alpha=.4, linestyle='--')
network4.train_network(train_data, 1000, epsilon)
plt.plot(x, predict(network4), c=color[1], label=str(1000) + " epoch")
plt.scatter(train_data[:, 0], train_data[:, 1], c="black", label="Input data")
plt.legend()
plt.show()
plt.clf()


# # # Plot approximation throughout epochs for training data 1
# time_jump = 400
# plt.title("Approximation throughout epochs\n" +
#           "(training data 1)")
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.grid(alpha=.4, linestyle='--')
# network4 = nn.NeuralNetwork(n_of_inputs, 6, n_of_outputs, 1, l_rate, m_rate, approximation)
# predict(network4)
# plt.plot(x, predict(network4), c=color[0], label=str(0) + " epoch")
# for j in range(1, 6):
#     network4.train_network(train_data, time_jump, epsilon)
#     plt.plot(x, predict(network4), c=color[j], label=str(j * time_jump) + " epoch")
# plt.scatter(train_data[:, 0], train_data[:, 1], c="black", label="Input data")
# plt.legend()
# plt.savefig('results/approximation/approximation_throughout_epochs_data_1.png')
# plt.clf()
#
# # Plot approximation throughout epochs for training data 2
# plt.title("Approximation throughout epochs\n" +
#           "(training data 2)")
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.grid(alpha=.4, linestyle='--')
# network4 = nn.NeuralNetwork(n_of_inputs, 6, n_of_outputs, 1, l_rate, m_rate, approximation)
# predict(network4)
# plt.plot(x, predict(network4), c=color[0], label=str(0) + " epoch")
# for j in range(1, 6):
#     network4.train_network(train_data2, time_jump, epsilon)
#     plt.plot(x, predict(network4), c=color[j], label=str(j * time_jump) + " epoch")
# plt.scatter(train_data2[:, 0], train_data2[:, 1], c="black", label="Input data")
# plt.legend()
# plt.savefig('results/approximation/approximation_throughout_epochs_data_2.png')
# plt.clf()


# Plot approximation
def plott(num, step, data):
    error_hist = list()
    epoch = list()
    color = ['g', 'r', 'c', 'm', 'y', 'lime', 'darkslategray']
    tmp = 0
    for i in range(1, num, step):
        network = nn.NeuralNetwork(n_of_inputs, i, n_of_outputs, 1, l_rate, m_rate, approximation)
        error, it = network.train_network(data, n_epoch, epsilon)
        lab = str(i) + ' neuron(s)'
        temp = list()
        for r in x:
            temp.append(network.predict([r]))
        plt.plot(x, temp, c=color[tmp], label=lab)
        error_hist.append(error)
        epoch.append(it)
        tmp += 1


def error(steps, neurons, data, test_data):
    color = ['g', 'r', 'c', 'm', 'y', 'lime', 'darkslategray']
    # train data error
    for i in range(steps):
        network = nn.NeuralNetwork(n_of_inputs, neurons[i], n_of_outputs, 1, l_rate, m_rate, approximation)
        error, it = network.train_network(data, n_epoch, epsilon)
        lab = str(neurons[i]) + ' (train)'
        plt.plot(range(it), error, label=lab, c=color[i])
    # test data error
    for i in range(steps):
        network = nn.NeuralNetwork(n_of_inputs, neurons[i], n_of_outputs, 1, l_rate, m_rate, approximation)
        error, it = network.train_network(test_data, n_epoch, epsilon)
        lab = str(neurons[i]) + ' (test)'
        plt.plot(range(it), error, label=lab, c=color[i + steps])


# # Training data 1
# plt.title("Approximation result for data 1")
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.grid(alpha=.4, linestyle='--')
# plt.scatter(test_data[:, 0], test_data[:, 1], c='b', label="Test data", s=3)
# plott(20, 4, train_data)
# plt.legend()
# plt.savefig('results/approximation/approximation_result_data_1.png')
# plt.clf()
#
# # data2
# plt.title("Approximation result for data 2")
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.grid(alpha=.4, linestyle='--')
# plt.scatter(test_data[:, 0], test_data[:, 1], c='b', label="Test data", s=3)
# plott(20, 4, train_data2)
# plt.legend()
# plt.savefig('results/approximation/approximation_result_data_2.png')
# plt.clf()
#
# # Error for 1, 5, 19 neurons in hidden layer, data1
# neurons = [1, 5, 19]
# steps = 3
# plt.title("Error - first training set" + '\nlearning rate = ' + str(l_rate) + ', momentum = ' + str(m_rate))
# plt.xlabel('epoch')
# plt.ylabel('MSE')
# plt.grid(alpha=.4, linestyle='--')
# error(steps, neurons, train_data, test_data)
# plt.legend()
# plt.savefig('results/approximation/approximation_error_1.png')
# plt.clf()
#
# # data2
# plt.title("Error - second training set" + '\nlearning rate = ' + str(l_rate) + ', momentum = ' + str(m_rate))
# plt.xlabel('epoch')
# plt.ylabel('MSE')
# plt.grid(alpha=.4, linestyle='--')
# error(steps, neurons, train_data2, test_data)
# plt.legend()
# plt.savefig('results/approximation/approximation_error_2.png')
# plt.clf()


def mse_avg_table(epoch, epsilon, lr, momentum, train_data, test_data, file_name):
    with open(file_name, 'w', newline='') as fileOut:
        writer = csv.writer(fileOut, delimiter=' ',
                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['hidden_neuron', 'avg_mse_td', 'standard_deviation_td', 'avg_mse_ed', 'standard_deviation_ed'])
        for i in range(20):  # THIS RUNS FOR A LONG TIME!!! BECAUSE ITS 100 times training the neural network
            sum_error_td = sum_error_ed = 0
            errors_td = []
            errors_ed = []
            print(i + 1)
            for j in range(3):
                network = nn.NeuralNetwork(1, i + 1, 1, 1, l_rate, m_rate, 1, 0)
                error_td = network.train_network(train_data, epoch, epsilon)[0][-1]
                sum_error_td += error_td
                errors_td.append(error_td)
                error_ed = network.train_network(test_data, epoch, epsilon)[0][-1]
                sum_error_ed += error_ed
                errors_ed.append(error_ed)
            errors_avg_td = sum_error_td / len(errors_td)
            errors_avg_ed = sum_error_ed / len(errors_ed)
            standard_deviation_td = standard_deviation_ed = 0
            for j in errors_td:
                standard_deviation_td += ((j - errors_avg_td) ** 2)
            for j in errors_ed:
                standard_deviation_ed += ((j - errors_avg_ed) ** 2)
            standard_deviation_td /= len(errors_td)
            standard_deviation_ed /= len(errors_ed)
            standard_deviation_td = np.sqrt(standard_deviation_td)
            standard_deviation_ed = np.sqrt(standard_deviation_ed)
            writer.writerow([i + 1, round(errors_avg_td, 4), round(standard_deviation_td, 4), round(errors_avg_ed, 4),
                             round(standard_deviation_ed, 4)])


epoch = 3000
epsilon = 0.0000001
# mse_avg_table(epoch, epsilon, l_rate, m_rate, train_data, test_data,
#               'results/approximation/approximation_avg_mse_train_data_1.txt')
# mse_avg_table(epoch, epsilon, l_rate, m_rate, train_data2, test_data,
#               'results/approximation/approximation_avg_mse_train_data_2.txt')
