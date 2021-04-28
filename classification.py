import csv
import os

import neural_network as nn
import matplotlib.pyplot as plt
import numpy as np

# Create target Directory
path = "results/classification"
try:
    os.makedirs(path)
    print("Directory ", path, " Created ")
except FileExistsError:
    print("Directory ", path, " already exists")

# Load TXT
filename = str("data/iris.data")
dataTrain = np.genfromtxt(filename, delimiter=',', dtype=['<f8', '<f8', '<f8', '<f8', 'U15'],
                          names=('sepal length', 'sepal width', 'petal length', 'petal width', 'label'))


# transform data from [1,1,1,1,Iris-setosa] to [1,1,1,1,0,0,1]
def transform_data(old_data, row_n, inputs_n, outputs_n, columns=None):
    if columns is None:
        columns = range(inputs_n)
    temp = np.zeros(shape=(len(old_data), len(columns)))
    new_data = np.ndarray(shape=(row_n, inputs_n + outputs_n))
    for i in range(len(old_data)):
        j = 0
        for col in columns:
            temp[i][j] = old_data[i][col]
            j += 1
        pom = temp[i]
        if old_data[i][-1] == 'Iris-setosa':
            new_data[i] = np.append(pom, [1, 0, 0])
        elif old_data[i][-1] == 'Iris-versicolor':
            new_data[i] = np.append(pom, [0, 1, 0])
        elif old_data[i][-1] == 'Iris-virginica':
            new_data[i] = np.append(pom, [0, 0, 1])
    return new_data


# division data to train_data and test_data
def data_division(dataTrain, train_data_percent):
    amount_of_data_types = len(dataTrain) / 3
    division = round(amount_of_data_types * train_data_percent)
    data_train = []
    data_test = []
    for i in range(0, division):
        data_train.append(dataTrain[i])
    for i in range(division, round(amount_of_data_types)):
        data_test.append(dataTrain[i])
    for i in range(round(amount_of_data_types), round(amount_of_data_types) + division):
        data_train.append(dataTrain[i])
    for i in range(round(amount_of_data_types) + division, round(amount_of_data_types) * 2):
        data_test.append(dataTrain[i])
    for i in range(round(amount_of_data_types) * 2, round(amount_of_data_types) * 2 + division):
        data_train.append(dataTrain[i])
    for i in range(round(amount_of_data_types) * 2 + division, len(dataTrain)):
        data_test.append(dataTrain[i])
    return data_train, data_test


# Testing network, testing data, result data, number of inputs, correctness percentage
def classification_test(t_network, t_data_conv, t_data, inputs_n, correct_p):
    i = correct = 0
    for row in t_data_conv:
        result = ''
        pred = t_network.predict(row[:inputs_n])
        if pred[0] > correct_p:
            result = 'Iris-setosa'
        elif pred[1] > correct_p:
            result = 'Iris-versicolor'
        elif pred[2] > correct_p:
            result = 'Iris-virginica'
        if result == t_data[i][-1]:
            correct += 1
        i += 1
    return correct * 100 / len(t_data)


def accuracy_ploting(inputs_n, tr_data, te_data, t_epoch, t_epsilon, correct_p, part):
    plt.title("Classification accuracy\n" +
              "(number of inputs " + str(inputs_n) + ", columns " + str(part) + ")")
    plt.xlabel('number of neurons')
    plt.ylabel('Percentage of correct classifications')
    plt.grid(alpha=.4, linestyle='--')
    for n_neurons in np.arange(1, 18, 4):
        net = nn.NeuralNetwork(inputs_n, n_neurons, 3, 1, 0.6, 0.1, 0)
        net.train_network(tr_data, t_epoch, t_epsilon)
        plt.bar(str(n_neurons), classification_test(net, te_data, data_test, inputs_n, correct_p))
    plt.savefig('results/classification/classification_accuracy_' + str(inputs_n) + '_inputs_c' + str(part) + '.png')
    plt.clf()


n_epoch = 1000
epsilon = 0.001
epoch = 150
accuracy = 0.5

# Prezentacja klasyfikacji
inputs = 4
hidden_neuron = 3
outputs = 3
bias = 1
l_rate = 0.1
m_rate = 0.05
approx = 0
seed = 0

network = nn.NeuralNetwork(inputs, hidden_neuron, outputs, bias, l_rate, m_rate, approx, seed)
data_train, data_test = data_division(dataTrain, 0.8)
data_train_conv = transform_data(data_train, len(data_train), 4, 3)
data_test_conv = transform_data(data_test, len(data_test), 4, 3)
network.train_network(data_train_conv, 1000, epsilon)
effect_train_data = classification_test(network, data_test_conv, data_test, 4, 0.5)
print('Skutecznosc klasyfikacji: ',effect_train_data)



# Testing neural network with 1 Input
for i in range(4):
    print("Testing neural network with 1 Input, for data column " + str(i))
    training_data = transform_data(data_train, len(data_train), 1, 3, [i])
    test_data = transform_data(data_test, len(data_test), 1, 3, [i])
    accuracy_ploting(1, training_data, test_data, epoch, epsilon, accuracy, i + 1)

# Testing neural network with 2 Inputs
for i in range(4):
    for j in range(i + 1, 4):
        print("Testing neural network with 2 Inputs, for data columns [" + str(i) + ", " + str(j) + "]")
        training_data = transform_data(data_train, len(data_train), 2, 3, [i, j])
        test_data = transform_data(data_test, len(data_test), 2, 3, [i, j])
        accuracy_ploting(2, training_data, test_data, epoch, epsilon, accuracy, [i, j])

# Testing neural network with 3 Inputs
for i in range(4):
    for j in range(i + 1, 4):
        for k in range(j + 1, 4):
            print("Testing neural network with 3 Inputs, for data columns [" + str(i) + ", " + str(j) + ", " + str(
                k) + "]")
            training_data = transform_data(data_train, len(data_train), 3, 3, [i, j, k])
            test_data = transform_data(data_test, len(data_test), 3, 3, [i, j, k])
            accuracy_ploting(3, training_data, test_data, epoch, epsilon, accuracy, [i, j, k])

# Testing neural network with 4 Inputs
print("Testing neural network with 4 Inputs, for data columns [0,1,2,3]")
train_data = transform_data(data_train, len(data_train), 4, 3)
test_data = transform_data(data_test, len(data_test), 4, 3)
accuracy_ploting(4, train_data, test_data, epoch, epsilon, accuracy, [0, 1, 2, 3])


error_history, epoch = network.train_network(data_train_conv,epoch,epsilon)
plt.title("Classification error")
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.grid(alpha=.4, linestyle='--')
plt.plot(range(epoch), error_history, 'red', label="Error")
plt.legend()
plt.savefig('results/classification/classification_error.png')
plt.clf()


# liczba epok, epsilon(musi byc bardzo maly), learn_rate, momentunm, dane do trenowania zamienione (1->1,0,0),
# dane do testowania, dane do testowania zamienione, liczba wejsc, dopuszczalny blad skutecznosci klasyf.,
# nazwa pliku wyjsciowego
def classification_test_table(epoch, epsilon, lr, momentum, train_data_conv, test_data, test_data_conv, n_inputs,
                              correct_p, file_name):
    with open(file_name, 'w', newline='') as fileOut:
        writer = csv.writer(fileOut, delimiter=' ',
                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ['hidden_neurons', 'effect_train', 'standard_deviation_train'])
        for i in range(20):  # THIS RUNS FOR A LONG TIME!!! BECAUSE ITS 100 times training the neural network
            print(i + 1)
            sum_effect_train_data = 0
            effects_train_data = []
            for j in range(3):
                t_network = nn.NeuralNetwork(n_inputs, i + 1, 3, 1, lr, momentum, 0, 0)
                t_network.train_network(train_data_conv, epoch, epsilon)
                effect_train_data = classification_test(t_network, test_data_conv, test_data, n_inputs, correct_p)
                sum_effect_train_data += effect_train_data
                effects_train_data.append(effect_train_data)
                print('Skutecznosc1: ', effect_train_data)
            effect_avg_train_data = sum_effect_train_data / len(effects_train_data)
            standard_deviation_train_data = 0
            for j in effects_train_data:
                standard_deviation_train_data += ((j - effect_avg_train_data) ** 2)
            standard_deviation_train_data /= len(effects_train_data)
            standard_deviation_train_data = np.sqrt(standard_deviation_train_data)
            print('Ilosc neuronow: ', i + 1, round(effect_avg_train_data, 4), round(standard_deviation_train_data, 4))
            writer.writerow([i + 1, round(effect_avg_train_data, 4), round(standard_deviation_train_data, 4)])


lr = 0.6
momentum = 0.1
classification_test_table(n_epoch, 0.0000001, lr, momentum, data_train_conv, data_test, data_test_conv, 4, 0.5,
'results/classification/classification_effect_table.txt')
