import numpy as np


class Neuron:
    def __init__(self, bias, l_rate, num_of_inputs, m_rate, index, seed):
        # Bias
        self.bias = bias
        # Learning rate
        self.l_rate = l_rate
        # Momentum rate
        self.m_rate = m_rate
        # Number of inputs
        self.num_of_inputs = num_of_inputs
        if seed:
            np.random.seed(4)
        self.weights = np.random.uniform(-1, 1, num_of_inputs + bias)
        # Backward propagation error
        self.current_error = 0
        # Backward propagation error (last iteration)
        self.last_error = np.zeros(self.num_of_inputs + bias)
        self.index = index

    def sigmoid(self, activation):
        return 1 / (1 + np.power(np.e, -activation))

    def derivative(self, x):
        fx = self.sigmoid(x)
        return fx * (1 - fx)

    def update(self, inputs):
        if self.bias:
            tmp = self.last_error[-1]
            self.last_error[-1] = self.current_error * self.l_rate - self.m_rate * tmp
            self.weights[-1] -= self.last_error[-1]
        for i in range(self.num_of_inputs):
            tmp = self.last_error[i]
            self.last_error[i] = self.current_error * self.l_rate * inputs[i] - self.m_rate * tmp
            self.weights[i] -= self.last_error[i]

    def weight_fun(self, row):
        weight_sum = 0
        if self.bias:
            weight_sum += self.weights[-1]
        for i in range(self.num_of_inputs):
            weight_sum += self.weights[i] * row[i]
        return weight_sum


class NeuronHidden(Neuron):
    def __init__(self, bias, l_rate, num_of_inputs, m_rate, index, seed, output_layer):
        Neuron.__init__(self, bias, l_rate, num_of_inputs, m_rate, index, seed)
        # List of neurons in the output layer
        self.outputLayer = output_layer

    def backward_propagation_error(self, activation_value):
        self.current_error = 0
        for neuron in self.outputLayer:
            self.current_error += neuron.weights[self.index] * neuron.current_error * self.derivative(activation_value)


class NeuronOutput(Neuron):
    def __init__(self, bias, l_rate, num_of_inputs, m_rate, index, seed, approximation=0):
        Neuron.__init__(self, bias, l_rate, num_of_inputs, m_rate, index, seed)
        self.approximation = approximation

    def sigmoid(self, activation):
        if self.approximation:
            return activation
        else:
            return 1 / (1 + np.power(np.e, -activation))

    def derivative(self, x):
        if self.approximation:
            return 1
        else:
            return self.sigmoid(x) * (1 - self.sigmoid(x))

    def backward_propagation_error(self, weight_value, expected_value):
        self.current_error = (self.sigmoid(weight_value) - expected_value) * self.derivative(weight_value)