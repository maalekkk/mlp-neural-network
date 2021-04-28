import numpy as np
import neuron as nn


class NeuralNetwork:
    def __init__(self, n_inputs, hidden_size, n_outputs, bias, l_rate=0.1, m_rate=0.0, approximation=0, seed=1):
        # Initialize list of hidden and output layers
        self.hidden_layer = []
        self.output_layer = []

        self.hidden_outputs = np.zeros(hidden_size)
        self.weight_value = np.zeros(n_outputs)

        self.num_of_inputs = n_inputs
        self.num_of_outputs = n_outputs
        self.approximation = approximation

        for i in range(n_outputs):
            self.output_layer.append(nn.NeuronOutput(bias, l_rate, hidden_size, m_rate, i, seed, approximation))
        for i in range(hidden_size):
            self.hidden_layer.append(nn.NeuronHidden(bias, l_rate, n_inputs, m_rate, i, seed, self.output_layer))

    def forward_prop(self, row):
        for neuron in self.hidden_layer:
            activation = neuron.weight_fun(row)
            self.hidden_outputs[neuron.index] = neuron.sigmoid(activation)
        outputs = np.zeros(len(self.output_layer))
        for neuron in self.output_layer:
            activation = neuron.weight_fun(self.hidden_outputs)
            self.weight_value[neuron.index] = activation
            outputs[neuron.index] = neuron.sigmoid(activation)
        return outputs

    def predict(self, row):
        outputs = self.forward_prop(row)
        return outputs

    def train_network(self, training_data, n_epoch, epsilon):
        data = training_data.copy()
        error_history = np.zeros(n_epoch)
        epoch = 0
        error_history[-1] = epsilon + 1
        while epoch < n_epoch and error_history[epoch - 1] > epsilon:
            sum_error = 0
            np.random.shuffle(data)
            for row in data:
                outputs = self.forward_prop(row[:self.num_of_inputs])
                sum_error += np.sum(np.square(outputs[:] - row[self.num_of_inputs:]))
                for neuron in self.output_layer:
                    neuron.backward_propagation_error(self.weight_value[neuron.index],
                                                      row[self.num_of_inputs + neuron.index])
                for neuron in self.hidden_layer:
                    neuron.backward_propagation_error(neuron.weight_fun(row))
                for neuron in self.hidden_layer:
                    neuron.update(row)
                for neuron in self.output_layer:
                    neuron.update(self.hidden_outputs)
            sum_error /= 2 * len(data)
            #if epoch % 1000 == 0:
            print('>epoch=%d, error=%.3f' % (epoch + 1, sum_error))
            error_history[epoch] = sum_error
            epoch += 1
        return error_history[:epoch], epoch
