import numpy as np


class Nn:
    def __init__(self, input_size, hidden_sizes, output_size, weights, biases):
        self.score = 0
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        if weights is None and biases is None:
            self.weights = []
            self.biases = []
            # Initialize weights and biases for each layer
            prev_size = input_size
            for size in hidden_sizes:
                layer_weights = np.random.randn(prev_size, size)
                layer_biases = np.random.randn(size)
                self.weights.append(layer_weights)
                self.biases.append(layer_biases)
                prev_size = size
            final_layer_weights = np.random.randn(prev_size, output_size)
            final_layer_biases = np.random.randn(output_size)
            self.weights.append(final_layer_weights)
            self.biases.append(final_layer_biases)
        else:
            self.weights = weights
            self.biases = biases

    def copy(self):
        return Nn(self.input_size, self.hidden_sizes, self.output_size, self.weights, self.biases)

    def forward(self, x):
        input_data = x
        for i in range(len(self.weights)):
            layer_weights = self.weights[i]
            layer_biases = self.biases[i]
            layer_output = np.dot(input_data, layer_weights) + layer_biases
            layer_output = self.sigmoid(layer_output)
            input_data = layer_output
        return input_data

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def evaluate_fitness(self, x, y):
        score = 0
        for i in range(len(x)):
            output = self.forward(x[i])
            if output >= 0.5:
                output = 1
            else:
                output = 0
            if output == y[i]:
                score += 1
        self.score = score
        return score

    def test_accuracy(self, test, test_label):
        # calculate accuracy based on test data
        score = 0
        for i in range(len(test)):
            output = self.forward(test[i])
            if output >= 0.5:
                output = 1
            else:
                output = 0
            if output == test_label[i]:
                score += 1
        accuracy = score / len(test)
        return accuracy
