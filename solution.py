import numpy as np


class Nn:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.bias1 = np.random.randn(self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias2 = np.random.randn(self.output_size)
